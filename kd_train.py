import torch
import argparse
from torchmetrics.classification import JaccardIndex
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import functional as F
from torch.nn.functional import interpolate
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from model import CustomSegmentationModel
from torchvision.models import segmentation
import torch.nn.functional as F_nn
import torch.nn as nn
import time

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_iou = None
        self.should_stop = False

    def __call__(self, val_iou):
        if self.best_iou is None:
            self.best_iou = val_iou
        elif val_iou < self.best_iou + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_iou = val_iou
            self.counter = 0

class TeacherWrapper(nn.Module):
    def __init__(self, teacher_model, student_channels):
        super().__init__()
        self.teacher = teacher_model
        self.layer1 = self.teacher.backbone.layer1
        self.layer2 = self.teacher.backbone.layer2
        self.layer3 = self.teacher.backbone.layer3
        self.layer4 = self.teacher.backbone.layer4
        self.classifier = self.teacher.classifier
        
        # Define 1x1 conv layers for channel projection
        self.proj1 = nn.Conv2d(256, student_channels[0], kernel_size=1)
        self.proj2 = nn.Conv2d(512, student_channels[1], kernel_size=1)
        self.proj3 = nn.Conv2d(1024, student_channels[2], kernel_size=1)
        self.proj4 = nn.Conv2d(2048, student_channels[3], kernel_size=1)

    def forward(self, x):
        x = self.teacher.backbone.conv1(x)
        x = self.teacher.backbone.bn1(x)
        x = self.teacher.backbone.relu(x)
        x = self.teacher.backbone.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        # Project feature maps to match student channels
        f1_proj = self.proj1(f1)
        f2_proj = self.proj2(f2)
        f3_proj = self.proj3(f3)
        f4_proj = self.proj4(f4)
        
        # Obtain the output logits
        out = self.classifier(f4)
        
        return {'features': [f1_proj, f2_proj, f3_proj, f4_proj], 'out': out}

def save_image_with_mask(image, mask, file_path, alpha=0.5, cmap='jet'):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Ensure image is in (H, W, C) format
    if image.shape[0] == 3:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))

    # Normalize the image to [0, 1] range for display
    image = (image - image.min()) / (image.max() - image.min())

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation='bilinear')
    plt.imshow(mask, cmap=cmap, alpha=alpha, interpolation='nearest')
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()

def train_one_epoch(model, teacher_model, train_loader, criterion, optimizer, device, epoch, distil_type, alpha=1.0, beta=0.1, tau=1.0):
    model.train()
    teacher_model.eval()
    running_loss = 0.0
    jaccard = JaccardIndex(task="multiclass", num_classes=21, ignore_index=21).to(device)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Student forward pass
        outputs_s = model(images)
        logits_s = outputs_s['out']  # Shape: [N, C, H, W]
        features_s = outputs_s['features']
        
        # Teacher forward pass
        with torch.no_grad():
            outputs_t = teacher_model(images)
            logits_t = outputs_t['out']  # Shape: [N, C, H, W]
            features_t = outputs_t['features']
        
        # Resize logits to match target size
        logits_s = interpolate(logits_s, size=(224, 224), mode='bilinear', align_corners=False)
        logits_t = interpolate(logits_t, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Standard cross-entropy loss
        loss_ce = criterion(logits_s, targets)

        # Initialize distillation losses
        loss_distill = 0.0
        loss_feature = 0.0

        # Response-based distillation
        if distil_type in ["response", "both"]:
            logits_s_flat = logits_s.permute(0, 2, 3, 1).reshape(-1, logits_s.shape[1])
            logits_t_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.shape[1])
            targets_flat = targets.view(-1)
            mask = targets_flat != 21  # Ignore invalid pixels
            logits_s_masked = logits_s_flat[mask]
            logits_t_masked = logits_t_flat[mask]
            
            log_probs = F_nn.log_softmax(logits_s_masked / tau, dim=1)
            soft_targets = F_nn.softmax(logits_t_masked / tau, dim=1)
            loss_distill = F_nn.kl_div(log_probs, soft_targets, reduction='batchmean') * (tau ** 2)
        
        # Feature-based distillation
        if distil_type in ["feature", "both"]:
            for f_s, f_t in zip(features_s, features_t):
                if f_s.shape != f_t.shape:
                    f_t = interpolate(f_t, size=f_s.shape[2:], mode='bilinear', align_corners=False)
                f_s_norm = F_nn.normalize(f_s, p=2, dim=1)
                f_t_norm = F_nn.normalize(f_t, p=2, dim=1)
                cos_sim = F_nn.cosine_similarity(f_s_norm, f_t_norm, dim=1)
                loss_feature += (1 - cos_sim.mean())
            loss_feature = loss_feature / len(features_s)

        # Total loss based on distillation type
        loss = alpha * loss_ce
        if distil_type in ["response", "both"]:
            loss += beta * loss_distill
        if distil_type in ["feature", "both"]:
            loss += beta * loss_feature
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        preds = logits_s.argmax(1)
        jaccard.update(preds, targets)
        
        pbar.set_postfix({
            'loss': loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_distill': loss_distill.item() if distil_type in ["response", "both"] else "N/A",
            'loss_feature': loss_feature.item() if distil_type in ["feature", "both"] else "N/A",
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_iou = jaccard.compute().item()
    return epoch_loss, epoch_iou

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    jaccard = JaccardIndex(task="multiclass", num_classes=21, ignore_index=21).to(device)
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)['out']
            outputs = interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            preds = outputs.argmax(1)
            jaccard.update(preds, targets)
    
    val_loss = val_loss / len(val_loader)
    val_iou = jaccard.compute().item()
    return val_loss, val_iou

def target_transform(target):
    target[target == 255] = 21  # Map ignored labels to 21
    return target

def get_student_channels(model):
    # Extract channel dimensions from model layers
    channels = [
        model.layer1[0].conv1.out_channels,
        model.layer2[0].conv1.out_channels,
        model.layer3[0].conv1.out_channels,
        model.layer4[0].conv1.out_channels
    ]
    return channels

class SegmentationTransform:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
    
    def __call__(self, image, target):
        # Random horizontal flip
        if random.random() < 0.5:
            image = F.hflip(image)
            target = F.hflip(target)
        
        # Random rotation
        angle = random.uniform(-10, 10)
        image = F.rotate(image, angle)
        target = F.rotate(target, angle, interpolation=F.InterpolationMode.NEAREST)
        
        # Random affine transformation
        translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-5, 5)
        image = F.affine(image, angle=0, translate=translate, scale=scale, shear=shear)
        target = F.affine(target, angle=0, translate=translate, scale=scale, shear=shear, interpolation=F.InterpolationMode.NEAREST)
        
        # Color jitter (only for image)
        image = self.color_jitter(image)
        
        # Resize
        image = F.resize(image, (224, 224))
        target = F.resize(target, (224, 224), interpolation=F.InterpolationMode.NEAREST)
        
        # Convert to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert target to tensor
        target = F.pil_to_tensor(target).squeeze(0).long()
        target[target == 255] = 21  # Map ignored labels to 21

        return image, target

def val_target_transform(target):
    target = F.resize(target, (224, 224), interpolation=F.InterpolationMode.NEAREST)
    target = F.pil_to_tensor(target).squeeze(0).long()
    target[target == 255] = 21  # Map ignored labels to 21
    return target

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--distil_type", type=str, choices=["response", "feature", "both"], default="both",
                        help="Type of distillation to apply: 'response', 'feature', or 'both'")
    args = parser.parse_args()

    distil_type = args.distil_type

    # Training hyperparameters
    num_epochs = 300
    learning_rate = 1e-3
    batch_size = 16
    alpha = 0.7
    beta = 0.3
    tau = 4  # Temperature for distillation
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load student model
    model = CustomSegmentationModel(num_classes=21)
    model = model.to(device)
    
    # Get student feature map channels
    student_channels = get_student_channels(model)
    
    # Load teacher model
    teacher_model_raw = segmentation.fcn_resnet50(weights='DEFAULT')
    teacher_model = TeacherWrapper(teacher_model_raw, student_channels).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Define transforms
    combined_transform = SegmentationTransform()

    DATA_DIR = "./data"
    DOWNLOAD = not os.path.exists(DATA_DIR)
    
    # Initialize datasets
    train_set = VOCSegmentation(
        root=DATA_DIR,
        year="2012",
        image_set="train",
        download=DOWNLOAD,
        transforms=combined_transform
    )
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_set = VOCSegmentation(
        root=DATA_DIR,
        year="2012",
        image_set="val",
        download=DOWNLOAD,
        transform=val_transform,
        target_transform=val_target_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=21)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True, min_lr=1e-7)
    
    
    # Create directory for checkpoints
    checkpoint_dir = f"./kd-{distil_type}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0.0

    early_stopping = EarlyStopping()

    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_iou = train_one_epoch(
            model, teacher_model, train_loader, criterion, optimizer, device, epoch+1, distil_type, alpha=alpha, beta=beta, tau=tau
        )
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        early_stopping(val_iou)
        if early_stopping.should_stop:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"lr: {scheduler.optimizer.param_groups[0]['lr']}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, checkpoint_path)
    
    end_time = time.time()
    total_training_time = end_time - start_time
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('IoU History')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{checkpoint_dir}/training_history.png')
    plt.close()

    # save training history in text file
    with open(f'{checkpoint_dir}/training_history.txt', 'w') as f:
        f.write(f"Best Validation IoU: {best_val_iou:.4f}\n")
        f.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch+1}: Train Loss={history['train_loss'][epoch]:.4f}, Val Loss={history['val_loss'][epoch]:.4f}, "
                    f"Train IoU={history['train_iou'][epoch]:.4f}, Val IoU={history['val_iou'][epoch]:.4f}\n")

if __name__ == "__main__":
    main()
