import torch
from torchmetrics.classification import JaccardIndex
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import interpolate
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from model import CustomSegmentationModel
import time
import os

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

def target_transform(target):
    target = F.resize(target, (224, 224), interpolation=F.InterpolationMode.NEAREST)
    target = F.pil_to_tensor(target).squeeze(0).long()
    target[target == 255] = 21  # Map ignored labels to 21
    return target

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    jaccard = JaccardIndex(task="multiclass", num_classes=21, ignore_index=21).to(device)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        outputs = interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Only compute loss on non-ignored pixels
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = outputs.argmax(1)
        jaccard.update(preds, targets)
        
        pbar.set_postfix({'loss': loss.item()})
    
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

class SegmentationTransform:
    def __init__(self, image_transform, target_transform):
        self.image_transform = image_transform
        self.target_transform = target_transform
        
    def __call__(self, image, target):
        seed = torch.randint(0, 2**32, (1,))[0].item()
        torch.manual_seed(seed)
        image = self.image_transform(image)
        torch.manual_seed(seed)
        target = self.target_transform(target)
        return image, target

def main():
    # Training hyperparameters
    num_epochs = 300
    learning_rate = 1e-3
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = CustomSegmentationModel(num_classes=21)
    model = model.to(device)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(
            degrees=(-10, 10),
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=(-5, 5)
        ),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    combined_transform = SegmentationTransform(transform, target_transform)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize datasets and dataloaders
    train_set = VOCSegmentation(root="./data", year="2012", image_set="train", 
                               transforms=combined_transform)
    val_set = VOCSegmentation(root="./data", year="2012", image_set="val",
                             transform=val_transform, target_transform=target_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Setup training
    criterion = torch.nn.CrossEntropyLoss(ignore_index=21)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-6)
    
    # Training loop
    best_val_iou = 0.0
    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}

    early_stopping = EarlyStopping()

    start_time = time.time()

    checkpoint_dir = 'kd-none'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, 
                                              optimizer, device, epoch+1)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        early_stopping(val_iou)
        if early_stopping.should_stop:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

        scheduler.step(val_loss)
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Train IoU={train_iou:.4f}, Val IoU={val_iou:.4f}, lr={scheduler.optimizer.param_groups[0]['lr']}")


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
        f.write(f"Best validation IoU: {best_val_iou:.4f}\n")
        f.write(f"Total training time: {total_training_time:.2f} seconds\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch+1}: Train Loss={history['train_loss'][epoch]:.4f}, Val Loss={history['val_loss'][epoch]:.4f}, "
                    f"Train IoU={history['train_iou'][epoch]:.4f}, Val IoU={history['val_iou'][epoch]:.4f}\n")

if __name__ == "__main__":
    main()
