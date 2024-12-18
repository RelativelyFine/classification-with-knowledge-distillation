import torch
from torchmetrics.classification import JaccardIndex
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import functional as F
from torch.nn.functional import interpolate

import matplotlib.pyplot as plt
import numpy as np
import os

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

    # Overlay the mask
    plt.imshow(mask, cmap=cmap, alpha=alpha, interpolation='nearest')

    # Remove axis for better visualization
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()

def main():
    # Load pretrained FCN-ResNet50 model
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

    # Define transforms for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Fix target transformation
    def target_transform(target):
        target = F.resize(target, (224, 224), interpolation=F.InterpolationMode.NEAREST)
        target = F.pil_to_tensor(target).squeeze(0).long()  # Convert to long tensor and squeeze channel dimension
        target[target == 255] = num_classes  # Optionally, remap ignored label (e.g., 255 to num_classes)
        return target

    # Initialize the dataset
    voc_test_set = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="val",
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    # Create DataLoader
    test_loader = DataLoader(voc_test_set, batch_size=32, shuffle=False, num_workers=0)

    # Initialize the Jaccard Index metric
    num_classes = 21  # VOC has 20 classes + background
    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=num_classes)  # Ignored pixels mapped to num_classes

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    model = model.eval().to(device)
    jaccard = jaccard.to(device)

    # Evaluate the model
    print("Evaluating the model...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)  # Targets already converted to long in target_transform

            outputs = model(images)['out']
            preds = interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False).argmax(1)

            # Print unique values in predictions and targets for debugging
            # print(f"Pred: {preds[0]}")
            # print(f"Target: {targets[0]}")

            # Update the Jaccard Index
            jaccard.update(preds, targets)

    # Compute the mean IoU
    miou = jaccard.compute().item()
    print(f"Mean IoU: {miou:.4f}")

    # Create a directory for saving output images
    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Counter for saved images
    saved_count = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)['out']
            preds = outputs.argmax(1)

            # Save a few sample images with masks
            for i in range(min(5, images.size(0))):  # Save up to 5 images from each batch
                image = images[i]
                pred_mask = preds[i]
                true_mask = targets[i]

                save_image_with_mask(
                    image=image,
                    mask=pred_mask,
                    file_path=os.path.join(output_dir, f"pred_{saved_count}.png")
                )

                save_image_with_mask(
                    image=image,
                    mask=true_mask,
                    file_path=os.path.join(output_dir, f"true_{saved_count}.png")
                )

                saved_count += 1

    print(f"Sample images saved in {output_dir}")

if __name__ == "__main__":
    main()
  
