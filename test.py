import torch
from model import CustomSegmentationModel
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F_nn
from torchvision.transforms import functional as F_transform
from torchmetrics.classification import JaccardIndex
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

VOC_LABELS = [
    "Background", "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus",
    "Car", "Cat", "Chair", "Cow", "Dining Table", "Dog", "Horse", "Motorbike",
    "Person", "Potted Plant", "Sheep", "Sofa", "Train", "TV/Monitor"
]

def create_color_map(num_classes):
    np.random.seed(42)
    color_map = np.random.rand(num_classes, 3)
    color_map[0] = [0, 0, 0]  # Set background to black
    return color_map

def compute_metrics(model, val_loader, device):
    model.eval()
    jaccard = JaccardIndex(task="multiclass", num_classes=21, ignore_index=21).to(device)
    class_jaccard = {i: JaccardIndex(task="multiclass", num_classes=21, ignore_index=21).to(device) 
                     for i in range(21)}
    
    print("Computing metrics...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)['out']
            outputs = F_nn.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
            preds = outputs.argmax(1)
            
            # Update global IoU
            jaccard.update(preds, targets)
            
            # Update per-class IoU
            for class_idx in range(21):
                class_jaccard[class_idx].update(preds, targets)
    
    # Compute final metrics
    mean_iou = jaccard.compute().item()
    class_ious = {i: class_jaccard[i].compute().item() for i in range(21)}
    
    return mean_iou, class_ious

def visualize_segmentation(image, pred_mask, true_mask, save_path):
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    # Ensure image is in correct format (H, W, C)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # Normalize image for display
    image = (image - image.min()) / (image.max() - image.min())

    # Create color map
    color_map = create_color_map(21)
    
    # Handle ignored labels (255 or 21) by masking them
    mask = true_mask != 21
    true_mask = true_mask * mask  # Zero out ignored labels
    
    # Create colored masks
    pred_colored = color_map[pred_mask]
    true_colored = color_map[true_mask]

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot prediction
    ax2.imshow(pred_colored)
    ax2.set_title('Prediction')
    ax2.axis('off')

    # Plot ground truth
    ax3.imshow(true_colored)
    ax3.set_title('Ground Truth')
    ax3.axis('off')

    # Add color bar with labels
    unique_classes = np.unique(np.concatenate([pred_mask, true_mask]))
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[i]) 
                      for i in unique_classes]
    fig.legend(legend_elements, 
              [VOC_LABELS[i] for i in unique_classes],
              loc='center right',
              bbox_to_anchor=(0.98, 0.5))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def target_transform(target):
    target = F_transform.resize(target, (224, 224), interpolation=F_transform.InterpolationMode.NEAREST)
    target = F_transform.pil_to_tensor(target).squeeze(0).long()
    target[target == 255] = 21
    return target

def main(weights_path):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model architecture
    model = CustomSegmentationModel(num_classes=21)
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)

    # Load state dict and move model to device
    try:
        model.load_state_dict(checkpoint)
    except:
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Initialize IoU metrics
    jaccard = JaccardIndex(task="multiclass", num_classes=21, ignore_index=21).to(device)
    class_jaccard = {i: JaccardIndex(task="multiclass", num_classes=21, ignore_index=21).to(device) 
                     for i in range(21)}

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load validation dataset
    val_set = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="val",
        download=False,
        transform=transform,
        target_transform=target_transform
    )

    # Create DataLoader
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Create output directory
    output_dir = args.weights.replace('.pth', '_eval')
    os.makedirs(output_dir, exist_ok=True)

    # Generate predictions, compute metrics, and create visualizations
    print("Generating visualizations and computing metrics...")
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            outputs = model(images)['out']
            outputs = F_nn.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
            preds = outputs.argmax(1)

            # Update IoU metrics
            jaccard.update(preds, targets)
            for class_idx in range(21):
                class_jaccard[class_idx].update(preds, targets)

            # Create visualization for first 10 images
            if idx < 10:
                # Move tensors back to CPU for visualization
                images_cpu = images.cpu()
                preds_cpu = preds.cpu()
                targets_cpu = targets.cpu()

                # Create visualization
                save_path = os.path.join(output_dir, f'visualization_{idx}.png')
                visualize_segmentation(
                    images_cpu[0],
                    preds_cpu[0],
                    targets_cpu[0],
                    save_path
                )
                print(f"Saved visualization {idx+1}/10")

    # Compute final metrics
    mean_iou = jaccard.compute().item()

    # Print and save metrics
    print("\nEvaluation Results:")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Mean IoU: {mean_iou:.4f}\n\n")

    print(f"\nResults saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation")
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True, 
        help="Path to the model weights file"
    )
    args = parser.parse_args()
    main(args.weights)