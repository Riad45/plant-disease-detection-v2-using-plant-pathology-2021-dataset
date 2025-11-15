"""
Test script to verify DataLoader works correctly
Displays sample images and their labels
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import get_dataloaders
from src.config import config


def denormalize(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def test_dataloader():
    print("="*70)
    print("TESTING DATALOADER")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        model_name='convnext_tiny'
    )
    
    # Test loading a batch
    print("\n🧪 Loading a batch from train set...")
    images, labels = next(iter(train_loader))
    
    print(f"\n✅ Batch loaded successfully!")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Images dtype: {images.dtype}")
    print(f"   Labels dtype: {labels.dtype}")
    print(f"   Images min/max: {images.min():.3f} / {images.max():.3f}")
    
    # Visualize some images
    print("\n🖼️ Visualizing sample images...")
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(min(8, len(images))):
        # Denormalize image
        img = denormalize(images[i])
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # Get label
        label_idx = labels[i].item()
        label_name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
        
        # Display
        axes[i].imshow(img_np)
        axes[i].set_title(f'Label: {label_name}\nClass: {label_idx}', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from Training Set', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    save_path = config.RESULTS_DIR / 'plots' / 'sample_training_images.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sample images saved to: {save_path}")
    
    plt.show()
    
    # Test all splits
    print("\n🧪 Testing all splits...")
    
    print(f"\nTrain loader:")
    images, labels = next(iter(train_loader))
    print(f"   ✅ Shape: {images.shape}, Labels: {labels.shape}")
    
    print(f"\nValidation loader:")
    images, labels = next(iter(val_loader))
    print(f"   ✅ Shape: {images.shape}, Labels: {labels.shape}")
    
    print(f"\nTest loader:")
    images, labels = next(iter(test_loader))
    print(f"   ✅ Shape: {images.shape}, Labels: {labels.shape}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n🚀 DataLoader is working correctly!")
    print("🚀 Ready to start training models!")


if __name__ == "__main__":
    test_dataloader()