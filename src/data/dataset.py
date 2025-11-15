"""
PyTorch Dataset class for Plant Pathology
Loads actual images from HuggingFace dataset based on CSV splits
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image
from datasets import load_dataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class PlantPathologyDataset(Dataset):
    """
    Custom Dataset for Plant Pathology
    Loads images from HuggingFace dataset based on sample IDs in CSV
    """
    
    def __init__(self, csv_file, split='train', img_size=224, augment=True):
        """
        Args:
            csv_file: Path to CSV file (train.csv, val.csv, or test.csv)
            split: 'train', 'val', or 'test'
            img_size: Size to resize images to
            augment: Whether to apply data augmentation (only for training)
        """
        self.df = pd.read_csv(csv_file)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # Load HuggingFace dataset (from cache - instant!)
        print(f"Loading HuggingFace dataset for {split} split...")
        dataset_dict = load_dataset(config.DATASET_NAME)
        self.hf_dataset = dataset_dict['train']  # We use train split only
        
        # Create transforms
        self.transform = self._get_transforms()
        
        print(f"✅ {split} dataset ready: {len(self.df)} samples")
    
    def _get_transforms(self):
        """Create image transformations"""
        if self.augment:
            # Training augmentations
            return T.Compose([
                T.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=30),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/Test - no augmentation
            return T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get one sample
        Returns: (image_tensor, label)
        """
        # Get row from CSV
        row = self.df.iloc[idx]
        sample_id = int(row['sample_id'])
        label = int(row['label_encoded'])
        
        # Load ACTUAL IMAGE from HuggingFace dataset
        image = self.hf_dataset[sample_id]['image']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(model_name='convnext_tiny', batch_size=None, num_workers=None):
    """
    Create train, validation, and test dataloaders
    
    Args:
        model_name: Name of model (to get image size)
        batch_size: Batch size (if None, uses config)
        num_workers: Number of workers (if None, uses config)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Get settings
    img_size = config.IMAGE_SIZES.get(model_name, 224)
    batch_size = batch_size or config.BATCH_SIZES.get(model_name, 16)
    num_workers = num_workers or config.NUM_WORKERS
    
    print(f"\n{'='*70}")
    print(f"Creating DataLoaders for {model_name}")
    print(f"{'='*70}")
    print(f"Image size: {img_size}×{img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    
    # Create datasets
    train_dataset = PlantPathologyDataset(
        csv_file=config.PROCESSED_DATA_DIR / 'train.csv',
        split='train',
        img_size=img_size,
        augment=True
    )
    
    val_dataset = PlantPathologyDataset(
        csv_file=config.PROCESSED_DATA_DIR / 'val.csv',
        split='val',
        img_size=img_size,
        augment=False
    )
    
    test_dataset = PlantPathologyDataset(
        csv_file=config.PROCESSED_DATA_DIR / 'test.csv',
        split='test',
        img_size=img_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    # Load class names
    import json
    with open(config.PROCESSED_DATA_DIR / 'class_names.json', 'r') as f:
        class_names = json.load(f)
    
    print(f"\n✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Classes: {len(class_names)}")
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset loading...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    
    # Get one batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Test successful!")
    print(f"   Batch image shape: {images.shape}")
    print(f"   Batch labels shape: {labels.shape}")
    print(f"   Image dtype: {images.dtype}")
    print(f"   Label dtype: {labels.dtype}")