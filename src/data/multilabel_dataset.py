"""
Multi-Label Dataset - GLOBAL CACHE FIX
Matches working multi-class pattern with caching
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
import json

# Fix Windows multiprocessing
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


# ✅ GLOBAL DATASET CACHE - Loaded once, shared by all workers
_GLOBAL_HF_DATASET = None

def _load_hf_dataset():
    """Load HuggingFace dataset once and cache globally"""
    global _GLOBAL_HF_DATASET
    if _GLOBAL_HF_DATASET is None:
        print(f"📥 Loading HuggingFace dataset (one-time cache)...")
        dataset_dict = load_dataset(config.DATASET_NAME)
        _GLOBAL_HF_DATASET = dataset_dict['train']
        print(f"✅ Dataset cached: {len(_GLOBAL_HF_DATASET)} images")
    return _GLOBAL_HF_DATASET


class PlantPathologyMultiLabelDataset(Dataset):
    """Multi-label dataset - EXACTLY matches multi-class pattern"""
    
    def __init__(self, csv_file, split='train', img_size=224, augment=True):
        self.df = pd.read_csv(csv_file)
        self.split = split
        self.img_size = img_size
        self.augment = bool(augment and (split == 'train'))
        
        # ✅ Use cached dataset instead of loading each time
        self.hf_dataset = _load_hf_dataset()
        
        self.transform = self._get_transforms()
        
        print(f"✅ {split} dataset ready: {len(self.df)} samples")

    def _get_transforms(self):
        """Create image transformations - EXACT COPY from multi-class"""
        if self.augment:
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
            return T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get one sample - OPTIMIZED"""
        row = self.df.iloc[idx]
        sample_id = int(row['sample_id'])
        
        # Multi-hot labels
        multihot_labels = row['multihot_labels']
        if isinstance(multihot_labels, str):
            multihot_labels = eval(multihot_labels)
        
        # Load image from cached dataset
        try:
            image = self.hf_dataset[sample_id]['image']
        except Exception as e:
            print(f"❌ Error loading image {sample_id}: {e}")
            image = Image.new('RGB', (224, 224), color='white')

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(multihot_labels, dtype=torch.float32)


def get_multilabel_dataloaders(model_name='convnext_tiny', batch_size=None, num_workers=None):
    """Create dataloaders - EXACT PATTERN from working multi-class"""
    
    # Set multiprocessing method for Windows
    if sys.platform == 'win32':
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Get settings
    img_size = config.IMAGE_SIZES.get(model_name, 224)
    batch_size = batch_size or config.BATCH_SIZES.get(model_name, 32)
    num_workers = num_workers or config.NUM_WORKERS

    print(f"\n{'='*70}")
    print(f"Creating MULTI-LABEL DataLoaders for {model_name} - FIXED VERSION")
    print(f"{'='*70}")
    print(f"Image size: {img_size}×{img_size}")
    print(f"Batch size: {batch_size} ⭐ (increased for performance)")
    print(f"Num workers: {num_workers} ⭐ (was 0)")
    print(f"Prefetch factor: {config.PREFETCH_FACTOR}")
    print(f"Persistent workers: {config.PERSISTENT_WORKERS}")

    # ✅ PRE-LOAD DATASET (critical!)
    _load_hf_dataset()

    # Create datasets
    train_dataset = PlantPathologyMultiLabelDataset(
        csv_file=config.PROCESSED_DATA_DIR / 'train_multilabel.csv',
        split='train',
        img_size=img_size,
        augment=True
    )

    val_dataset = PlantPathologyMultiLabelDataset(
        csv_file=config.PROCESSED_DATA_DIR / 'val_multilabel.csv',
        split='val', 
        img_size=img_size,
        augment=False
    )

    test_dataset = PlantPathologyMultiLabelDataset(
        csv_file=config.PROCESSED_DATA_DIR / 'test_multilabel.csv',
        split='test',
        img_size=img_size,
        augment=False
    )

    # Create dataloaders - EXACT PATTERN from multi-class
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if num_workers > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if num_workers > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if num_workers > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None
    )

    # Load disease info
    with open(config.PROCESSED_DATA_DIR / 'multilabel_info.json', 'r') as f:
        disease_info = json.load(f)
    
    base_diseases = disease_info['base_diseases']

    print(f"\n✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print(f"   Classes: {len(base_diseases)}")
    print(f"   Expected GPU utilization: 70-95% ⭐")
    print(f"   Expected epoch time: 2-5 minutes ⭐")

    return train_loader, val_loader, test_loader, base_diseases


if __name__ == "__main__":
    print("🧪 Testing dataset with performance fixes...")
    train_loader, val_loader, test_loader, class_names = get_multilabel_dataloaders()
    
    images, labels = next(iter(train_loader))
    print(f"✅ Test passed! Batch shape: {images.shape}")