"""
Model Factory - Load pretrained models from timm
Supports: ConvNeXt-Tiny, Swin-Tiny, DeiT-Small
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


def create_model(model_name, num_classes=None, pretrained=True):
    """
    Create model from timm with custom classifier head
    
    Args:
        model_name: 'convnext_tiny', 'swin_tiny', or 'deit_small'
        num_classes: Number of output classes (default: from config)
        pretrained: Load pretrained weights
        
    Returns:
        model: PyTorch model ready for training
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    # Get model config
    model_config = config.MODEL_CONFIGS.get(model_name)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(config.MODEL_CONFIGS.keys())}")
    
    timm_name = model_config['timm_name']
    drop_rate = model_config.get('drop_rate', 0.0)
    drop_path_rate = model_config.get('drop_path_rate', 0.0)
    
    print(f"\n{'='*70}")
    print(f"Creating Model: {model_name}")
    print(f"{'='*70}")
    print(f"TIMM name: {timm_name}")
    print(f"Pretrained: {pretrained}")
    print(f"Num classes: {num_classes}")
    print(f"Drop rate: {drop_rate}")
    print(f"Drop path rate: {drop_path_rate}")
    
    # Create model from timm
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"{'='*70}\n")
    
    return model


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model from checkpoint
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        model: Model with loaded weights
        checkpoint: Full checkpoint dict (for resume training)
    """
    print(f"\n📥 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")
    
    return model, checkpoint


if __name__ == "__main__":
    # Test model creation
    from src.config import config
    config.update_num_classes(12, None)
    
    print("\n" + "="*70)
    print("TESTING MODEL FACTORY")
    print("="*70)
    
    for model_name in ['convnext_tiny', 'swin_tiny', 'deit_small']:
        model = create_model(model_name, num_classes=12, pretrained=True)
        print(f"✅ {model_name} created successfully!\n")