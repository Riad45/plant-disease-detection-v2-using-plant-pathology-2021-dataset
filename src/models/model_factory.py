"""
Model Factory - SIMPLIFIED AND CLEAN
Works for both multi-class and multi-label
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


def create_multilabel_model(model_name='convnext_tiny', num_diseases=6, pretrained=True):
    """
    Create a multi-label classification model
    
    Args:
        model_name: Model architecture (convnext_tiny, swin_tiny, deit_small)
        num_diseases: Number of output classes (6 for multi-label)
        pretrained: Use pretrained weights
    """
    print(f"\n{'='*70}")
    print(f"Creating MULTI-LABEL Model: {model_name}")
    print(f"{'='*70}")
    
    # Get model config
    if model_name not in config.MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = config.MODEL_CONFIGS[model_name]
    timm_name = model_config['timm_name']
    
    # Create model with TIMM (let it handle the head)
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_diseases,  # TIMM handles everything
        drop_rate=model_config.get('drop_rate', 0.0),
        drop_path_rate=model_config.get('drop_path_rate', 0.0)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    disease_names = ['healthy', 'scab', 'frog_eye_leaf_spot', 'complex', 'rust', 'powdery_mildew']
    
    print(f"✅ Multi-label model created:")
    print(f"   Base model: {timm_name}")
    print(f"   Output: {num_diseases} binary classifiers")
    print(f"   Diseases: {', '.join(disease_names)}")
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model_size_mb:.2f} MB (FP32)")
    print(f"{'='*70}\n")
    
    return model


# Test
if __name__ == "__main__":
    print("Testing Multi-Label Model Creation...")
    
    model = create_multilabel_model('convnext_tiny', num_diseases=6, pretrained=False)
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\n✅ Forward pass test:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: torch.Size([2, 6])")
    
    assert output.shape == torch.Size([2, 6]), f"Wrong output shape! Got {output.shape}"
    print(f"\n✅ Model creation test passed!")