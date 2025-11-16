"""
Loss functions with class weighting for imbalanced datasets
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


def calculate_class_weights(method='inverse_freq'):
    """Calculate class weights from training data distribution"""
    stats_path = config.PROCESSED_DATA_DIR / 'statistics.json'
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    class_dist = stats['class_distribution']
    num_classes = stats['num_classes']
    class_names = stats['class_names']
    
    # Get counts in correct order (match label encoder)
    counts = np.array([class_dist[name] for name in class_names], dtype=np.float32)
    
    # Calculate inverse frequency weights
    weights = 1.0 / (counts + 1e-6)  # Add epsilon to avoid division by zero
    weights = weights / weights.sum() * num_classes  # Normalize
    
    # Convert to tensor
    weights_tensor = torch.FloatTensor(weights)
    
    print(f"\n{'='*70}")
    print(f"Class Weights Calculated (method={method})")
    print(f"{'='*70}")
    print(f"{'Class':<15} {'Count':>8} {'Weight':>10}")
    print(f"{'-'*35}")
    for i, (name, count, weight) in enumerate(zip(class_names, counts, weights)):
        print(f"{name:<15} {int(count):>8} {weight:>10.4f}")
    print(f"{'='*70}\n")
    
    return weights_tensor


def get_loss_function(use_weighted=True, label_smoothing=0.1):
    """Get loss function for training"""
    
    if use_weighted:
        weights = calculate_class_weights(method='inverse_freq')
        
        # Move to GPU
        weights = weights.to(config.DEVICE)
        
        # Validate shape
        assert len(weights) == config.NUM_CLASSES, \
            f"Weight shape mismatch: {len(weights)} != {config.NUM_CLASSES}"
        
        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        
        print(f"✅ Using Weighted CrossEntropyLoss")
        print(f"   Label smoothing: {label_smoothing}")
        print(f"   Num classes: {config.NUM_CLASSES}")
        print(f"   Weights on device: {weights.device}\n")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"✅ Using CrossEntropyLoss (label_smoothing={label_smoothing})\n")
    
    return criterion