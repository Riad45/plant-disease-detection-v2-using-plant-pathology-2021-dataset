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
    """
    Calculate class weights from training data distribution
    
    Args:
        method: 'inverse_freq' or 'effective_samples'
        
    Returns:
        weights: Tensor of class weights
    """
    # Load class distribution from statistics
    stats_path = config.PROCESSED_DATA_DIR / 'statistics.json'
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    class_dist = stats['class_distribution']
    num_classes = stats['num_classes']
    
    # Get counts in correct order (by class index)
    class_names = stats['class_names']
    counts = np.array([class_dist[name] for name in class_names])
    
    if method == 'inverse_freq':
        # Inverse frequency: weight = 1 / count
        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes  # Normalize
        
    elif method == 'effective_samples':
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\n{'='*70}")
    print(f"Class Weights Calculated (method={method})")
    print(f"{'='*70}")
    print(f"{'Class':<15} {'Count':>8} {'Weight':>10}")
    print(f"{'-'*35}")
    for i, (name, count, weight) in enumerate(zip(class_names, counts, weights)):
        print(f"{name:<15} {count:>8} {weight:>10.4f}")
    print(f"{'='*70}\n")
    
    return torch.FloatTensor(weights)


def get_loss_function(use_weighted=True, label_smoothing=0.1):
    """
    Get loss function for training
    
    Args:
        use_weighted: Use class weights
        label_smoothing: Label smoothing factor
        
    Returns:
        criterion: Loss function
    """
    if use_weighted:
        weights = calculate_class_weights(method='inverse_freq')
        weights = weights.to(config.DEVICE)
        
        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=label_smoothing
        )
        print(f"✅ Using Weighted CrossEntropyLoss with label_smoothing={label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"✅ Using CrossEntropyLoss with label_smoothing={label_smoothing}")
    
    return criterion


if __name__ == "__main__":
    # Test loss function
    print("\nTesting loss function...")
    criterion = get_loss_function(use_weighted=True, label_smoothing=0.1)
    print(f"✅ Loss function created: {criterion}")