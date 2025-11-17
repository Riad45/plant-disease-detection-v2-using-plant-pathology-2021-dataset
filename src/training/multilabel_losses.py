"""
Multi-Label Loss Functions - STABLE VERSION
Fixed numerical stability issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class FocalLoss(nn.Module):
    """
    Focal Loss - FIXED for numerical stability
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Store alpha as buffer (not parameter)
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (batch_size, num_classes)
            targets: binary labels (batch_size, num_classes)
        """
        # Get probabilities (numerically stable)
        p = torch.sigmoid(inputs)
        
        # Clamp for numerical stability
        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        
        # Calculate focal loss components
        ce_loss = F.binary_cross_entropy(p, targets, reduction='none')
        
        # p_t: probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Alpha weighting per class
            alpha_t = self.alpha.unsqueeze(0) * targets + (1 - self.alpha.unsqueeze(0)) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy - SIMPLE AND STABLE"""
    
    def __init__(self, pos_weights=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
        # Store pos_weights as buffer
        if pos_weights is not None:
            self.register_buffer('pos_weights', pos_weights)
        else:
            self.pos_weights = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (batch_size, num_classes)
            targets: binary labels (batch_size, num_classes)
        """
        # Use PyTorch's built-in weighted BCE with logits (most stable)
        if self.pos_weights is not None:
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets, 
                pos_weight=self.pos_weights,
                reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets,
                reduction=self.reduction
            )
        
        return loss


def calculate_multilabel_weights(method='balanced'):
    """Calculate weights for multi-label class imbalance"""
    
    # Load disease info
    info_path = config.PROCESSED_DATA_DIR / 'multilabel_info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    total_samples = info['total_samples']
    disease_counts = info['disease_counts']
    base_diseases = info['base_diseases']
    
    pos_weights = []
    
    print(f"\n📊 Multi-label Class Weights Calculation ({method}):")
    print(f"{'Disease':<25} {'Pos Samples':>12} {'Neg Samples':>12} {'Pos Weight':>12}")
    print("-" * 70)
    
    for disease in base_diseases:
        pos_count = disease_counts[disease]
        neg_count = total_samples - pos_count
        
        if method == 'balanced':
            # PyTorch convention: neg_count / pos_count
            pos_weight = neg_count / (pos_count + 1e-5)
        elif method == 'sqrt_balanced':
            # Softer weighting
            pos_weight = np.sqrt(neg_count / (pos_count + 1e-5))
        else:
            pos_weight = 1.0
        
        pos_weights.append(pos_weight)
        
        print(f"{disease:<25} {pos_count:>12} {neg_count:>12} {pos_weight:>12.2f}")
    
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
    return pos_weights


def get_multilabel_loss(loss_type='weighted_bce', gamma=2.0):
    """
    Get loss function for multi-label classification
    
    Args:
        loss_type: 'weighted_bce' (stable) or 'focal' (experimental)
        gamma: Focal loss gamma parameter
    """
    
    pos_weights = calculate_multilabel_weights(method='balanced')
    pos_weights = pos_weights.to(config.DEVICE)
    
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=pos_weights, gamma=gamma)
        print(f"\n🎯 USING FOCAL LOSS (gamma={gamma})")
        print(f"   ⚠️ Experimental - watch for NaN loss values")
        
    elif loss_type == 'weighted_bce':
        criterion = WeightedBCELoss(pos_weights=pos_weights)
        print(f"\n🎯 USING WEIGHTED BCE LOSS")
        print(f"   ✅ Most stable for multi-label classification")
        
    else:  # standard_bce
        criterion = nn.BCEWithLogitsLoss()
        print(f"\n🎯 USING STANDARD BCE LOSS")
        print(f"   No class weighting")
    
    return criterion


# Test
if __name__ == "__main__":
    print("\nTesting Multi-Label Loss Functions...")
    
    # Fake data
    batch_size, num_diseases = 4, 6
    logits = torch.randn(batch_size, num_diseases)
    targets = torch.randint(0, 2, (batch_size, num_diseases)).float()
    
    # Test losses
    print("\n1. Weighted BCE:")
    criterion = WeightedBCELoss(pos_weights=torch.ones(num_diseases) * 2.0)
    loss = criterion(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n2. Focal Loss:")
    criterion = FocalLoss(alpha=torch.ones(num_diseases) * 0.25, gamma=2.0)
    loss = criterion(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✅ Loss functions test passed!")