"""
Metrics calculation for multi-class classification
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score
)


class MetricsCalculator:
    """Calculate and store metrics for classification"""
    
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, predictions, labels, probabilities=None):
        """
        Update with batch predictions
        
        Args:
            predictions: Predicted class indices (batch_size,)
            labels: True class indices (batch_size,)
            probabilities: Class probabilities (batch_size, num_classes) - optional
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.cpu().numpy()
        
        self.all_preds.extend(predictions)
        self.all_labels.extend(labels)
        if probabilities is not None:
            self.all_probs.extend(probabilities)
    
    def compute(self, average='macro'):
        """
        Compute all metrics
        
        Args:
            average: 'macro', 'micro', or 'weighted'
            
        Returns:
            dict: Dictionary of metrics
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # Overall accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Balanced accuracy (for imbalanced datasets)
        balanced_acc = balanced_accuracy_score(labels, preds)
        
        # Precision, Recall, F1 (macro, micro, weighted)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=average, zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            f'{average}_precision': precision,
            f'{average}_recall': recall,
            f'{average}_f1': f1,
            'per_class': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1': per_class_f1.tolist(),
                'support': per_class_support.tolist(),
            },
            'confusion_matrix': cm.tolist(),
        }
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        return confusion_matrix(labels, preds)
    
    def print_metrics(self, metrics, phase="Validation"):
        """Pretty print metrics"""
        print(f"\n{'='*70}")
        print(f"{phase} Metrics")
        print(f"{'='*70}")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
        print(f"Macro F1:          {metrics['macro_f1']:.4f}")
        print(f"{'='*70}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"{'-'*60}")
        for i, class_name in enumerate(self.class_names):
            p = metrics['per_class']['precision'][i]
            r = metrics['per_class']['recall'][i]
            f = metrics['per_class']['f1'][i]
            s = metrics['per_class']['support'][i]
            print(f"{class_name:<15} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test metrics calculator
    print("\nTesting MetricsCalculator...")
    
    num_classes = 12
    calculator = MetricsCalculator(num_classes)
    
    # Simulate some predictions
    fake_preds = torch.randint(0, num_classes, (100,))
    fake_labels = torch.randint(0, num_classes, (100,))
    
    calculator.update(fake_preds, fake_labels)
    metrics = calculator.compute(average='macro')
    calculator.print_metrics(metrics)
    
    print("✅ MetricsCalculator test passed!")