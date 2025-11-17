"""
Multi-label metrics calculation
Per-disease metrics + micro/macro averages + example-based metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, hamming_loss,
    jaccard_score, zero_one_loss
)
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class MultiLabelMetricsCalculator:
    """Calculate comprehensive metrics for multi-label classification"""
    
    def __init__(self, disease_names):
        self.disease_names = disease_names
        self.num_diseases = len(disease_names)
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_preds = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions, targets, probabilities=None):
        """
        Update with batch predictions
        
        Args:
            predictions: Binary predictions (batch_size, num_diseases)
            targets: Ground truth labels (batch_size, num_diseases)  
            probabilities: Prediction probabilities (batch_size, num_diseases) - optional
        """
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.cpu().numpy()
        
        self.all_preds.extend(predictions)
        self.all_targets.extend(targets)
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
    
    def compute(self, threshold=0.5):
        """Compute comprehensive multi-label metrics"""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        # Ensure binary predictions
        if preds.max() > 1 or preds.min() < 0:
            preds = (preds > threshold).astype(int)
        
        metrics = {
            'per_disease': {},
            'macro': {},
            'micro': {},
            'example_based': {},
            'confusion_matrices': {}
        }
        
        # Per-disease metrics
        disease_precisions = []
        disease_recalls = [] 
        disease_f1s = []
        disease_accuracies = []
        disease_supports = []
        
        for i, disease in enumerate(self.disease_names):
            disease_preds = preds[:, i]
            disease_targets = targets[:, i]
            
            # Avoid division by zero for metrics
            if len(np.unique(disease_targets)) > 1:
                precision = precision_score(disease_targets, disease_preds, zero_division=0)
                recall = recall_score(disease_targets, disease_preds, zero_division=0)
                f1 = f1_score(disease_targets, disease_preds, zero_division=0)
                accuracy = accuracy_score(disease_targets, disease_preds)
            else:
                precision = recall = f1 = accuracy = 0.0
            
            support = int(disease_targets.sum())
            
            metrics['per_disease'][disease] = {
                'precision': precision,
                'recall': recall, 
                'f1': f1,
                'accuracy': accuracy,
                'support': support
            }
            
            disease_precisions.append(precision)
            disease_recalls.append(recall)
            disease_f1s.append(f1)
            disease_accuracies.append(accuracy)
            disease_supports.append(support)
            
            # Per-disease confusion matrix
            cm = confusion_matrix(disease_targets, disease_preds)
            metrics['confusion_matrices'][disease] = cm.tolist()
        
        # Macro averages (average of per-disease metrics)
        metrics['macro']['precision'] = np.mean(disease_precisions)
        metrics['macro']['recall'] = np.mean(disease_recalls)
        metrics['macro']['f1'] = np.mean(disease_f1s)
        metrics['macro']['accuracy'] = np.mean(disease_accuracies)
        
        # Micro averages (pool all predictions across diseases)
        metrics['micro']['precision'] = precision_score(targets, preds, average='micro', zero_division=0)
        metrics['micro']['recall'] = recall_score(targets, preds, average='micro', zero_division=0)
        metrics['micro']['f1'] = f1_score(targets, preds, average='micro', zero_division=0)
        metrics['micro']['accuracy'] = accuracy_score(targets, preds)
        
        # Example-based metrics
        exact_match = (preds == targets).all(axis=1).mean()
        hamming_loss_val = hamming_loss(targets, preds)
        jaccard_micro = jaccard_score(targets, preds, average='micro', zero_division=0)
        jaccard_macro = jaccard_score(targets, preds, average='macro', zero_division=0)
        
        metrics['example_based']['exact_match_ratio'] = exact_match
        metrics['example_based']['hamming_loss'] = hamming_loss_val
        metrics['example_based']['jaccard_micro'] = jaccard_micro
        metrics['example_based']['jaccard_macro'] = jaccard_macro
        
        # Overall metrics
        metrics['overall'] = {
            'total_samples': len(preds),
            'avg_diseases_per_sample': targets.sum(axis=1).mean(),
            'disease_coverage': (targets.sum(axis=0) > 0).sum() / self.num_diseases
        }
        
        return metrics
    
    def get_detailed_classification_report(self, threshold=0.5):
        """Get detailed sklearn classification report"""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        if preds.max() > 1 or preds.min() < 0:
            preds = (preds > threshold).astype(int)
        
        report = classification_report(
            targets, preds, 
            target_names=self.disease_names,
            output_dict=True,
            zero_division=0
        )
        return report
    
    def print_metrics(self, metrics, phase="Validation"):
        """Print formatted multi-label metrics"""
        print(f"\n{'='*80}")
        print(f"{phase} - Multi-Label Metrics")
        print(f"{'='*80}")
        
        print(f"\n📊 Per-Disease Metrics:")
        print(f"{'Disease':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'Support':>10}")
        print("-" * 85)
        for disease in self.disease_names:
            m = metrics['per_disease'][disease]
            print(f"{disease:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['accuracy']:>10.4f} {m['support']:>10}")
        
        print(f"\n📈 Averages:")
        print(f"Macro Precision:  {metrics['macro']['precision']:.4f}")
        print(f"Macro Recall:     {metrics['macro']['recall']:.4f}")
        print(f"Macro F1:         {metrics['macro']['f1']:.4f}")
        print(f"Micro F1:         {metrics['micro']['f1']:.4f}")
        print(f"Micro Accuracy:   {metrics['micro']['accuracy']:.4f}")
        
        print(f"\n🎯 Example-Based Metrics:")
        print(f"Exact Match Ratio: {metrics['example_based']['exact_match_ratio']:.4f}")
        print(f"Hamming Loss:      {metrics['example_based']['hamming_loss']:.4f}")
        print(f"Jaccard Micro:     {metrics['example_based']['jaccard_micro']:.4f}")
        print(f"Jaccard Macro:     {metrics['example_based']['jaccard_macro']:.4f}")
        
        print(f"\n📋 Overall Statistics:")
        print(f"Total Samples:     {metrics['overall']['total_samples']}")
        print(f"Avg Diseases/Sample: {metrics['overall']['avg_diseases_per_sample']:.2f}")
        print(f"Disease Coverage:  {metrics['overall']['disease_coverage']:.2%}")
        
        print(f"{'='*80}\n")


# Test the metrics calculator
if __name__ == "__main__":
    print("\nTesting MultiLabelMetricsCalculator...")
    
    # Create fake data
    disease_names = ["healthy", "scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
    calculator = MultiLabelMetricsCalculator(disease_names)
    
    # Generate fake predictions
    batch_size = 100
    num_diseases = len(disease_names)
    
    # Random predictions and targets
    preds = np.random.randint(0, 2, (batch_size, num_diseases))
    targets = np.random.randint(0, 2, (batch_size, num_diseases))
    probs = np.random.random((batch_size, num_diseases))
    
    calculator.update(preds, targets, probs)
    metrics = calculator.compute()
    calculator.print_metrics(metrics, "Test")
    
    print("✅ MultiLabelMetricsCalculator test passed!")