"""
Model Evaluation Script - Generate all thesis figures and metrics
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.data.dataset import get_dataloaders
from src.models.model_factory import create_model
from src.training.metrics import MetricsCalculator


def load_best_model(model_name, num_classes):
    """Load the best model from training"""
    model_path = config.MODEL_DIR / model_name / 'best_model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")
    
    # Create model architecture
    model = create_model(model_name, num_classes, pretrained=False)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    
    print(f"✅ Loaded best model from epoch {checkpoint['epoch']}")
    print(f"✅ Best Macro F1: {checkpoint['best_metric']:.4f}")
    
    return model, checkpoint


def evaluate_model(model, test_loader, class_names, model_name):
    """Comprehensive model evaluation"""
    metrics_calculator = MetricsCalculator(len(class_names), class_names)
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_name.upper()} ON TEST SET")
    print(f"{'='*70}")
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            metrics_calculator.update(predictions, targets, outputs)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Compute metrics
    metrics = metrics_calculator.compute(average='macro')
    metrics_calculator.print_metrics(metrics, phase="Test Set")
    
    return metrics, np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)


def plot_confusion_matrix(all_targets, all_predictions, class_names, model_name, save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title(f'Confusion Matrix - {model_name}\n(Normalized by True Class)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
    
    plt.close()
    
    return cm


def plot_class_performance(metrics, class_names, model_name, save_path=None):
    """Plot per-class performance metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision per class
    axes[0].bar(range(len(class_names)), metrics['per_class']['precision'], color='skyblue', alpha=0.7)
    axes[0].set_title('Precision per Class')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Recall per class
    axes[1].bar(range(len(class_names)), metrics['per_class']['recall'], color='lightgreen', alpha=0.7)
    axes[1].set_title('Recall per Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score per class
    axes[2].bar(range(len(class_names)), metrics['per_class']['f1'], color='salmon', alpha=0.7)
    axes[2].set_title('F1-Score per Class')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Per-Class Performance Metrics - {model_name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Class performance plot saved: {save_path}")
    
    plt.close()


def save_detailed_report(metrics, class_names, model_name, save_path=None):
    """Save detailed classification report"""
    if save_path is None:
        save_path = config.METRICS_DIR / f'{model_name}_detailed_report.json'
    
    report = {
        'model_name': model_name,
        'overall_metrics': {
            'accuracy': metrics['accuracy'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'macro_f1': metrics['macro_f1'],
            'balanced_accuracy': metrics['balanced_accuracy'],
        },
        'per_class_metrics': {},
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    for i, class_name in enumerate(class_names):
        report['per_class_metrics'][class_name] = {
            'precision': metrics['per_class']['precision'][i],
            'recall': metrics['per_class']['recall'][i],
            'f1_score': metrics['per_class']['f1'][i],
            'support': metrics['per_class']['support'][i]
        }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Detailed report saved: {save_path}")
    return report


def main():
    """Main evaluation function"""
    model_name = 'convnext_tiny'
    
    print("\n" + "="*80)
    print("MODEL EVALUATION - THESIS FIGURES GENERATION")
    print("="*80)
    
    # Load dataset statistics
    stats_path = config.PROCESSED_DATA_DIR / 'statistics.json'
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    config.update_num_classes(stats['num_classes'], stats['class_names'])
    
    # Get test dataloader
    print("\n📦 Loading test dataset...")
    _, _, test_loader, class_names = get_dataloaders(model_name)
    
    # Load best model
    print("\n🤖 Loading best model...")
    model, checkpoint = load_best_model(model_name, config.NUM_CLASSES)
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    metrics, all_preds, all_targets, all_probs = evaluate_model(
        model, test_loader, class_names, model_name
    )
    
    # Generate visualizations
    print("\n📊 Generating thesis figures...")
    
    # 1. Confusion Matrix
    cm_path = config.CONFUSION_DIR / f'{model_name}_confusion_matrix.png'
    plot_confusion_matrix(all_targets, all_preds, class_names, model_name, cm_path)
    
    # 2. Class Performance
    perf_path = config.PLOTS_DIR / f'{model_name}_class_performance.png'
    plot_class_performance(metrics, class_names, model_name, perf_path)
    
    # 3. Detailed Report
    report_path = config.METRICS_DIR / f'{model_name}_evaluation_report.json'
    report = save_detailed_report(metrics, class_names, model_name, report_path)
    
    # Print summary for thesis
    print(f"\n{'='*80}")
    print("📋 THESIS RESULTS SUMMARY - ConvNeXt-Tiny")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   ✅ {cm_path}")
    print(f"   ✅ {perf_path}")
    print(f"   ✅ {report_path}")
    print(f"   ✅ Training curves (from training)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()