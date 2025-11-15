"""
Visualization utilities for plots and charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        history: Training history dict
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics curves
    val_f1 = [m['macro_f1'] for m in history['val_metrics']]
    val_acc = [m['accuracy'] for m in history['val_metrics']]
    
    axes[1].plot(epochs, val_acc, 'g-', label='Accuracy', linewidth=2)
    axes[1].plot(epochs, val_f1, 'purple', linestyle='--', label='Macro F1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion'}
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(metrics, class_names, save_path=None):
    """
    Plot per-class F1 scores
    
    Args:
        metrics: Metrics dictionary
        class_names: List of class names
        save_path: Path to save plot
    """
    f1_scores = metrics['per_class']['f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    bars = ax.bar(x, f1_scores, color='steelblue', alpha=0.8)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if f1_scores[i] >= 0.8:
            bar.set_color('green')
        elif f1_scores[i] >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (≥0.8)')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.6)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("✅ Visualization module loaded successfully!")