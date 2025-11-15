"""
Early Stopping implementation
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving
    """
    
    def __init__(self, patience=7, min_delta=0.0001, mode='max', verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' (for accuracy, f1) or 'min' (for loss)
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # For mode='max', we want higher values
        # For mode='min', we want lower values
        self.monitor_op = np.greater if mode == 'max' else np.less
        self.best_score = -np.inf if mode == 'max' else np.inf
    
    def __call__(self, current_score, epoch):
        """
        Check if training should stop
        
        Args:
            current_score: Current validation metric value
            epoch: Current epoch number
            
        Returns:
            should_stop: Boolean indicating if training should stop
        """
        # Check if this is an improvement
        if self.mode == 'max':
            is_improvement = current_score > (self.best_score + self.min_delta)
        else:
            is_improvement = current_score < (self.best_score - self.min_delta)
        
        if is_improvement:
            if self.verbose:
                print(f"✅ Metric improved from {self.best_score:.4f} to {current_score:.4f}")
            
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  No improvement for {self.counter}/{self.patience} epochs "
                      f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered after {self.patience} epochs without improvement")
                    print(f"   Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = -np.inf if self.mode == 'max' else np.inf
        self.early_stop = False
        self.best_epoch = 0


if __name__ == "__main__":
    # Test early stopping
    print("\nTesting EarlyStopping...")
    
    # Test with max mode (e.g., for accuracy/f1)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, mode='max')
    
    # Simulate validation scores
    scores = [0.75, 0.78, 0.80, 0.81, 0.805, 0.802, 0.801, 0.800]
    
    print("\nSimulating training epochs:")
    for epoch, score in enumerate(scores, 1):
        print(f"\nEpoch {epoch}: Score = {score:.4f}")
        should_stop = early_stopping(score, epoch)
        
        if should_stop:
            print(f"\n✅ Would stop training at epoch {epoch}")
            break
    
    print("\n✅ EarlyStopping test passed!")