"""
Trainer Class - Core training engine
Handles training loop, validation, checkpointing, logging
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.training.metrics import MetricsCalculator
from src.training.losses import get_loss_function
from src.training.early_stopping import EarlyStopping


class Trainer:
    """
    Main Trainer class for model training
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        model_name,
        config_obj=None,
        logger=None
    ):
        """
        Initialize Trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of the model (for saving)
            config_obj: Configuration object (default: global config)
            logger: WandB logger (optional)
        """
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.cfg = config_obj or config
        self.logger = logger
        
        # Get model-specific settings
        self.model_config = config.get_model_config(model_name)
        self.gradient_accumulation_steps = self.model_config['gradient_accumulation']
        
        # Loss function (weighted for class imbalance)
        self.criterion = get_loss_function(
            use_weighted=True,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
            eps=config.EPS
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.USE_AMP else None
        self.use_amp = config.USE_AMP
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=config.NUM_CLASSES,
            class_names=config.CLASS_NAMES
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode='max',  # We monitor macro_f1 (higher is better)
            verbose=True
        ) if config.EARLY_STOPPING else None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Paths
        self.checkpoint_dir = config.CHECKPOINT_DIR / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_save_dir = config.MODEL_DIR / model_name
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"TRAINER INITIALIZED: {model_name}")
        print(f"{'='*70}")
        print(f"Device: {config.DEVICE}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}x")
        print(f"Effective Batch Size: {self.model_config['batch_size'] * self.gradient_accumulation_steps}")
        print(f"Optimizer: {config.OPTIMIZER}")
        print(f"Scheduler: {config.LR_SCHEDULER}")
        print(f"Early Stopping: {config.EARLY_STOPPING} (patience={config.EARLY_STOPPING_PATIENCE})")
        print(f"{'='*70}\n")
    
    def _get_scheduler(self):
        """Create learning rate scheduler"""
        if config.LR_SCHEDULER == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS,
                eta_min=config.LR_MIN
            )
        elif config.LR_SCHEDULER == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.LEARNING_RATE,
                epochs=config.NUM_EPOCHS,
                steps_per_epoch=len(self.train_loader)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        running_loss = 0.0
        
        # Progress bar
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [TRAIN]",
            ncols=100
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.GRADIENT_CLIP_NORM
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            
            else:
                # Standard training (no mixed precision)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.GRADIENT_CLIP_NORM
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update running loss
            running_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Clear cache periodically
            if (batch_idx + 1) % config.EMPTY_CACHE_EVERY_N_BATCHES == 0:
                torch.cuda.empty_cache()
        
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """
        Validate model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average validation loss
            metrics: Dictionary of metrics
        """
        self.model.eval()
        running_loss = 0.0
        
        # Reset metrics calculator
        self.metrics_calculator.reset()
        
        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [VAL]  ",
            ncols=100
        )
        
        for images, labels in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions
            _, predictions = torch.max(outputs, 1)
            
            # Update metrics
            self.metrics_calculator.update(predictions, labels, outputs)
            
            # Update progress bar
            avg_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_loss = running_loss / len(self.val_loader)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute(average='macro')
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'config': {
                'model_name': self.model_name,
                'num_classes': config.NUM_CLASSES,
                'image_size': self.model_config['image_size'],
            }
        }
        
        # Save periodic checkpoint
        if epoch % config.SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.model_save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved: {best_path} (Macro F1: {metrics['macro_f1']:.4f})")
        
        # Save last checkpoint (always)
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)
    
    def train(self):
        """
        Main training loop
        """
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING: {self.model_name}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            
            # Warmup learning rate
            if epoch <= config.WARMUP_EPOCHS:
                warmup_factor = epoch / config.WARMUP_EPOCHS
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config.LEARNING_RATE * warmup_factor
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_metrics.append(metrics)
            
            # Learning rate scheduler step
            if self.scheduler and epoch > config.WARMUP_EPOCHS:
                self.scheduler.step()
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{config.NUM_EPOCHS} Summary")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Accuracy:   {metrics['accuracy']:.4f}")
            print(f"Macro F1:   {metrics['macro_f1']:.4f} ⭐")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"{'='*70}\n")
            
            # Log to WandB
            if self.logger:
                self.logger.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': metrics['accuracy'],
                    'val_balanced_accuracy': metrics['balanced_accuracy'],
                    'val_macro_f1': metrics['macro_f1'],
                    'val_macro_precision': metrics['macro_precision'],
                    'val_macro_recall': metrics['macro_recall'],
                    'learning_rate': current_lr,
                })
            
            # Check if best model
            current_metric = metrics['macro_f1']
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best=is_best)
            
            # Early stopping check
            if self.early_stopping:
                should_stop = self.early_stopping(current_metric, epoch)
                if should_stop:
                    print(f"\n{'='*70}")
                    print(f"🛑 EARLY STOPPING at Epoch {epoch}")
                    print(f"{'='*70}")
                    print(f"Best Macro F1: {self.best_metric:.4f} at Epoch {self.best_epoch}")
                    print(f"{'='*70}\n")
                    break
        
        # Training finished
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETED: {self.model_name}")
        print(f"{'='*70}")
        print(f"Total epochs: {self.current_epoch}")
        print(f"Best Macro F1: {self.best_metric:.4f} (Epoch {self.best_epoch})")
        print(f"Training time: {total_time/3600:.2f} hours")
        print(f"{'='*70}\n")
        
        # Save training history
        history = {
            'model_name': self.model_name,
            'total_epochs': self.current_epoch,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'training_time_hours': total_time / 3600,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
        }
        
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"📊 Training history saved: {history_path}")
        
        return history


if __name__ == "__main__":
    print("✅ Trainer module loaded successfully!")
    print("This file should be imported, not run directly.")