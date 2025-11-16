"""
Trainer Class - Core training engine with FIXED progress bars
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time
import json
from pathlib import Path
import sys
import os

# Fix for Git Bash progress bars
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.training.metrics import MetricsCalculator
from src.training.losses import get_loss_function
from src.training.early_stopping import EarlyStopping


class Trainer:
    """Main Trainer class for model training - FIXED PROGRESS BARS"""

    def __init__(self, model, train_loader, val_loader, model_name, config_obj=None, logger=None):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.cfg = config_obj or config
        self.logger = logger
        
        # Model-specific settings
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
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.LR_MIN
        )
        
        # Mixed precision
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
            mode='max',
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
        print(f"TRAINER INITIALIZED: {model_name} - FIXED VERSION")
        print(f"{'='*70}")
        print(f"Device: {config.DEVICE}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Batch Size: {self.model_config['batch_size']} ⭐")
        print(f"Num Workers: {config.NUM_WORKERS} ⭐")
        print(f"Expected Epoch Time: 2-5 minutes (was 20 minutes)")
        print(f"{'='*70}\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch - FIXED PROGRESS BAR"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # FIXED: Simple progress bar that works in Git Bash
        pbar = tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch} Train",
            ncols=80,  # Fixed width
            leave=False,  # Don't leave progress bar after completion
            mininterval=0.5,  # Update every 0.5 seconds
            file=sys.stdout
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.GRADIENT_CLIP_NORM
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
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
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update running loss
            running_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar - FIXED: Simple postfix
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            
            pbar.set_postfix({
                'loss': f'{current_loss:.3f}',
                'acc': f'{current_acc:.1f}%'
            })
            pbar.update(1)
        
        pbar.close()
        
        avg_loss = running_loss / len(self.train_loader)
        avg_acc = 100.0 * correct / total
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model - FIXED PROGRESS BAR"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Reset metrics calculator
        self.metrics_calculator.reset()
        
        # FIXED: Simple validation progress bar
        pbar = tqdm(
            total=len(self.val_loader),
            desc=f"Epoch {epoch} Val  ",
            ncols=80,
            leave=False,
            mininterval=0.5,
            file=sys.stdout
        )
        
        for images, labels in self.val_loader:
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            # Update metrics
            self.metrics_calculator.update(predictions, labels, outputs)
            
            # Update progress bar - FIXED: Simple postfix
            current_loss = running_loss / (pbar.n + 1)
            current_acc = 100.0 * correct / total
            
            pbar.set_postfix({
                'loss': f'{current_loss:.3f}',
                'acc': f'{current_acc:.1f}%'
            })
            pbar.update(1)
        
        pbar.close()
        
        avg_loss = running_loss / len(self.val_loader)
        avg_acc = 100.0 * correct / total
        
        # Compute metrics
        metrics = self.metrics_calculator.compute(average='macro')
        
        return avg_loss, avg_acc, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
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
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING: {self.model_name} - FIXED VERSION")
        print(f"{'='*70}\n")
        
        # Print GPU info
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Batch Size: {self.model_config['batch_size']} ⭐")
            print(f"Num Workers: {config.NUM_WORKERS} ⭐")
            print(f"Expected Speed: 2-5 minutes per epoch ⭐\n")
        
        start_time = time.time()
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            
            # Warmup learning rate
            if epoch <= config.WARMUP_EPOCHS:
                warmup_factor = epoch / config.WARMUP_EPOCHS
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config.LEARNING_RATE * warmup_factor
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc, metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_metrics.append(metrics)
            
            # Learning rate scheduler step
            if self.scheduler and epoch > config.WARMUP_EPOCHS:
                self.scheduler.step()
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch}/{config.NUM_EPOCHS} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Macro F1:   {metrics['macro_f1']:.4f} ⭐")
            print(f"   Learning Rate: {current_lr:.2e}")
            
            # Log to WandB
            if self.logger:
                self.logger.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
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
                    print(f"\n🛑 EARLY STOPPING at Epoch {epoch}")
                    print(f"Best Macro F1: {self.best_metric:.4f} at Epoch {self.best_epoch}")
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