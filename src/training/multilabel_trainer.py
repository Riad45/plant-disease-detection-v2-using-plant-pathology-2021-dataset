"""
Multi-Label Trainer - MATCHES WORKING MULTI-CLASS PATTERN EXACTLY
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
from src.training.multilabel_metrics import MultiLabelMetricsCalculator
from src.training.multilabel_losses import get_multilabel_loss
from src.training.early_stopping import EarlyStopping


class MultiLabelTrainer:
    """Multi-label trainer - EXACT PATTERN from working multi-class"""
    
    def __init__(self, model, train_loader, val_loader, model_name, disease_names, config_obj=None, logger=None):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.disease_names = disease_names
        self.num_diseases = len(disease_names)
        self.cfg = config_obj or config
        self.logger = logger
        
        config_model_name = model_name.replace('_multilabel', '')
        self.model_config = config.get_model_config(config_model_name)
        self.gradient_accumulation_steps = self.model_config['gradient_accumulation']
        
        # Loss function
        self.criterion = get_multilabel_loss(loss_type='weighted_bce')
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
            eps=config.EPS
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.LR_MIN
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.USE_AMP else None
        self.use_amp = config.USE_AMP
        
        # Metrics
        self.metrics_calculator = MultiLabelMetricsCalculator(disease_names)
        
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
        print(f"Expected Epoch Time: 2-5 minutes (was 25 minutes)")
        print(f"{'='*70}\n")
    
    def train_epoch(self, epoch):
        """Train one epoch - EXACT COPY from working multi-class"""
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        # EXACT PROGRESS BAR from working code
        pbar = tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch} Train",
            ncols=80,
            leave=False,
            mininterval=0.5,
            file=sys.stdout
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_NORM)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.gradient_accumulation_steps
            
            # Calculate accuracy (per-label)
            with torch.no_grad():
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                running_correct += (predictions == labels).sum().item()
                running_total += labels.numel()
            
            # Update progress bar - EXACT PATTERN
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * running_correct / running_total
            
            pbar.set_postfix({
                'loss': f'{current_loss:.3f}',
                'acc': f'{current_acc:.1f}%'
            })
            pbar.update(1)
        
        pbar.close()
        
        avg_loss = running_loss / len(self.train_loader)
        avg_acc = 100.0 * running_correct / running_total
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate - EXACT COPY from working multi-class"""
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        self.metrics_calculator.reset()
        
        # EXACT PROGRESS BAR from working code
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
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Metrics
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            self.metrics_calculator.update(predictions, labels, probabilities)
            
            running_correct += (predictions == labels).sum().item()
            running_total += labels.numel()
            
            # Update progress bar - EXACT PATTERN
            current_loss = running_loss / (pbar.n + 1)
            current_acc = 100.0 * running_correct / running_total
            
            pbar.set_postfix({
                'loss': f'{current_loss:.3f}',
                'acc': f'{current_acc:.1f}%'
            })
            pbar.update(1)
        
        pbar.close()
        
        avg_loss = running_loss / len(self.val_loader)
        avg_acc = 100.0 * running_correct / running_total
        metrics = self.metrics_calculator.compute()
        
        return avg_loss, avg_acc, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
        }
        
        if epoch % config.SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.model_save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved: {best_path} (Macro F1: {metrics['macro']['f1']:.4f})")
        
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)
    
    def train(self):
        """Main training loop - EXACT COPY from working multi-class"""
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING: {self.model_name} - FIXED VERSION")
        print(f"{'='*70}\n")
        
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Batch Size: {self.model_config['batch_size']} ⭐")
            print(f"Num Workers: {config.NUM_WORKERS} ⭐")
            print(f"Expected Speed: 2-5 minutes per epoch ⭐\n")
        
        start_time = time.time()
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            
            # Warmup
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
            
            # Scheduler
            if self.scheduler and epoch > config.WARMUP_EPOCHS:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch}/{config.NUM_EPOCHS} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Macro F1:   {metrics['macro']['f1']:.4f} ⭐")
            print(f"   Learning Rate: {current_lr:.2e}")
            
            # Log to WandB
            if self.logger:
                self.logger.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_macro_f1': metrics['macro']['f1'],
                    'val_macro_precision': metrics['macro']['precision'],
                    'val_macro_recall': metrics['macro']['recall'],
                    'learning_rate': current_lr,
                })
            
            # Check best
            current_metric = metrics['macro']['f1']
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
            
            # Save
            self.save_checkpoint(epoch, metrics, is_best=is_best)
            
            # Early stopping
            if self.early_stopping:
                should_stop = self.early_stopping(current_metric, epoch)
                if should_stop:
                    print(f"\n🛑 EARLY STOPPING at Epoch {epoch}")
                    print(f"Best Macro F1: {self.best_metric:.4f} at Epoch {self.best_epoch}")
                    break
        
        # Finished
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETED: {self.model_name}")
        print(f"{'='*70}")
        print(f"Total epochs: {self.current_epoch}")
        print(f"Best Macro F1: {self.best_metric:.4f} (Epoch {self.best_epoch})")
        print(f"Training time: {total_time/3600:.2f} hours")
        print(f"{'='*70}\n")
        
        # Save history
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