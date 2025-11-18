"""
Training Script for Swin-Tiny - MULTI-LABEL (WINDOWS OPTIMIZED)
Plant Disease Detection Thesis Project

FIXED VERSION:
- Correct JSON key reading
- Git Bash progress bar compatibility
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config

# ============================================================================
# SWIN-SPECIFIC OPTIMIZATIONS FOR RTX 3050 6GB + WINDOWS
# ============================================================================
config.USE_AMP = False  # Disabled for Windows stability
config.NUM_WORKERS = 2  # Optimal for Windows
config.BATCH_SIZES['swin_tiny'] = 24  # Same as ConvNeXt (works perfectly!)
config.PERSISTENT_WORKERS = True
config.PREFETCH_FACTOR = 2

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 128

# Import after config modifications
from src.data.multilabel_dataset import get_multilabel_dataloaders
from src.models.model_factory import create_multilabel_model
from src.training.multilabel_trainer import MultiLabelTrainer
from src.utils.logger import WandbLogger
from src.utils.visualize import plot_training_curves

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import json


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if config.CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   💾 GPU Memory:")
        print(f"      - Allocated: {allocated:.2f} GB")
        print(f"      - Reserved:  {reserved:.2f} GB")
        print(f"      - Peak:      {max_allocated:.2f} GB")


def print_section_header(title, emoji=""):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{emoji} {title}")
    print("="*80)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function - complete workflow"""
    
    # Set random seed
    set_seed(config.SEED)
    
    # Model configuration
    MODEL_NAME = 'swin_tiny'
    SAVE_NAME = 'swin_tiny_multilabel'
    BATCH_SIZE = 24  # Same as ConvNeXt
    NUM_WORKERS = 2  # Optimal for Windows

    # ========================================================================
    # HEADER
    # ========================================================================
    print_section_header("PLANT DISEASE DETECTION - MULTI-LABEL THESIS PROJECT", "🌱")
    print(f"Model:        Swin-Tiny Transformer (Multi-Label)")
    print(f"Architecture: Shifted Window Self-Attention")
    print(f"Output:       6 binary classifiers for base diseases")
    print(f"Dataset:      {config.DATASET_NAME}")
    print(f"Device:       {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Batch Size:   {BATCH_SIZE} (same as ConvNeXt) 🚀")
    print(f"Num Workers:  {NUM_WORKERS}")
    print("="*80)

    # ========================================================================
    # STEP 1: Load Multi-Label Dataset Information (FIXED)
    # ========================================================================
    print("\n📊 STEP 1: Loading Multi-Label Dataset Information...")

    info_path = config.PROCESSED_DATA_DIR / 'multilabel_info.json'

    if not info_path.exists():
        print("\n❌ ERROR: Multi-label data not found!")
        print("Please run this first:")
        print("   python src/data/prepare_dataset_multilabel.py")
        return

    with open(info_path, 'r') as f:
        disease_info = json.load(f)

    base_diseases = disease_info['base_diseases']
    total_samples = disease_info['total_samples']
    
    # ✅ FIXED: Read correct keys from JSON
    train_samples = disease_info.get('train_samples', 'N/A')
    val_samples = disease_info.get('val_samples', 'N/A')
    test_samples = disease_info.get('test_samples', 'N/A')

    print(f"✅ Base Diseases: {len(base_diseases)}")
    print(f"   {', '.join(base_diseases)}")
    print(f"✅ Total Samples: {total_samples:,}")
    print(f"✅ Train Samples: {train_samples:,}" if isinstance(train_samples, int) else f"✅ Train Samples: {train_samples}")
    print(f"✅ Val Samples:   {val_samples:,}" if isinstance(val_samples, int) else f"✅ Val Samples:   {val_samples}")
    print(f"✅ Test Samples:  {test_samples:,}" if isinstance(test_samples, int) else f"✅ Test Samples:  {test_samples}")

    # Update config
    config.update_num_classes(len(base_diseases), base_diseases)

    # ========================================================================
    # STEP 2: Initialize WandB Logger
    # ========================================================================
    print("\n📝 STEP 2: Initializing WandB Logger...")

    logger = WandbLogger(
        model_name=MODEL_NAME,
        project=config.WANDB_PROJECT,
        tags=[
            'swin-tiny',
            'transformer',
            'multilabel',
            'plant-pathology',
            'thesis',
            'windows-optimized',
            'batch-24'
        ]
    )

    # ========================================================================
    # STEP 3: Create Data Loaders
    # ========================================================================
    print("\n📦 STEP 3: Creating Multi-Label Data Loaders...")
    print(f"   Configuration:")
    print(f"   - Model:       {MODEL_NAME}")
    print(f"   - Image Size:  224×224")
    print(f"   - Batch Size:  {BATCH_SIZE} 🚀")
    print(f"   - Num Workers: {NUM_WORKERS}")
    print(f"   - Prefetch:    {config.PREFETCH_FACTOR}")
    print(f"   - Persistent:  {config.PERSISTENT_WORKERS}")
    print(f"   - Pin Memory:  {config.PIN_MEMORY}")
    
    # ✅ Note about Git Bash progress bars
    if 'MSYSTEM' in os.environ or 'MINGW' in os.environ.get('MSYSTEM', ''):
        print(f"\n   ℹ️  Git Bash Detected:")
        print(f"      Progress bars may print multiple lines (visual issue only)")
        print(f"      Training speed is NOT affected!")
        print(f"      Use Windows Terminal or CMD for cleaner output (optional)")

    try:
        train_loader, val_loader, test_loader, disease_names = get_multilabel_dataloaders(
            model_name=MODEL_NAME,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to create data loaders!")
        print(f"   {e}")
        return

    print(f"\n✅ Data Loaders Created:")
    print(f"   - Train Batches: {len(train_loader)}")
    print(f"   - Val Batches:   {len(val_loader)}")
    print(f"   - Test Batches:  {len(test_loader)}")
    print(f"   - Diseases:      {len(disease_names)}")

    # ========================================================================
    # STEP 4: Create Swin Transformer Model
    # ========================================================================
    print("\n🤖 STEP 4: Creating Swin-Tiny Transformer Model...")

    try:
        model = create_multilabel_model(
            model_name=MODEL_NAME,
            num_diseases=len(base_diseases),
            pretrained=True
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to create model!")
        print(f"   {e}")
        return

    # ========================================================================
    # STEP 5: Initialize Trainer
    # ========================================================================
    print("\n⚙️  STEP 5: Initializing Multi-Label Trainer...")

    try:
        trainer = MultiLabelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=SAVE_NAME,
            disease_names=disease_names,
            config_obj=config,
            logger=logger
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize trainer!")
        print(f"   {e}")
        return

    # ========================================================================
    # STEP 6: Validation & Memory Check
    # ========================================================================
    print("\n🔍 STEP 6: Pre-Training Validation & GPU Memory Check...")

    try:
        print("   Testing forward pass with one batch...")
        with torch.no_grad():
            # Clear cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Get test batch
            test_images, test_labels = next(iter(train_loader))
            test_images = test_images.to(config.DEVICE)
            
            # Forward pass
            test_outputs = model(test_images)
            
            print(f"\n   ✅ Forward Pass Successful!")
            print(f"   - Input Shape:  {test_images.shape}")
            print(f"   - Output Shape: {test_outputs.shape}")
            print(f"   - Labels Shape: {test_labels.shape}")
            print(f"   - Expected:     torch.Size([{BATCH_SIZE}, 6])")
            
            # Memory check
            print()
            print_gpu_memory()
            
            # Verify output is valid
            assert test_outputs.shape == torch.Size([BATCH_SIZE, 6]), \
                f"Wrong output shape! Got {test_outputs.shape}"
            
            # Check memory is safe
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                if allocated_gb > 5.5:
                    print(f"\n   ⚠️  WARNING: High memory usage ({allocated_gb:.2f} GB)")
                    print(f"   Risk of OOM during training!")
                    print(f"   Consider reducing batch size to 20")
                else:
                    print(f"\n   ✅ Memory usage is safe for training!")
            
            # Cleanup
            del test_images, test_labels, test_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n   ❌ OUT OF MEMORY during validation!")
            print(f"\n   💡 SOLUTION:")
            print(f"   - Batch size {BATCH_SIZE} is too large for your GPU")
            print(f"   - Edit this file (train_swin_multilabel.py)")
            print(f"   - Change line 37: config.BATCH_SIZES['swin_tiny'] = 20")
            print(f"   - Or change line 68: BATCH_SIZE = 20")
            print(f"   - Then restart training")
            logger.finish()
            return
        else:
            print(f"\n   ❌ Runtime error: {e}")
            logger.finish()
            return
            
    except Exception as e:
        print(f"\n   ❌ Validation failed: {e}")
        logger.finish()
        return

    # ========================================================================
    # STEP 7: Start Training
    # ========================================================================
    print_section_header("STARTING MULTI-LABEL TRAINING", "🚀")
    print("CONFIGURATION:")
    print(f"   • Model:        Swin-Tiny Transformer")
    print(f"   • Batch Size:   {BATCH_SIZE}")
    print(f"   • Max Epochs:   {config.NUM_EPOCHS}")
    print(f"   • Learning Rate: {config.LEARNING_RATE}")
    print(f"   • Optimizer:    {config.OPTIMIZER}")
    print(f"   • Scheduler:    {config.LR_SCHEDULER}")
    print(f"   • Device:       {config.DEVICE}")
    print()
    print("EXPECTED PERFORMANCE:")
    print(f"   • Epoch Time:   3-6 minutes")
    print(f"   • GPU Utilization: 70-95%")
    print(f"   • Total Time:   2.5-5 hours (50 epochs)")
    print()
    print("ARCHITECTURE:")
    print(f"   • Shifted Window Self-Attention")
    print(f"   • 6 binary classifiers (multi-label)")
    print(f"   • Each image can have multiple diseases")
    print()
    print("CONTROLS:")
    print(f"   • Press Ctrl+C to stop early (checkpoint saved)")
    print(f"   • Monitor GPU: 'nvidia-smi -l 1' in another terminal")
    print()
    if 'MSYSTEM' in os.environ or 'MINGW' in os.environ.get('MSYSTEM', ''):
        print("ℹ️  GIT BASH NOTE:")
        print(f"   Progress bars print multiple lines (cosmetic issue)")
        print(f"   Training works perfectly - just looks messy!")
        print(f"   Actual speed: ~1.15 it/s (fast!) ✅")
    print("="*80 + "\n")

    # Train with comprehensive error handling
    try:
        history = trainer.train()
        
        # ====================================================================
        # STEP 8: Save Results
        # ====================================================================
        print("\n📊 STEP 8: Saving Training Results...")
        
        # Save plots
        plot_save_path = config.PLOTS_DIR / f'{SAVE_NAME}_training_curves.png'
        plot_training_curves(history, save_path=plot_save_path)
        print(f"✅ Training curves saved: {plot_save_path}")
        
        # Log to WandB
        if logger.enabled:
            import wandb
            wandb.log({"training_curves": wandb.Image(str(plot_save_path))})
            print(f"✅ Results logged to WandB")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠️  TRAINING INTERRUPTED BY USER")
        print("="*80)
        print("💾 Saving checkpoint...")
        
        try:
            dummy_metrics = {
                'macro': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'micro': {'f1': 0.0},
                'example_based': {'exact_match_ratio': 0.0}
            }
            trainer.save_checkpoint(
                epoch=trainer.current_epoch,
                metrics=dummy_metrics,
                is_best=False
            )
            print("✅ Checkpoint saved successfully")
            print(f"   Location: {config.CHECKPOINT_DIR / SAVE_NAME / 'last_checkpoint.pth'}")
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n\n" + "="*80)
            print("❌ OUT OF MEMORY ERROR DURING TRAINING")
            print("="*80)
            
            # Try to save checkpoint
            print("💾 Attempting to save checkpoint...")
            try:
                dummy_metrics = {
                    'macro': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                    'micro': {'f1': 0.0},
                    'example_based': {'exact_match_ratio': 0.0}
                }
                trainer.save_checkpoint(
                    epoch=trainer.current_epoch,
                    metrics=dummy_metrics,
                    is_best=False
                )
                print("✅ Checkpoint saved")
            except:
                print("❌ Could not save checkpoint")
            
            print("\n💡 SOLUTION:")
            print("   1. Edit this file: scripts/train_swin_multilabel.py")
            print("   2. Change line 37: config.BATCH_SIZES['swin_tiny'] = 20")
            print("   3. Or change line 68: BATCH_SIZE = 20")
            print("   4. Restart training")
            print("="*80)
        else:
            print(f"\n❌ Runtime error: {e}")
            
    except Exception as e:
        print(f"\n\n❌ TRAINING ERROR: {e}")
        print("💾 Attempting to save recovery checkpoint...")
        
        try:
            dummy_metrics = {
                'macro': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                'micro': {'f1': 0.0},
                'example_based': {'exact_match_ratio': 0.0}
            }
            trainer.save_checkpoint(
                epoch=trainer.current_epoch,
                metrics=dummy_metrics,
                is_best=False
            )
            print("✅ Recovery checkpoint saved")
        except:
            print("❌ Could not save recovery checkpoint")
            
    finally:
        # ====================================================================
        # STEP 9: Cleanup
        # ====================================================================
        print("\n🧹 STEP 9: Cleanup...")
        logger.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Cleanup complete")

    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    print_section_header("TRAINING COMPLETED", "✅")
    print(f"Model:       {SAVE_NAME}")
    print(f"Best Metric: {trainer.best_metric:.4f} (Epoch {trainer.best_epoch})")
    print(f"Total Epochs: {trainer.current_epoch}")
    print()
    print("📁 SAVED FILES:")
    print(f"   • Best Model:     {config.MODEL_DIR / SAVE_NAME / 'best_model.pth'}")
    print(f"   • Last Checkpoint: {config.CHECKPOINT_DIR / SAVE_NAME / 'last_checkpoint.pth'}")
    print(f"   • Training History: {config.CHECKPOINT_DIR / SAVE_NAME / 'training_history.json'}")
    print(f"   • Training Curves:  {config.PLOTS_DIR / f'{SAVE_NAME}_training_curves.png'}")
    print()
    print("📊 NEXT STEPS:")
    print("   1. Compare results with ConvNeXt")
    print("   2. Evaluate on test set: python scripts/evaluate_model.py")
    print("   3. Analyze confusion matrices")
    print("   4. Generate thesis figures")
    print("="*80 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Import os here
    import os
    
    # Windows multiprocessing fix
    if sys.platform == 'win32':
        torch.multiprocessing.freeze_support()
    
    # Run training
    main()