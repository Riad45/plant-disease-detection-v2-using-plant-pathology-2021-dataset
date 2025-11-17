"""
Training Script for ConvNeXt-Tiny - MULTI-LABEL (WINDOWS OPTIMIZED)
Complete fix for hanging and progress bar issues
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import config

# =========================================================================
# CRITICAL WINDOWS OPTIMIZATIONS
# =========================================================================
config.USE_AMP = False  # Disable AMP for stability on Windows
config.NUM_WORKERS = 2  # Optimized for Windows stability
config.BATCH_SIZES['convnext_tiny'] = 24  # Reduced for RTX 3050 6GB
config.PERSISTENT_WORKERS = True
config.PREFETCH_FACTOR = 2

# Set CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 128

from src.data.multilabel_dataset import get_multilabel_dataloaders
from src.models.model_factory import create_multilabel_model
from src.training.multilabel_trainer import MultiLabelTrainer
from src.utils.logger import WandbLogger
from src.utils.visualize import plot_training_curves

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if config.CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(config.SEED)


def main():
    """Main training function - WINDOWS OPTIMIZED"""
    
    # Model names
    model_name = 'convnext_tiny'
    save_name = 'convnext_tiny_multilabel'

    print("\n" + "="*80)
    print("PLANT DISEASE DETECTION - MULTI-LABEL THESIS PROJECT")
    print("="*80)
    print(f"Model: ConvNeXt-Tiny (Multi-Label)")
    print(f"Output: 6 binary classifiers for base diseases") 
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Device: {config.DEVICE}")
    print("="*80 + "\n")

    # =========================================================================
    # STEP 1: Load Multi-Label Dataset Information
    # =========================================================================
    print("📊 STEP 1: Loading Multi-Label Dataset Information...")

    import json
    info_path = config.PROCESSED_DATA_DIR / 'multilabel_info.json'

    if not info_path.exists():
        print("❌ Multi-label data not found! Please run prepare_dataset_multilabel.py first")
        return

    with open(info_path, 'r') as f:
        disease_info = json.load(f)

    base_diseases = disease_info['base_diseases']
    total_samples = disease_info['total_samples']

    print(f"✅ Base Diseases: {len(base_diseases)}")
    print(f"✅ Total Samples: {total_samples}")
    print(f"✅ Disease Classes: {base_diseases}\n")

    # Update config for multi-label (6 diseases)
    config.update_num_classes(len(base_diseases), base_diseases)

    # =========================================================================
    # STEP 2: Initialize WandB Logger
    # =========================================================================
    print("📝 STEP 2: Initializing WandB Logger...")

    logger = WandbLogger(
        model_name=model_name,
        project=config.WANDB_PROJECT,
        tags=['convnext', 'multilabel', 'plant-pathology', 'thesis', 'windows_optimized']
    )
    print()

    # =========================================================================
    # STEP 3: WINDOWS OPTIMIZED DATA LOADERS
    # =========================================================================
    print("📦 STEP 3: Creating Windows-Optimized Data Loaders...")

    # CRITICAL: Windows-specific optimizations
    optimized_batch_size = 24  # Reduced for RTX 3050 6GB stability
    optimized_workers = 2      # Optimal for Windows
    
    print(f"   🎯 Windows Optimizations Applied:")
    print(f"   - Batch Size: {optimized_batch_size} (reduced for stability)")
    print(f"   - Num Workers: {optimized_workers} (optimal for Windows)")
    print(f"   - Mixed Precision: Disabled (better stability)")
    print(f"   - Persistent Workers: Enabled")
    
    train_loader, val_loader, test_loader, disease_names = get_multilabel_dataloaders(
        model_name=model_name,
        batch_size=optimized_batch_size,
        num_workers=optimized_workers
    )
    print()

    # =========================================================================
    # STEP 4: Create Multi-Label Model
    # =========================================================================
    print("🤖 STEP 4: Creating Multi-Label ConvNeXt-Tiny Model...")

    model = create_multilabel_model(
        model_name=model_name,
        num_diseases=len(base_diseases),
        pretrained=True
    )
    print()

    # =========================================================================
    # STEP 5: Initialize Multi-Label Trainer
    # =========================================================================
    print("⚙️  STEP 5: Initializing Multi-Label Trainer...")

    trainer = MultiLabelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=save_name,
        disease_names=disease_names,
        config_obj=config,
        logger=logger
    )
    print()

    # =========================================================================
    # STEP 6: PERFORMANCE VALIDATION
    # =========================================================================
    print("🔍 STEP 6: Performance Validation...")
    
    # Test one batch to ensure everything works
    print("   Testing one training batch...")
    try:
        with torch.no_grad():
            test_images, test_labels = next(iter(train_loader))
            test_images = test_images.to(config.DEVICE)
            test_outputs = model(test_images)
            
            print(f"   ✅ Validation passed!")
            print(f"   - Input shape: {test_images.shape}")
            print(f"   - Output shape: {test_outputs.shape}")
            print(f"   - Labels shape: {test_labels.shape}")
            print(f"   - Expected: [batch, 6] binary outputs")
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return
    
    print()

    # =========================================================================
    # STEP 7: Start Multi-Label Training
    # =========================================================================
    print("🚀 STEP 7: Starting Multi-Label Training...")
    print("="*80)
    print("WINDOWS OPTIMIZED TRAINING - EXPECTED PERFORMANCE:")
    print(f"   • Epoch Time: 3-6 minutes (optimized)")
    print(f"   • GPU Memory: ~4-5 GB (stable)")
    print(f"   • No hanging with {optimized_workers} workers")
    print(f"   • Clean progress bars")
    print("="*80)
    print("CORRECT FORMULATION: 6 binary classifiers for disease detection")
    print("Each image can have multiple diseases simultaneously")
    print("Press Ctrl+C to stop training early (checkpoint will be saved)")
    print("="*80 + "\n")

    try:
        history = trainer.train()
        
        # =====================================================================
        # STEP 8: Save Training Plots
        # =====================================================================
        print("\n📊 STEP 8: Generating Training Plots...")
        
        plot_save_path = config.PLOTS_DIR / f'{save_name}_training_curves.png'
        plot_training_curves(history, save_path=plot_save_path)
        
        if logger.enabled:
            import wandb
            wandb.log({"training_curves": wandb.Image(str(plot_save_path))})
        
        print(f"✅ Training plots saved: {plot_save_path}")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("💾 Saving checkpoint...")
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

    except Exception as e:
        print(f"\n❌ Training error: {e}")
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
        # =====================================================================
        # STEP 9: Cleanup
        # =====================================================================
        print("\n🧹 STEP 9: Cleanup...")
        logger.finish()
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("✅ CONVNEXT-TINY MULTI-LABEL TRAINING COMPLETED!")
    print("="*80)
    print(f"📁 Model saved in: {config.MODEL_DIR / save_name}")
    print(f"📁 Checkpoints saved in: {config.CHECKPOINT_DIR / save_name}") 
    print(f"📁 Plots saved in: {config.PLOTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()