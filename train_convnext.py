"""
Training Script for ConvNeXt-Tiny
Plant Disease Detection Thesis Project
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config import config
# force-disable AMP for ConvNeXt to avoid illegal memory access on some setups
config.USE_AMP = False


from src.data.dataset import get_dataloaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.utils.logger import WandbLogger
from src.utils.visualize import plot_training_curves

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
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
    """Main training function for ConvNeXt-Tiny"""
    
    model_name = 'convnext_tiny'
    
    print("\n" + "="*80)
    print("PLANT DISEASE DETECTION - THESIS PROJECT")
    print("="*80)
    print(f"Model: ConvNeXt-Tiny")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Device: {config.DEVICE}")
    print("="*80 + "\n")
    
    # =========================================================================
    # STEP 1: Load Dataset Statistics
    # =========================================================================
    print("📊 STEP 1: Loading Dataset Statistics...")
    
    import json
    stats_path = config.PROCESSED_DATA_DIR / 'statistics.json'
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    config.update_num_classes(stats['num_classes'], stats['class_names'])
    
    print(f"✅ Dataset: {stats['total_samples_used']} samples")
    print(f"✅ Classes: {stats['num_classes']}")
    print(f"✅ Train: {stats['train_samples']} | Val: {stats['val_samples']} | Test: {stats['test_samples']}\n")
    
    # =========================================================================
    # STEP 2: Initialize WandB Logger
    # =========================================================================
    print("📝 STEP 2: Initializing WandB Logger...")
    
    logger = WandbLogger(
        model_name=model_name,
        project=config.WANDB_PROJECT,
        tags=['convnext', 'tiny', 'plant-pathology']
    )
    print()
    
    # =========================================================================
    # STEP 3: Create Data Loaders
    # =========================================================================
    print("📦 STEP 3: Creating Data Loaders...")
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        model_name=model_name,
        batch_size=config.BATCH_SIZES[model_name],
        num_workers=config.NUM_WORKERS
    )
    print()
    
    # =========================================================================
    # STEP 4: Create Model
    # =========================================================================
    print("🤖 STEP 4: Creating ConvNeXt-Tiny Model...")
    
    model = create_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    print()
    
    # =========================================================================
    # STEP 5: Initialize Trainer
    # =========================================================================
    print("⚙️  STEP 5: Initializing Trainer...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        config_obj=config,
        logger=logger
    )
    print()
    
    # =========================================================================
    # STEP 6: Start Training
    # =========================================================================
    print("🚀 STEP 6: Starting Training...")
    print("="*80)
    print("Press Ctrl+C to stop training early (checkpoint will be saved)")
    print("="*80 + "\n")
    
    try:
        history = trainer.train()
        
        # =====================================================================
        # STEP 7: Save Training Plots
        # =====================================================================
        print("\n📊 STEP 7: Generating Training Plots...")
        
        plot_save_path = config.PLOTS_DIR / f'{model_name}_training_curves.png'
        plot_training_curves(history, save_path=plot_save_path)
        
        # Log to WandB
        if logger.enabled:
            import wandb
            wandb.log({"training_curves": wandb.Image(str(plot_save_path))})
        
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            metrics={'macro_f1': 0.0},
            is_best=False
        )
    
    finally:
        # =====================================================================
        # STEP 8: Cleanup
        # =====================================================================
        print("\n🧹 STEP 8: Cleanup...")
        logger.finish()
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("✅ CONVNEXT-TINY TRAINING COMPLETED!")
    print("="*80)
    print(f"📁 Model saved in: {config.MODEL_DIR / model_name}")
    print(f"📁 Checkpoints saved in: {config.CHECKPOINT_DIR / model_name}")
    print(f"📁 Plots saved in: {config.PLOTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()