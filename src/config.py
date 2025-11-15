"""
FINAL Optimized Configuration for Plant Disease Detection Thesis
Hardware: RTX 3050 6GB | Dataset: Plant Pathology 2021
Models: ConvNeXt-Tiny, Swin-Tiny, DeiT-Small

Author: Your Name
Date: 2024
"""

import torch
import os
from pathlib import Path
from datetime import datetime


class Config:
    """Main configuration class - all hyperparameters in one place"""
    
    # ============================================================================
    # PROJECT PATHS
    # ============================================================================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    MODEL_DIR = BASE_DIR / "models"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    RESULTS_DIR = BASE_DIR / "results"
    PLOTS_DIR = RESULTS_DIR / "plots"
    LOGS_DIR = RESULTS_DIR / "logs"
    METRICS_DIR = RESULTS_DIR / "metrics"
    CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
    
    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================
    DATASET_NAME = "timm/plant-pathology-2021"
    USE_ONLY_TRAIN_SPLIT = True  # Use only train split, create own 80/10/10
    
    # Will be set dynamically after data loading
    NUM_CLASSES = None
    CLASS_NAMES = None
    
    # Split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # ============================================================================
    # HARDWARE CONFIGURATION (RTX 3050 6GB Optimized)
    # ============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataLoader settings
    NUM_WORKERS = 2  # Works well on your system
    PIN_MEMORY = True if torch.cuda.is_available() else False
    PERSISTENT_WORKERS = False
    
    # Mixed Precision Training (FP16) - ESSENTIAL for 6GB VRAM
    USE_AMP = torch.cuda.is_available()
    
    # Gradient Accumulation (to simulate larger batch sizes)
    GRADIENT_ACCUMULATION_STEPS = {
        'convnext_tiny': 1,  # Effective batch = 20
        'swin_tiny': 2,      # Effective batch = 32 (16*2)
        'deit_small': 1      # Effective batch = 22
    }
    
    # ============================================================================
    # MODEL-SPECIFIC CONFIGURATIONS
    # ============================================================================
    
    # Image sizes (native resolution for each model)
    IMAGE_SIZES = {
        'convnext_tiny': 224,
        'swin_tiny': 224,
        'deit_small': 224,
    }
    
    # Batch sizes (optimized for RTX 3050 6GB with mixed precision)
    BATCH_SIZES = {
        'convnext_tiny': 20,  # ConvNeXt is efficient
        'swin_tiny': 16,      # Swin needs more memory
        'deit_small': 22,     # DeiT-Small is lighter
    }
    
    # Model architecture configs
    MODEL_CONFIGS = {
        'convnext_tiny': {
            'timm_name': 'convnext_tiny.fb_in22k_ft_in1k',
            'pretrained': True,
            'drop_rate': 0.2,
            'drop_path_rate': 0.1,
        },
        'swin_tiny': {
            'timm_name': 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
            'pretrained': True,
            'drop_rate': 0.2,
            'drop_path_rate': 0.2,
        },
        'deit_small': {
            'timm_name': 'deit_small_patch16_224.fb_in1k',
            'pretrained': True,
            'drop_rate': 0.1,
        }
    }
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    
    # Epochs - Set to 50 with early stopping for safety
    NUM_EPOCHS = 50
    WARMUP_EPOCHS = 3
    
    # Optimizer
    OPTIMIZER = 'adamw'
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # Learning rate scheduler
    LR_SCHEDULER = 'cosine'  # 'cosine' or 'step' or 'plateau'
    LR_MIN = 1e-6  # Minimum learning rate for cosine annealing
    
    # For ReduceLROnPlateau (backup scheduler)
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    
    # Loss function
    LABEL_SMOOTHING = 0.1  # Helps with generalization
    
    # Regularization
    GRADIENT_CLIP_NORM = 1.0  # Gradient clipping
    
    # Early stopping - Will stop training if no improvement
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
    EARLY_STOPPING_MIN_DELTA = 0.0001
    
    # ============================================================================
    # DATA AUGMENTATION
    # ============================================================================
    
    AUGMENTATION_CONFIG = {
        'train': {
            # Geometric transformations
            'random_resized_crop_scale': (0.8, 1.0),
            'random_resized_crop_ratio': (0.9, 1.1),
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob': 0.5,
            'rotation_degrees': 30,
            'rotate_prob': 0.5,
            
            # Color transformations
            'color_jitter_prob': 0.3,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            
            # Noise and occlusion
            'random_erasing_prob': 0.2,
            'random_erasing_scale': (0.02, 0.2),
        },
        'val': None,
        'test': None
    }
    
    # ============================================================================
    # LOGGING AND CHECKPOINTING
    # ============================================================================
    
    # Logging frequency
    LOG_INTERVAL = 50  # Log every N batches
    SAVE_CHECKPOINT_EVERY_N_EPOCHS = 5
    
    # What to save
    SAVE_BEST_ONLY = False  # Save checkpoints every N epochs + best
    SAVE_LAST_K_CHECKPOINTS = 3  # Keep only last 3 checkpoints
    
    # Metrics to track
    METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
    TRACK_PER_CLASS_METRICS = True
    
    # Visualization
    PLOT_TRAINING_CURVES = True
    PLOT_CONFUSION_MATRIX = True
    PLOT_PER_CLASS_ACCURACY = True
    
    # ============================================================================
    # REPRODUCIBILITY
    # ============================================================================
    SEED = 42
    CUDNN_DETERMINISTIC = True
    CUDNN_BENCHMARK = False
    
    # ============================================================================
    # EXPERIMENT TRACKING
    # ============================================================================
    EXPERIMENT_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_NAME = "plant_disease_comparison"
    
    # Weights & Biases (Optional)
    USE_WANDB = False
    WANDB_PROJECT = "plant-disease-thesis"
    WANDB_ENTITY = None
    
    # TensorBoard
    USE_TENSORBOARD = True
    
    # ============================================================================
    # TESTING AND INFERENCE
    # ============================================================================
    TEST_TIME_AUGMENTATION = False
    TTA_TRANSFORMS = 5
    
    # ============================================================================
    # PAPER/THESIS SPECIFIC
    # ============================================================================
    SAVE_PREDICTIONS = True
    SAVE_MISCLASSIFIED = True
    COMPUTE_CLASS_ACTIVATION_MAPS = False
    
    # ============================================================================
    # GPU MEMORY OPTIMIZATION
    # ============================================================================
    EMPTY_CACHE_EVERY_N_BATCHES = 100
    USE_GRADIENT_CHECKPOINTING = False
    
    # ============================================================================
    # METHODS
    # ============================================================================
    
    @classmethod
    def update_num_classes(cls, num_classes, class_names):
        """Update number of classes after loading data"""
        cls.NUM_CLASSES = num_classes
        cls.CLASS_NAMES = class_names
    
    @classmethod
    def get_model_config(cls, model_name):
        """Get all settings for a specific model"""
        return {
            'name': model_name,
            'timm_name': cls.MODEL_CONFIGS[model_name]['timm_name'],
            'pretrained': cls.MODEL_CONFIGS[model_name]['pretrained'],
            'num_classes': cls.NUM_CLASSES,
            'image_size': cls.IMAGE_SIZES[model_name],
            'batch_size': cls.BATCH_SIZES[model_name],
            'drop_rate': cls.MODEL_CONFIGS[model_name].get('drop_rate', 0.0),
            'drop_path_rate': cls.MODEL_CONFIGS[model_name].get('drop_path_rate', 0.0),
            'gradient_accumulation': cls.GRADIENT_ACCUMULATION_STEPS[model_name],
        }
    
    @classmethod
    def save_config(cls, filepath=None):
        """Save configuration to JSON file"""
        if filepath is None:
            filepath = cls.LOGS_DIR / f'config_{cls.EXPERIMENT_ID}.json'
        
        config_dict = {
            'dataset': cls.DATASET_NAME,
            'num_classes': cls.NUM_CLASSES,
            'class_names': cls.CLASS_NAMES,
            'hardware': {
                'device': str(cls.DEVICE),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'mixed_precision': cls.USE_AMP,
            },
            'models': cls.MODEL_CONFIGS,
            'image_sizes': cls.IMAGE_SIZES,
            'batch_sizes': cls.BATCH_SIZES,
            'training': {
                'epochs': cls.NUM_EPOCHS,
                'learning_rate': cls.LEARNING_RATE,
                'optimizer': cls.OPTIMIZER,
                'scheduler': cls.LR_SCHEDULER,
                'label_smoothing': cls.LABEL_SMOOTHING,
                'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            },
            'experiment_id': cls.EXPERIMENT_ID,
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return filepath
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        print(f"\n📊 Dataset:")
        print(f"   Name: {cls.DATASET_NAME}")
        print(f"   Classes: {cls.NUM_CLASSES}")
        print(f"   Split: {cls.TRAIN_RATIO}/{cls.VAL_RATIO}/{cls.TEST_RATIO}")
        
        print(f"\n🖥️ Hardware:")
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   Mixed Precision: {cls.USE_AMP}")
            print(f"   Num Workers: {cls.NUM_WORKERS}")
        else:
            print(f"   Device: CPU")
        
        print(f"\n🤖 Models:")
        for model_name in cls.MODEL_CONFIGS.keys():
            cfg = cls.get_model_config(model_name)
            print(f"   {model_name}:")
            print(f"      Image: {cfg['image_size']}×{cfg['image_size']}")
            print(f"      Batch: {cfg['batch_size']}")
            print(f"      Accum: {cfg['gradient_accumulation']}x")
            print(f"      Effective: {cfg['batch_size'] * cfg['gradient_accumulation']}")
        
        print(f"\n⚙️ Training:")
        print(f"   Max Epochs: {cls.NUM_EPOCHS}")
        print(f"   Learning Rate: {cls.LEARNING_RATE}")
        print(f"   Optimizer: {cls.OPTIMIZER}")
        print(f"   Scheduler: {cls.LR_SCHEDULER}")
        print(f"   Early Stopping: {cls.EARLY_STOPPING} (patience={cls.EARLY_STOPPING_PATIENCE})")
        print(f"   Label Smoothing: {cls.LABEL_SMOOTHING}")
        
        print(f"\n📁 Paths:")
        print(f"   Data: {cls.PROCESSED_DATA_DIR}")
        print(f"   Models: {cls.MODEL_DIR}")
        print(f"   Results: {cls.RESULTS_DIR}")
        
        print(f"\n🆔 Experiment: {cls.EXPERIMENT_ID}")
        print("="*70 + "\n")


# Initialize config
config = Config()

# Print GPU info on import
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA: {torch.version.cuda}")
    print(f"✅ Mixed Precision: Enabled")
else:
    print("⚠️ No GPU detected - Training will be very slow!")