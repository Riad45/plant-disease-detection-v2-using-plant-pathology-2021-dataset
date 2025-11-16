"""
OPTIMIZED Configuration for Plant Disease Detection
Speed + Accuracy Balanced for RTX 3050 6GB - FIXED VERSION
"""

import torch
from pathlib import Path
from datetime import datetime


class Config:
    """Main configuration class - all hyperparameters in one place"""

    # ========================================================================
    # PROJECT PATHS
    # ========================================================================
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

    # ========================================================================
    # DATASET CONFIGURATION
    # ========================================================================
    DATASET_NAME = "timm/plant-pathology-2021"
    USE_ONLY_TRAIN_SPLIT = True

    NUM_CLASSES = None
    CLASS_NAMES = None

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # ========================================================================
    # HARDWARE CONFIGURATION (Windows Optimized - FIXED!)
    # ========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader settings (FIXED FOR PERFORMANCE)
    NUM_WORKERS = 2  # CHANGED: 0 → 2 for better GPU utilization
    PIN_MEMORY = True if torch.cuda.is_available() else False
    PERSISTENT_WORKERS = True  # CHANGED: False → True for better performance
    PREFETCH_FACTOR = 2  # ADDED: Prefetching for faster data loading

    # Mixed Precision Training (FP16) - ESSENTIAL for 6GB GPU
    USE_AMP = torch.cuda.is_available()

    # Gradient Accumulation steps per model
    GRADIENT_ACCUMULATION_STEPS = {
        'convnext_tiny': 1,
        'swin_tiny': 1,  # CHANGED: 2 → 1 for stability
        'deit_small': 1
    }

    # ========================================================================
    # MODEL-SPECIFIC CONFIGURATIONS
    # ========================================================================
    IMAGE_SIZES = {
        'convnext_tiny': 224,
        'swin_tiny': 224,
        'deit_small': 224,
    }

    # Batch sizes (Optimized for RTX 3050 6GB - FIXED!)
    BATCH_SIZES = {
        'convnext_tiny': 32,  # CHANGED: 16 → 32 (2x increase)
        'swin_tiny': 24,      # CHANGED: 12 → 24 (2x increase)
        'deit_small': 32,     # CHANGED: 18 → 32 (almost 2x increase)
    }

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

    # ========================================================================
    # TRAINING HYPERPARAMETERS
    # ========================================================================
    NUM_EPOCHS = 50
    WARMUP_EPOCHS = 3

    OPTIMIZER = 'adamw'
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    LR_SCHEDULER = 'cosine'
    LR_MIN = 1e-6

    LABEL_SMOOTHING = 0.1
    GRADIENT_CLIP_NORM = 1.0

    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.0001

    # ========================================================================
    # DATA AUGMENTATION
    # ========================================================================
    AUGMENTATION_CONFIG = {
        'train': {
            'random_resized_crop_scale': (0.8, 1.0),
            'random_resized_crop_ratio': (0.9, 1.1),
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob': 0.5,
            'rotation_degrees': 30,
            'rotate_prob': 0.5,
            'color_jitter_prob': 0.3,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'random_erasing_prob': 0.2,
            'random_erasing_scale': (0.02, 0.2),
        },
        'val': None,
        'test': None
    }

    # ========================================================================
    # LOGGING AND CHECKPOINTING
    # ========================================================================
    LOG_INTERVAL = 50
    SAVE_CHECKPOINT_EVERY_N_EPOCHS = 5
    SAVE_BEST_ONLY = False
    SAVE_LAST_K_CHECKPOINTS = 3

    METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
    TRACK_PER_CLASS_METRICS = True

    PLOT_TRAINING_CURVES = True
    PLOT_CONFUSION_MATRIX = True
    PLOT_PER_CLASS_ACCURACY = True

    # ========================================================================
    # REPRODUCIBILITY vs SPEED
    # ========================================================================
    SEED = 42
    CUDNN_DETERMINISTIC = False
    CUDNN_BENCHMARK = True

    # ========================================================================
    # EXPERIMENT TRACKING
    # ========================================================================
    EXPERIMENT_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_NAME = "plant_disease_comparison"

    USE_WANDB = True
    WANDB_PROJECT = "plant-disease-thesis"
    WANDB_ENTITY = None

    USE_TENSORBOARD = True

    # ========================================================================
    # TESTING / INFERENCE
    # ========================================================================
    TEST_TIME_AUGMENTATION = False
    TTA_TRANSFORMS = 5

    # ========================================================================
    # PAPER / THESIS SPECIFIC
    # ========================================================================
    SAVE_PREDICTIONS = True
    SAVE_MISCLASSIFIED = True
    COMPUTE_CLASS_ACTIVATION_MAPS = False

    # ========================================================================
    # GPU MEMORY OPTIMIZATION
    # ========================================================================
    EMPTY_CACHE_EVERY_N_BATCHES = 100
    USE_GRADIENT_CHECKPOINTING = False

    # ========================================================================
    # WINDOWS MULTIPROCESSING FIX
    # ========================================================================
    MULTIPROCESSING_START_METHOD = 'spawn'  # Required for Windows

    # ========================================================================
    # METHODS
    # ========================================================================
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
                'num_workers': cls.NUM_WORKERS,
                'batch_sizes': cls.BATCH_SIZES,
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
        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY - FIXED VERSION")
        print("=" * 70)

        print(f"\n📊 Dataset:")
        print(f"   Name: {cls.DATASET_NAME}")
        print(f"   Classes: {cls.NUM_CLASSES}")
        print(f"   Split: {cls.TRAIN_RATIO}/{cls.VAL_RATIO}/{cls.TEST_RATIO}")

        print(f"\n🖥️ Hardware (FIXED FOR PERFORMANCE):")
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   Mixed Precision: {cls.USE_AMP}")
            print(f"   Num Workers: {cls.NUM_WORKERS} ⭐ (was 0)")
            print(f"   Prefetch Factor: {cls.PREFETCH_FACTOR}")
            print(f"   Speed Mode: benchmark={cls.CUDNN_BENCHMARK}")
        else:
            print(f"   Device: CPU")

        print(f"\n🤖 Models (OPTIMIZED BATCH SIZES):")
        for model_name in cls.MODEL_CONFIGS.keys():
            cfg = cls.get_model_config(model_name)
            print(f"   {model_name}:")
            print(f"      Image: {cfg['image_size']}×{cfg['image_size']}")
            print(f"      Batch: {cfg['batch_size']} ⭐ (increased)")
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
        print(f"\n🎯 EXPECTED PERFORMANCE:")
        print(f"   Epoch Time: 2-5 minutes (was 20 minutes)")
        print(f"   GPU Utilization: 70-95% (was 0-30%)")
        print(f"   Total Training: 2-4 hours (was 16+ hours)")
        print("=" * 70 + "\n")


# Initialize config instance
config = Config()

# Print performance improvements
if torch.cuda.is_available():
    print("🚀 PERFORMANCE FIXES APPLIED:")
    print("   ✅ num_workers = 2 (was 0)")
    print("   ✅ Batch sizes increased 2x")
    print("   ✅ Prefetching enabled")
    print("   ✅ Persistent workers enabled")
    print("   Expected speed improvement: 5-10x faster!")
else:
    print("⚠️  No GPU detected - training will be slow")