"""
Logging utilities - WandB integration
"""

import wandb
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class WandbLogger:
    """
    Weights & Biases logger wrapper
    """
    
    def __init__(self, model_name, project=None, entity=None, tags=None):
        """
        Initialize WandB logger
        
        Args:
            model_name: Name of the model
            project: WandB project name
            entity: WandB entity (username/team)
            tags: List of tags
        """
        self.model_name = model_name
        
        if config.USE_WANDB:
            # Initialize wandb
            wandb.init(
                project=project or config.WANDB_PROJECT,
                entity=entity or config.WANDB_ENTITY,
                name=f"{model_name}_{config.EXPERIMENT_ID}",
                tags=tags or [model_name, 'plant-pathology'],
                config={
                    'model': model_name,
                    'dataset': config.DATASET_NAME,
                    'num_classes': config.NUM_CLASSES,
                    'epochs': config.NUM_EPOCHS,
                    'batch_size': config.BATCH_SIZES[model_name],
                    'learning_rate': config.LEARNING_RATE,
                    'optimizer': config.OPTIMIZER,
                    'scheduler': config.LR_SCHEDULER,
                    'image_size': config.IMAGE_SIZES[model_name],
                    'gradient_accumulation': config.GRADIENT_ACCUMULATION_STEPS[model_name],
                    'mixed_precision': config.USE_AMP,
                    'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
                    'label_smoothing': config.LABEL_SMOOTHING,
                }
            )
            
            print(f"\n✅ WandB initialized: {wandb.run.name}")
            print(f"   Project: {wandb.run.project}")
            print(f"   URL: {wandb.run.url}\n")
            
            self.enabled = True
        else:
            print("\n⚠️  WandB disabled in config")
            self.enabled = False
    
    def log(self, metrics, step=None):
        """Log metrics to WandB"""
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_image(self, key, image):
        """Log image to WandB"""
        if self.enabled:
            wandb.log({key: wandb.Image(image)})
    
    def log_confusion_matrix(self, cm, class_names):
        """Log confusion matrix to WandB"""
        if self.enabled:
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=None,
                    preds=None,
                    class_names=class_names
                )
            })
    
    def finish(self):
        """Finish WandB run"""
        if self.enabled:
            wandb.finish()
            print("\n✅ WandB run finished")


if __name__ == "__main__":
    print("✅ Logger module loaded successfully!")