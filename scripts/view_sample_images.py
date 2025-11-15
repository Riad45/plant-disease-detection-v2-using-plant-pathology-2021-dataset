"""
View actual images from the dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

from src.config import config


def view_samples():
    print("="*70)
    print("VIEWING ACTUAL DATASET IMAGES")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_dict = load_dataset(config.DATASET_NAME)
    hf_dataset = dataset_dict['train']
    
    # Load train CSV to see which images we're using
    train_df = pd.read_csv(config.PROCESSED_DATA_DIR / 'train.csv')
    
    print(f"✅ Dataset loaded: {len(hf_dataset)} images")
    print(f"✅ Train CSV loaded: {len(train_df)} samples")
    
    # Show first 8 images from our training set
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    print("\nDisplaying first 8 training images...")
    
    for i in range(8):
        # Get sample from our train split
        row = train_df.iloc[i]
        sample_id = int(row['sample_id'])
        label_name = row['label_name']
        
        # Load ACTUAL image from HuggingFace
        image = hf_dataset[sample_id]['image']
        
        # Display
        axes[i].imshow(image)
        axes[i].set_title(f'ID: {sample_id}\n{label_name}', fontsize=10)
        axes[i].axis('off')
        
        print(f"   Image {i}: sample_id={sample_id}, label={label_name}, size={image.size}")
    
    plt.suptitle('Actual Images from HuggingFace Dataset', fontsize=14)
    plt.tight_layout()
    
    save_path = config.RESULTS_DIR / 'plots' / 'raw_dataset_images.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Images saved to: {save_path}")
    
    plt.show()
    
    print("\n✅ These are the ACTUAL images your model will train on!")


if __name__ == "__main__":
    view_samples()