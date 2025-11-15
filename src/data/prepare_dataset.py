"""
Data Preparation Script
Creates train/val/test splits (80/10/10) from train split only
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from collections import Counter
from tqdm import tqdm

from src.config import config

def main():
    print("="*70)
    print("PLANT PATHOLOGY 2021 - DATA PREPARATION")
    print("Using ONLY Train Split (16,768 samples)")
    print("Creating 80/10/10 splits")
    print("="*70)
    
    # STEP 1: Load dataset
    print("\n📥 STEP 1: Loading dataset from HuggingFace cache...")
    dataset_dict = load_dataset(config.DATASET_NAME)
    
    print(f"\n📊 Dataset splits available:")
    for split_name, split_data in dataset_dict.items():
        print(f"   {split_name}: {len(split_data)} samples")
    
    # Use ONLY train split
    train_data = dataset_dict['train']
    
    print(f"\n✅ Using: TRAIN split with {len(train_data)} samples")
    print(f"   (Official validation: {len(dataset_dict['validation'])} samples - reserved for future use)")
    
    # STEP 2: Process labels
    print("\n🏷️ STEP 2: Extracting and processing labels...")
    records = []
    
    for idx in tqdm(range(len(train_data)), desc="Processing"):
        sample = train_data[idx]
        labels = sample['labels']
        
        # Convert labels to string format
        if isinstance(labels, list):
            label_str = '_'.join(map(str, sorted(labels)))
        else:
            label_str = str(labels)
        
        records.append({
            'sample_id': idx,
            'label': label_str
        })
    
    df = pd.DataFrame(records)
    print(f"✅ Processed {len(df)} samples")
    
    # STEP 3: Encode labels
    print("\n🔢 STEP 3: Encoding labels to integers...")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()
    
    print(f"✅ Number of classes: {num_classes}")
    
    # Show class distribution
    print("\n📊 Class distribution:")
    class_dist = df['label'].value_counts()
    print(f"\n{'Class':<35} {'Count':>8} {'Percentage':>12}")
    print("-" * 58)
    for label, count in class_dist.items():
        pct = count/len(df)*100
        print(f"{label:<35} {count:>8} {pct:>11.1f}%")
    
    # Save label encoder
    encoder_path = config.PROCESSED_DATA_DIR / 'label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    class_names_path = config.PROCESSED_DATA_DIR / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"\n✅ Label encoder saved: {encoder_path}")
    print(f"✅ Class names saved: {class_names_path}")
    
    # Update config
    config.NUM_CLASSES = num_classes
    config.CLASS_NAMES = class_names
    
    # STEP 4: Create stratified splits
    print(f"\n✂️ STEP 4: Creating stratified splits (80% / 10% / 10%)...")
    
    X = df['sample_id'].values
    y = df['label_encoded'].values
    
    # Set random seed
    np.random.seed(config.SEED)
    
    # First split: 90% temporary, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.10, 
        stratify=y, 
        random_state=config.SEED
    )
    
    # Second split: 80% train, 10% validation (from the 90%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=0.111,  # 10/90 = 0.111
        stratify=y_temp, 
        random_state=config.SEED
    )
    
    # Create DataFrames for each split
    train_df = pd.DataFrame({
        'sample_id': X_train,
        'label_encoded': y_train,
        'label_name': label_encoder.inverse_transform(y_train)
    })
    
    val_df = pd.DataFrame({
        'sample_id': X_val,
        'label_encoded': y_val,
        'label_name': label_encoder.inverse_transform(y_val)
    })
    
    test_df = pd.DataFrame({
        'sample_id': X_test,
        'label_encoded': y_test,
        'label_name': label_encoder.inverse_transform(y_test)
    })
    
    # STEP 5: Save split files
    print(f"\n💾 STEP 5: Saving split CSV files...")
    
    train_df.to_csv(config.PROCESSED_DATA_DIR / 'train.csv', index=False)
    val_df.to_csv(config.PROCESSED_DATA_DIR / 'val.csv', index=False)
    test_df.to_csv(config.PROCESSED_DATA_DIR / 'test.csv', index=False)
    
    print(f"✅ {config.PROCESSED_DATA_DIR / 'train.csv'}")
    print(f"✅ {config.PROCESSED_DATA_DIR / 'val.csv'}")
    print(f"✅ {config.PROCESSED_DATA_DIR / 'test.csv'}")
    
    # STEP 6: Verify splits
    print(f"\n🔍 STEP 6: Verifying splits...")
    total = len(df)
    
    print(f"\n{'Split':<12} {'Samples':>10} {'Percentage':>12} {'Classes':>10}")
    print("-" * 48)
    print(f"{'Total':<12} {total:>10,} {100.0:>11.1f}% {num_classes:>10}")
    print(f"{'Train':<12} {len(train_df):>10,} {len(train_df)/total*100:>11.1f}% {train_df['label_encoded'].nunique():>10}")
    print(f"{'Validation':<12} {len(val_df):>10,} {len(val_df)/total*100:>11.1f}% {val_df['label_encoded'].nunique():>10}")
    print(f"{'Test':<12} {len(test_df):>10,} {len(test_df)/total*100:>11.1f}% {test_df['label_encoded'].nunique():>10}")
    
    # Check if all classes present
    if (train_df['label_encoded'].nunique() == num_classes and 
        val_df['label_encoded'].nunique() == num_classes and 
        test_df['label_encoded'].nunique() == num_classes):
        print(f"\n✅ All {num_classes} classes present in all splits!")
    else:
        print(f"\n⚠️ Warning: Some classes missing in splits!")
    
    # STEP 7: Save statistics
    print(f"\n📊 STEP 7: Saving statistics...")
    
    stats = {
        'dataset_source': 'Plant Pathology 2021 (timm/plant-pathology-2021)',
        'split_strategy': 'Use train split only (Option 1)',
        'original_train_samples': len(train_data),
        'official_validation_samples': len(dataset_dict['validation']),
        'total_samples_used': len(df),
        'num_classes': num_classes,
        'class_names': class_names,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'split_ratio': '80/10/10',
        'stratified': True,
        'random_seed': config.SEED,
        'class_distribution': class_dist.to_dict()
    }
    
    stats_path = config.PROCESSED_DATA_DIR / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Statistics saved: {stats_path}")
    
    # Show sample data
    print(f"\n📋 Sample data from each split:")
    
    print(f"\n{'='*70}")
    print("TRAIN Split (first 5):")
    print(train_df[['sample_id', 'label_name', 'label_encoded']].head().to_string(index=False))
    
    print(f"\n{'='*70}")
    print("VALIDATION Split (first 5):")
    print(val_df[['sample_id', 'label_name', 'label_encoded']].head().to_string(index=False))
    
    print(f"\n{'='*70}")
    print("TEST Split (first 5):")
    print(test_df[['sample_id', 'label_name', 'label_encoded']].head().to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Files created in: {config.PROCESSED_DATA_DIR}")
    print(f"\n   ✅ train.csv - {len(train_df):,} samples")
    print(f"   ✅ val.csv - {len(val_df):,} samples")
    print(f"   ✅ test.csv - {len(test_df):,} samples")
    print(f"   ✅ label_encoder.pkl")
    print(f"   ✅ class_names.json")
    print(f"   ✅ statistics.json")
    print(f"\n🚀 Ready for model training!")

if __name__ == "__main__":
    main()