"""
CORRECT Data Preparation for Multi-Label Plant Disease Classification
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from collections import Counter
from tqdm import tqdm

from src.config import config

def labels_to_multihot(labels, num_diseases=6):
    """
    Convert list of labels to multi-hot encoding
    Example: [0, 2] → [1, 0, 1, 0, 0, 0]
    """
    if isinstance(labels, int):
        labels = [labels]
    
    multihot = np.zeros(num_diseases, dtype=np.float32)
    for label in labels:
        if label < num_diseases:  # Only consider base diseases
            multihot[label] = 1.0
    return multihot.tolist()

def multihot_to_name(multihot):
    """Convert multi-hot encoding to readable name"""
    disease_names = [
        "healthy", "scab", "frog_eye_leaf_spot", 
        "complex", "rust", "powdery_mildew"
    ]
    
    present_diseases = []
    for i, present in enumerate(multihot):
        if present > 0.5:
            present_diseases.append(disease_names[i])
    
    if not present_diseases:
        return "healthy"
    elif len(present_diseases) == 1:
        return present_diseases[0]
    else:
        return "_and_".join(present_diseases)

def main():
    print("="*70)
    print("PLANT PATHOLOGY 2021 - MULTI-LABEL PREPARATION")
    print("CORRECT approach: 6 binary outputs for base diseases")
    print("="*70)
    
    # STEP 1: Load dataset
    print("\n📥 STEP 1: Loading dataset...")
    dataset_dict = load_dataset(config.DATASET_NAME)
    train_data = dataset_dict['train']
    
    # STEP 2: Process as MULTI-LABEL
    print("\n🏷️ STEP 2: Converting to multi-label format...")
    records = []
    
    for idx in tqdm(range(len(train_data)), desc="Processing"):
        sample = train_data[idx]
        original_labels = sample['labels']
        
        # Convert to multi-hot encoding
        multihot_labels = labels_to_multihot(original_labels)
        readable_name = multihot_to_name(multihot_labels)
        
        records.append({
            'sample_id': idx,
            'multihot_labels': multihot_labels,  # [1, 0, 1, 0, 0, 0]
            'readable_name': readable_name,      # "scab_and_frog_eye_leaf_spot"
            'original_labels': original_labels   # [0, 2]
        })
    
    df = pd.DataFrame(records)
    
    # STEP 3: Analyze distribution
    print("\n📊 Disease Distribution Analysis:")
    base_diseases = ["healthy", "scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
    
    disease_counts = {disease: 0 for disease in base_diseases}
    for multihot in df['multihot_labels']:
        for i, present in enumerate(multihot):
            if present > 0.5:
                disease_counts[base_diseases[i]] += 1
    
    print(f"\n{'Disease':<25} {'Count':>8} {'Percentage':>12}")
    print("-" * 50)
    for disease, count in disease_counts.items():
        pct = count/len(df)*100
        print(f"{disease:<25} {count:>8} {pct:>11.1f}%")
    
    # Count combinations
    combination_counts = Counter(df['readable_name'])
    print(f"\n📊 Most Common Disease Combinations:")
    for combo, count in combination_counts.most_common(10):
        print(f"   {combo:<45} {count:>6} samples")
    
    # STEP 4: Create splits
    print(f"\n✂️ STEP 4: Creating splits...")
    
    X = df['sample_id'].values
    y = np.array(df['multihot_labels'].tolist())  # Multi-hot labels
    
    # Simple random split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.SEED
    )
    
    # Create DataFrames
    train_df = pd.DataFrame({
        'sample_id': X_train,
        'multihot_labels': [y_train[i].tolist() for i in range(len(y_train))],
        'readable_name': [multihot_to_name(y_train[i]) for i in range(len(y_train))]
    })
    
    val_df = pd.DataFrame({
        'sample_id': X_val, 
        'multihot_labels': [y_val[i].tolist() for i in range(len(y_val))],
        'readable_name': [multihot_to_name(y_val[i]) for i in range(len(y_val))]
    })
    
    test_df = pd.DataFrame({
        'sample_id': X_test,
        'multihot_labels': [y_test[i].tolist() for i in range(len(y_test))],
        'readable_name': [multihot_to_name(y_test[i]) for i in range(len(y_test))]
    })
    
    # STEP 5: Save files
    print(f"\n💾 STEP 5: Saving multi-label datasets...")
    
    train_df.to_csv(config.PROCESSED_DATA_DIR / 'train_multilabel.csv', index=False)
    val_df.to_csv(config.PROCESSED_DATA_DIR / 'val_multilabel.csv', index=False) 
    test_df.to_csv(config.PROCESSED_DATA_DIR / 'test_multilabel.csv', index=False)
    
    # Save disease information
    disease_info = {
        'base_diseases': base_diseases,
        'num_diseases': len(base_diseases),
        'disease_counts': disease_counts,
        'combination_counts': dict(combination_counts),
        'total_samples': len(df)
    }
    
    with open(config.PROCESSED_DATA_DIR / 'multilabel_info.json', 'w') as f:
        json.dump(disease_info, f, indent=2)
    
    print(f"\n✅ Multi-label dataset prepared!")
    print(f"   Base diseases: {len(base_diseases)}")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Output: 6 binary classifications (multi-label)")

if __name__ == "__main__":
    main()