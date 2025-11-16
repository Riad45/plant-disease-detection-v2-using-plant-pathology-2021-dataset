"""
Verify all labels are in valid range before training
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import config

print("\n" + "="*70)
print("VERIFYING LABEL INTEGRITY")
print("="*70 + "\n")

# Load stats
stats_path = config.PROCESSED_DATA_DIR / 'statistics.json'
import json
with open(stats_path, 'r') as f:
    stats = json.load(f)

num_classes = stats['num_classes']
print(f"Number of classes: {num_classes}")
print(f"Valid label range: [0, {num_classes-1}]\n")

all_valid = True

for split in ['train', 'val', 'test']:
    csv_path = config.PROCESSED_DATA_DIR / f'{split}.csv'
    df = pd.read_csv(csv_path)
    
    labels = df['label_encoded'].values
    
    min_label = labels.min()
    max_label = labels.max()
    invalid = (labels < 0) | (labels >= num_classes)
    
    print(f"{split.upper()}:")
    print(f"  Samples: {len(labels)}")
    print(f"  Min label: {min_label}")
    print(f"  Max label: {max_label}")
    print(f"  Invalid labels: {invalid.sum()}")
    
    if invalid.sum() > 0:
        print(f"  ❌ ERROR: Found {invalid.sum()} invalid labels!")
        print(f"  Invalid rows: {df[invalid][['sample_id', 'label_encoded', 'label_name']].head(10)}")
        all_valid = False
    else:
        print(f"  ✅ All labels valid")
    print()

print("="*70)
if all_valid:
    print("✅ ALL LABELS VALID - SAFE TO TRAIN!")
else:
    print("❌ INVALID LABELS FOUND - FIX BEFORE TRAINING!")
print("="*70 + "\n")