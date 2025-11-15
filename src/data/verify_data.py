"""
Quick script to verify data preparation was successful
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import pickle
import json

from src.config import config

print("="*70)
print("DATA VERIFICATION")
print("="*70)

# Check if files exist
files_to_check = [
    'train.csv',
    'val.csv',
    'test.csv',
    'label_encoder.pkl',
    'class_names.json',
    'statistics.json'
]

print("\n📁 Checking files...")
all_exist = True
for filename in files_to_check:
    filepath = config.PROCESSED_DATA_DIR / filename
    if filepath.exists():
        print(f"   ✅ {filename}")
    else:
        print(f"   ❌ {filename} - NOT FOUND!")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing!")
    print("Run: python src/data/prepare_dataset.py")
    sys.exit(1)

# Load and verify data
print("\n📊 Loading data...")

train_df = pd.read_csv(config.PROCESSED_DATA_DIR / 'train.csv')
val_df = pd.read_csv(config.PROCESSED_DATA_DIR / 'val.csv')
test_df = pd.read_csv(config.PROCESSED_DATA_DIR / 'test.csv')

with open(config.PROCESSED_DATA_DIR / 'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open(config.PROCESSED_DATA_DIR / 'class_names.json', 'r') as f:
    class_names = json.load(f)

with open(config.PROCESSED_DATA_DIR / 'statistics.json', 'r') as f:
    stats = json.load(f)

print("\n✅ Summary:")
print(f"   Total samples: {stats['total_samples_used']:,}")
print(f"   Number of classes: {stats['num_classes']}")
print(f"   Train samples: {len(train_df):,}")
print(f"   Val samples: {len(val_df):,}")
print(f"   Test samples: {len(test_df):,}")

print("\n✅ Classes:")
for i, cls in enumerate(class_names, 1):
    print(f"   {i}. {cls}")

print("\n✅ All data verified successfully!")
print("\n🚀 Ready to start training!")