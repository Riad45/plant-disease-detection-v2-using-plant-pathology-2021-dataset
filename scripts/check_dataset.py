"""Check the actual Plant Pathology 2021 dataset structure"""

from datasets import load_dataset

print("Checking dataset structure...")
dataset = load_dataset("timm/plant-pathology-2021", split="train")

print(f"\nTotal samples: {len(dataset)}")
print(f"\nDataset features: {dataset.features}")
print(f"\nFirst sample:")
sample = dataset[0]
for key, value in sample.items():
    if key == 'image':
        print(f"  {key}: PIL Image object")
    else:
        print(f"  {key}: {value}")

print("\nChecking a few more samples...")
for i in range(3):
    print(f"\nSample {i}:")
    print(f"  labels: {dataset[i]['labels']}")
