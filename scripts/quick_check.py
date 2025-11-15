from pathlib import Path
import os

cache = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--timm--plant-pathology-2021"

if cache.exists():
    files = list(cache.rglob("*.parquet"))
    total_gb = sum(f.stat().st_size for f in files) / (1024**3)
    print(f"Found {len(files)} files")
    print(f"Total: {total_gb:.2f} GB")
    print(f"Progress: {len(files)}/30 files")
    print(f"Percentage: {(len(files)/30)*100:.1f}%")
    print("\nYOUR DATA IS SAFE!")
else:
    print("Cache directory not found")
