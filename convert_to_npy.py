"""
One-time: Convert chestmnist_224.npz → .npy files for true memory mapping.

Writes in small chunks to avoid disk/OneDrive write limits.

Run this ONCE. After this, all workers share the same physical RAM pages.

Usage:
    python convert_to_npy.py
"""

import os
import numpy as np

DATA_ROOT = "data"
NPZ_PATH = os.path.join(DATA_ROOT, "chestmnist_224.npz")
NPY_DIR = os.path.join(DATA_ROOT, "chestmnist_npy")

os.makedirs(NPY_DIR, exist_ok=True)

print(f"Loading {NPZ_PATH}...")
data = np.load(NPZ_PATH)

for key in data.files:
    arr = data[key]
    out_path = os.path.join(NPY_DIR, f"{key}.npy")
    print(f"  Converting {key} → {out_path}")
    print(f"    Shape: {arr.shape}, Size: {arr.nbytes / 1024**3:.2f} GB")
    
    # Create memory-mapped output file and write in chunks
    # This avoids the "0 written" error on OneDrive/limited filesystems
    fp = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=arr.dtype, shape=arr.shape
    )
    
    # Write 500 images at a time to avoid large single writes
    chunk_size = 500
    total = len(arr)
    for i in range(0, total, chunk_size):
        end = min(i + chunk_size, total)
        fp[i:end] = arr[i:end]
        if (i // chunk_size) % 20 == 0:
            print(f"    Progress: {end}/{total} ({end/total*100:.0f}%)")
    
    fp.flush()
    del fp
    print(f"    ✅ Done")

print(f"\n✅ All .npy files saved to {NPY_DIR}/")
print(f"   You can now use num_workers=4+ without memory issues.")
print(f"\n   Next: python train_simclr.py")
