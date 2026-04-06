import os
import medmnist
from medmnist import INFO

# Download ChestMNIST only (chest X-ray, multi-label, 14 classes)
# ~3.9 GB at 224×224 resolution (compressed .npz)
# Stored locally in ./data/ (not ~/.medmnist/)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_ROOT, exist_ok=True)

dataset_name = "chestmnist"
DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])

for split in ['train', 'val', 'test']:
    dataset = DataClass(split=split, download=True, size=224, root=DATA_ROOT)
    print(f"✓ {dataset_name} [{split}] — {len(dataset)} images downloaded")

print(f"\n✅ Done! ChestMNIST saved to {DATA_ROOT}/")
print(f"   File: chestmnist_224.npz")
print(f"   Total images: 112,120 | Classes: 14 (multi-label)")
print(f"   Resolution: 224×224 | Task: multi-label binary classification")