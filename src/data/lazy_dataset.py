"""
Lazy-loading ChestMNIST dataset — uses .npy files with memory mapping.

Why .npy instead of .npz?
- .npz = compressed ZIP -> must decompress entirely into RAM per worker
- .npy = uncompressed -> true mmap, OS shares pages across ALL workers

With this, 4 workers share the same ~3.67 GB of physical RAM.
No duplication, no OOM.

Prerequisites: Run 'python convert_to_npy.py' once to create the .npy files.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LazyChestMNIST(Dataset):
    """
    Memory-efficient ChestMNIST that supports num_workers > 0 on Windows.
    
    Uses .npy files with memory-mapping:
    - Only the file path is pickled (tiny, fast serialization)
    - Each worker memory-maps the .npy file independently
    - OS shares physical RAM pages across all workers automatically
    - Total RAM used: ~3.67 GB regardless of num_workers
    """
    
    def __init__(self, root: str, split: str, transform=None, size: int = 224):
        """
        Args:
            root: Directory containing chestmnist_npy/ subfolder
            split: "train", "val", or "test"
            transform: torchvision transforms to apply
            size: Image resolution (224)
        """
        self.npy_dir = os.path.join(root, "chestmnist_npy")
        self.split = split
        self.transform = transform
        
        self._img_path = os.path.join(self.npy_dir, f"{split}_images.npy")
        self._lbl_path = os.path.join(self.npy_dir, f"{split}_labels.npy")
        
        if not os.path.exists(self._img_path):
            raise FileNotFoundError(
                f"Dataset not found at {self._img_path}. "
                f"Run 'python scripts/convert_to_npy.py' first."
            )
        
        # Get length without loading data
        self._length = np.load(self._img_path, mmap_mode="r").shape[0]
        print(f"✅ {split} set ready. ({self._length} images, memory-mapped)")
        
        # Actual data — loaded lazily via mmap in each worker
        self._images = None
        self._labels = None
    
    def _ensure_loaded(self):
        """Memory-map the .npy files (called once per worker process)."""
        if self._images is None:
            # mmap_mode="r" = read-only memory mapping
            # OS handles paging — only accessed pages use physical RAM
            # All workers share the same physical pages automatically
            self._images = np.load(self._img_path, mmap_mode="r")
            self._labels = np.load(self._lbl_path, mmap_mode="r")
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        self._ensure_loaded()
        
        # Read just ONE image from the mmap — only this page enters RAM
        img = np.array(self._images[idx])  # Copy from mmap to regular array
        label = np.array(self._labels[idx])
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __getstate__(self):
        """Only pickle paths — NOT data arrays. Workers reload via mmap."""
        state = self.__dict__.copy()
        state["_images"] = None
        state["_labels"] = None
        return state
    
    def __setstate__(self, state):
        """Unpickle in worker — data will be mmap'd on first access."""
        self.__dict__.update(state)