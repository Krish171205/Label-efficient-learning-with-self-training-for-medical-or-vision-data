"""
ChestMNIST data loading with label-fraction simulation.

This module handles:
1. Loading ChestMNIST from the local data/ directory
2. Splitting training data into labeled + unlabeled pools
3. Applying augmentations
4. Creating DataLoaders for all phases
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO


# Standard ImageNet normalization (used with pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ConvertToRGB:
    """Picklable replacement for transforms.Lambda(lambda img: img.convert('RGB')).
    Lambda functions can't be pickled, which breaks num_workers > 0 on Windows."""
    def __call__(self, img):
        return img.convert("RGB")


def get_transforms(split: str = "train", image_size: int = 224):
    """
    Get image transforms for different phases.
    
    Args:
        split: "train" (with augmentation) or "test" (no augmentation)
        image_size: Target image size
    
    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            ConvertToRGB(),                                      # Grayscale → 3ch
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            ConvertToRGB(),                                      # Grayscale → 3ch
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ChestMNISTDataset:
    """
    Wrapper around MedMNIST's ChestMNIST with label-fraction simulation.
    
    Key feature: split training data into a small labeled pool and a large
    unlabeled pool to simulate the low-label regime.
    """
    
    def __init__(self, data_root: str = "data", image_size: int = 224, 
                 label_fraction: float = 0.01, seed: int = 42):
        """
        Args:
            data_root: Path to directory containing chestmnist_224.npz
            image_size: Image resolution (should be 224)
            label_fraction: Fraction of training data to label (0.01 = 1%)
            seed: Random seed for reproducible splits
        """
        self.data_root = data_root
        self.image_size = image_size
        self.label_fraction = label_fraction
        self.seed = seed
        
        # Get dataset info
        self.num_classes = 14  # ChestMNIST has 14 disease labels
        
        # Use lazy-loading dataset (fixes Windows multiprocessing)
        from src.data.lazy_dataset import LazyChestMNIST
        
        self.train_dataset = LazyChestMNIST(
            root=data_root, split="train",
            transform=get_transforms("train", image_size), size=image_size
        )
        self.val_dataset = LazyChestMNIST(
            root=data_root, split="val",
            transform=get_transforms("test", image_size), size=image_size
        )
        self.test_dataset = LazyChestMNIST(
            root=data_root, split="test",
            transform=get_transforms("test", image_size), size=image_size
        )
        
        # Split training data into labeled + unlabeled
        self._split_labeled_unlabeled()
        
        print(f"✓ ChestMNIST loaded from {data_root}/")
        print(f"  Total training images: {len(self.train_dataset)}")
        print(f"  Labeled pool:   {len(self.labeled_indices)} ({label_fraction*100:.0f}%)")
        print(f"  Unlabeled pool: {len(self.unlabeled_indices)} ({(1-label_fraction)*100:.0f}%)")
        print(f"  Validation:     {len(self.val_dataset)}")
        print(f"  Test:           {len(self.test_dataset)}")
        print(f"  Classes:        {self.num_classes}")
    
    def _split_labeled_unlabeled(self):
        """Split training indices into labeled and unlabeled pools."""
        rng = np.random.RandomState(self.seed)
        total = len(self.train_dataset)
        all_indices = np.arange(total)
        rng.shuffle(all_indices)
        
        n_labeled = max(1, int(total * self.label_fraction))
        
        self.labeled_indices = all_indices[:n_labeled].tolist()
        self.unlabeled_indices = all_indices[n_labeled:].tolist()
    
    def get_labeled_loader(self, batch_size: int = 64, num_workers: int = 4,
                           pin_memory: bool = True) -> DataLoader:
        """DataLoader for the small labeled pool (for supervised training)."""
        labeled_subset = Subset(self.train_dataset, self.labeled_indices)
        return DataLoader(
            labeled_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            persistent_workers=(num_workers > 0),
        )
    
    def get_unlabeled_loader(self, batch_size: int = 64, num_workers: int = 4,
                             pin_memory: bool = True) -> DataLoader:
        """DataLoader for the unlabeled pool (for pseudo-labeling / SimCLR)."""
        unlabeled_subset = Subset(self.train_dataset, self.unlabeled_indices)
        return DataLoader(
            unlabeled_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
            persistent_workers=(num_workers > 0),
        )
    
    def get_full_train_loader(self, batch_size: int = 64, num_workers: int = 4,
                              pin_memory: bool = True) -> DataLoader:
        """DataLoader for ALL training data (used in SimCLR pretraining)."""
        return DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            persistent_workers=(num_workers > 0),
        )
    
    def get_val_loader(self, batch_size: int = 64, num_workers: int = 4,
                       pin_memory: bool = True) -> DataLoader:
        """DataLoader for validation."""
        return DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
            persistent_workers=(num_workers > 0),
        )
    
    def get_test_loader(self, batch_size: int = 64, num_workers: int = 4,
                        pin_memory: bool = True) -> DataLoader:
        """DataLoader for final test evaluation."""
        return DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
            persistent_workers=(num_workers > 0),
        )
    
    def add_pseudo_labeled(self, indices: list, pseudo_labels: np.ndarray):
        """
        Add pseudo-labeled indices to the labeled pool.
        Called during each round of self-training.
        
        Args:
            indices: Indices in the unlabeled pool that passed the threshold
            pseudo_labels: Their predicted labels, shape (len(indices), 14)
        """
        # Move from unlabeled to labeled pool
        for idx in indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.append(idx)
        
        print(f"  → Added {len(indices)} pseudo-labeled images")
        print(f"    Labeled pool now: {len(self.labeled_indices)}")
        print(f"    Unlabeled pool now: {len(self.unlabeled_indices)}")
