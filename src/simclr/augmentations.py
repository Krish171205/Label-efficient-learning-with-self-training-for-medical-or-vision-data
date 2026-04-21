"""
SimCLR augmentation pipeline — GPU Accelerated with Kornia.

SimCLR requires STRONG augmentations to work — the harder it is for the model
to recognize that two views come from the same image, the more meaningful
features it's forced to learn.

Rather than running these transforms on the CPU (which bottlenecks Kaggle/Colab),
this module allows loading raw tensors and applying the transforms massively in
parallel directly on the GPU using Kornia.
"""

from torchvision import transforms
import torch.nn as nn
import kornia.augmentation as K

from src.data.chest_mnist import ConvertToRGB  # Picklable RGB converter

# ImageNet normalization standard (used by PyTorch Image Models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KorniaDualViewTransform(nn.Module):
    """
    Creates two randomly augmented views of the same batch directly on the GPU.
    
    Expects input tensors of shape (B, C, H, W) normalized to [0, 1].
    """
    def __init__(self, image_size: int = 224):
        super().__init__()
        
        # We wrap Kornia augmentations in nn.Sequential
        self.transform = nn.Sequential(
            K.RandomResizedCrop(size=(image_size, image_size), scale=(0.2, 1.0), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0), p=0.5),
            K.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )

    def forward(self, x):
        """
        Apply transform twice to get two different views of the entire batch.
        x shape: (B, C, H, W)
        """
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


def get_simclr_transform(image_size: int = 224):
    """
    Get the BASE SimCLR transform for the CPU DataLoader.
    
    This NO LONGER does augmentations. It solely converts the raw image to RGB
    and turns it into a generic float tensor [0, 1] for Kornia to ingest on the GPU.
    
    Args:
        image_size: Target image size (Resize just in case, but ChestMNIST is 224)
    
    Returns:
        torchvision.transforms.Compose instance
    """
    return transforms.Compose([
        ConvertToRGB(),                                
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),                         
    ])
