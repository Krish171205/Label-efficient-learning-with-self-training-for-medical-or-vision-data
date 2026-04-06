"""
SimCLR augmentation pipeline.

SimCLR requires STRONG augmentations to work — the harder it is for the model
to recognize that two views come from the same image, the more meaningful
features it's forced to learn.

This creates a transform that produces two different augmented views
of the same input image.
"""

from torchvision import transforms
from src.data.chest_mnist import ConvertToRGB  # Picklable RGB converter


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DualViewTransform:
    """
    Creates two randomly augmented views of the same image.
    
    SimCLR needs (view1, view2) pairs. This class applies the same
    augmentation pipeline independently twice to produce different views.
    
    Handles grayscale → RGB conversion for ChestMNIST.
    Fully picklable — works with num_workers > 0 on Windows.
    """
    
    def __init__(self, image_size: int = 224):
        self.view_transform = transforms.Compose([
            ConvertToRGB(),                                      # Grayscale → 3ch (picklable)
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    def __call__(self, img):
        """Apply transform twice to get two different views."""
        view1 = self.view_transform(img)
        view2 = self.view_transform(img)
        return view1, view2


def get_simclr_transform(image_size: int = 224):
    """
    Get SimCLR dual-view transform.
    
    Returns a DualViewTransform that produces two augmented views
    from one input image, with grayscale → RGB conversion built in.
    
    Args:
        image_size: Target image size (224)
    
    Returns:
        DualViewTransform instance
    """
    return DualViewTransform(image_size=image_size)
