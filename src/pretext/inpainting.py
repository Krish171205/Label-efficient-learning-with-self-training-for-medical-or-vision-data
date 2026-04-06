"""
Inpainting pretext task.

The idea: mask a random square patch in the image, then train the model
to reconstruct what was under the mask. To do this, the backbone MUST learn
spatial coherence — what a region of a lung should look like given its neighbors.

Architecture:
    Masked image → ResNet-18 backbone → features → Decoder → reconstructed patch
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


from src.data.chest_mnist import ConvertToRGB  # Picklable RGB converter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InpaintingDataset(Dataset):
    """
    Wraps any image dataset to produce (masked_image, original_image) pairs.
    
    For each image, masks a random square patch. The model must reconstruct
    the full original image from the masked input.
    
    Labels are FREE — the original image IS the target.
    """
    
    def __init__(self, base_dataset, image_size: int = 224, mask_size: int = 56):
        """
        Args:
            base_dataset: A MedMNIST dataset (raw, without transforms)
            image_size: Target image size (224)
            mask_size: Size of the square mask (56 = 25% of image)
        """
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.mask_size = mask_size
        
        # Transform to get the clean target image
        self.transform = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Note: no normalization for the target — we reconstruct raw pixels
        ])
        
        # Transform for the masked input (with normalization for backbone)
        self.input_transform = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        # Get clean target
        target = self.transform(img)  # (3, 224, 224), range [0, 1]
        
        # Get input (normalized for backbone)
        input_img = self.input_transform(img)  # (3, 224, 224), normalized
        
        # Create random mask
        mask = torch.ones(1, self.image_size, self.image_size)
        
        # Random position for the mask
        max_pos = self.image_size - self.mask_size
        top = np.random.randint(0, max_pos + 1)
        left = np.random.randint(0, max_pos + 1)
        
        # Zero out the masked region in input
        mask[:, top:top+self.mask_size, left:left+self.mask_size] = 0
        masked_input = input_img * mask  # Masked region becomes 0
        
        return masked_input, target, mask


class InpaintingModel(nn.Module):
    """
    Inpainting model = backbone (encoder) + decoder.
    
    The decoder upsamples backbone features to reconstruct the full image.
    After pretraining, discard the decoder and keep the backbone.
    """
    
    def __init__(self, backbone_name: str = "resnet18", image_size: int = 224):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
        )
        
        self.feature_dim = self.backbone.num_features  # 512 for ResNet-18
        
        # Decoder: takes 512-d features → reconstructs (3, 224, 224) image
        # We use a simple MLP + reshape approach for efficiency:
        # 512 → 2048 → 7x7 spatial → upsample to 224x224
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling layers to go from 7x7 to 224x224
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 64, kernel_size=4, stride=2, padding=1),   # 7→14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14→28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 28→56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 56→112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),    # 112→224
            nn.Sigmoid(),  # Output in [0, 1] to match target
        )
        
        # MSE loss for pixel reconstruction
        self.criterion = nn.MSELoss()
        
        print(f"✓ InpaintingModel created")
        print(f"  Backbone: {backbone_name} (random init)")
        print(f"  Task: reconstruct masked image regions")
        print(f"  Params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Masked input image, shape (B, 3, 224, 224)
        
        Returns:
            Reconstructed image, shape (B, 3, 224, 224)
        """
        features = self.backbone(x)           # (B, 512)
        decoded = self.decoder(features)       # (B, 3*7*7)
        decoded = decoded.view(-1, 3, 7, 7)    # (B, 3, 7, 7)
        reconstructed = self.upsample(decoded) # (B, 3, 224, 224)
        return reconstructed
    
    def compute_loss(self, reconstructed: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE), optionally weighted on masked region.
        
        Args:
            reconstructed: Model output, shape (B, 3, 224, 224)
            target: Original image, shape (B, 3, 224, 224)
            mask: Binary mask (1=visible, 0=masked), shape (B, 1, 224, 224)
        """
        if mask is not None:
            # Focus loss on the masked (inpainted) region
            inverted_mask = 1.0 - mask  # 1 where masked, 0 where visible
            masked_loss = ((reconstructed - target) ** 2 * inverted_mask).sum()
            masked_loss = masked_loss / (inverted_mask.sum() + 1e-8)
            
            # Also small loss on visible region for global coherence
            visible_loss = ((reconstructed - target) ** 2 * mask).sum()
            visible_loss = visible_loss / (mask.sum() + 1e-8)
            
            return 0.8 * masked_loss + 0.2 * visible_loss
        else:
            return self.criterion(reconstructed, target)
    
    def get_backbone_state_dict(self) -> dict:
        return self.backbone.state_dict()
