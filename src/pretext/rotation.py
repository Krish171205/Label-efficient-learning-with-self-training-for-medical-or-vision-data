"""
Rotation prediction pretext task.

The idea: rotate an image by 0°, 90°, 180°, or 270°, then train the model
to predict which rotation was applied. To solve this, the backbone MUST learn
what "upright" anatomy looks like — it learns orientation-invariant features.

This is a 4-class classification problem with FREE labels (rotation angle).
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


class RotationDataset(Dataset):
    """
    Wraps any image dataset to produce (rotated_image, rotation_label) pairs.
    
    For each image, randomly picks one of 4 rotations:
        0 → 0°,  1 → 90°,  2 → 180°,  3 → 270°
    
    The labels are FREE — no human annotation needed.
    """
    
    def __init__(self, base_dataset, image_size: int = 224):
        """
        Args:
            base_dataset: A MedMNIST dataset (raw, without transforms)
            image_size: Target image size
        """
        self.base_dataset = base_dataset
        self.image_size = image_size
        
        # Transform applied AFTER rotation
        self.transform = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        self.rotations = [0, 90, 180, 270]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get raw image (MedMNIST returns (image, label))
        img, _ = self.base_dataset[idx]
        
        # img might be a PIL Image or tensor depending on base_dataset transform
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        # Pick random rotation
        rot_idx = np.random.randint(0, 4)
        angle = self.rotations[rot_idx]
        
        # Rotate
        rotated = img.rotate(angle)
        
        # Apply transforms
        rotated = self.transform(rotated)
        
        return rotated, rot_idx  # image, rotation_class (0-3)


class RotationModel(nn.Module):
    """
    Rotation prediction model = backbone + 4-class head.
    
    Architecture:
        Image → ResNet-18 backbone → 512-d features → FC(512 → 4)
    
    After pretraining, discard the rotation head and keep the backbone.
    """
    
    def __init__(self, backbone_name: str = "resnet18"):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
        )
        
        self.feature_dim = self.backbone.num_features
        
        # 4-class rotation head
        self.rotation_head = nn.Linear(self.feature_dim, 4)
        
        # Cross-entropy loss (standard classification)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✓ RotationModel created")
        print(f"  Backbone: {backbone_name} (random init)")
        print(f"  Task: 4-class rotation prediction (0°, 90°, 180°, 270°)")
        print(f"  Params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.rotation_head(features)
        return logits
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)
    
    def get_backbone_state_dict(self) -> dict:
        return self.backbone.state_dict()
