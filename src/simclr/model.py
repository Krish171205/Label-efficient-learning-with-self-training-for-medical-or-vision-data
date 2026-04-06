"""
SimCLR contrastive pretraining for chest X-ray backbone.

SimCLR learns visual representations WITHOUT labels by:
1. Taking each image and creating two augmented views
2. Training the backbone to produce similar embeddings for both views
3. While pushing embeddings of different images apart

After pretraining, the backbone understands chest X-ray structure
(lung texture, opacity patterns, anatomy) even before seeing a single label.

Uses the lightly library for the SimCLR framework.
"""

import torch
import torch.nn as nn
import timm
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss


class SimCLRModel(nn.Module):
    """
    SimCLR model = backbone + projection head.
    
    Architecture:
        Image → ResNet-18 backbone → 512-d features → projection head → 128-d embedding
    
    The projection head is discarded after pretraining.
    Only the backbone is kept for downstream classification.
    """
    
    def __init__(self, backbone_name: str = "resnet18", projection_dim: int = 128):
        """
        Args:
            backbone_name: timm model name
            projection_dim: Output dimension of projection head (default 128)
        """
        super().__init__()
        
        # Backbone WITHOUT pretrained weights (we're learning from scratch on our data)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,    # No ImageNet — learn from unlabeled chest X-rays
            num_classes=0,       # Remove classification head
        )
        
        self.feature_dim = self.backbone.num_features  # 512 for ResNet-18
        
        # SimCLR projection head: 512 → 512 → 128
        # This is discarded after pretraining — it exists only to make
        # the contrastive loss work better (proven in the SimCLR paper)
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=self.feature_dim,
            output_dim=projection_dim,
        )
        
        print(f"✓ SimCLR model created")
        print(f"  Backbone: {backbone_name} (random init — no ImageNet)")
        print(f"  Features: {self.feature_dim} → projection → {projection_dim}")
        print(f"  Params:   {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone + projection head.
        
        Args:
            x: Augmented images, shape (B, 3, 224, 224)
        
        Returns:
            Projected embeddings, shape (B, 128)
        """
        features = self.backbone(x)                # (B, 512)
        projections = self.projection_head(features)  # (B, 128)
        return projections
    
    def get_backbone_state_dict(self) -> dict:
        """
        Extract backbone weights for downstream use.
        Call this after pretraining to save just the backbone.
        
        Returns:
            State dict of the backbone only (no projection head)
        """
        return self.backbone.state_dict()
