"""
Multi-label classifier for ChestMNIST.

Architecture:
    timm ResNet-18 backbone → Global Average Pool → FC(512 → 14) → Sigmoid

The backbone can be:
    1. ImageNet-pretrained (transfer learning baseline)
    2. Randomly initialized (scratch baseline)
    3. Loaded from a SimCLR/pretext checkpoint (our main approach)
"""

import torch
import torch.nn as nn
import timm


class ChestClassifier(nn.Module):
    """
    Multi-label chest X-ray classifier.
    
    Uses a timm backbone (ResNet-18 by default) with a custom classification
    head for 14 binary disease labels.
    """
    
    def __init__(self, backbone_name: str = "resnet18", num_classes: int = 14,
                 pretrained_imagenet: bool = True, drop_rate: float = 0.2):
        """
        Args:
            backbone_name: timm model name (e.g., "resnet18", "resnet50")
            num_classes: Number of output labels (14 for ChestMNIST)
            pretrained_imagenet: If True, load ImageNet weights
            drop_rate: Dropout rate before final FC layer
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        
        # Create backbone WITHOUT its default classification head
        # num_classes=0 tells timm to remove the final FC layer
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_imagenet,
            num_classes=0,          # Remove default head → outputs features only
            drop_rate=drop_rate,
        )
        
        # Get the feature dimension from the backbone
        # For ResNet-18: 512, ResNet-50: 2048
        self.feature_dim = self.backbone.num_features
        
        # Custom multi-label head
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.feature_dim, num_classes),
            # No sigmoid here — we apply it separately:
            #   - During training: BCEWithLogitsLoss (numerically stable, includes sigmoid)
            #   - During inference: torch.sigmoid() for probabilities
        )
        
        # Loss function: BCE with logits for multi-label
        self.criterion = nn.BCEWithLogitsLoss()
        
        pretrain_status = "ImageNet" if pretrained_imagenet else "random"
        print(f"✓ ChestClassifier created")
        print(f"  Backbone: {backbone_name} ({pretrain_status} init)")
        print(f"  Features: {self.feature_dim} → {num_classes} labels")
        print(f"  Params:   {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images, shape (B, 3, 224, 224)
        
        Returns:
            Logits (NOT probabilities), shape (B, 14)
            Apply torch.sigmoid() for probabilities during inference.
        """
        features = self.backbone(x)         # (B, 512) for ResNet-18
        logits = self.classifier(features)  # (B, 14)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features without the classification head.
        Used for active learning (uncertainty/core-set) and visualization.
        
        Args:
            x: Input images, shape (B, 3, 224, 224)
        
        Returns:
            Feature vectors, shape (B, 512) for ResNet-18
        """
        return self.backbone(x)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE loss for multi-label classification.
        
        Args:
            logits: Raw model output (before sigmoid), shape (B, 14)
            targets: Binary labels, shape (B, 14)
        
        Returns:
            Scalar loss value
        """
        return self.criterion(logits, targets.float())
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sigmoid probabilities (for inference / pseudo-labeling).
        
        Args:
            x: Input images, shape (B, 3, 224, 224)
        
        Returns:
            Probabilities, shape (B, 14), each value in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def load_backbone_weights(self, checkpoint_path: str):
        """
        Load backbone-only weights from a SimCLR or pretext checkpoint.
        This replaces ImageNet weights with self-supervised pretrained weights.
        
        Args:
            checkpoint_path: Path to .pth file containing backbone state_dict
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # Handle different checkpoint formats
        if "backbone" in state_dict:
            backbone_weights = state_dict["backbone"]
        elif "state_dict" in state_dict:
            # Lightning-style checkpoint — filter backbone keys
            backbone_weights = {
                k.replace("backbone.", ""): v 
                for k, v in state_dict["state_dict"].items() 
                if k.startswith("backbone.")
            }
        else:
            backbone_weights = state_dict
        
        missing, unexpected = self.backbone.load_state_dict(backbone_weights, strict=False)
        
        print(f"✓ Backbone weights loaded from {checkpoint_path}")
        if missing:
            print(f"  ⚠ Missing keys: {len(missing)}")
        if unexpected:
            print(f"  ⚠ Unexpected keys: {len(unexpected)}")
    
    def freeze_backbone(self):
        """
        Freeze backbone weights — only train the classification head.
        Useful for linear probing after self-supervised pretraining.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Backbone frozen — {trainable:,} trainable params remaining")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Backbone unfrozen — {trainable:,} trainable params")


def build_classifier(cfg) -> ChestClassifier:
    """
    Build classifier from config.
    
    Args:
        cfg: Config object with training.backbone, data.num_classes, etc.
    
    Returns:
        ChestClassifier instance
    """
    return ChestClassifier(
        backbone_name=cfg.training.backbone,
        num_classes=cfg.data.num_classes,
        pretrained_imagenet=cfg.training.pretrained_imagenet,
    )
