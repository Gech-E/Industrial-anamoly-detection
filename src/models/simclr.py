"""
SimCLR Model Architecture.
ResNet-50 encoder backbone + MLP projection head for contrastive learning.
"""

import torch
import torch.nn as nn
from torchvision import models
import logging

logger = logging.getLogger(__name__)


class ResNetEncoder(nn.Module):
    """
    ResNet-50 feature encoder backbone.
    Removes the final fully connected layer to output feature vectors.
    
    Output: 512-dimensional feature vector per image.
    """
    
    def __init__(self, pretrained: bool = True, backbone: str = "resnet50"):
        super().__init__()
        
        # Load ResNet with or without ImageNet pretrained weights
        if backbone == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                resnet = models.resnet50(weights=weights)
                logger.info("ResNet-50 encoder initialized with ImageNet pretrained weights")
            else:
                resnet = models.resnet50(weights=None)
                logger.info("ResNet-50 encoder initialized randomly")
            self.feature_dim = 2048  # ResNet-50 outputs 2048-dim features
        elif backbone == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                resnet = models.resnet18(weights=weights)
                logger.info("ResNet-18 encoder initialized with ImageNet pretrained weights")
            else:
                resnet = models.resnet18(weights=None)
                logger.info("ResNet-18 encoder initialized randomly")
            self.feature_dim = 512   # ResNet-18 outputs 512-dim features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Store individual layers instead of sequential block
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Multi-scale feature concatenation dimension
        concat_dim = 1536 if backbone == "resnet50" else 384
        target_dim = 512
        
        # Adaptive pooling to reduce channel dimension
        if concat_dim > target_dim:
            self.dim_reduction = nn.AdaptiveAvgPool1d(target_dim)
        else:
            self.dim_reduction = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Feature tensor of shape (B, 2048)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    import torch.nn.functional as F

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale patch features for anomaly detection.
        Combines layer2 and layer3 features, and applies channel-wise dimensionality reduction.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        
        # Upsample layer3 to layer2 resolution
        feat3_up = torch.nn.functional.interpolate(
            feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False
        )
        
        # Concatenate multi-scale features
        features = torch.cat([feat2, feat3_up], dim=1)  # (B, C1+C2, H, W)
        
        B, C, H, W = features.shape
        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
        features = features.view(B, C, H * W).transpose(1, 2)
        
        # Dimensionality reduction on channels
        features = self.dim_reduction(features)  # (B, H*W, target_dim)
        
        # We need to reshape back to spatial for heatmap generation logic later
        # But memory bank usually wants (N_patches, D).
        # We will return the dense patches (B, H, W, D) for easier manipulation downstream.
        reduced_dim = features.shape[-1]
        features = features.transpose(1, 2).view(B, reduced_dim, H, W)
        return features


class ProjectionHead(nn.Module):
    """
    MLP Projection Head for SimCLR.
    Maps encoder features to a lower-dimensional space where contrastive loss is applied.
    
    Architecture: Linear → BatchNorm → ReLU → Linear
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
        logger.info(
            f"Projection head: {input_dim} -> {hidden_dim} -> {output_dim}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to contrastive learning space.
        
        Args:
            x: Feature tensor of shape (B, input_dim)
            
        Returns:
            Projected tensor of shape (B, output_dim)
        """
        return self.projection(x)


class SimCLRModel(nn.Module):
    """
    Complete SimCLR model combining encoder and projection head.
    
    During training: encoder → projection head → NT-Xent loss
    During inference: encoder only → feature extraction for anomaly scoring
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        if config is None:
            config = {}
        
        model_cfg = config.get("model", {})
        
        pretrained = model_cfg.get("pretrained", True)
        backbone = model_cfg.get("backbone", "resnet50")
        
        self.encoder = ResNetEncoder(pretrained=pretrained, backbone=backbone)
        
        feature_dim = self.encoder.feature_dim
        projection_hidden = model_cfg.get("projection_hidden_dim", 256)
        projection_dim = model_cfg.get("projection_dim", 128)
        
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=projection_hidden,
            output_dim=projection_dim,
        )
        
        logger.info(f"SimCLR model initialized (feature_dim={feature_dim}, projection_dim={projection_dim})")
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass through encoder + projection head.
        Used during SimCLR training.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (features [B, 2048], projections [B, 128])
        """
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections
    
    def encode(self, x: torch.Tensor, use_patches: bool = False) -> torch.Tensor:
        """
        Extract features.
        If use_patches is True, returns dense spatial feature maps (B, D, H, W)
        Otherwise, returns pooled image-level features (B, 2048).
        """
        if use_patches:
            return self.encoder.extract_patches(x)
        return self.encoder(x)
