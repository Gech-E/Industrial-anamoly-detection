"""
SimCLR Model Architecture with PatchCore-style Patch Feature Extraction.

ResNet-50 encoder backbone + MLP projection head for contrastive learning.
Supports:
    - Single-layer global feature extraction (layer4 → 2048-dim)
    - Multi-layer global feature extraction (layer2+layer3+layer4 → 3584-dim)
    - Patch-level spatial feature extraction (PatchCore-style) for anomaly localization

Research note:
    PatchCore (Roth et al., 2022) extracts spatial feature maps and builds a
    patch-wise memory bank. This gives dramatically better AUROC than global
    features because anomalies are typically local — a small defect in a large
    normal image gets averaged out in global pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ResNetEncoder(nn.Module):
    """
    ResNet feature encoder backbone.
    Removes the final fully connected layer to output feature vectors.

    Supports:
        - Single-layer output (standard forward): 2048-dim for ResNet-50
        - Multi-layer extraction (layer2 + layer3 + layer4): 3584-dim for ResNet-50
        - Patch-level spatial feature extraction from intermediate layers
    """

    def __init__(self, pretrained: bool = True, backbone: str = "resnet50"):
        super().__init__()
        self.backbone_name = backbone

        # Load ResNet with or without ImageNet pretrained weights
        if backbone == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                resnet = models.resnet50(weights=weights)
                logger.info("ResNet-50 encoder initialized with ImageNet pretrained weights")
            else:
                resnet = models.resnet50(weights=None)
                logger.info("ResNet-50 encoder initialized randomly")
            self.feature_dim = 2048
            self._layer_dims = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        elif backbone == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                resnet = models.resnet18(weights=weights)
                logger.info("ResNet-18 encoder initialized with ImageNet pretrained weights")
            else:
                resnet = models.resnet18(weights=None)
                logger.info("ResNet-18 encoder initialized randomly")
            self.feature_dim = 512
            self._layer_dims = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Store individual layers for flexible feature extraction
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def get_multi_layer_dim(self, layers: List[str]) -> int:
        """Get the total feature dimension for multi-layer extraction."""
        return sum(self._layer_dims[layer] for layer in layers)

    def get_layer_dim(self, layer: str) -> int:
        """Get feature dimension for a single layer."""
        return self._layer_dims[layer]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass — returns layer4 pooled features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, feature_dim)
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

    @torch.no_grad()
    def extract_multi_layer(
        self, x: torch.Tensor, layers: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Extract and concatenate features from multiple ResNet layers.

        Each layer's spatial feature map is average-pooled to a vector,
        then all vectors are concatenated and L2-normalized.

        For ResNet-50 with layers=[layer2, layer3, layer4]:
            512 + 1024 + 2048 = 3584-dimensional output.

        Args:
            x: Input tensor of shape (B, 3, H, W)
            layers: List of layer names to extract from.
                    Default: ["layer2", "layer3", "layer4"]

        Returns:
            L2-normalized feature tensor of shape (B, sum_of_layer_dims)
        """
        if layers is None:
            layers = ["layer2", "layer3", "layer4"]

        # Forward through shared stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        if "layer1" in layers:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            features.append(pooled)

        x = self.layer2(x)
        if "layer2" in layers:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            features.append(pooled)

        x = self.layer3(x)
        if "layer3" in layers:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            features.append(pooled)

        x = self.layer4(x)
        if "layer4" in layers:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            features.append(pooled)

        # Concatenate all layer features
        concatenated = torch.cat(features, dim=1)

        # L2 normalize
        concatenated = F.normalize(concatenated, p=2, dim=1)

        return concatenated

    @torch.no_grad()
    def extract_patch_features(
        self,
        x: torch.Tensor,
        layers: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract patch-level spatial features for PatchCore-style anomaly detection.

        Instead of global average pooling (which discards spatial info), this
        returns per-patch feature vectors from intermediate feature maps.

        Multi-scale fusion:
            - layer2 features (28×28 for 224 input) provide fine-grained detail
            - layer3 features (14×14) provide semantic context
            - layer3 is upsampled to layer2 resolution, then concatenated

        Research note (PatchCore):
            Using layers 2+3 gives the best trade-off between local detail and
            semantic meaning. Layer4 is too coarse (7×7) for localization.
            Layer1 is too early (low-level, noisy).

        Args:
            x: Input tensor of shape (B, 3, H, W)
            layers: Layers to use. Default: ["layer2", "layer3"]

        Returns:
            Tuple of:
                - patch_features: (B, H_patch * W_patch, C_total) patch embeddings
                - patch_shape: (H_patch, W_patch) spatial dimensions of patch grid
        """
        if layers is None:
            layers = ["layer2", "layer3"]

        # Forward through shared stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Extract intermediate feature maps
        feat_maps = {}
        x = self.layer1(x)
        if "layer1" in layers:
            feat_maps["layer1"] = x

        x = self.layer2(x)
        if "layer2" in layers:
            feat_maps["layer2"] = x

        x = self.layer3(x)
        if "layer3" in layers:
            feat_maps["layer3"] = x

        x = self.layer4(x)
        if "layer4" in layers:
            feat_maps["layer4"] = x

        if not feat_maps:
            raise ValueError(f"No valid layers found in {layers}")

        # Determine target spatial resolution (use highest-res feature map)
        # Layer order by resolution: layer1 > layer2 > layer3 > layer4
        resolution_order = ["layer1", "layer2", "layer3", "layer4"]
        target_layer = None
        for layer_name in resolution_order:
            if layer_name in feat_maps:
                target_layer = layer_name
                break

        target_h, target_w = feat_maps[target_layer].shape[2:]

        # Upsample all feature maps to target resolution and concatenate
        aligned_features = []
        for layer_name in resolution_order:
            if layer_name in feat_maps:
                fmap = feat_maps[layer_name]
                if fmap.shape[2:] != (target_h, target_w):
                    # Bilinear upsample to match target resolution
                    fmap = F.interpolate(
                        fmap,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                aligned_features.append(fmap)

        # Concatenate along channel dimension: (B, C_total, H, W)
        multi_scale = torch.cat(aligned_features, dim=1)

        B, C, H, W = multi_scale.shape

        # Reshape to patch embeddings: (B, H*W, C)
        patch_features = multi_scale.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # L2 normalize each patch embedding
        patch_features = F.normalize(patch_features, p=2, dim=2)

        return patch_features, (H, W)


class ProjectionHead(nn.Module):
    """
    MLP Projection Head for SimCLR.
    Maps encoder features to a lower-dimensional space where contrastive loss is applied.

    Architecture: Linear -> BatchNorm -> ReLU -> Linear
    Hidden dim increased to 512 per SimCLR paper recommendations.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
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

    During training: encoder -> projection head -> NT-Xent loss
    During inference: encoder only -> feature extraction for anomaly scoring

    Supports:
        - Standard single-layer encoding (layer4)
        - Multi-layer feature extraction (layer2 + layer3 + layer4)
        - Patch-level feature extraction (PatchCore-style)
    """

    def __init__(self, config: dict = None):
        super().__init__()

        if config is None:
            config = {}

        model_cfg = config.get("model", {})

        pretrained = model_cfg.get("pretrained", True)
        backbone = model_cfg.get("backbone", "resnet50")

        self.encoder = ResNetEncoder(pretrained=pretrained, backbone=backbone)

        # Multi-layer config
        self.multi_layer = model_cfg.get("multi_layer", True)
        self.feature_layers = model_cfg.get(
            "feature_layers", ["layer2", "layer3", "layer4"]
        )

        # Patch feature config
        self.patch_layers = model_cfg.get(
            "patch_layers", ["layer2", "layer3"]
        )

        feature_dim = self.encoder.feature_dim
        projection_hidden = model_cfg.get("projection_hidden_dim", 512)
        projection_dim = model_cfg.get("projection_dim", 128)

        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=projection_hidden,
            output_dim=projection_dim,
        )

        # Log config
        if self.multi_layer:
            ml_dim = self.encoder.get_multi_layer_dim(self.feature_layers)
            logger.info(
                f"SimCLR model initialized | backbone={backbone} | "
                f"multi-layer={self.feature_layers} -> {ml_dim}-dim | "
                f"projection_dim={projection_dim}"
            )
        else:
            logger.info(
                f"SimCLR model initialized | backbone={backbone} | "
                f"feature_dim={feature_dim} | projection_dim={projection_dim}"
            )

        # Log patch feature dim
        patch_dim = self.encoder.get_multi_layer_dim(self.patch_layers)
        logger.info(
            f"Patch features: layers={self.patch_layers} -> {patch_dim}-dim per patch"
        )

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using single-layer (layer4) encoding.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        return self.encoder(x)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for anomaly detection (global, pooled).

        Uses multi-layer extraction if enabled in config,
        otherwise falls back to single-layer encoding.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            L2-normalized feature tensor
        """
        if self.multi_layer:
            return self.encoder.extract_multi_layer(x, self.feature_layers)
        else:
            features = self.encoder(x)
            return F.normalize(features, p=2, dim=1)

    @torch.no_grad()
    def extract_patch_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract patch-level features for PatchCore anomaly detection.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Tuple of:
                - patch_features: (B, num_patches, C) patch embeddings
                - patch_shape: (H_patch, W_patch) spatial dimensions
        """
        return self.encoder.extract_patch_features(x, self.patch_layers)
