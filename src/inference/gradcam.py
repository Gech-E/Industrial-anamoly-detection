"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for anomaly localization.
Generates heatmaps showing WHERE in the image the anomaly is detected.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for ResNet-18 anomaly localization.
    
    Hooks into a target convolutional layer (default: layer4) and computes
    gradient-weighted activation maps to visualize which regions of the
    image contribute most to the anomaly score.
    """
    
    def __init__(self, model, target_layer_name: str = "layer4"):
        """
        Args:
            model: SimCLR model (uses encoder)
            target_layer_name: Name of the target layer in ResNet encoder
        """
        self.model = model
        self.model.eval()
        
        # Get target layer
        self.target_layer = self._get_target_layer(target_layer_name)
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"Grad-CAM initialized with target layer: {target_layer_name}")
    
    def _get_target_layer(self, layer_name: str):
        """Get the target layer from the encoder."""
        encoder = self.model.encoder
        
        if hasattr(encoder, layer_name):
            return getattr(encoder, layer_name)
        
        # Try to find in features sequential
        layer_map = {
            "layer1": 4,
            "layer2": 5,
            "layer3": 6,
            "layer4": 7,
        }
        
        if layer_name in layer_map:
            return encoder.features[layer_map[layer_name]]
        
        raise ValueError(f"Target layer '{layer_name}' not found in encoder")
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        device: torch.device,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.
        
        We use the L2 norm of the feature vector as the target for backpropagation.
        For anomaly detection, high feature norms in unusual directions indicate anomalies.
        
        Args:
            input_tensor: Preprocessed image tensor, shape (1, 3, H, W)
            device: Device for computation
            
        Returns:
            Heatmap as numpy array, shape (H, W), values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        features = self.model.encode(input_tensor)
        
        # Use L2 norm of features as target
        target = torch.norm(features, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Grad-CAM computation
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)
        
        # Global average pooling of gradients → channel importance weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, h, w)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def generate_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.
        
        Args:
            image: Original image as numpy array (H, W, 3), RGB, values [0, 255]
            heatmap: Grad-CAM heatmap (H, W), values [0, 1]
            alpha: Transparency of the heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Overlaid image as numpy array (H, W, 3), RGB, values [0, 255]
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = np.float32(heatmap_colored) * alpha + np.float32(image) * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay
    
    def visualize(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        anomaly_score: float,
        label: str,
        save_path: Optional[str] = None,
    ):
        """
        Create a side-by-side visualization: original | heatmap | overlay.
        
        Args:
            original_image: RGB image (H, W, 3), values [0, 255]
            heatmap: Grad-CAM heatmap (H, W), values [0, 1]
            anomaly_score: Anomaly score for the image
            label: Predicted label string
            save_path: Optional path to save the figure
        """
        overlay = self.generate_overlay(original_image, heatmap)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
        axes[0].axis("off")
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Anomaly Heatmap", fontsize=13, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(overlay)
        color = "#F44336" if label == "Anomaly" else "#4CAF50"
        axes[2].set_title(
            f"Overlay | {label} (Score: {anomaly_score:.4f})",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Grad-CAM visualization saved: {save_path}")
        
        plt.close(fig)
