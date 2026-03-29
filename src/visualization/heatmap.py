"""
Anomaly Heatmap Generator.

Generates pixel-level anomaly localization maps from patch-level scores.
Provides superior localization compared to Grad-CAM because it uses direct
patch attribution rather than gradient-based approximation.

Research note:
    In PatchCore, each patch gets a kNN distance score. These scores are
    arranged on the spatial grid, upsampled to image resolution, and
    Gaussian-smoothed to produce pixel-level anomaly maps.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AnomalyHeatmapGenerator:
    """
    Generates anomaly localization heatmaps from patch-level anomaly scores.

    Pipeline:
        1. Reshape patch scores into spatial grid (H_patch × W_patch)
        2. Upsample to original image resolution (bilinear interpolation)
        3. Apply Gaussian smoothing for visual quality
        4. Normalize to [0, 1]
        5. Apply colormap and overlay on original image

    Usage:
        generator = AnomalyHeatmapGenerator(sigma=4.0)
        heatmap = generator.generate(patch_scores, patch_shape, image_size=(224, 224))
        overlay = generator.overlay(original_image, heatmap)
    """

    def __init__(
        self,
        sigma: float = 4.0,
        colormap: str = "jet",
        alpha: float = 0.4,
    ):
        """
        Args:
            sigma: Gaussian smoothing sigma. Higher = smoother heatmap.
                   Typical range: 2.0–8.0. Default 4.0 works well for 224×224.
            colormap: Matplotlib colormap for heatmap visualization.
            alpha: Overlay transparency (0 = original only, 1 = heatmap only).
        """
        self.sigma = sigma
        self.colormap = colormap
        self.alpha = alpha

        # OpenCV colormap mapping
        self._cv2_colormaps = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "inferno": cv2.COLORMAP_INFERNO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "turbo": cv2.COLORMAP_TURBO,
        }

    def generate(
        self,
        patch_scores: np.ndarray,
        patch_shape: Tuple[int, int],
        image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Generate pixel-level anomaly heatmap from patch scores.

        Args:
            patch_scores: Anomaly scores per patch, shape (num_patches,)
            patch_shape: Spatial dimensions (H_patch, W_patch)
            image_size: Target image resolution (H, W)

        Returns:
            Heatmap as numpy array, shape (H, W), values normalized to [0, 1]
        """
        H_patch, W_patch = patch_shape

        # Reshape to spatial grid
        score_map = patch_scores.reshape(H_patch, W_patch)

        # Upsample to image resolution using bilinear interpolation
        heatmap = cv2.resize(
            score_map.astype(np.float32),
            (image_size[1], image_size[0]),  # cv2 uses (W, H)
            interpolation=cv2.INTER_LINEAR,
        )

        # Gaussian smoothing for visual quality
        if self.sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.sigma)

        # Normalize to [0, 1]
        hmap_min = heatmap.min()
        hmap_max = heatmap.max()
        if hmap_max > hmap_min:
            heatmap = (heatmap - hmap_min) / (hmap_max - hmap_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def generate_from_2d(
        self,
        score_map_2d: np.ndarray,
        image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Generate heatmap from a pre-shaped 2D score map.

        Args:
            score_map_2d: 2D anomaly score map, shape (H_patch, W_patch)
            image_size: Target image resolution (H, W)

        Returns:
            Heatmap as numpy array, shape (H, W), values in [0, 1]
        """
        heatmap = cv2.resize(
            score_map_2d.astype(np.float32),
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.sigma)

        hmap_min = heatmap.min()
        hmap_max = heatmap.max()
        if hmap_max > hmap_min:
            heatmap = (heatmap - hmap_min) / (hmap_max - hmap_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def colorize(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Apply colormap to heatmap.

        Args:
            heatmap: Normalized heatmap (H, W) in [0, 1]

        Returns:
            Colored heatmap (H, W, 3) as uint8 RGB
        """
        cmap = self._cv2_colormaps.get(self.colormap, cv2.COLORMAP_JET)
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cmap)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return colored

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image (H, W, 3) as uint8 RGB
            heatmap: Normalized heatmap (H, W) in [0, 1]
            alpha: Override transparency. None uses self.alpha.

        Returns:
            Overlaid image (H, W, 3) as uint8 RGB
        """
        if alpha is None:
            alpha = self.alpha

        # Resize heatmap to match image if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(
                heatmap, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Colorize
        colored = self.colorize(heatmap)

        # Blend
        blended = (
            np.float32(colored) * alpha + np.float32(image) * (1 - alpha)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def visualize(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        anomaly_score: float,
        label: str,
        confidence: float = 0.0,
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Create side-by-side visualization: Original | Heatmap | Overlay.

        Args:
            original_image: RGB image (H, W, 3), values [0, 255]
            heatmap: Normalized heatmap (H, W), values [0, 1]
            anomaly_score: Raw anomaly score
            label: Predicted label ("Normal" or "Anomaly")
            confidence: Calibrated confidence [0, 1]
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure (if not saved, for Streamlit display)
        """
        overlay_img = self.overlay(original_image, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        # Heatmap
        im = axes[1].imshow(heatmap, cmap=self.colormap, vmin=0, vmax=1)
        axes[1].set_title("Anomaly Heatmap", fontsize=13, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(overlay_img)
        color = "#F44336" if label == "Anomaly" else "#4CAF50"
        conf_str = f" | Conf: {confidence:.0%}" if confidence > 0 else ""
        axes[2].set_title(
            f"{label} (Score: {anomaly_score:.4f}{conf_str})",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Heatmap visualization saved: {save_path}")
            plt.close(fig)
            return None

        return fig

    def visualize_comparison(
        self,
        original_image: np.ndarray,
        patch_heatmap: np.ndarray,
        gradcam_heatmap: Optional[np.ndarray],
        anomaly_score: float,
        label: str,
        save_path: Optional[str] = None,
    ):
        """
        Side-by-side comparison: Original | Patch Heatmap | Grad-CAM | Overlay.
        Useful for ablation studies comparing localization methods.
        """
        n_cols = 4 if gradcam_heatmap is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        axes[0].imshow(original_image)
        axes[0].set_title("Original", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        im1 = axes[1].imshow(patch_heatmap, cmap=self.colormap, vmin=0, vmax=1)
        axes[1].set_title("Patch Heatmap", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        if gradcam_heatmap is not None:
            im2 = axes[2].imshow(gradcam_heatmap, cmap="jet", vmin=0, vmax=1)
            axes[2].set_title("Grad-CAM", fontsize=12, fontweight="bold")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        overlay = self.overlay(original_image, patch_heatmap)
        axes[-1].imshow(overlay)
        color = "#F44336" if label == "Anomaly" else "#4CAF50"
        axes[-1].set_title(
            f"Overlay | {label} ({anomaly_score:.4f})",
            fontsize=12, fontweight="bold", color=color,
        )
        axes[-1].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.close(fig)
