"""
Production Anomaly Predictor API with PatchCore support.

Provides clean predict() and predict_batch() methods for inference,
with calibrated confidence scores and anomaly heatmaps.

Supports two pipelines:
    1. PatchAnomalyPredictor: PatchCore patch-level detection (recommended)
    2. AnomalyPredictor: Global feature detection (legacy)

Usage:
    predictor = PatchAnomalyPredictor.from_config("configs/config.yaml", "bottle")
    result = predictor.predict(pil_image)
    # result = {"score": 0.42, "label": "Anomaly", "confidence": 0.87,
    #           "confidence_pct": 87.0, "heatmap": np.array(...), ...}
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.utils import load_config, get_device, load_checkpoint
from src.models.simclr import SimCLRModel
from src.memory.memory_bank import (
    MemoryBank, AnomalyScorer,
    PatchMemoryBank, PatchAnomalyScorer,
)
from src.scoring.calibration import ScoreCalibrator
from src.visualization.heatmap import AnomalyHeatmapGenerator
from src.training.augmentations import get_eval_transform

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# PatchCore Predictor (PRIMARY — high AUROC + localization)
# ═══════════════════════════════════════════════════════════════

class PatchAnomalyPredictor:
    """
    Production-quality PatchCore anomaly detection predictor.

    Features:
        - Patch-level anomaly detection for high AUROC
        - Anomaly heatmap generation for localization
        - Calibrated confidence scores (0–100%)
        - Thread-safe after initialization

    Example:
        predictor = PatchAnomalyPredictor.from_config("configs/config.yaml", "bottle")
        result = predictor.predict(Image.open("test.png"))
        # result["label"] = "Anomaly"
        # result["confidence_pct"] = 87.2
        # result["heatmap"] = np.array(...)  # (224, 224) anomaly map
    """

    def __init__(
        self,
        model: SimCLRModel,
        scorer: PatchAnomalyScorer,
        calibrator: ScoreCalibrator,
        heatmap_gen: AnomalyHeatmapGenerator,
        transform,
        threshold: float,
        device: torch.device,
        category: str = "unknown",
        image_size: int = 224,
    ):
        self.model = model
        self.scorer = scorer
        self.calibrator = calibrator
        self.heatmap_gen = heatmap_gen
        self.transform = transform
        self.threshold = threshold
        self.device = device
        self.category = category
        self.image_size = image_size

        self.model.eval()
        logger.info(
            f"PatchAnomalyPredictor ready: category={category}, "
            f"threshold={threshold:.4f}, device={device}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/config.yaml",
        category: str = "bottle",
        device: Optional[torch.device] = None,
    ) -> "PatchAnomalyPredictor":
        """Create a PatchCore predictor from config and saved checkpoints."""
        config = load_config(config_path)

        if device is None:
            device = get_device()

        checkpoint_dir = config.get("output", {}).get(
            "checkpoints_dir", "outputs/checkpoints"
        )

        # Load model
        checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                f"Run training first: python scripts/train.py --category {category}"
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict", {})
        is_resnet50 = any("conv3" in key for key in state_dict.keys())

        if "model" not in config:
            config["model"] = {}
        config["model"]["backbone"] = "resnet50" if is_resnet50 else "resnet18"

        model = SimCLRModel(config)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Load patch memory bank
        patch_bank_path = os.path.join(
            checkpoint_dir, f"{category}_patch_bank.pt"
        )
        if not os.path.exists(patch_bank_path):
            raise FileNotFoundError(
                f"Patch memory bank not found: {patch_bank_path}. "
                f"Run training with patch mode first."
            )

        patch_bank = PatchMemoryBank()
        patch_bank.load(patch_bank_path)

        # Create scorer
        scoring_cfg = config.get("scoring", {})
        scorer = PatchAnomalyScorer(
            k_neighbors=scoring_cfg.get("k_neighbors", 3),
            weight_knn=scoring_cfg.get("weight_knn", 1.0),
            weight_mahalanobis=scoring_cfg.get("weight_mahalanobis", 0.0),
            weight_cosine=scoring_cfg.get("weight_cosine", 0.0),
        )
        scorer.fit(patch_bank)

        # Load calibrator
        calibrator = ScoreCalibrator(
            method=config.get("calibration", {}).get("method", "minmax_sigmoid"),
        )

        # Try to load calibration from metrics
        results_dir = config.get("output", {}).get("results_dir", "outputs/results")
        metrics_path = os.path.join(results_dir, f"{category}_metrics.json")
        threshold = 0.5

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                threshold = float(metrics.get("threshold", 0.5))

                if "calibration" in metrics:
                    calibrator.load_params(metrics["calibration"])
                logger.info(f"Loaded threshold={threshold:.4f} from metrics")
            except Exception:
                pass

        # Heatmap generator
        loc_cfg = config.get("localization", {})
        heatmap_gen = AnomalyHeatmapGenerator(
            sigma=loc_cfg.get("gaussian_sigma", 4.0),
            colormap=loc_cfg.get("colormap", "jet"),
            alpha=loc_cfg.get("overlay_alpha", 0.4),
        )

        # Transform
        transform = get_eval_transform(config)
        image_size = config.get("dataset", {}).get("image_size", 224)

        return cls(
            model=model,
            scorer=scorer,
            calibrator=calibrator,
            heatmap_gen=heatmap_gen,
            transform=transform,
            threshold=threshold,
            device=device,
            category=category,
            image_size=image_size,
        )

    def predict(self, image: Image.Image) -> Dict:
        """
        Run PatchCore anomaly detection on a single image.

        Args:
            image: PIL Image (RGB).

        Returns:
            Dict with keys:
                - score: float raw anomaly score
                - label: "Normal" or "Anomaly"
                - threshold: float threshold used
                - confidence: float in [0, 1]
                - confidence_pct: float in [0, 100]
                - is_anomaly: bool
                - heatmap: np.ndarray (H, W) anomaly heatmap in [0, 1]
                - overlay: np.ndarray (H, W, 3) heatmap overlaid on image
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        original_np = np.array(
            image.resize((self.image_size, self.image_size))
        )

        with torch.no_grad():
            patch_features, patch_shape = self.model.extract_patch_features(
                input_tensor
            )

        # Score patches
        patches = patch_features[0].cpu()  # (num_patches, C)
        patch_scores, image_score = self.scorer.score_patches(patches)

        # Classification
        is_anomaly = image_score >= self.threshold
        label = "Anomaly" if is_anomaly else "Normal"

        # Calibrated confidence
        confidence = self.calibrator.calibrate(image_score)

        # Generate heatmap
        heatmap = self.heatmap_gen.generate(
            patch_scores, patch_shape,
            image_size=(self.image_size, self.image_size),
        )
        overlay = self.heatmap_gen.overlay(original_np, heatmap)

        return {
            "score": image_score,
            "label": label,
            "threshold": self.threshold,
            "confidence": confidence,
            "confidence_pct": self.calibrator.to_percentage(confidence),
            "confidence_label": self.calibrator.get_confidence_label(confidence),
            "is_anomaly": is_anomaly,
            "heatmap": heatmap,
            "overlay": overlay,
            "original": original_np,
            "patch_scores": patch_scores,
            "patch_shape": patch_shape,
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Run anomaly detection on a batch of images."""
        return [self.predict(img) for img in images]


# ═══════════════════════════════════════════════════════════════
# Global Feature Predictor (legacy, backward-compatible)
# ═══════════════════════════════════════════════════════════════

class AnomalyPredictor:
    """
    Global-feature anomaly detection predictor (legacy).
    Kept for backward compatibility with existing checkpoints.
    """

    def __init__(
        self,
        model: SimCLRModel,
        scorer: AnomalyScorer,
        transform,
        threshold: float,
        device: torch.device,
        category: str = "unknown",
        calibrator: Optional[ScoreCalibrator] = None,
    ):
        self.model = model
        self.scorer = scorer
        self.transform = transform
        self.threshold = threshold
        self.device = device
        self.category = category
        self.calibrator = calibrator

        self.model.eval()
        logger.info(
            f"AnomalyPredictor ready: category={category}, "
            f"threshold={threshold:.4f}, device={device}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/config.yaml",
        category: str = "bottle",
        device: Optional[torch.device] = None,
    ) -> "AnomalyPredictor":
        """Create predictor from config file and saved checkpoints."""
        config = load_config(config_path)

        if device is None:
            device = get_device()

        checkpoint_dir = config.get("output", {}).get(
            "checkpoints_dir", "outputs/checkpoints"
        )

        # Load model
        checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                f"Run training first: python scripts/train.py --category {category}"
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict", {})
        is_resnet50 = any("conv3" in key for key in state_dict.keys())

        if "model" not in config:
            config["model"] = {}
        config["model"]["backbone"] = "resnet50" if is_resnet50 else "resnet18"

        model = SimCLRModel(config)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Load memory bank
        bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
        if not os.path.exists(bank_path):
            raise FileNotFoundError(f"Memory bank not found: {bank_path}")

        memory_bank = MemoryBank()
        memory_bank.load(bank_path)

        # Create scorer
        ad_cfg = config.get("anomaly_detection", {})
        scorer = AnomalyScorer(
            method=ad_cfg.get("method", "mahalanobis"),
            k_neighbors=ad_cfg.get("k_neighbors", 5),
        )
        scorer.fit(memory_bank)

        # Load threshold and calibrator
        threshold = ad_cfg.get("score_threshold")
        calibrator = ScoreCalibrator(
            method=config.get("calibration", {}).get("method", "minmax_sigmoid")
        )

        results_dir = config.get("output", {}).get("results_dir", "outputs/results")
        metrics_path = os.path.join(results_dir, f"{category}_metrics.json")

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                if threshold is None:
                    threshold = float(metrics.get("threshold", 0.5))
                if "calibration" in metrics:
                    calibrator.load_params(metrics["calibration"])
            except Exception:
                if threshold is None:
                    threshold = 0.5
        elif threshold is None:
            threshold = 0.5

        transform = get_eval_transform(config)

        return cls(
            model=model,
            scorer=scorer,
            transform=transform,
            threshold=threshold,
            device=device,
            category=category,
            calibrator=calibrator,
        )

    def predict(self, image: Image.Image) -> Dict:
        """Run anomaly detection on a single image."""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(input_tensor)
            else:
                features = self.model.encode(input_tensor)

        features_cpu = features.cpu()
        score = float(self.scorer.score(features_cpu)[0])

        is_anomaly = score >= self.threshold
        label = "Anomaly" if is_anomaly else "Normal"

        # Calibrated confidence
        if self.calibrator and self.calibrator._fitted:
            confidence = self.calibrator.calibrate(score)
        else:
            confidence = min(
                abs(score - self.threshold) / max(self.threshold, 1e-6), 1.0
            )

        return {
            "score": score,
            "label": label,
            "threshold": self.threshold,
            "confidence": confidence,
            "confidence_pct": round(confidence * 100, 1),
            "is_anomaly": is_anomaly,
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Run anomaly detection on a batch of images."""
        if not images:
            return []

        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(batch)
            else:
                features = self.model.encode(batch)

        features_cpu = features.cpu()
        scores = self.scorer.score(features_cpu)

        results = []
        for score_val in scores:
            score_val = float(score_val)
            is_anomaly = score_val >= self.threshold

            if self.calibrator and self.calibrator._fitted:
                confidence = self.calibrator.calibrate(score_val)
            else:
                confidence = min(
                    abs(score_val - self.threshold) / max(self.threshold, 1e-6),
                    1.0,
                )

            results.append({
                "score": score_val,
                "label": "Anomaly" if is_anomaly else "Normal",
                "threshold": self.threshold,
                "confidence": confidence,
                "confidence_pct": round(confidence * 100, 1),
                "is_anomaly": is_anomaly,
            })

        return results


def create_predictor(
    config_path: str = "configs/config.yaml",
    category: str = "bottle",
    device: Optional[torch.device] = None,
):
    """
    Factory function: auto-detects the best available predictor.

    Prefers PatchCore if patch_bank exists, falls back to global.
    """
    config = load_config(config_path)
    checkpoint_dir = config.get("output", {}).get(
        "checkpoints_dir", "outputs/checkpoints"
    )

    patch_bank_path = os.path.join(checkpoint_dir, f"{category}_patch_bank.pt")

    if os.path.exists(patch_bank_path):
        logger.info(f"Using PatchCore predictor for {category}")
        return PatchAnomalyPredictor.from_config(config_path, category, device)
    else:
        logger.info(f"Using global predictor for {category}")
        return AnomalyPredictor.from_config(config_path, category, device)
