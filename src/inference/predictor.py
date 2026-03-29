"""
Production Anomaly Predictor API.
Provides clean predict() and predict_batch() methods for inference.

Usage:
    predictor = AnomalyPredictor.from_config("configs/config.yaml", "bottle")
    result = predictor.predict(pil_image)
    results = predictor.predict_batch([img1, img2, img3])
"""

import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.utils import load_config, get_device, load_checkpoint
from src.models.simclr import SimCLRModel
from src.memory.memory_bank import MemoryBank, AnomalyScorer
from src.training.augmentations import get_eval_transform

logger = logging.getLogger(__name__)


class AnomalyPredictor:
    """
    Production-quality anomaly detection predictor.

    Encapsulates model loading, feature extraction, scoring, and
    threshold-based classification in a clean API.

    Thread-safe after initialization (all state is read-only during inference).

    Example:
        predictor = AnomalyPredictor.from_config("configs/config.yaml", "bottle")
        result = predictor.predict(Image.open("test.png"))
        # result = {"score": 12.34, "label": "Anomaly", "threshold": 8.5, "confidence": 0.87}
    """

    def __init__(
        self,
        model: SimCLRModel,
        scorer: AnomalyScorer,
        transform,
        threshold: float,
        device: torch.device,
        category: str = "unknown",
    ):
        self.model = model
        self.scorer = scorer
        self.transform = transform
        self.threshold = threshold
        self.device = device
        self.category = category

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
        """
        Create a predictor from config file and saved checkpoints.

        Automatically loads model, memory bank, scorer, and threshold.

        Args:
            config_path: Path to YAML config file.
            category: MVTec AD category name.
            device: Torch device. If None, auto-detect.

        Returns:
            Configured AnomalyPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint or memory bank is missing.
        """
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

        # Auto-detect backbone from checkpoint
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

        logger.info(
            f"Model loaded from {checkpoint_path} "
            f"(epoch {checkpoint.get('epoch', '?')})"
        )

        # Load memory bank
        bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
        if not os.path.exists(bank_path):
            raise FileNotFoundError(
                f"Memory bank not found: {bank_path}. "
                f"Run training first."
            )

        memory_bank = MemoryBank()
        memory_bank.load(bank_path)

        # Create scorer
        ad_cfg = config.get("anomaly_detection", {})
        scorer = AnomalyScorer(
            method=ad_cfg.get("method", "mahalanobis"),
            k_neighbors=ad_cfg.get("k_neighbors", 5),
        )
        scorer.fit(memory_bank)

        # Load threshold
        threshold = ad_cfg.get("score_threshold")

        # Try to load calibrated threshold from evaluation metrics
        if threshold is None:
            results_dir = config.get("output", {}).get(
                "results_dir", "outputs/results"
            )
            metrics_path = os.path.join(results_dir, f"{category}_metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    threshold = float(metrics.get("threshold", 0.5))
                    logger.info(
                        f"Loaded calibrated threshold from metrics: {threshold:.4f}"
                    )
                except Exception:
                    threshold = 0.5
            else:
                threshold = 0.5
                logger.warning(
                    "No calibrated threshold found — using default 0.5. "
                    "Run evaluation to calibrate."
                )

        # Transform
        transform = get_eval_transform(config)

        return cls(
            model=model,
            scorer=scorer,
            transform=transform,
            threshold=threshold,
            device=device,
            category=category,
        )

    def predict(self, image: Image.Image) -> Dict:
        """
        Run anomaly detection on a single image.

        Args:
            image: PIL Image (RGB).

        Returns:
            Dict with keys:
                - score: float anomaly score
                - label: "Normal" or "Anomaly"
                - threshold: float threshold used
                - confidence: float in [0, 1], how far from threshold
                - is_anomaly: bool
        """
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

        # Confidence: normalized distance from threshold
        confidence = min(
            abs(score - self.threshold) / max(self.threshold, 1e-6), 1.0
        )

        return {
            "score": score,
            "label": label,
            "threshold": self.threshold,
            "confidence": confidence,
            "is_anomaly": is_anomaly,
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """
        Run anomaly detection on a batch of images.

        Args:
            images: List of PIL Images (RGB).

        Returns:
            List of prediction dicts (same format as predict()).
        """
        if not images:
            return []

        # Prepare batch tensor
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
        for i, score in enumerate(scores):
            score_val = float(score)
            is_anomaly = score_val >= self.threshold
            confidence = min(
                abs(score_val - self.threshold) / max(self.threshold, 1e-6),
                1.0,
            )
            results.append({
                "score": score_val,
                "label": "Anomaly" if is_anomaly else "Normal",
                "threshold": self.threshold,
                "confidence": confidence,
                "is_anomaly": is_anomaly,
            })

        return results
