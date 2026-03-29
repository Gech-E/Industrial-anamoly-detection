"""
Score Calibration Module for Anomaly Detection.

Converts raw anomaly scores into interpretable confidence values (0-100%).
Supports multiple calibration methods:
    - Min-max normalization (per-category)
    - Temperature scaling (Platt scaling variant)
    - Sigmoid calibration
    - Percentile-based thresholding

Research note:
    Raw Mahalanobis/kNN scores vary wildly across categories (e.g., 3.2 for
    bottle vs 150.0 for grid). Calibration normalizes scores so that
    "confidence = 85%" means the same thing regardless of category.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid

logger = logging.getLogger(__name__)


class ScoreCalibrator:
    """
    Calibrates raw anomaly scores to interpretable confidence values.

    Fit on training/validation scores, then apply to test scores.
    Outputs confidence in [0, 1] where 1 = definitely anomalous.

    Usage:
        calibrator = ScoreCalibrator(method="minmax_sigmoid")
        calibrator.fit(train_scores, train_labels)

        # During inference:
        confidence = calibrator.calibrate(raw_score)
        # confidence in [0, 1], interpretable as probability of anomaly
    """

    def __init__(
        self,
        method: str = "minmax_sigmoid",
        temperature: float = 1.0,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
    ):
        """
        Args:
            method: Calibration method. One of:
                - "minmax": Simple min-max to [0, 1]
                - "minmax_sigmoid": Min-max + sigmoid (recommended)
                - "temperature": Temperature scaling
                - "percentile": Percentile-based normalization
            temperature: Initial temperature for temperature scaling
            percentile_low: Lower percentile for clipping
            percentile_high: Upper percentile for clipping
        """
        self.method = method
        self.temperature = temperature
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        # Fitted parameters
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._score_mean: float = 0.0
        self._score_std: float = 1.0
        self._threshold: float = 0.5
        self._normal_mean: float = 0.0
        self._normal_std: float = 1.0
        self._anomaly_mean: float = 1.0
        self._anomaly_std: float = 1.0
        self._fitted: bool = False

        logger.info(f"ScoreCalibrator initialized: method={method}")

    def fit(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ):
        """
        Fit calibration parameters from training/validation scores.

        Args:
            scores: Raw anomaly scores, shape (N,)
            labels: Optional ground truth labels (0=normal, 1=anomaly)
            threshold: Optional pre-computed threshold
        """
        scores = np.asarray(scores, dtype=np.float64)

        # Robust min/max using percentiles to avoid outlier sensitivity
        self._score_min = float(np.percentile(scores, self.percentile_low))
        self._score_max = float(np.percentile(scores, self.percentile_high))

        # Prevent zero range
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1.0

        self._score_mean = float(np.mean(scores))
        self._score_std = float(np.std(scores)) + 1e-8

        if labels is not None:
            labels = np.asarray(labels)
            normal_mask = labels == 0
            anomaly_mask = labels == 1

            if normal_mask.any():
                self._normal_mean = float(np.mean(scores[normal_mask]))
                self._normal_std = float(np.std(scores[normal_mask])) + 1e-8

            if anomaly_mask.any():
                self._anomaly_mean = float(np.mean(scores[anomaly_mask]))
                self._anomaly_std = float(np.std(scores[anomaly_mask])) + 1e-8

            # Auto-compute threshold as midpoint between class means
            if normal_mask.any() and anomaly_mask.any():
                self._threshold = float(
                    (self._normal_mean + self._anomaly_mean) / 2
                )
            elif threshold is not None:
                self._threshold = threshold
        elif threshold is not None:
            self._threshold = threshold

        # Temperature scaling: optimize temperature on labeled data
        if self.method == "temperature" and labels is not None:
            self._optimize_temperature(scores, labels)

        self._fitted = True

        logger.info(
            f"Calibrator fitted: min={self._score_min:.4f}, "
            f"max={self._score_max:.4f}, threshold={self._threshold:.4f}, "
            f"method={self.method}"
        )

    def _optimize_temperature(self, scores: np.ndarray, labels: np.ndarray):
        """Optimize temperature to minimize NLL (Platt scaling)."""
        def nll(temp):
            # Normalize scores to [0, 1]
            normed = (scores - self._score_min) / (self._score_max - self._score_min)
            normed = np.clip(normed, 0, 1)
            # Apply temperature-scaled sigmoid
            logits = (normed - 0.5) / max(temp, 1e-6)
            probs = expit(logits)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            # Negative log-likelihood
            loss = -np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            )
            return loss

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.temperature = float(result.x)
        logger.info(f"Optimized temperature: {self.temperature:.4f}")

    def calibrate(self, score: float) -> float:
        """
        Calibrate a single raw score to confidence in [0, 1].

        Args:
            score: Raw anomaly score

        Returns:
            Calibrated confidence in [0, 1]. Higher = more anomalous.
        """
        if not self._fitted:
            logger.warning("Calibrator not fitted — returning raw score clipped to [0,1]")
            return float(np.clip(score, 0, 1))

        if self.method == "minmax":
            return self._calibrate_minmax(score)
        elif self.method == "minmax_sigmoid":
            return self._calibrate_minmax_sigmoid(score)
        elif self.method == "temperature":
            return self._calibrate_temperature(score)
        elif self.method == "percentile":
            return self._calibrate_minmax(score)
        else:
            return self._calibrate_minmax_sigmoid(score)

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """
        Calibrate a batch of raw scores.

        Args:
            scores: Raw anomaly scores, shape (N,)

        Returns:
            Calibrated confidences, shape (N,), values in [0, 1]
        """
        return np.array([self.calibrate(float(s)) for s in scores])

    def _calibrate_minmax(self, score: float) -> float:
        """Simple min-max normalization to [0, 1]."""
        normed = (score - self._score_min) / (self._score_max - self._score_min)
        return float(np.clip(normed, 0.0, 1.0))

    def _calibrate_minmax_sigmoid(self, score: float) -> float:
        """Min-max normalization followed by sigmoid for smoother calibration."""
        normed = (score - self._score_min) / (self._score_max - self._score_min)
        normed = np.clip(normed, -3.0, 3.0)  # Prevent extreme values

        # Center around threshold (mapped to 0.5 in normalized space)
        threshold_normed = (self._threshold - self._score_min) / (
            self._score_max - self._score_min
        )
        centered = (normed - threshold_normed) * 6.0  # Scale for sigmoid range

        confidence = float(expit(centered / max(self.temperature, 1e-6)))
        return float(np.clip(confidence, 0.0, 1.0))

    def _calibrate_temperature(self, score: float) -> float:
        """Temperature-scaled sigmoid calibration."""
        normed = (score - self._score_min) / (self._score_max - self._score_min)
        normed = np.clip(normed, 0, 1)
        logit = (normed - 0.5) / max(self.temperature, 1e-6)
        return float(expit(logit))

    def get_confidence_label(self, confidence: float) -> str:
        """Get human-readable confidence label."""
        if confidence < 0.2:
            return "Very Low"
        elif confidence < 0.4:
            return "Low"
        elif confidence < 0.6:
            return "Moderate"
        elif confidence < 0.8:
            return "High"
        else:
            return "Very High"

    def to_percentage(self, confidence: float) -> float:
        """Convert [0, 1] confidence to [0, 100] percentage."""
        return round(confidence * 100, 1)

    def save_params(self) -> Dict:
        """Export calibration parameters for serialization."""
        return {
            "method": self.method,
            "temperature": self.temperature,
            "score_min": self._score_min,
            "score_max": self._score_max,
            "score_mean": self._score_mean,
            "score_std": self._score_std,
            "threshold": self._threshold,
            "normal_mean": self._normal_mean,
            "normal_std": self._normal_std,
            "anomaly_mean": self._anomaly_mean,
            "anomaly_std": self._anomaly_std,
            "fitted": self._fitted,
        }

    def load_params(self, params: Dict):
        """Load calibration parameters from serialized dict."""
        self.method = params.get("method", self.method)
        self.temperature = params.get("temperature", self.temperature)
        self._score_min = params.get("score_min", self._score_min)
        self._score_max = params.get("score_max", self._score_max)
        self._score_mean = params.get("score_mean", self._score_mean)
        self._score_std = params.get("score_std", self._score_std)
        self._threshold = params.get("threshold", self._threshold)
        self._normal_mean = params.get("normal_mean", self._normal_mean)
        self._normal_std = params.get("normal_std", self._normal_std)
        self._anomaly_mean = params.get("anomaly_mean", self._anomaly_mean)
        self._anomaly_std = params.get("anomaly_std", self._anomaly_std)
        self._fitted = params.get("fitted", self._fitted)
        logger.info("Calibrator parameters loaded")
