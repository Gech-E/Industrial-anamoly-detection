"""
Feature Memory Bank and Anomaly Scoring.
Stores normal-sample features and scores test images using
Mahalanobis distance or k-NN distance as the anomaly score.

Supports:
    - Multi-layer feature extraction
    - PCA dimensionality reduction for CPU efficiency
    - Coreset subsampling (greedy k-center)
    - Mahalanobis distance scoring (primary)
    - k-NN distance scoring (fallback)
"""

import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class MemoryBank:
    """
    Stores feature representations of normal training images.
    Used as the reference distribution for anomaly scoring.

    Features:
        - Build from model + dataloader
        - Optional PCA dimensionality reduction
        - Optional coreset subsampling
        - Save/load with PCA transform
    """

    def __init__(
        self,
        use_pca: bool = False,
        pca_components: int = 256,
        subsample_size: Optional[int] = None,
        **kwargs,
    ):
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.subsample_size = subsample_size

        self.features: Optional[torch.Tensor] = None
        self.count: int = 0

        # PCA transform (fitted during build)
        self._pca: Optional[PCA] = None

    @torch.no_grad()
    def build(
        self,
        model,
        dataloader,
        device: torch.device,
    ):
        """
        Extract features from all normal training images and store them.

        Uses model.extract_features() for multi-layer extraction if available,
        otherwise falls back to model.encode().

        Args:
            model: Trained SimCLR model
            dataloader: DataLoader for normal training images
            device: Device to run inference on
        """
        model.eval()
        all_features = []

        logger.info("Extracting features for memory bank...")
        start_time = time.time()

        for batch_idx, (images, labels, indices) in enumerate(dataloader):
            images = images.to(device)

            # Use multi-layer extraction if available
            if hasattr(model, "extract_features"):
                features = model.extract_features(images)
            else:
                features = model.encode(images)
                features = F.normalize(features, p=2, dim=1)

            all_features.append(features.cpu())

        raw_features = torch.cat(all_features, dim=0)
        extract_time = time.time() - start_time

        logger.info(
            f"Feature extraction: {raw_features.shape[0]} samples, "
            f"dim={raw_features.shape[1]}, time={extract_time:.1f}s"
        )

        # Apply PCA dimensionality reduction
        if self.use_pca and raw_features.shape[1] > self.pca_components:
            n_samples, n_features = raw_features.shape
            # Cap PCA components to valid range
            max_components = min(n_samples, n_features) - 1
            actual_components = min(self.pca_components, max_components)

            if actual_components < 2:
                logger.warning(
                    f"Too few samples ({n_samples}) for PCA — skipping"
                )
            else:
                if actual_components < self.pca_components:
                    logger.info(
                        f"PCA components capped: {self.pca_components} -> "
                        f"{actual_components} (n_samples={n_samples})"
                    )

                logger.info(
                    f"Applying PCA: {n_features} -> {actual_components} dims"
                )
                features_np = raw_features.numpy()

                self._pca = PCA(n_components=actual_components)
                features_reduced = self._pca.fit_transform(features_np)

                explained_var = sum(self._pca.explained_variance_ratio_) * 100
                logger.info(f"PCA explained variance: {explained_var:.1f}%")

                raw_features = torch.from_numpy(features_reduced).float()

                # Re-normalize after PCA
                raw_features = F.normalize(raw_features, p=2, dim=1)

        # Coreset subsampling
        if (
            self.subsample_size is not None
            and raw_features.shape[0] > self.subsample_size
        ):
            logger.info(
                f"Coreset subsampling: {raw_features.shape[0]} -> "
                f"{self.subsample_size} features"
            )
            indices = self._greedy_coreset(raw_features, self.subsample_size)
            raw_features = raw_features[indices]

        self.features = raw_features
        self.count = self.features.shape[0]

        total_time = time.time() - start_time
        logger.info(
            f"Memory bank built: {self.count} features, "
            f"dim={self.features.shape[1]}, total_time={total_time:.1f}s"
        )

    def _greedy_coreset(
        self, features: torch.Tensor, target_size: int
    ) -> np.ndarray:
        """
        Greedy k-center coreset selection.

        Iteratively selects the point furthest from the current coreset,
        ensuring maximum coverage of the feature space.

        Args:
            features: Feature tensor of shape (N, D)
            target_size: Number of features to select

        Returns:
            Array of selected indices
        """
        n = features.shape[0]
        if target_size >= n:
            return np.arange(n)

        # Start with a random point
        selected = [np.random.randint(n)]
        min_distances = torch.full((n,), float("inf"))

        for _ in range(target_size - 1):
            last = features[selected[-1]].unsqueeze(0)
            distances = torch.cdist(features, last).squeeze(1)
            min_distances = torch.minimum(min_distances, distances)
            next_idx = torch.argmax(min_distances).item()
            selected.append(next_idx)

        return np.array(selected)

    def save(self, save_path: str):
        """Save memory bank and PCA transform to file."""
        save_data = {
            "features": self.features,
            "count": self.count,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
        }

        # Save PCA components if fitted
        if self._pca is not None:
            save_data["pca_components_"] = self._pca.components_
            save_data["pca_mean_"] = self._pca.mean_
            save_data["pca_explained_variance_"] = self._pca.explained_variance_

        torch.save(save_data, save_path)
        logger.info(f"Memory bank saved to {save_path}")

    def load(self, load_path: str):
        """Load memory bank and PCA transform from file."""
        data = torch.load(load_path, map_location="cpu", weights_only=False)
        self.features = data["features"]
        self.count = data["count"]
        self.use_pca = data.get("use_pca", False)
        self.pca_components = data.get("pca_components", 256)

        # Restore PCA if saved
        if "pca_components_" in data:
            n_components = data["pca_components_"].shape[0]
            n_features = data["pca_components_"].shape[1]
            self._pca = PCA(n_components=n_components)
            self._pca.components_ = data["pca_components_"]
            self._pca.mean_ = data["pca_mean_"]
            self._pca.explained_variance_ = data["pca_explained_variance_"]
            # Set n_features_in_ for transform
            self._pca.n_features_in_ = n_features
            logger.info(f"PCA transform restored: {n_features} -> {n_components} dims")

        logger.info(
            f"Memory bank loaded from {load_path}: "
            f"{self.count} features, dim={self.features.shape[1]}"
        )

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the same PCA transform used during build to new features.

        Args:
            features: Raw features of shape (N, D_original)

        Returns:
            Transformed features of shape (N, D_reduced)
        """
        if self._pca is not None:
            features_np = features.numpy()
            features_reduced = self._pca.transform(features_np)
            features = torch.from_numpy(features_reduced).float()
            features = F.normalize(features, p=2, dim=1)
        return features


class AnomalyScorer:
    """
    Scores test images based on their distance to the normal feature memory bank.
    Higher scores indicate more anomalous samples.

    Methods:
        - Mahalanobis: Mahalanobis distance to the mean of normal features
          (numerically stable, vectorized, with shrinkage regularization)
        - kNN: Average distance to k nearest neighbors

    Usage:
        scorer = AnomalyScorer(method="mahalanobis")
        scorer.fit(memory_bank)
        scores = scorer.score(features)
    """

    def __init__(
        self,
        method: str = "mahalanobis",
        k_neighbors: int = 5,
    ):
        self.method = method.lower()
        self.k = k_neighbors
        self.memory_bank: Optional[MemoryBank] = None

        # Pre-computed statistics for Mahalanobis distance
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._cholesky: Optional[np.ndarray] = None  # For efficient computation

        logger.info(
            f"Anomaly scorer initialized: method={method}, k={k_neighbors}"
        )

    def fit(self, memory_bank: MemoryBank):
        """
        Fit the scorer using the memory bank of normal features.

        For Mahalanobis: computes mean and inverse covariance matrix
        with Ledoit-Wolf shrinkage for numerical stability.

        Args:
            memory_bank: MemoryBank containing normal features
        """
        self.memory_bank = memory_bank

        if self.method == "mahalanobis":
            self._fit_mahalanobis(memory_bank)

        logger.info(f"Scorer fitted with {memory_bank.count} reference features")

    def _fit_mahalanobis(self, memory_bank: MemoryBank):
        """
        Compute Mahalanobis distance statistics with regularized covariance.

        Uses shrinkage regularization (cov + lambda * I) for numerical stability,
        especially important when feature dimension > number of samples.
        """
        features_np = memory_bank.features.numpy().astype(np.float64)
        n_samples, n_features = features_np.shape

        # Compute mean
        self._mean = np.mean(features_np, axis=0)

        # Compute covariance with regularization
        cov = np.cov(features_np, rowvar=False)

        # Adaptive regularization: larger lambda when n_samples < n_features
        if n_samples < n_features:
            reg_strength = 0.1
            logger.info(
                f"Mahalanobis: n_samples ({n_samples}) < n_features ({n_features}), "
                f"using strong regularization (lambda={reg_strength})"
            )
        else:
            reg_strength = 1e-5

        cov_reg = cov + reg_strength * np.eye(n_features, dtype=np.float64)

        # Compute inverse using Cholesky decomposition (more stable than direct inverse)
        try:
            L = np.linalg.cholesky(cov_reg)
            L_inv = np.linalg.inv(L)
            self._cov_inv = L_inv.T @ L_inv
            self._cholesky = L_inv
            logger.info(
                f"Mahalanobis statistics computed via Cholesky decomposition "
                f"(dim={n_features}, reg={reg_strength})"
            )
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            logger.warning(
                "Cholesky failed — using pseudo-inverse for covariance"
            )
            self._cov_inv = np.linalg.pinv(cov_reg)
            self._cholesky = None

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute anomaly scores for the given features.

        Features should be raw model output — PCA transform is applied
        automatically if the memory bank was built with PCA.

        Args:
            features: Test features, shape (N, D)

        Returns:
            Anomaly scores, shape (N,). Higher = more anomalous.
        """
        # Apply PCA if memory bank uses it
        if self.memory_bank is not None:
            features = self.memory_bank.transform_features(features)
        else:
            features = F.normalize(features, p=2, dim=1)

        if self.method == "mahalanobis":
            return self._mahalanobis_score(features)
        elif self.method == "knn":
            return self._knn_score(features)
        else:
            raise ValueError(f"Unknown scoring method: {self.method}")

    def _mahalanobis_score(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute vectorized Mahalanobis distance anomaly score.

        score(x) = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))

        Vectorized computation — no Python loop over individual samples.
        """
        features_np = features.numpy().astype(np.float64)

        # Center features
        diff = features_np - self._mean  # (N, D)

        if self._cholesky is not None:
            # Efficient: ||L_inv @ (x - mu)||_2
            transformed = diff @ self._cholesky.T  # (N, D)
            scores = np.sqrt(np.sum(transformed ** 2, axis=1))
        else:
            # Fallback: full Mahalanobis
            left = diff @ self._cov_inv  # (N, D)
            scores = np.sqrt(np.sum(left * diff, axis=1))

        return scores.astype(np.float32)

    def _knn_score(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute k-NN anomaly score.
        Score = average distance to k nearest neighbors in memory bank.
        """
        bank = self.memory_bank.features  # (M, D)

        # Cosine similarity then convert to distance
        sim_matrix = torch.mm(features, bank.t())  # (N, M)
        distances = 1.0 - sim_matrix  # cosine distance

        # Get k smallest distances
        k = min(self.k, distances.shape[1])
        topk_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)

        scores = topk_distances.mean(dim=1).numpy()
        return scores

    @torch.no_grad()
    def score_batch(
        self,
        model,
        dataloader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score an entire dataset.

        Args:
            model: Trained SimCLR model
            dataloader: Test DataLoader
            device: Device for inference

        Returns:
            Tuple of (anomaly_scores, true_labels)
        """
        model.eval()
        all_scores = []
        all_labels = []

        for images, labels, indices in dataloader:
            images = images.to(device)

            # Use multi-layer extraction if available
            if hasattr(model, "extract_features"):
                features = model.extract_features(images)
            else:
                features = model.encode(images)

            features_cpu = features.cpu()
            scores = self.score(features_cpu)
            all_scores.append(scores)
            all_labels.append(labels.numpy())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        return all_scores, all_labels
