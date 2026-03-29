"""
Feature Memory Bank and Anomaly Scoring.

Supports two modes:
    1. Global Memory Bank (original): Stores globally-pooled features,
       scores using Mahalanobis or kNN distance.
    2. Patch Memory Bank (PatchCore-style): Stores patch-level spatial features,
       scores using kNN on per-patch embeddings. Produces anomaly maps for localization.

Research note:
    PatchCore (Roth et al., CVPR 2022) achieves state-of-the-art on MVTec AD
    (99.1% AUROC) by using patch-level features with coreset subsampling.
    The key insight is that anomalies are LOCAL — they affect specific spatial
    regions, not the entire image. Global pooling averages out these local signals.
"""

import logging
import os
import time
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Global Memory Bank (original, backward-compatible)
# ═══════════════════════════════════════════════════════════════════

class MemoryBank:
    """
    Stores feature representations of normal training images (global features).
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
        """
        model.eval()
        all_features = []

        logger.info("Extracting features for memory bank...")
        start_time = time.time()

        for batch_idx, (images, labels, indices) in enumerate(dataloader):
            images = images.to(device)

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
        """Greedy k-center coreset selection."""
        n = features.shape[0]
        if target_size >= n:
            return np.arange(n)

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

        if "pca_components_" in data:
            n_components = data["pca_components_"].shape[0]
            n_features = data["pca_components_"].shape[1]
            self._pca = PCA(n_components=n_components)
            self._pca.components_ = data["pca_components_"]
            self._pca.mean_ = data["pca_mean_"]
            self._pca.explained_variance_ = data["pca_explained_variance_"]
            self._pca.n_features_in_ = n_features
            logger.info(f"PCA transform restored: {n_features} -> {n_components} dims")

        logger.info(
            f"Memory bank loaded from {load_path}: "
            f"{self.count} features, dim={self.features.shape[1]}"
        )

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the same PCA transform used during build to new features."""
        if self._pca is not None:
            features_np = features.numpy()
            features_reduced = self._pca.transform(features_np)
            features = torch.from_numpy(features_reduced).float()
            features = F.normalize(features, p=2, dim=1)
        return features


# ═══════════════════════════════════════════════════════════════════
# Global Anomaly Scorer (original, backward-compatible)
# ═══════════════════════════════════════════════════════════════════

class AnomalyScorer:
    """
    Scores test images based on their distance to the normal feature memory bank.
    Higher scores indicate more anomalous samples.

    Methods:
        - Mahalanobis: Mahalanobis distance to the mean of normal features
        - kNN: Average distance to k nearest neighbors
    """

    def __init__(
        self,
        method: str = "mahalanobis",
        k_neighbors: int = 5,
    ):
        self.method = method.lower()
        self.k = k_neighbors
        self.memory_bank: Optional[MemoryBank] = None

        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._cholesky: Optional[np.ndarray] = None

        logger.info(
            f"Anomaly scorer initialized: method={method}, k={k_neighbors}"
        )

    def fit(self, memory_bank: MemoryBank):
        """Fit the scorer using the memory bank of normal features."""
        self.memory_bank = memory_bank

        if self.method == "mahalanobis":
            self._fit_mahalanobis(memory_bank)

        logger.info(f"Scorer fitted with {memory_bank.count} reference features")

    def _fit_mahalanobis(self, memory_bank: MemoryBank):
        """Compute Mahalanobis distance statistics with regularized covariance."""
        features_np = memory_bank.features.numpy().astype(np.float64)
        n_samples, n_features = features_np.shape

        self._mean = np.mean(features_np, axis=0)
        cov = np.cov(features_np, rowvar=False)

        if n_samples < n_features:
            reg_strength = 0.1
            logger.info(
                f"Mahalanobis: n_samples ({n_samples}) < n_features ({n_features}), "
                f"using strong regularization (lambda={reg_strength})"
            )
        else:
            reg_strength = 1e-5

        cov_reg = cov + reg_strength * np.eye(n_features, dtype=np.float64)

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
            logger.warning(
                "Cholesky failed — using pseudo-inverse for covariance"
            )
            self._cov_inv = np.linalg.pinv(cov_reg)
            self._cholesky = None

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> np.ndarray:
        """Compute anomaly scores for the given features."""
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
        """Compute vectorized Mahalanobis distance anomaly score."""
        features_np = features.numpy().astype(np.float64)
        diff = features_np - self._mean

        if self._cholesky is not None:
            transformed = diff @ self._cholesky.T
            scores = np.sqrt(np.sum(transformed ** 2, axis=1))
        else:
            left = diff @ self._cov_inv
            scores = np.sqrt(np.sum(left * diff, axis=1))

        return scores.astype(np.float32)

    def _knn_score(self, features: torch.Tensor) -> np.ndarray:
        """Compute k-NN anomaly score."""
        bank = self.memory_bank.features
        sim_matrix = torch.mm(features, bank.t())
        distances = 1.0 - sim_matrix

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
        """Score an entire dataset."""
        model.eval()
        all_scores = []
        all_labels = []

        for images, labels, indices in dataloader:
            images = images.to(device)

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


# ═══════════════════════════════════════════════════════════════════
# Patch Memory Bank (PatchCore-style) — PRIMARY method for high AUROC
# ═══════════════════════════════════════════════════════════════════

class PatchMemoryBank:
    """
    PatchCore-style patch-level memory bank.

    Instead of storing one global feature per image, stores individual
    patch embeddings from spatial feature maps. This preserves spatial
    information critical for detecting local anomalies.

    Pipeline:
        1. Extract patch features from all training images → (N*H*W, C) matrix
        2. Optional PCA for dimensionality reduction
        3. Greedy coreset subsampling to reduce memory (keep ~10%)
        4. Store as reference for kNN scoring

    Research note:
        PatchCore achieves 99.1% AUROC on MVTec AD by:
        - Using layer2+layer3 features (multi-scale)
        - Coreset subsampling (efficient, <1% AUROC loss)
        - kNN scoring with k=1 (or small k)
    """

    def __init__(
        self,
        use_pca: bool = True,
        pca_components: int = 256,
        coreset_ratio: float = 0.1,
        coreset_max: int = 5000,
    ):
        """
        Args:
            use_pca: Whether to apply PCA to patch features
            pca_components: Target dimensions after PCA
            coreset_ratio: Fraction of patches to keep (0.1 = 10%)
            coreset_max: Maximum number of patches in coreset
        """
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.coreset_ratio = coreset_ratio
        self.coreset_max = coreset_max

        self.features: Optional[torch.Tensor] = None
        self.patch_shape: Optional[Tuple[int, int]] = None
        self.count: int = 0
        self.feature_dim: int = 0

        self._pca: Optional[PCA] = None

    @torch.no_grad()
    def build(
        self,
        model,
        dataloader,
        device: torch.device,
    ):
        """
        Build patch memory bank from training data.

        Extracts patch-level features from all training images,
        applies PCA, then selects a representative coreset.
        """
        model.eval()
        all_patches = []

        logger.info("Building patch memory bank...")
        start_time = time.time()

        for batch_idx, (images, labels, indices) in enumerate(dataloader):
            images = images.to(device)

            # Extract patch features: (B, num_patches, C)
            patch_features, patch_shape = model.extract_patch_features(images)
            self.patch_shape = patch_shape

            # Reshape to (B*num_patches, C)
            B, N, C = patch_features.shape
            patches_flat = patch_features.reshape(B * N, C)
            all_patches.append(patches_flat.cpu())

        raw_patches = torch.cat(all_patches, dim=0)
        self.feature_dim = raw_patches.shape[1]
        extract_time = time.time() - start_time

        logger.info(
            f"Patch extraction: {raw_patches.shape[0]} patches, "
            f"dim={raw_patches.shape[1]}, patch_shape={self.patch_shape}, "
            f"time={extract_time:.1f}s"
        )

        # Apply PCA dimensionality reduction
        if self.use_pca and raw_patches.shape[1] > self.pca_components:
            n_patches, n_features = raw_patches.shape
            max_components = min(n_patches, n_features) - 1
            actual_components = min(self.pca_components, max_components)

            if actual_components >= 2:
                logger.info(
                    f"Applying PCA to patches: {n_features} -> {actual_components} dims"
                )
                patches_np = raw_patches.numpy()

                self._pca = PCA(n_components=actual_components)
                patches_reduced = self._pca.fit_transform(patches_np)

                explained_var = sum(self._pca.explained_variance_ratio_) * 100
                logger.info(f"PCA explained variance: {explained_var:.1f}%")

                raw_patches = torch.from_numpy(patches_reduced).float()
                raw_patches = F.normalize(raw_patches, p=2, dim=1)
            else:
                logger.warning("Too few patches for PCA — skipping")

        # Coreset subsampling
        target_size = min(
            int(raw_patches.shape[0] * self.coreset_ratio),
            self.coreset_max,
        )
        target_size = max(target_size, 100)  # Keep at least 100 patches

        if raw_patches.shape[0] > target_size:
            logger.info(
                f"Coreset subsampling: {raw_patches.shape[0]} -> {target_size} patches"
            )
            indices = self._greedy_coreset(raw_patches, target_size)
            raw_patches = raw_patches[indices]

        self.features = raw_patches
        self.count = self.features.shape[0]

        total_time = time.time() - start_time
        logger.info(
            f"Patch memory bank built: {self.count} patches, "
            f"dim={self.features.shape[1]}, total_time={total_time:.1f}s"
        )

    def _greedy_coreset(
        self, features: torch.Tensor, target_size: int
    ) -> np.ndarray:
        """
        Greedy k-center coreset selection for patch features.

        Uses batch processing to avoid OOM on large patch sets.
        """
        n = features.shape[0]
        if target_size >= n:
            return np.arange(n)

        # Start with a random point
        selected = [np.random.randint(n)]
        min_distances = torch.full((n,), float("inf"))

        # Process in chunks to avoid memory issues
        batch_size = 5000

        for i in range(target_size - 1):
            last_feat = features[selected[-1]].unsqueeze(0)

            # Compute distances in batches
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                chunk = features[start:end]
                dist = torch.cdist(chunk, last_feat).squeeze(1)
                min_distances[start:end] = torch.minimum(
                    min_distances[start:end], dist
                )

            next_idx = torch.argmax(min_distances).item()
            selected.append(next_idx)

            # Log progress every 500 iterations
            if (i + 1) % 500 == 0:
                logger.info(f"  Coreset progress: {i + 1}/{target_size}")

        return np.array(selected)

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PCA transform to new patch features."""
        if self._pca is not None:
            features_np = features.numpy()
            features_reduced = self._pca.transform(features_np)
            features = torch.from_numpy(features_reduced).float()
            features = F.normalize(features, p=2, dim=1)
        return features

    def save(self, save_path: str):
        """Save patch memory bank."""
        save_data = {
            "features": self.features,
            "count": self.count,
            "feature_dim": self.feature_dim,
            "patch_shape": self.patch_shape,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "coreset_ratio": self.coreset_ratio,
            "type": "patch",
        }

        if self._pca is not None:
            save_data["pca_components_"] = self._pca.components_
            save_data["pca_mean_"] = self._pca.mean_
            save_data["pca_explained_variance_"] = self._pca.explained_variance_

        torch.save(save_data, save_path)
        logger.info(f"Patch memory bank saved to {save_path}")

    def load(self, load_path: str):
        """Load patch memory bank."""
        data = torch.load(load_path, map_location="cpu", weights_only=False)
        self.features = data["features"]
        self.count = data["count"]
        self.feature_dim = data.get("feature_dim", self.features.shape[1])
        self.patch_shape = data.get("patch_shape", None)
        self.use_pca = data.get("use_pca", False)
        self.pca_components = data.get("pca_components", 256)
        self.coreset_ratio = data.get("coreset_ratio", 0.1)

        if "pca_components_" in data:
            n_components = data["pca_components_"].shape[0]
            n_features = data["pca_components_"].shape[1]
            self._pca = PCA(n_components=n_components)
            self._pca.components_ = data["pca_components_"]
            self._pca.mean_ = data["pca_mean_"]
            self._pca.explained_variance_ = data["pca_explained_variance_"]
            self._pca.n_features_in_ = n_features
            logger.info(f"Patch PCA transform restored: {n_features} -> {n_components}")

        logger.info(
            f"Patch memory bank loaded: {self.count} patches, "
            f"dim={self.features.shape[1]}, patch_shape={self.patch_shape}"
        )


# ═══════════════════════════════════════════════════════════════════
# Patch Anomaly Scorer — Hybrid kNN + Mahalanobis scoring
# ═══════════════════════════════════════════════════════════════════

class PatchAnomalyScorer:
    """
    PatchCore-style anomaly scorer with hybrid scoring.

    For each test image:
        1. Extract patch features → (num_patches, C)
        2. For each patch, find k nearest neighbors in memory bank
        3. Patch score = distance to nearest neighbor(s)
        4. Image score = max(patch_scores) [most anomalous patch]
        5. Anomaly map = reshaped patch scores (for localization)

    Scoring methods (ensemble):
        - kNN distance (primary): L2 distance to nearest patches
        - Mahalanobis (secondary): Distance from global distribution
        - Cosine distance (optional): Angular distance

    Final score = w_knn * knn_score + w_maha * maha_score + w_cos * cos_score
    """

    def __init__(
        self,
        k_neighbors: int = 3,
        weight_knn: float = 1.0,
        weight_mahalanobis: float = 0.0,
        weight_cosine: float = 0.0,
    ):
        """
        Args:
            k_neighbors: Number of nearest neighbors for kNN scoring
            weight_knn: Weight for kNN distance score
            weight_mahalanobis: Weight for Mahalanobis distance score
            weight_cosine: Weight for cosine distance score
        """
        self.k = k_neighbors
        self.weight_knn = weight_knn
        self.weight_maha = weight_mahalanobis
        self.weight_cos = weight_cosine

        self.memory_bank: Optional[PatchMemoryBank] = None

        # Mahalanobis statistics (fitted on patch features)
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None

        # Try to use faiss for fast kNN
        self._faiss_index = None
        self._use_faiss = False

        logger.info(
            f"PatchAnomalyScorer initialized: k={k_neighbors}, "
            f"weights=(knn={weight_knn}, maha={weight_mahalanobis}, "
            f"cos={weight_cosine})"
        )

    def fit(self, memory_bank: PatchMemoryBank):
        """Fit the scorer with the patch memory bank."""
        self.memory_bank = memory_bank

        features_np = memory_bank.features.numpy().astype(np.float64)

        # Build faiss index if available
        try:
            import faiss
            dim = memory_bank.features.shape[1]
            self._faiss_index = faiss.IndexFlatL2(dim)
            self._faiss_index.add(memory_bank.features.numpy().astype(np.float32))
            self._use_faiss = True
            logger.info(f"Faiss index built with {memory_bank.count} patches")
        except ImportError:
            self._use_faiss = False
            logger.info("Faiss not available — using torch for kNN (slower)")

        # Fit Mahalanobis if weighted
        if self.weight_maha > 0:
            self._fit_mahalanobis(features_np)

        logger.info(
            f"PatchAnomalyScorer fitted with {memory_bank.count} patches"
        )

    def _fit_mahalanobis(self, features_np: np.ndarray):
        """Compute Mahalanobis statistics on patch features."""
        n_samples, n_features = features_np.shape
        self._mean = np.mean(features_np, axis=0)

        cov = np.cov(features_np, rowvar=False)
        reg = 0.1 if n_samples < n_features else 1e-5
        cov_reg = cov + reg * np.eye(n_features, dtype=np.float64)

        try:
            self._cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            self._cov_inv = np.linalg.pinv(cov_reg)

    def score_patches(
        self, patch_features: torch.Tensor
    ) -> Tuple[np.ndarray, float]:
        """
        Score individual patches and compute image-level score.

        Args:
            patch_features: Patch embeddings (num_patches, C)
                           Raw features — PCA transform applied internally.

        Returns:
            Tuple of:
                - patch_scores: Per-patch anomaly scores (num_patches,)
                - image_score: Image-level anomaly score (max of patches)
        """
        # Apply PCA if memory bank uses it
        if self.memory_bank is not None:
            patch_features = self.memory_bank.transform_features(patch_features)

        # kNN scoring
        knn_scores = np.zeros(patch_features.shape[0], dtype=np.float32)
        if self.weight_knn > 0:
            knn_scores = self._knn_patch_score(patch_features)

        # Mahalanobis scoring
        maha_scores = np.zeros(patch_features.shape[0], dtype=np.float32)
        if self.weight_maha > 0 and self._mean is not None:
            maha_scores = self._mahalanobis_patch_score(patch_features)

        # Cosine scoring
        cos_scores = np.zeros(patch_features.shape[0], dtype=np.float32)
        if self.weight_cos > 0:
            cos_scores = self._cosine_patch_score(patch_features)

        # Weighted ensemble
        total_weight = self.weight_knn + self.weight_maha + self.weight_cos
        if total_weight == 0:
            total_weight = 1.0

        patch_scores = (
            self.weight_knn * knn_scores
            + self.weight_maha * maha_scores
            + self.weight_cos * cos_scores
        ) / total_weight

        # Image score = max over all patches
        image_score = float(np.max(patch_scores))

        return patch_scores, image_score

    def _knn_patch_score(self, patches: torch.Tensor) -> np.ndarray:
        """kNN distance scoring for patches."""
        if self._use_faiss and self._faiss_index is not None:
            return self._knn_faiss(patches)
        else:
            return self._knn_torch(patches)

    def _knn_faiss(self, patches: torch.Tensor) -> np.ndarray:
        """kNN using faiss (fast)."""
        import faiss
        queries = patches.numpy().astype(np.float32)
        k = min(self.k, self.memory_bank.count)
        distances, _ = self._faiss_index.search(queries, k)
        # Average distance to k neighbors
        scores = np.mean(np.sqrt(np.maximum(distances, 0)), axis=1)
        return scores.astype(np.float32)

    def _knn_torch(self, patches: torch.Tensor) -> np.ndarray:
        """kNN using torch cdist (fallback, works on CPU)."""
        bank = self.memory_bank.features
        k = min(self.k, bank.shape[0])

        # Process in batches to avoid OOM
        batch_size = 256
        all_scores = []

        for start in range(0, patches.shape[0], batch_size):
            end = min(start + batch_size, patches.shape[0])
            chunk = patches[start:end]

            # L2 distance
            distances = torch.cdist(chunk, bank)  # (chunk_size, bank_size)

            # k smallest distances
            topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
            scores = topk_dist.mean(dim=1).numpy()
            all_scores.append(scores)

        return np.concatenate(all_scores).astype(np.float32)

    def _mahalanobis_patch_score(self, patches: torch.Tensor) -> np.ndarray:
        """Mahalanobis distance for each patch."""
        patches_np = patches.numpy().astype(np.float64)
        diff = patches_np - self._mean
        left = diff @ self._cov_inv
        scores = np.sqrt(np.sum(left * diff, axis=1))
        # Normalize to similar range as kNN
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-8)
        return scores.astype(np.float32)

    def _cosine_patch_score(self, patches: torch.Tensor) -> np.ndarray:
        """Cosine distance scoring for patches."""
        bank = self.memory_bank.features
        k = min(self.k, bank.shape[0])

        # Cosine similarity
        sim = torch.mm(patches, bank.t())  # (N, M)
        # Convert to distance
        distances = 1.0 - sim

        topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
        scores = topk_dist.mean(dim=1).numpy()
        return scores.astype(np.float32)

    @torch.no_grad()
    def score_batch(
        self,
        model,
        dataloader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Score an entire dataset with patch-level scoring.

        Returns:
            Tuple of:
                - image_scores: (N,) image-level anomaly scores
                - labels: (N,) ground truth labels
                - all_patch_scores: List of (patch_scores, patch_shape) per image
        """
        model.eval()
        all_image_scores = []
        all_labels = []
        all_patch_info = []

        for images, labels, indices in dataloader:
            images = images.to(device)

            # Extract patch features: (B, num_patches, C)
            patch_features, patch_shape = model.extract_patch_features(images)

            for i in range(images.shape[0]):
                # Score patches for this image
                patches_i = patch_features[i]  # (num_patches, C)
                patch_scores, image_score = self.score_patches(patches_i)

                all_image_scores.append(image_score)
                all_patch_info.append((patch_scores, patch_shape))

            all_labels.append(labels.numpy())

        image_scores = np.array(all_image_scores, dtype=np.float32)
        labels = np.concatenate(all_labels)

        return image_scores, labels, all_patch_info
