"""
Feature Memory Bank and Anomaly Scoring.
After SimCLR pretraining, stores normal-sample features and scores test images
using k-NN distance as the anomaly score.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


class MemoryBank:
    """
    Stores feature representations of normal training images.
    Used as the reference distribution for anomaly scoring.
    """
    
    def __init__(
        self,
        use_patches: bool = False,
        patch_layer: str = "layer2",
        subsample_size: Optional[float] = None,
        feature_compression: Optional[int] = None,
        **kwargs
    ):
        self.use_patches = use_patches
        self.patch_layer = patch_layer
        self.subsample_size = subsample_size
        self.feature_compression = feature_compression
        self.features = None
        self.count = 0
    
    @torch.no_grad()
    def build(self, model, dataloader, device: torch.device):
        """
        Extract features from all normal training images and store them.
        
        Args:
            model: Trained SimCLR model (uses encoder only)
            dataloader: DataLoader for normal training images
            device: Device to run inference on
        """
        model.eval()
        all_features = []
        
        for batch_idx, (images, labels, indices) in enumerate(dataloader):
            images = images.to(device)
            features = model.encode(images)
            features = F.normalize(features, dim=1)
            all_features.append(features.cpu())
        
        self.features = torch.cat(all_features, dim=0)
        self.count = self.features.shape[0]
        
        logger.info(
            f"Memory bank built: {self.count} features, "
            f"shape={self.features.shape}"
        )
    
    def save(self, save_path: str):
        """Save memory bank to file."""
        torch.save({
            "features": self.features,
            "count": self.count,
        }, save_path)
        logger.info(f"Memory bank saved to {save_path}")
    
    def load(self, load_path: str):
        """Load memory bank from file."""
        data = torch.load(load_path, map_location="cpu", weights_only=True)
        self.features = data["features"]
        self.count = data["count"]
        logger.info(f"Memory bank loaded from {load_path}: {self.count} features")


class AnomalyScorer:
    """
    Scores test images based on their distance to the normal feature memory bank.
    Higher scores indicate more anomalous samples.
    
    Methods:
        - k-NN: Average distance to k nearest neighbors
        - Mahalanobis: Mahalanobis distance to the mean of normal features
    """
    
    def __init__(
        self,
        method: str = "knn",
        k_neighbors: int = 5,
    ):
        self.method = method
        self.k = k_neighbors
        self.memory_bank = None
        
        # Pre-computed statistics for Mahalanobis distance
        self._mean = None
        self._cov_inv = None
        
        logger.info(f"Anomaly scorer initialized: method={method}, k={k_neighbors}")
    
    def fit(self, memory_bank: MemoryBank):
        """
        Fit the scorer using the memory bank of normal features.
        
        Args:
            memory_bank: MemoryBank containing normal features
        """
        self.memory_bank = memory_bank
        
        if self.method == "mahalanobis":
            features_np = memory_bank.features.numpy()
            self._mean = np.mean(features_np, axis=0)
            cov = np.cov(features_np, rowvar=False) + np.eye(features_np.shape[1]) * 1e-6
            self._cov_inv = np.linalg.inv(cov)
            logger.info("Mahalanobis statistics computed")
    
    @torch.no_grad()
    def score(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute anomaly scores for the given features.
        
        Args:
            features: Test features, shape (N, D), L2-normalized
            
        Returns:
            Anomaly scores, shape (N,). Higher = more anomalous.
        """
        features = F.normalize(features, dim=1)
        
        if self.method == "knn":
            return self._knn_score(features)
        elif self.method == "mahalanobis":
            return self._mahalanobis_score(features)
        else:
            raise ValueError(f"Unknown scoring method: {self.method}")
    
    def _knn_score(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute k-NN anomaly score.
        Score = average distance to k nearest neighbors in memory bank.
        """
        bank = self.memory_bank.features  # (M, D)
        
        # Compute pairwise distances: (N, M)
        # Using cosine distance: 1 - cosine_similarity
        sim_matrix = torch.mm(features, bank.t())  # (N, M)
        distances = 1.0 - sim_matrix  # cosine distance
        
        # Get k smallest distances for each test sample
        topk_distances, _ = torch.topk(distances, k=self.k, dim=1, largest=False)
        
        # Average k-NN distance as anomaly score
        scores = topk_distances.mean(dim=1).numpy()
        
        return scores
    
    def _mahalanobis_score(self, features: torch.Tensor) -> np.ndarray:
        """
        Compute Mahalanobis distance anomaly score.
        """
        features_np = features.numpy()
        scores = np.array([
            mahalanobis(f, self._mean, self._cov_inv)
            for f in features_np
        ])
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
            features = model.encode(images)
            features = features.cpu()
            
            scores = self.score(features)
            all_scores.append(scores)
            all_labels.append(labels.numpy())
        
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        return all_scores, all_labels
