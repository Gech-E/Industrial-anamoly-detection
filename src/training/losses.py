"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
The core contrastive loss function used in SimCLR training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    For a batch of N images producing 2N augmented views:
    - Each image i has a positive pair (its other augmented view)
    - All other 2(N-1) views are treated as negatives
    - Loss encourages positive pairs to be similar, negatives to be dissimilar
    
    Args:
        temperature: Controls the sharpness of the similarity distribution.
                     Lower temperature → sharper distribution (harder negatives).
        
    Reference: "A Simple Framework for Contrastive Learning of Visual Representations"
               (Chen et al., 2020) — SimCLR paper
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        logger.info(f"NT-Xent loss initialized with temperature={temperature}")
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs.
        
        Args:
            z_i: Projected features from view 1, shape (N, D)
            z_j: Projected features from view 2, shape (N, D)
            
        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # L2 normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate both views: [z_i; z_j] → shape (2N, D)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        
        # Compute cosine similarity matrix: (2N, 2N)
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)
        
        # Create mask to exclude self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -float("inf"))
        
        # Create positive pair labels
        # For z_i[k], the positive pair is z_j[k] at index k + batch_size
        # For z_j[k], the positive pair is z_i[k] at index k
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),  # z_i → z_j
            torch.arange(0, batch_size, device=device),               # z_j → z_i
        ])  # (2N,)
        
        # Cross entropy loss treats the similarity matrix rows as logits
        # and the labels indicate which column is the positive pair
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Alternative simpler contrastive loss (for comparison/ablation).
    Directly uses cosine similarity with margin.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            z_i: Projected features from view 1, shape (N, D)
            z_j: Projected features from view 2, shape (N, D)
            
        Returns:
            Scalar loss value
        """
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Positive pair similarity (should be high → close to 1)
        pos_sim = F.cosine_similarity(z_i, z_j, dim=1)  # (N,)
        
        # Loss: encourage positive pairs to be similar
        loss = -torch.log(
            torch.exp(pos_sim / self.temperature) /
            torch.exp(torch.tensor(1.0 / self.temperature, device=z_i.device))
        ).mean()
        
        return loss
