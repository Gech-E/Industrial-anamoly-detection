"""
PyTorch Dataset classes for MVTec AD anomaly detection.
Handles loading, organizing, and serving data for both training and evaluation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.training.augmentations import ContrastiveAugmentation, get_eval_transform


logger = logging.getLogger(__name__)


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset.
    
    Directory structure expected:
        root_dir/
        └── category/
            ├── train/
            │   └── good/        # Normal training images
            └── test/
                ├── good/        # Normal test images  
                └── defect_1/    # Anomalous test images (various defect types)
                └── defect_2/
                └── ...
    
    Args:
        root_dir: Root directory of MVTec AD dataset
        category: Product category (e.g., 'bottle', 'cable')
        split: 'train' or 'test'
        transform: Optional transform to apply to images
        is_contrastive: If True, returns two augmented views (for SimCLR training)
    """
    
    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        transform=None,
        is_contrastive: bool = False,
    ):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.is_contrastive = is_contrastive
        
        self.category_dir = os.path.join(root_dir, category, split)
        
        if not os.path.exists(self.category_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.category_dir}. "
                f"Please download the MVTec AD dataset first."
            )
        
        self.image_paths = []
        self.labels = []       # 0 = normal, 1 = anomaly
        self.defect_types = [] # 'good' or specific defect name
        
        self._load_samples()
        
        logger.info(
            f"MVTecDataset: category={category}, split={split}, "
            f"samples={len(self.image_paths)}, "
            f"normal={self.labels.count(0)}, anomaly={self.labels.count(1)}"
        )
    
    def _load_samples(self):
        """Scan directory structure and load image paths with labels."""
        skipped = 0
        for defect_type in sorted(os.listdir(self.category_dir)):
            defect_dir = os.path.join(self.category_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            
            label = 0 if defect_type == "good" else 1
            
            for img_name in sorted(os.listdir(defect_dir)):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    img_path = os.path.join(defect_dir, img_name)
                    # Validate image integrity
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        self.defect_types.append(defect_type)
                    except Exception as e:
                        skipped += 1
                        logger.warning(f"Skipping corrupt image: {img_path} ({e})")
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} corrupt image(s) in {self.category_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image as RGB
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}, using random fallback")
            # Return a random valid sample as fallback
            fallback_idx = (idx + 1) % len(self)
            return self.__getitem__(fallback_idx)
        
        if self.is_contrastive and self.transform is not None:
            # Return two augmented views for contrastive learning
            view_1, view_2 = self.transform(image)
            return view_1, view_2, label, idx
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx
    
    def get_image_path(self, idx: int) -> str:
        """Get the file path for a given index."""
        return self.image_paths[idx]
    
    def get_defect_type(self, idx: int) -> str:
        """Get the defect type for a given index."""
        return self.defect_types[idx]


def create_train_dataloader(
    config: dict,
    category: str,
) -> DataLoader:
    """
    Create DataLoader for SimCLR contrastive training (normal samples only).
    
    Returns:
        DataLoader yielding (view_1, view_2, label, idx) tuples
    """
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    
    contrastive_transform = ContrastiveAugmentation(config)
    
    dataset = MVTecDataset(
        root_dir=dataset_cfg["root_dir"],
        category=category,
        split="train",
        transform=contrastive_transform,
        is_contrastive=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


def create_test_dataloader(
    config: dict,
    category: str,
) -> DataLoader:
    """
    Create DataLoader for evaluation/testing.
    
    Returns:
        DataLoader yielding (image, label, idx) tuples
    """
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    
    eval_transform = get_eval_transform(config)
    
    dataset = MVTecDataset(
        root_dir=dataset_cfg["root_dir"],
        category=category,
        split="test",
        transform=eval_transform,
        is_contrastive=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    
    return dataloader


def create_feature_dataloader(
    config: dict,
    category: str,
) -> DataLoader:
    """
    Create DataLoader for feature extraction from training set (no augmentation).
    Used to build the memory bank after SimCLR pretraining.
    
    Returns:
        DataLoader yielding (image, label, idx) tuples
    """
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    
    eval_transform = get_eval_transform(config)
    
    dataset = MVTecDataset(
        root_dir=dataset_cfg["root_dir"],
        category=category,
        split="train",
        transform=eval_transform,
        is_contrastive=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    
    return dataloader
