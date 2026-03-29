"""
SimCLR-style contrastive augmentation pipeline.
Generates two differently augmented views of the same image for contrastive learning.
"""

import torch
from torchvision import transforms


class ContrastiveAugmentation:
    """
    Generates two augmented views of the same image using SimCLR-style transforms.
    
    Each view undergoes: RandomResizedCrop → RandomHorizontalFlip → ColorJitter → 
    RandomGrayscale → GaussianBlur → Normalize.
    """
    
    def __init__(self, config: dict):
        aug_cfg = config.get("augmentations", {})
        img_size = config.get("dataset", {}).get("image_size", 224)
        
        transform_list = []
        
        # Random Resized Crop
        crop_cfg = aug_cfg.get("random_crop", {})
        if crop_cfg.get("enabled", True):
            scale = tuple(crop_cfg.get("scale", [0.2, 1.0]))
            transform_list.append(
                transforms.RandomResizedCrop(size=img_size, scale=scale)
            )
        else:
            transform_list.append(transforms.Resize((img_size, img_size)))
        
        # Random Horizontal Flip
        flip_cfg = aug_cfg.get("horizontal_flip", {})
        if flip_cfg.get("enabled", True):
            p = flip_cfg.get("probability", 0.5)
            transform_list.append(transforms.RandomHorizontalFlip(p=p))
        
        # Color Jitter
        jitter_cfg = aug_cfg.get("color_jitter", {})
        if jitter_cfg.get("enabled", True):
            p = jitter_cfg.get("probability", 0.8)
            transform_list.append(
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=jitter_cfg.get("brightness", 0.4),
                        contrast=jitter_cfg.get("contrast", 0.4),
                        saturation=jitter_cfg.get("saturation", 0.4),
                        hue=jitter_cfg.get("hue", 0.1),
                    )
                ], p=p)
            )
        
        # Random Grayscale
        gray_cfg = aug_cfg.get("grayscale", {})
        if gray_cfg.get("enabled", True):
            p = gray_cfg.get("probability", 0.2)
            transform_list.append(transforms.RandomGrayscale(p=p))
        
        # Gaussian Blur
        blur_cfg = aug_cfg.get("gaussian_blur", {})
        if blur_cfg.get("enabled", True):
            p = blur_cfg.get("probability", 0.5)
            kernel_size = blur_cfg.get("kernel_size", 23)
            # Ensure kernel_size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            transform_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=kernel_size,
                        sigma=(0.1, 2.0)
                    )
                ], p=p)
            )
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        norm_cfg = aug_cfg.get("normalize", {})
        mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
        std = norm_cfg.get("std", [0.229, 0.224, 0.225])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, image):
        """Return two differently augmented views of the same image."""
        view_1 = self.transform(image)
        view_2 = self.transform(image)
        return view_1, view_2


def get_eval_transform(config: dict) -> transforms.Compose:
    """
    Get the evaluation/inference transform (no random augmentation).
    Only resize, center crop, and normalize.
    """
    img_size = config.get("dataset", {}).get("image_size", 224)
    norm_cfg = config.get("augmentations", {}).get("normalize", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_inverse_normalize(config: dict):
    """Get inverse normalization transform for visualization."""
    norm_cfg = config.get("augmentations", {}).get("normalize", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])
    
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    
    return transforms.Normalize(mean=inv_mean, std=inv_std)
