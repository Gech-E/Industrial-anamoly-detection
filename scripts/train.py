"""
Training entry point for SimCLR contrastive pretraining.
Compatible with enhanced config (ResNet50 + patch features + compression).

Usage:
python scripts/train.py --config configs/config.yaml --category bottle
"""

import os
import sys
import argparse
import logging
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import (
    load_config, set_seed, get_device,
    setup_logging, ensure_dirs, count_parameters
)
from src.models.simclr import SimCLRModel
from src.training.dataset import create_train_dataloader, create_feature_dataloader
from src.training.trainer import SimCLRTrainer
from src.memory.memory_bank import MemoryBank

logger = logging.getLogger(__name__)


def train_category(config: dict, category: str, device: torch.device):
    """Train SimCLR model for a single MVTec category."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Training SimCLR for category: {category}")
    logger.info(f"{'='*60}")

    # 1. Data
    logger.info("Creating training dataloader...")
    train_dataloader = create_train_dataloader(config, category)
    logger.info(f"Training samples: {len(train_dataloader.dataset)}")

    # 2. Model
    model = SimCLRModel(config)

    param_info = count_parameters(model)
    logger.info(
        f"Model parameters: {param_info['total_millions']:.2f}M total, "
        f"{param_info['trainable_millions']:.2f}M trainable"
    )

    # Move model to device early (important for feature extraction later)
    model.to(device)

    # 3. Trainer
    trainer = SimCLRTrainer(model, config, device)
    history = trainer.train(train_dataloader, category)

    # 4. Build Memory Bank (Patch + Compression aware)
    logger.info("Building feature memory bank from normal training images...")

    feature_dataloader = create_feature_dataloader(config, category)

    memory_bank = MemoryBank(
        use_patches=config.get("model", {}).get("use_patches", False),
        patch_layer=config.get("model", {}).get("patch_layer", "layer2"),
        subsample_size=config.get("memory_bank", {}).get("subsample", None),
        feature_compression=config.get("model", {}).get("feature_compression", None),  # NEW
    )

    memory_bank.build(
        model=model,
        dataloader=feature_dataloader,
        device=device,
    )

    # 5. Save Memory Bank
    checkpoint_dir = config.get("output", {}).get(
        "checkpoints_dir", "outputs/checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
    memory_bank.save(bank_path)

    logger.info(f"Memory bank saved: {bank_path}")

    # 6. Save Metadata (NEW - IMPORTANT)
    meta = {
        "category": category,
        "use_patches": config.get("model", {}).get("use_patches", False),
        "patch_layer": config.get("model", {}).get("patch_layer", "layer2"),
        "feature_compression": config.get("model", {}).get("feature_compression", None),
        "k_neighbors": config.get("anomaly_detection", {}).get("k_neighbors", 3),
        "aggregation": config.get("anomaly_detection", {}).get("score_aggregation", "max"),
        "threshold_method": config.get("thresholding", {}).get("method", None),
        "threshold_value": config.get("thresholding", {}).get("value", None),
    }

    meta_path = os.path.join(checkpoint_dir, f"{category}_meta.pt")
    torch.save(meta, meta_path)

    logger.info(f"Metadata saved: {meta_path}")
    logger.info(f"Training complete for category: {category}")

    return model, memory_bank, history


def main():
    parser = argparse.ArgumentParser(description="SimCLR Contrastive Pretraining")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="MVTec category to train on",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # Setup
    setup_logging(config.get("output", {}).get("logs_dir", "outputs/logs"))
    set_seed(config["training"].get("seed", 42))
    ensure_dirs(config)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Categories
    if args.category:
        categories = [args.category]
    else:
        categories = config["dataset"]["categories"]

    # Train Loop
    for category in categories:
        train_category(config, category, device)

    logger.info("\n" + "=" * 60)
    logger.info("All training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()