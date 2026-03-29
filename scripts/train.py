"""
Training entry point for SimCLR contrastive pretraining + feature extraction.

Supports two modes:
    1. Feature extraction only (recommended): Uses ImageNet pretrained features
       directly with multi-layer extraction. Fast, no GPU needed, high AUROC.
    2. SimCLR training: Full contrastive pretraining with early stopping.

Usage:
    # Feature extraction only (default, recommended):
    python scripts/train.py --category bottle

    # SimCLR contrastive training:
    python scripts/train.py --category bottle --train-simclr

    # Resume training from checkpoint:
    python scripts/train.py --category bottle --train-simclr --resume
"""

import os
import sys
import argparse
import logging
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import (
    load_config, set_seed, get_device,
    setup_logging, ensure_dirs, count_parameters, format_time,
)
from src.models.simclr import SimCLRModel
from src.training.dataset import (
    create_train_dataloader, create_feature_dataloader, create_test_dataloader,
)
from src.training.trainer import SimCLRTrainer
from src.memory.memory_bank import MemoryBank, AnomalyScorer
from src.evaluation.evaluator import AnomalyEvaluator

logger = logging.getLogger(__name__)


def build_memory_bank(
    config: dict,
    model: SimCLRModel,
    category: str,
    device: torch.device,
) -> MemoryBank:
    """
    Build feature memory bank from normal training images.

    Uses multi-layer feature extraction if configured.
    Applies PCA and coreset subsampling if configured.
    """
    logger.info("Building feature memory bank from normal training images...")

    feature_dataloader = create_feature_dataloader(config, category)

    mb_cfg = config.get("memory_bank", {})
    memory_bank = MemoryBank(
        use_pca=mb_cfg.get("use_pca", False),
        pca_components=mb_cfg.get("pca_components", 256),
        subsample_size=mb_cfg.get("subsample", None),
    )

    memory_bank.build(
        model=model,
        dataloader=feature_dataloader,
        device=device,
    )

    return memory_bank


def quick_evaluate(
    config: dict,
    model: SimCLRModel,
    memory_bank: MemoryBank,
    category: str,
    device: torch.device,
) -> dict:
    """Run quick evaluation after training to report AUROC immediately."""
    logger.info("Running quick evaluation...")

    ad_cfg = config.get("anomaly_detection", {})
    scorer = AnomalyScorer(
        method=ad_cfg.get("method", "mahalanobis"),
        k_neighbors=ad_cfg.get("k_neighbors", 5),
    )
    scorer.fit(memory_bank)

    test_dataloader = create_test_dataloader(config, category)
    scores, labels = scorer.score_batch(model, test_dataloader, device)

    # Quick metrics
    results_dir = config.get("output", {}).get("results_dir", "outputs/results")
    threshold_cfg = config.get("thresholding", {})

    evaluator = AnomalyEvaluator(
        output_dir=results_dir,
        threshold_method=threshold_cfg.get("method", "youden"),
        percentile_value=threshold_cfg.get("value", 97),
    )

    metrics = evaluator.generate_full_report(scores, labels, category)

    return metrics


def train_category(config: dict, category: str, device: torch.device, args):
    """Train/extract features for a single MVTec category."""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing category: {category}")
    logger.info(f"{'=' * 60}")

    pipeline_start = time.time()

    # Determine mode
    feature_only = config.get("model", {}).get("feature_extraction_only", True)
    if args.train_simclr:
        feature_only = False

    # 1. Create model
    model = SimCLRModel(config)
    model = model.to(device)

    param_info = count_parameters(model)
    logger.info(
        f"Model parameters: {param_info['total_millions']:.2f}M total, "
        f"{param_info['trainable_millions']:.2f}M trainable"
    )

    checkpoint_dir = config.get("output", {}).get(
        "checkpoints_dir", "outputs/checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    if feature_only:
        # ── Feature extraction only mode ──
        logger.info(
            "Mode: Feature extraction only (ImageNet pretrained features)"
        )
        logger.info(
            "Using multi-layer extraction for high-quality representations"
        )
        model.eval()

        # Save model checkpoint (pretrained weights)
        best_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 0,
                "loss": 0.0,
                "category": category,
                "mode": "feature_extraction_only",
            },
            best_path,
        )
        logger.info(f"Pretrained model saved: {best_path}")

    else:
        # ── SimCLR contrastive training ──
        logger.info("Mode: SimCLR contrastive training")

        train_dataloader = create_train_dataloader(config, category)
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")

        trainer = SimCLRTrainer(model, config, device)

        # Resume if requested
        history = trainer.train(
            train_dataloader, category, resume=args.resume
        )

    # 2. Build memory bank
    memory_bank = build_memory_bank(config, model, category, device)

    # Save memory bank
    bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
    memory_bank.save(bank_path)

    # 3. Save metadata
    model_cfg = config.get("model", {})
    meta = {
        "category": category,
        "mode": "feature_extraction_only" if feature_only else "simclr",
        "multi_layer": model_cfg.get("multi_layer", True),
        "feature_layers": model_cfg.get(
            "feature_layers", ["layer2", "layer3", "layer4"]
        ),
        "use_pca": config.get("memory_bank", {}).get("use_pca", False),
        "pca_components": config.get("memory_bank", {}).get("pca_components", 256),
        "scoring_method": config.get("anomaly_detection", {}).get("method", "mahalanobis"),
        "memory_bank_size": memory_bank.count,
        "feature_dim": memory_bank.features.shape[1],
    }

    meta_path = os.path.join(checkpoint_dir, f"{category}_meta.pt")
    torch.save(meta, meta_path)
    logger.info(f"Metadata saved: {meta_path}")

    # 4. Quick evaluation
    try:
        metrics = quick_evaluate(config, model, memory_bank, category, device)
        logger.info(
            f"\n  >>> Quick Eval: AUROC={metrics['auroc']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )
    except Exception as e:
        logger.warning(f"Quick evaluation failed: {e}")
        metrics = None

    pipeline_time = time.time() - pipeline_start
    logger.info(f"Total pipeline time for {category}: {format_time(pipeline_time)}")

    return model, memory_bank, metrics


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR Anomaly Detection Training Pipeline"
    )

    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="MVTec category to train on (default: all)",
    )
    parser.add_argument(
        "--train-simclr", action="store_true",
        help="Run SimCLR contrastive training instead of feature extraction only",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
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

    # Process each category
    all_results = {}
    for category in categories:
        _, _, metrics = train_category(config, category, device, args)
        if metrics:
            all_results[category] = metrics

    # Summary
    if all_results:
        logger.info(f"\n{'=' * 70}")
        logger.info(
            f"{'Category':<15} {'AUROC':>8} {'F1':>8} "
            f"{'Prec':>8} {'Recall':>8}"
        )
        logger.info(f"{'-' * 70}")

        aurocs = []
        for cat, m in all_results.items():
            logger.info(
                f"{cat:<15} {m['auroc']:>8.4f} {m['f1_score']:>8.4f} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f}"
            )
            aurocs.append(m["auroc"])

        logger.info(f"{'-' * 70}")
        logger.info(f"{'MEAN':<15} {sum(aurocs) / len(aurocs):>8.4f}")
        logger.info(f"{'=' * 70}")

    logger.info("\nAll processing complete!")


if __name__ == "__main__":
    main()