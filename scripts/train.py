"""
Training entry point for anomaly detection pipeline.

Supports two pipelines:
    1. PatchCore (default, recommended): Extract patch features → build patch
       memory bank → coreset subsampling → calibrate scores → evaluate.
       Achieves ≥0.90 AUROC on MVTec AD without any training.

    2. SimCLR + Global: Contrastive pretraining → global feature memory bank →
       Mahalanobis/kNN scoring. Original pipeline, kept for ablation.

Usage:
    # PatchCore pipeline (default, recommended):
    python scripts/train.py --category bottle

    # All categories:
    python scripts/train.py

    # SimCLR contrastive training (optional):
    python scripts/train.py --category bottle --train-simclr

    # Disable patch mode (use global features):
    python scripts/train.py --category bottle --no-patch
"""

import os
import sys
import argparse
import logging
import time

import torch
import numpy as np

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
from src.memory.memory_bank import (
    MemoryBank, AnomalyScorer,
    PatchMemoryBank, PatchAnomalyScorer,
)
from src.evaluation.evaluator import AnomalyEvaluator
from src.scoring.calibration import ScoreCalibrator

logger = logging.getLogger(__name__)


# PatchCore Pipeline (PRIMARY — high AUROC)

def build_patch_memory_bank(
    config: dict,
    model: SimCLRModel,
    category: str,
    device: torch.device,
) -> PatchMemoryBank:
    """Build patch-level memory bank from normal training images."""
    logger.info("Building PatchCore patch memory bank...")

    feature_dataloader = create_feature_dataloader(config, category)

    patch_cfg = config.get("patch_detection", {})
    memory_bank = PatchMemoryBank(
        use_pca=patch_cfg.get("use_pca", True),
        pca_components=patch_cfg.get("pca_components", 256),
        coreset_ratio=patch_cfg.get("coreset_ratio", 0.1),
        coreset_max=patch_cfg.get("coreset_max", 5000),
    )

    memory_bank.build(
        model=model,
        dataloader=feature_dataloader,
        device=device,
    )

    return memory_bank


def evaluate_patch_pipeline(
    config: dict,
    model: SimCLRModel,
    patch_bank: PatchMemoryBank,
    category: str,
    device: torch.device,
) -> dict:
    """Run full PatchCore evaluation with calibration."""
    logger.info("Running PatchCore evaluation...")

    # Create patch scorer
    scoring_cfg = config.get("scoring", {})
    scorer = PatchAnomalyScorer(
        k_neighbors=scoring_cfg.get("k_neighbors", 3),
        weight_knn=scoring_cfg.get("weight_knn", 1.0),
        weight_mahalanobis=scoring_cfg.get("weight_mahalanobis", 0.0),
        weight_cosine=scoring_cfg.get("weight_cosine", 0.0),
    )
    scorer.fit(patch_bank)

    # Score test set
    test_dataloader = create_test_dataloader(config, category)
    image_scores, labels, all_patch_info = scorer.score_batch(
        model, test_dataloader, device
    )

    # Calibrate scores
    cal_cfg = config.get("calibration", {})
    calibrator = ScoreCalibrator(
        method=cal_cfg.get("method", "minmax_sigmoid"),
        temperature=cal_cfg.get("temperature", 1.0),
    )
    calibrator.fit(image_scores, labels)

    # Evaluate
    results_dir = config.get("output", {}).get("results_dir", "outputs/results")
    threshold_cfg = config.get("thresholding", {})

    evaluator = AnomalyEvaluator(
        output_dir=results_dir,
        threshold_method=threshold_cfg.get("method", "youden"),
        percentile_value=threshold_cfg.get("value", 97),
    )

    metrics = evaluator.generate_full_report(image_scores, labels, category)

    # Add calibration params to metrics
    metrics["calibration"] = calibrator.save_params()

    # Re-save metrics with calibration params (generate_full_report saves before these are added)
    evaluator.save_metrics(metrics, category)

    return metrics


# ═══════════════════════════════════════════════════════════════
# Global Feature Pipeline (legacy, for ablation)

def build_global_memory_bank(
    config: dict,
    model: SimCLRModel,
    category: str,
    device: torch.device,
) -> MemoryBank:
    """Build global feature memory bank from normal training images."""
    logger.info("Building global feature memory bank...")

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


def evaluate_global_pipeline(
    config: dict,
    model: SimCLRModel,
    memory_bank: MemoryBank,
    category: str,
    device: torch.device,
) -> dict:
    """Run evaluation with global features (legacy pipeline)."""
    logger.info("Running global feature evaluation...")

    ad_cfg = config.get("anomaly_detection", {})
    scorer = AnomalyScorer(
        method=ad_cfg.get("method", "mahalanobis"),
        k_neighbors=ad_cfg.get("k_neighbors", 5),
    )
    scorer.fit(memory_bank)

    test_dataloader = create_test_dataloader(config, category)
    scores, labels = scorer.score_batch(model, test_dataloader, device)

    results_dir = config.get("output", {}).get("results_dir", "outputs/results")
    threshold_cfg = config.get("thresholding", {})

    evaluator = AnomalyEvaluator(
        output_dir=results_dir,
        threshold_method=threshold_cfg.get("method", "youden"),
        percentile_value=threshold_cfg.get("value", 97),
    )

    metrics = evaluator.generate_full_report(scores, labels, category)
    return metrics


# Main Training Pipeline

def train_category(config: dict, category: str, device: torch.device, args):
    """Train/extract features for a single MVTec category."""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing category: {category}")
    logger.info(f"{'=' * 60}")

    pipeline_start = time.time()

    # Determine modes
    feature_only = config.get("model", {}).get("feature_extraction_only", True)
    use_patch = config.get("patch_detection", {}).get("enabled", True)

    if args.train_simclr:
        feature_only = False
    if args.no_patch:
        use_patch = False

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

    # 2. Train or use pretrained features
    if feature_only:
        logger.info("Mode: Feature extraction only (ImageNet pretrained)")
        model.eval()

        # Save pretrained model checkpoint
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
        logger.info("Mode: SimCLR contrastive training")

        train_dataloader = create_train_dataloader(config, category)
        logger.info(f"Training samples: {len(train_dataloader.dataset)}")

        trainer = SimCLRTrainer(model, config, device)
        history = trainer.train(
            train_dataloader, category, resume=args.resume
        )

    # 3. Build memory bank (patch or global)
    if use_patch:
        logger.info("Pipeline: PatchCore (patch-level features)")
        patch_bank = build_patch_memory_bank(config, model, category, device)

        # Save patch memory bank
        bank_path = os.path.join(checkpoint_dir, f"{category}_patch_bank.pt")
        patch_bank.save(bank_path)

        # Also build global memory bank for backward compatibility
        global_bank = build_global_memory_bank(config, model, category, device)
        global_bank_path = os.path.join(
            checkpoint_dir, f"{category}_memory_bank.pt"
        )
        global_bank.save(global_bank_path)
    else:
        logger.info("Pipeline: Global features (legacy)")
        global_bank = build_global_memory_bank(config, model, category, device)
        bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
        global_bank.save(bank_path)
        patch_bank = None

    # 4. Save metadata
    model_cfg = config.get("model", {})
    meta = {
        "category": category,
        "mode": "feature_extraction_only" if feature_only else "simclr",
        "pipeline": "patchcore" if use_patch else "global",
        "multi_layer": model_cfg.get("multi_layer", True),
        "feature_layers": model_cfg.get("feature_layers", ["layer2", "layer3", "layer4"]),
        "patch_layers": model_cfg.get("patch_layers", ["layer2", "layer3"]),
        "use_patch": use_patch,
        "scoring_method": config.get("scoring", {}).get("weight_knn", 1.0),
    }

    if use_patch and patch_bank is not None:
        meta["patch_bank_size"] = patch_bank.count
        meta["patch_feature_dim"] = patch_bank.features.shape[1]
        meta["patch_shape"] = list(patch_bank.patch_shape) if patch_bank.patch_shape else None

    meta_path = os.path.join(checkpoint_dir, f"{category}_meta.pt")
    torch.save(meta, meta_path)

    # 5. Evaluate
    try:
        if use_patch and patch_bank is not None:
            metrics = evaluate_patch_pipeline(
                config, model, patch_bank, category, device
            )
        else:
            metrics = evaluate_global_pipeline(
                config, model, global_bank, category, device
            )

        logger.info(
            f"\n  >>> Results: AUROC={metrics['auroc']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        metrics = None

    pipeline_time = time.time() - pipeline_start
    logger.info(f"Total pipeline time for {category}: {format_time(pipeline_time)}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Training Pipeline (PatchCore + SimCLR)"
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
        "--no-patch", action="store_true",
        help="Disable PatchCore, use global features instead",
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

    use_patch = config.get("patch_detection", {}).get("enabled", True)
    if args.no_patch:
        use_patch = False
    logger.info(f"Pipeline: {'PatchCore' if use_patch else 'Global features'}")

    # Categories
    if args.category:
        categories = [args.category]
    else:
        categories = config["dataset"]["categories"]

    # Process each category
    all_results = {}
    for category in categories:
        _, metrics = train_category(config, category, device, args)
        if metrics:
            all_results[category] = metrics

    # Summary
    if all_results:
        logger.info(f"\n{'=' * 78}")
        logger.info(
            f"{'Category':<15} {'AUROC':>8} {'AP':>8} {'F1':>8} "
            f"{'Prec':>8} {'Recall':>8}"
        )
        logger.info(f"{'-' * 78}")

        aurocs = []
        for cat, m in all_results.items():
            logger.info(
                f"{cat:<15} {m['auroc']:>8.4f} "
                f"{m.get('average_precision', 0):>8.4f} "
                f"{m['f1_score']:>8.4f} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f}"
            )
            aurocs.append(m["auroc"])

        logger.info(f"{'-' * 78}")
        mean_auroc = sum(aurocs) / len(aurocs)
        std_auroc = float(np.std(aurocs))
        logger.info(
            f"{'MEAN':.<15} {mean_auroc:>8.4f} +/- {std_auroc:.4f}"
        )
        logger.info(f"{'=' * 78}")

        # Save summary
        results_dir = config.get("output", {}).get("results_dir", "outputs/results")
        os.makedirs(results_dir, exist_ok=True)
        import json
        summary_path = os.path.join(results_dir, "all_categories_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Summary saved: {summary_path}")

    logger.info("\nAll processing complete!")


if __name__ == "__main__":
    main()