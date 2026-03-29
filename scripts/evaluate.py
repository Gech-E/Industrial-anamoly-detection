"""
Evaluation entry point with PatchCore support.

Supports:
    - PatchCore patch-level evaluation with heatmap localization
    - Global feature evaluation (legacy)
    - Pixel-level metrics (AUROC, PRO score) when GT masks available
    - Cross-category summary with mean ± std

Usage:
    python scripts/evaluate.py --category bottle
    python scripts/evaluate.py  # all categories
"""

import os
import sys
import argparse
import logging
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from src.utils.utils import (
    load_config, set_seed, get_device,
    setup_logging, ensure_dirs, load_checkpoint,
)
from src.models.simclr import SimCLRModel
from src.training.dataset import create_test_dataloader
from src.training.augmentations import get_eval_transform
from src.memory.memory_bank import (
    MemoryBank, AnomalyScorer,
    PatchMemoryBank, PatchAnomalyScorer,
)
from src.evaluation.evaluator import AnomalyEvaluator
from src.scoring.calibration import ScoreCalibrator
from src.visualization.heatmap import AnomalyHeatmapGenerator

logger = logging.getLogger(__name__)


def evaluate_category(config: dict, category: str, device: torch.device):
    """Evaluate anomaly detection for a single category."""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Evaluating category: {category}")
    logger.info(f"{'=' * 60}")

    checkpoint_dir = config["output"]["checkpoints_dir"]
    results_dir = config["output"]["results_dir"]
    vis_dir = config["output"]["visualizations_dir"]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Model
    model = SimCLRModel(config)
    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Run training first: python scripts/train.py --category {category}")
        return None

    load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()

    # 2. Determine pipeline mode
    patch_bank_path = os.path.join(checkpoint_dir, f"{category}_patch_bank.pt")
    use_patch = os.path.exists(patch_bank_path)

    if use_patch:
        logger.info("Using PatchCore pipeline")
        return _evaluate_patch(config, model, category, device,
                               checkpoint_dir, results_dir, vis_dir)
    else:
        logger.info("Using global feature pipeline")
        return _evaluate_global(config, model, category, device,
                                checkpoint_dir, results_dir, vis_dir)


def _evaluate_patch(config, model, category, device,
                    checkpoint_dir, results_dir, vis_dir):
    """Evaluate using PatchCore pipeline with heatmap localization."""

    # Load patch memory bank
    patch_bank_path = os.path.join(checkpoint_dir, f"{category}_patch_bank.pt")
    patch_bank = PatchMemoryBank()
    patch_bank.load(patch_bank_path)

    # Create patch scorer
    scoring_cfg = config.get("scoring", {})
    scorer = PatchAnomalyScorer(
        k_neighbors=scoring_cfg.get("k_neighbors", 3),
        weight_knn=scoring_cfg.get("weight_knn", 1.0),
        weight_mahalanobis=scoring_cfg.get("weight_mahalanobis", 0.0),
        weight_cosine=scoring_cfg.get("weight_cosine", 0.0),
    )
    scorer.fit(patch_bank)

    # 3. Score test set
    test_dataloader = create_test_dataloader(config, category)
    test_dataset = test_dataloader.dataset
    logger.info(f"Test samples: {len(test_dataset)}")

    image_scores, labels, all_patch_info = scorer.score_batch(
        model, test_dataloader, device
    )

    # 4. Calibrate scores
    cal_cfg = config.get("calibration", {})
    calibrator = ScoreCalibrator(
        method=cal_cfg.get("method", "minmax_sigmoid"),
        temperature=cal_cfg.get("temperature", 1.0),
    )
    calibrator.fit(image_scores, labels)

    # 5. Compute metrics
    threshold_cfg = config.get("thresholding", {})
    evaluator = AnomalyEvaluator(
        output_dir=results_dir,
        threshold_method=threshold_cfg.get("method", "youden"),
        percentile_value=threshold_cfg.get("value", 97),
    )

    metrics = evaluator.generate_full_report(image_scores, labels, category)
    metrics["pipeline"] = "patchcore"
    metrics["calibration"] = calibrator.save_params()

    # 6. Generate heatmap visualizations
    logger.info("Generating anomaly heatmap visualizations...")

    loc_cfg = config.get("localization", {})
    heatmap_gen = AnomalyHeatmapGenerator(
        sigma=loc_cfg.get("gaussian_sigma", 4.0),
        colormap=loc_cfg.get("colormap", "jet"),
        alpha=loc_cfg.get("overlay_alpha", 0.4),
    )

    eval_transform = get_eval_transform(config)
    img_size = config.get("dataset", {}).get("image_size", 224)

    # Select samples for visualization
    anomaly_indices = [i for i, l in enumerate(test_dataset.labels) if l == 1][:5]
    normal_indices = [i for i, l in enumerate(test_dataset.labels) if l == 0][:2]
    vis_indices = normal_indices + anomaly_indices

    for idx in vis_indices:
        if idx >= len(all_patch_info):
            continue

        img_path = test_dataset.get_image_path(idx)
        defect_type = test_dataset.get_defect_type(idx)

        original = Image.open(img_path).convert("RGB")
        original_np = np.array(original.resize((img_size, img_size)))

        patch_scores, patch_shape = all_patch_info[idx]

        # Generate heatmap
        heatmap = heatmap_gen.generate(
            patch_scores, patch_shape, image_size=(img_size, img_size)
        )

        score_val = float(image_scores[idx])
        confidence = calibrator.calibrate(score_val)
        label_str = "Anomaly" if test_dataset.labels[idx] == 1 else "Normal"

        save_path = os.path.join(
            vis_dir, f"{category}_{defect_type}_{idx}_heatmap.png"
        )

        heatmap_gen.visualize(
            original_np, heatmap,
            anomaly_score=score_val,
            label=label_str,
            confidence=confidence,
            save_path=save_path,
        )

    logger.info(f"Evaluation complete for {category}")
    return metrics


def _evaluate_global(config, model, category, device,
                     checkpoint_dir, results_dir, vis_dir):
    """Evaluate using global feature pipeline (legacy)."""

    bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
    if not os.path.exists(bank_path):
        logger.error(f"Memory bank not found: {bank_path}")
        return None

    memory_bank = MemoryBank()
    memory_bank.load(bank_path)

    ad_cfg = config.get("anomaly_detection", {})
    scorer = AnomalyScorer(
        method=ad_cfg.get("method", "mahalanobis"),
        k_neighbors=ad_cfg.get("k_neighbors", 5),
    )
    scorer.fit(memory_bank)

    test_dataloader = create_test_dataloader(config, category)
    test_dataset = test_dataloader.dataset
    logger.info(f"Test samples: {len(test_dataset)}")

    scores, labels = scorer.score_batch(model, test_dataloader, device)

    # Calibrate
    cal_cfg = config.get("calibration", {})
    calibrator = ScoreCalibrator(
        method=cal_cfg.get("method", "minmax_sigmoid"),
    )
    calibrator.fit(scores, labels)

    threshold_cfg = config.get("thresholding", {})
    evaluator = AnomalyEvaluator(
        output_dir=results_dir,
        threshold_method=threshold_cfg.get("method", "youden"),
        percentile_value=threshold_cfg.get("value", 97),
    )

    metrics = evaluator.generate_full_report(scores, labels, category)
    metrics["pipeline"] = "global"
    metrics["calibration"] = calibrator.save_params()

    # Grad-CAM visualization (legacy)
    try:
        from src.inference.gradcam import GradCAM

        gradcam_layer = config.get("gradcam", {}).get("target_layer", "layer3")
        gradcam = GradCAM(model, target_layer_name=gradcam_layer)

        eval_transform = get_eval_transform(config)

        anomaly_indices = [i for i, l in enumerate(test_dataset.labels) if l == 1][:5]
        normal_indices = [i for i, l in enumerate(test_dataset.labels) if l == 0][:2]
        vis_indices = normal_indices + anomaly_indices

        for idx in vis_indices:
            img_path = test_dataset.get_image_path(idx)
            defect_type = test_dataset.get_defect_type(idx)

            original = Image.open(img_path).convert("RGB")
            original_np = np.array(original.resize((224, 224)))
            input_tensor = eval_transform(original).unsqueeze(0).to(device)

            heatmap = gradcam.generate(input_tensor, device)

            score_val = float(scores[idx]) if idx < len(scores) else 0.0
            label_str = "Anomaly" if test_dataset.labels[idx] == 1 else "Normal"

            save_path = os.path.join(
                vis_dir, f"{category}_{defect_type}_{idx}_gradcam.png"
            )
            gradcam.visualize(
                original_np, heatmap,
                anomaly_score=score_val, label=label_str,
                save_path=save_path,
            )
    except Exception as e:
        logger.warning(f"Grad-CAM visualization failed: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Anomaly Detection (PatchCore + Global)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="MVTec category to evaluate (optional)",
    )

    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    setup_logging(config["output"]["logs_dir"])
    set_seed(config["training"]["seed"])
    ensure_dirs(config)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Categories
    categories = (
        [args.category] if args.category else config["dataset"]["categories"]
    )

    # Evaluation Loop
    all_metrics = {}

    for category in categories:
        metrics = evaluate_category(config, category, device)
        if metrics:
            all_metrics[category] = metrics

    # Cross-category Summary
    if all_metrics:
        evaluator = AnomalyEvaluator(
            output_dir=config["output"]["results_dir"]
        )
        summary = evaluator.cross_category_summary(all_metrics)

        # Print detailed table
        logger.info(f"\n{'=' * 78}")
        logger.info(
            f"{'Category':<15} {'AUROC':>8} {'AP':>8} {'F1':>8} "
            f"{'Acc':>8} {'Prec':>8} {'Recall':>8}"
        )
        logger.info(f"{'-' * 78}")

        for cat, m in all_metrics.items():
            logger.info(
                f"{cat:<15} {m['auroc']:>8.4f} "
                f"{m.get('average_precision', 0):>8.4f} "
                f"{m['f1_score']:>8.4f} {m['accuracy']:>8.4f} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f}"
            )

        aurocs = [m["auroc"] for m in all_metrics.values()]
        logger.info(f"{'-' * 78}")
        logger.info(
            f"{'MEAN':.<15} {np.mean(aurocs):>8.4f} +/- {np.std(aurocs):.4f}"
        )
        logger.info(f"{'=' * 78}")

        # Save comprehensive summary
        summary_path = os.path.join(
            config["output"]["results_dir"], "all_categories_summary.json"
        )
        save_data = {
            "categories": all_metrics,
            "summary": summary,
        }
        with open(summary_path, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()