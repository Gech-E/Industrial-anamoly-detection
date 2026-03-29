"""
Evaluation entry point.
Usage: python scripts/evaluate.py --config configs/config.yaml --category bottle
"""

import os
import sys
import argparse
import logging
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.utils import (
    load_config, set_seed, get_device,
    setup_logging, ensure_dirs, load_checkpoint
)
from src.models.simclr import SimCLRModel
from src.training.dataset import create_test_dataloader
from src.memory.memory_bank import MemoryBank, AnomalyScorer
from src.evaluation.evaluator import AnomalyEvaluator
from src.inference.gradcam import GradCAM
from src.training.augmentations import get_eval_transform

logger = logging.getLogger(__name__)


def evaluate_category(config: dict, category: str, device: torch.device):
    """Evaluate anomaly detection for a single category."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating category: {category}")
    logger.info(f"{'='*60}")

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

    # 2. Load Memory Bank (MATCH TRAIN CONFIG)
    bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
    if not os.path.exists(bank_path):
        logger.error(f"Memory bank not found: {bank_path}")
        return None

    memory_bank = MemoryBank(
        use_patches=config["model"].get("use_patches", False),
        patch_layer=config["model"].get("patch_layer", "layer2"),
    )
    memory_bank.load(bank_path)

    # 3. Anomaly Scorer
    ad_cfg = config["anomaly_detection"]

    scorer = AnomalyScorer(
        method=ad_cfg.get("method", "knn"),
        k_neighbors=ad_cfg.get("k_neighbors", 5),
    )
    scorer.fit(memory_bank)

    # 4. Test Data
    test_dataloader = create_test_dataloader(config, category)
    test_dataset = test_dataloader.dataset

    logger.info(f"Test samples: {len(test_dataset)}")

    # 5. Scoring
    scores, labels = scorer.score_batch(model, test_dataloader, device)

    # 6. Evaluation Metrics
    evaluator = AnomalyEvaluator(output_dir=results_dir)
    metrics = evaluator.generate_full_report(scores, labels, category)

    # 7. Grad-CAM Visualization (CONFIG-DRIVEN)
    logger.info("Generating Grad-CAM visualizations...")

    gradcam_layer = config.get("gradcam", {}).get("target_layer", "layer4")
    gradcam = GradCAM(model, target_layer_name=gradcam_layer)

    eval_transform = get_eval_transform(config)

    # Select samples
    anomaly_indices = [i for i, l in enumerate(test_dataset.labels) if l == 1][:5]
    normal_indices = [i for i, l in enumerate(test_dataset.labels) if l == 0][:2]

    vis_indices = normal_indices + anomaly_indices

    from PIL import Image
    import numpy as np

    for idx in vis_indices:
        img_path = test_dataset.get_image_path(idx)
        defect_type = test_dataset.get_defect_type(idx)

        original = Image.open(img_path).convert("RGB")
        original_np = np.array(original.resize((224, 224)))

        input_tensor = eval_transform(original).unsqueeze(0).to(device)

        # Generate heatmap
        heatmap = gradcam.generate(input_tensor, device)

        # Safe score fetch
        score_val = float(scores[idx]) if idx < len(scores) else 0.0
        label_str = "Anomaly" if test_dataset.labels[idx] == 1 else "Normal"

        save_path = os.path.join(
            vis_dir,
            f"{category}_{defect_type}_{idx}_gradcam.png"
        )

        gradcam.visualize(
            original_np,
            heatmap,
            anomaly_score=score_val,
            label=label_str,
            save_path=save_path,
        )

    logger.info(f"Evaluation complete for {category}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection")

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
    categories = [args.category] if args.category else config["dataset"]["categories"]

    # Evaluation Loop
    all_metrics = {}

    for category in categories:
        metrics = evaluate_category(config, category, device)
        if metrics:
            all_metrics[category] = metrics

    # Summary Report
    if all_metrics:
        results_dir = config["output"]["results_dir"]

        logger.info(f"\n{'='*70}")
        logger.info(f"{'Category':<15} {'AUROC':>8} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8}")
        logger.info(f"{'-'*70}")

        aurocs = []

        for cat, m in all_metrics.items():
            logger.info(
                f"{cat:<15} {m['auroc']:>8.4f} {m['f1_score']:>8.4f} "
                f"{m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}"
            )
            aurocs.append(m["auroc"])

        logger.info(f"{'-'*70}")
        logger.info(f"{'MEAN':<15} {sum(aurocs)/len(aurocs):>8.4f}")
        logger.info(f"{'='*70}")

        summary_path = os.path.join(results_dir, "all_categories_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()