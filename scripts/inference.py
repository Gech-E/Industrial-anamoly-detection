"""
Single-image or batch inference script.
Usage: python scripts/inference.py --image path/to/image.png --category bottle
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from src.utils.utils import load_config, get_device, setup_logging, load_checkpoint
from src.models.simclr import SimCLRModel
from src.memory.memory_bank import MemoryBank, AnomalyScorer
from src.inference.gradcam import GradCAM
from src.training.augmentations import get_eval_transform

logger = logging.getLogger(__name__)


def run_inference(
    image_path: str,
    category: str,
    config: dict,
    device: torch.device,
    model: SimCLRModel = None,
    scorer: AnomalyScorer = None,
    gradcam_module: GradCAM = None,
    output_dir: str = "outputs/visualizations",
):
    """Run anomaly detection inference on a single image."""

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = config["output"]["checkpoints_dir"]

    # 1. Load Model
    if model is None:
        model = SimCLRModel(config)

        checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None

        load_checkpoint(model, checkpoint_path, device=device)
        model = model.to(device)

    model.eval()

    # 2. Load Memory Bank + Scorer (MATCH TRAIN CONFIG)
    if scorer is None:
        bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")

        if not os.path.exists(bank_path):
            logger.error(f"Memory bank not found: {bank_path}")
            return None

        memory_bank = MemoryBank(
            use_patches=config["model"].get("use_patches", False),
            patch_layer=config["model"].get("patch_layer", "layer2"),
        )
        memory_bank.load(bank_path)

        ad_cfg = config["anomaly_detection"]

        scorer = AnomalyScorer(
            method=ad_cfg.get("method", "knn"),
            k_neighbors=ad_cfg.get("k_neighbors", 5),
        )
        scorer.fit(memory_bank)

    # 3. Load + Preprocess Image
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original.resize((224, 224)))

    eval_transform = get_eval_transform(config)
    input_tensor = eval_transform(original).unsqueeze(0).to(device)

    # 4. Feature Extraction (PATCH-AWARE)
    with torch.no_grad():
        if config["model"].get("use_patches", False):
            features = model.extract_patch_features(input_tensor)
        else:
            features = model.encode(input_tensor)

    features_cpu = features.cpu()

    # 5. Anomaly Scoring
    anomaly_score = float(scorer.score(features_cpu)[0])

    threshold = config["anomaly_detection"].get("score_threshold", 0.5)
    predicted_label = "Anomaly" if anomaly_score >= threshold else "Normal"

    # 6. Grad-CAM Visualization (CONFIG-DRIVEN)
    if gradcam_module is None:
        target_layer = config.get("gradcam", {}).get("target_layer", "layer4")
        gradcam_module = GradCAM(model, target_layer_name=target_layer)

    heatmap = gradcam_module.generate(input_tensor, device)

    # 7. Save Output
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{category}_{img_name}_result.png")

    gradcam_module.visualize(
        original_np,
        heatmap,
        anomaly_score=anomaly_score,
        label=predicted_label,
        save_path=save_path,
    )

    # 8. Logging
    logger.info(f"\n{'='*50}")
    logger.info("Inference Result")
    logger.info(f"{'='*50}")
    logger.info(f"Image:          {image_path}")
    logger.info(f"Category:       {category}")
    logger.info(f"Anomaly Score:  {anomaly_score:.6f}")
    logger.info(f"Threshold:      {threshold:.4f}")
    logger.info(f"Prediction:     {predicted_label}")
    logger.info(f"Visualization:  {save_path}")
    logger.info(f"{'='*50}")

    return {
        "image_path": image_path,
        "category": category,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "predicted_label": predicted_label,
        "heatmap_path": save_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Inference")

    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--category", type=str, required=True,
        help="MVTec category (e.g., bottle, cable, capsule)",
    )

    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--output_dir", type=str, default="outputs/visualizations",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    setup_logging(config["output"]["logs_dir"])
    device = get_device()

    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return

    # Run Inference
    result = run_inference(
        image_path=args.image,
        category=args.category,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )

    if result:
        print(
            f"\n✓ Prediction: {result['predicted_label']} "
            f"(score: {result['anomaly_score']:.4f})"
        )


if __name__ == "__main__":
    main()