"""
Single-image or batch inference script with PatchCore support.

Generates anomaly heatmaps and calibrated confidence scores.

Usage:
    # Single image (auto-detects PatchCore vs Global):
    python scripts/inference.py --image path/to/image.png --category bottle

    # Batch (directory):
    python scripts/inference.py --image_dir path/to/images/ --category bottle
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from src.utils.utils import load_config, get_device, setup_logging
from src.inference.predictor import create_predictor, PatchAnomalyPredictor
from src.visualization.heatmap import AnomalyHeatmapGenerator

logger = logging.getLogger(__name__)


def run_single_inference(
    image_path: str,
    predictor,
    config: dict,
    output_dir: str,
):
    """Run inference on a single image with heatmap visualization."""

    original = Image.open(image_path).convert("RGB")
    result = predictor.predict(original)

    # Save heatmap visualization if patch predictor
    if isinstance(predictor, PatchAnomalyPredictor) and "heatmap" in result:
        loc_cfg = config.get("localization", {})
        heatmap_gen = AnomalyHeatmapGenerator(
            sigma=loc_cfg.get("gaussian_sigma", 4.0),
            colormap=loc_cfg.get("colormap", "jet"),
            alpha=loc_cfg.get("overlay_alpha", 0.4),
        )

        img_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(
            output_dir, f"{predictor.category}_{img_name}_result.png"
        )

        heatmap_gen.visualize(
            result["original"],
            result["heatmap"],
            anomaly_score=result["score"],
            label=result["label"],
            confidence=result["confidence"],
            save_path=save_path,
        )
    else:
        # Legacy Grad-CAM
        try:
            from src.inference.gradcam import GradCAM
            from src.training.augmentations import get_eval_transform

            gradcam = GradCAM(predictor.model, target_layer_name="layer3")
            eval_transform = get_eval_transform(config)
            original_np = np.array(original.resize((224, 224)))
            input_tensor = eval_transform(original).unsqueeze(0).to(predictor.device)

            heatmap = gradcam.generate(input_tensor, predictor.device)

            img_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(
                output_dir, f"{predictor.category}_{img_name}_result.png"
            )

            gradcam.visualize(
                original_np, heatmap,
                anomaly_score=result["score"],
                label=result["label"],
                save_path=save_path,
            )
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            save_path = "N/A"

    # Print result
    logger.info(f"\n{'=' * 50}")
    logger.info("Inference Result")
    logger.info(f"{'=' * 50}")
    logger.info(f"Image:          {image_path}")
    logger.info(f"Category:       {predictor.category}")
    logger.info(f"Anomaly Score:  {result['score']:.6f}")
    logger.info(f"Threshold:      {result['threshold']:.4f}")
    logger.info(f"Prediction:     {result['label']}")
    logger.info(f"Confidence:     {result.get('confidence_pct', 0):.1f}%")
    if "confidence_label" in result:
        logger.info(f"Confidence:     {result['confidence_label']}")
    logger.info(f"{'=' * 50}")

    return result


def run_batch_inference(
    image_dir: str,
    predictor,
    output_dir: str,
):
    """Run inference on all images in a directory."""

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return []

    logger.info(f"Processing {len(image_files)} images from {image_dir}")

    results = []
    for f in image_files:
        path = os.path.join(image_dir, f)
        try:
            img = Image.open(path).convert("RGB")
            result = predictor.predict(img)
            result["filename"] = f
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")

    # Print summary table
    logger.info(f"\n{'=' * 78}")
    logger.info(
        f"{'File':<30} {'Score':>10} {'Label':>10} "
        f"{'Confidence':>12}"
    )
    logger.info(f"{'-' * 78}")

    anomaly_count = 0
    for result in results:
        fname = result.get("filename", "?")
        logger.info(
            f"{fname:<30} {result['score']:>10.4f} "
            f"{result['label']:>10} "
            f"{result.get('confidence_pct', 0):>10.1f}%"
        )
        if result["is_anomaly"]:
            anomaly_count += 1

    logger.info(f"{'-' * 78}")
    logger.info(
        f"Total: {len(results)} | "
        f"Normal: {len(results) - anomaly_count} | "
        f"Anomaly: {anomaly_count}"
    )
    logger.info(f"{'=' * 78}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Inference (PatchCore + Global)"
    )

    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to single input image",
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Path to directory of images for batch inference",
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

    if args.image is None and args.image_dir is None:
        parser.error("Must specify either --image or --image_dir")

    # Setup
    config = load_config(args.config)
    setup_logging(config.get("output", {}).get("logs_dir", "outputs/logs"))
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()

    # Create predictor (auto-detects PatchCore vs Global)
    try:
        predictor = create_predictor(args.config, args.category, device)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Image not found: {args.image}")
            return

        result = run_single_inference(
            args.image, predictor, config, args.output_dir
        )

        if result:
            icon = "✅" if result["label"] == "Normal" else "⚠️"
            print(
                f"\n{icon} Prediction: {result['label']} "
                f"(score: {result['score']:.4f}, "
                f"confidence: {result.get('confidence_pct', 0):.1f}%)"
            )

    elif args.image_dir:
        if not os.path.isdir(args.image_dir):
            logger.error(f"Directory not found: {args.image_dir}")
            return

        run_batch_inference(args.image_dir, predictor, args.output_dir)


if __name__ == "__main__":
    main()