"""
Single-image or batch inference script using the AnomalyPredictor API.

Usage:
    # Single image:
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
from src.inference.predictor import AnomalyPredictor
from src.inference.gradcam import GradCAM
from src.training.augmentations import get_eval_transform

logger = logging.getLogger(__name__)


def run_single_inference(
    image_path: str,
    predictor: AnomalyPredictor,
    gradcam: GradCAM,
    config: dict,
    output_dir: str,
):
    """Run inference on a single image with Grad-CAM visualization."""

    original = Image.open(image_path).convert("RGB")
    result = predictor.predict(original)

    # Grad-CAM
    eval_transform = get_eval_transform(config)
    original_np = np.array(original.resize((224, 224)))
    input_tensor = eval_transform(original).unsqueeze(0).to(predictor.device)

    heatmap = gradcam.generate(input_tensor, predictor.device)

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(
        output_dir, f"{predictor.category}_{img_name}_result.png"
    )

    gradcam.visualize(
        original_np,
        heatmap,
        anomaly_score=result["score"],
        label=result["label"],
        save_path=save_path,
    )

    logger.info(f"\n{'=' * 50}")
    logger.info("Inference Result")
    logger.info(f"{'=' * 50}")
    logger.info(f"Image:          {image_path}")
    logger.info(f"Category:       {predictor.category}")
    logger.info(f"Anomaly Score:  {result['score']:.6f}")
    logger.info(f"Threshold:      {result['threshold']:.4f}")
    logger.info(f"Prediction:     {result['label']}")
    logger.info(f"Confidence:     {result['confidence']:.2%}")
    logger.info(f"Visualization:  {save_path}")
    logger.info(f"{'=' * 50}")

    return result


def run_batch_inference(
    image_dir: str,
    predictor: AnomalyPredictor,
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

    images = []
    paths = []
    for f in image_files:
        path = os.path.join(image_dir, f)
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            paths.append(path)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    results = predictor.predict_batch(images)

    # Print summary table
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{'File':<30} {'Score':>10} {'Label':>10} {'Confidence':>12}")
    logger.info(f"{'-' * 70}")

    anomaly_count = 0
    for path, result in zip(paths, results):
        fname = os.path.basename(path)
        logger.info(
            f"{fname:<30} {result['score']:>10.4f} "
            f"{result['label']:>10} {result['confidence']:>11.2%}"
        )
        if result["is_anomaly"]:
            anomaly_count += 1

    logger.info(f"{'-' * 70}")
    logger.info(
        f"Total: {len(results)} | "
        f"Normal: {len(results) - anomaly_count} | "
        f"Anomaly: {anomaly_count}"
    )
    logger.info(f"{'=' * 70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Inference")

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

    # Create predictor
    try:
        predictor = AnomalyPredictor.from_config(
            args.config, args.category, device
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Image not found: {args.image}")
            return

        gradcam_layer = config.get("gradcam", {}).get("target_layer", "layer3")
        gradcam = GradCAM(predictor.model, target_layer_name=gradcam_layer)

        result = run_single_inference(
            args.image, predictor, gradcam, config, args.output_dir
        )

        if result:
            print(
                f"\n{'✅' if result['label'] == 'Normal' else '⚠️'} "
                f"Prediction: {result['label']} "
                f"(score: {result['score']:.4f}, "
                f"confidence: {result['confidence']:.2%})"
            )

    elif args.image_dir:
        if not os.path.isdir(args.image_dir):
            logger.error(f"Directory not found: {args.image_dir}")
            return

        run_batch_inference(args.image_dir, predictor, args.output_dir)


if __name__ == "__main__":
    main()