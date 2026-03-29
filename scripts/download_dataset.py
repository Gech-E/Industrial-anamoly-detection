"""
Dataset download script for MVTec AD from Kaggle.
Usage: python scripts/download_dataset.py
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import load_config, setup_logging


def download_mvtec_dataset(config: dict):
    """Download MVTec AD dataset from Kaggle."""
    dataset_cfg = config["dataset"]
    root_dir = dataset_cfg["root_dir"]
    kaggle_dataset = dataset_cfg.get("kaggle_dataset", "alex000kim/mvtec-ad")
    
    logger = logging.getLogger(__name__)
    
    # Create data directory
    os.makedirs(root_dir, exist_ok=True)
    
    # Check if dataset already exists
    if os.path.exists(root_dir) and len(os.listdir(root_dir)) > 0:
        existing = os.listdir(root_dir)
        logger.info(f"Dataset directory already contains {len(existing)} items: {existing[:5]}...")
        user_input = input("Dataset directory is not empty. Re-download? (y/n): ").strip().lower()
        if user_input != "y":
            logger.info("Skipping download.")
            return
    
    logger.info(f"Downloading MVTec AD dataset from Kaggle: {kaggle_dataset}")
    logger.info(f"Saving to: {root_dir}")
    
    try:
        import opendatasets as od
        od.download(
            f"https://www.kaggle.com/datasets/{kaggle_dataset}",
            data_dir=os.path.dirname(root_dir),
        )
        logger.info("Download complete!")
        
    except ImportError:
        logger.error(
            "opendatasets not installed. Install with: pip install opendatasets\n"
            "Alternatively, download manually from:\n"
            f"  https://www.kaggle.com/datasets/{kaggle_dataset}\n"
            f"Extract to: {root_dir}"
        )
        return
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(
            "You can download manually from Kaggle:\n"
            f"  https://www.kaggle.com/datasets/{kaggle_dataset}\n"
            f"Extract to: {root_dir}"
        )
        return
    
    # Validate download
    validate_dataset(root_dir, dataset_cfg["categories"])


def validate_dataset(root_dir: str, expected_categories: list):
    """Validate that the dataset has the expected structure."""
    logger = logging.getLogger(__name__)
    
    found_categories = []
    missing_categories = []
    
    for category in expected_categories:
        cat_dir = os.path.join(root_dir, category)
        if os.path.isdir(cat_dir):
            train_dir = os.path.join(cat_dir, "train", "good")
            test_dir = os.path.join(cat_dir, "test")
            
            if os.path.isdir(train_dir) and os.path.isdir(test_dir):
                train_count = len([
                    f for f in os.listdir(train_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                found_categories.append(f"{category} ({train_count} train images)")
            else:
                missing_categories.append(f"{category} (missing train/test subdirs)")
        else:
            missing_categories.append(category)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Dataset Validation Report")
    logger.info(f"{'='*50}")
    logger.info(f"Found {len(found_categories)}/{len(expected_categories)} categories:")
    for cat in found_categories:
        logger.info(f"  ✓ {cat}")
    
    if missing_categories:
        logger.warning(f"Missing {len(missing_categories)} categories:")
        for cat in missing_categories:
            logger.warning(f"  ✗ {cat}")
    else:
        logger.info("All categories found! Dataset is ready.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging()
    download_mvtec_dataset(config)
