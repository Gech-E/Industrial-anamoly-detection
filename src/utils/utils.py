"""
Utility functions for the anomaly detection project.
Provides logging, seed setting, device selection, checkpoint management, and helpers.
"""

import os
import random
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


def setup_logging(log_dir: str = "outputs/logs", level: int = logging.INFO):
    """Setup logging configuration with file and console handlers."""
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers = []

    # File handler (UTF-8 encoding for Windows compatibility)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    **kwargs,
):
    """Save model checkpoint with full training state."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)
    logging.info(
        f"Checkpoint saved: {save_path} (epoch {epoch}, loss {loss:.4f})"
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load model checkpoint.

    Supports loading with or without optimizer state (for training resume).
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", "?")
    loss = checkpoint.get("loss", 0.0)

    logging.info(
        f"Checkpoint loaded: {checkpoint_path} "
        f"(epoch {epoch}, loss {loss:.4f})"
    )
    return checkpoint


def ensure_dirs(config: dict):
    """Create all necessary output directories from config."""
    output_cfg = config.get("output", {})
    dirs = [
        output_cfg.get("root_dir", "outputs"),
        output_cfg.get("checkpoints_dir", "outputs/checkpoints"),
        output_cfg.get("logs_dir", "outputs/logs"),
        output_cfg.get("results_dir", "outputs/results"),
        output_cfg.get("visualizations_dir", "outputs/visualizations"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 0:
        return "0.0s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
