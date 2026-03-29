"""
SimCLR Trainer with Early Stopping and Checkpoint Resume.
Handles the full contrastive learning training loop with logging,
checkpointing, early stopping, and learning rate scheduling.
Optimized for CPU-only environments.
"""

import os
import time
import logging
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.simclr import SimCLRModel
from src.training.losses import NTXentLoss
from src.utils.utils import save_checkpoint, load_checkpoint, format_time

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to terminate training when loss plateaus.

    Monitors training loss and stops when no improvement
    exceeding min_delta is observed for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

        logger.info(
            f"Early stopping initialized: patience={patience}, min_delta={min_delta}"
        )

    def step(self, loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            loss: Current epoch loss.

        Returns:
            True if training should stop.
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement (best_loss={self.best_loss:.4f})"
                )

        return self.should_stop


class SimCLRTrainer:
    """
    Trainer for SimCLR contrastive pretraining.

    Training pipeline:
    1. For each batch of N images, generate 2N augmented views
    2. Pass all views through encoder + projection head
    3. Compute NT-Xent loss between positive pairs
    4. Update model parameters via backpropagation

    Features:
    - Early stopping with configurable patience
    - Checkpoint resume support
    - CPU-optimized (auto-disables AMP on CPU)
    - Per-epoch timing and training summary
    """

    def __init__(
        self,
        model: SimCLRModel,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        training_cfg = config["training"]

        # Optimizer
        self.optimizer = self._create_optimizer(training_cfg)

        # Loss function
        self.criterion = NTXentLoss(
            temperature=training_cfg.get("temperature", 0.07)
        )

        # Learning rate scheduler
        self.scheduler = None
        self.warmup_epochs = training_cfg.get("warmup_epochs", 5)
        self.total_epochs = training_cfg.get("epochs", 50)

        # Mixed precision — auto-disable on CPU
        self.use_amp = (
            training_cfg.get("use_amp", True) and device.type == "cuda"
        )
        self.scaler = GradScaler(device.type) if self.use_amp else None

        if device.type == "cpu":
            logger.info("CPU detected — AMP disabled, optimizing for CPU inference")

        # Gradient clipping
        self.grad_clip = training_cfg.get("gradient_clip_max_norm", 1.0)

        # Gradient accumulation for effective larger batches on CPU
        self.accum_steps = training_cfg.get("gradient_accumulation_steps", 1)

        # Early stopping
        es_cfg = training_cfg.get("early_stopping", {})
        self.early_stopping = None
        if es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 15),
                min_delta=es_cfg.get("min_delta", 0.001),
            )

        # Checkpointing
        self.checkpoint_interval = training_cfg.get("checkpoint_interval", 10)
        self.checkpoint_dir = config.get("output", {}).get(
            "checkpoints_dir", "outputs/checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # TensorBoard
        log_dir = config.get("output", {}).get("logs_dir", "outputs/logs")
        self.writer = SummaryWriter(log_dir=log_dir)

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        logger.info(
            f"SimCLR Trainer initialized | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | "
            f"Device={device} | "
            f"Epochs={self.total_epochs} | "
            f"Early stopping={'ON' if self.early_stopping else 'OFF'}"
        )

    def _create_optimizer(self, training_cfg: dict) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_name = training_cfg.get("optimizer", "adam").lower()
        lr = training_cfg.get("learning_rate", 3e-4)
        wd = training_cfg.get("weight_decay", 1e-4)

        if opt_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif opt_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _create_scheduler(self, dataloader_len: int):
        """Create cosine annealing LR scheduler with linear warmup."""
        total_steps = self.total_epochs * dataloader_len
        warmup_steps = self.warmup_epochs * dataloader_len

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            else:
                progress = (step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                return 0.5 * (
                    1 + torch.cos(torch.tensor(progress * 3.14159)).item()
                )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def resume_from_checkpoint(self, category: str) -> bool:
        """
        Resume training from the latest checkpoint.

        Args:
            category: MVTec category name.

        Returns:
            True if successfully resumed, False otherwise.
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{category}_best.pt"
        )

        if not os.path.exists(checkpoint_path):
            logger.info("No checkpoint found for resume — starting from scratch")
            return False

        try:
            checkpoint = load_checkpoint(
                self.model, checkpoint_path,
                optimizer=self.optimizer,
                device=self.device,
            )
            self.start_epoch = checkpoint.get("epoch", 0)
            self.best_loss = checkpoint.get("loss", float("inf"))
            self.global_step = checkpoint.get("global_step", 0)

            logger.info(
                f"Resumed from epoch {self.start_epoch} "
                f"(loss={self.best_loss:.4f})"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to resume checkpoint: {e} — starting fresh")
            return False

    def train(
        self,
        train_dataloader,
        category: str,
        resume: bool = False,
    ) -> Dict:
        """
        Run the full SimCLR training loop.

        Args:
            train_dataloader: DataLoader yielding (view_1, view_2, label, idx)
            category: MVTec category name (for saving checkpoints)
            resume: Whether to resume from last checkpoint

        Returns:
            Training history dict with loss values per epoch
        """
        # Resume if requested
        if resume:
            self.resume_from_checkpoint(category)

        self._create_scheduler(len(train_dataloader))

        history = {"epoch_losses": [], "learning_rates": [], "epoch_times": []}

        logger.info(f"Starting SimCLR training for category: {category}")
        logger.info(
            f"Epochs: {self.start_epoch + 1}->{self.total_epochs} | "
            f"Batch size: {self.config['training']['batch_size']} | "
            f"Batches/epoch: {len(train_dataloader)}"
        )

        start_time = time.time()

        for epoch in range(self.start_epoch, self.total_epochs):
            epoch_start = time.time()

            epoch_loss = self._train_one_epoch(train_dataloader, epoch)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            history["epoch_losses"].append(epoch_loss)
            history["learning_rates"].append(current_lr)
            history["epoch_times"].append(epoch_time)

            # TensorBoard logging
            self.writer.add_scalar(f"{category}/train_loss", epoch_loss, epoch)
            self.writer.add_scalar(f"{category}/learning_rate", current_lr, epoch)
            self.writer.add_scalar(f"{category}/epoch_time", epoch_time, epoch)

            elapsed = time.time() - start_time
            remaining_epochs = self.total_epochs - epoch - 1
            eta = (elapsed / (epoch - self.start_epoch + 1)) * remaining_epochs

            logger.info(
                f"Epoch [{epoch + 1}/{self.total_epochs}] | "
                f"Loss: {epoch_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {format_time(epoch_time)} | "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(eta)}"
            )

            # Save best checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                best_path = os.path.join(
                    self.checkpoint_dir, f"{category}_best.pt"
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    save_path=best_path,
                    category=category,
                    global_step=self.global_step,
                )

            # Save periodic checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                save_path = os.path.join(
                    self.checkpoint_dir,
                    f"{category}_epoch_{epoch + 1}.pt",
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    save_path=save_path,
                    category=category,
                    global_step=self.global_step,
                )

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping.step(epoch_loss):
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}/{self.total_epochs}"
                    )
                    break

        total_time = time.time() - start_time
        actual_epochs = len(history["epoch_losses"])

        # Training summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Summary — {category}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Total epochs:    {actual_epochs}/{self.total_epochs}")
        logger.info(f"  Total time:      {format_time(total_time)}")
        logger.info(f"  Best loss:       {self.best_loss:.4f}")
        logger.info(
            f"  Avg epoch time:  {format_time(sum(history['epoch_times']) / max(actual_epochs, 1))}"
        )
        if self.early_stopping and self.early_stopping.should_stop:
            logger.info(f"  Early stopped:   Yes (patience={self.early_stopping.patience})")
        logger.info(f"{'=' * 60}")

        self.writer.close()
        return history

    def _train_one_epoch(self, dataloader, epoch: int) -> float:
        """Train for one epoch with gradient accumulation support."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{self.total_epochs}",
            leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (view_1, view_2, labels, indices) in enumerate(pbar):
            view_1 = view_1.to(self.device)
            view_2 = view_2.to(self.device)

            if self.use_amp:
                with autocast(self.device.type):
                    _, z_i = self.model(view_1)
                    _, z_j = self.model(view_2)
                    loss = self.criterion(z_i, z_j)
                    # Scale loss for gradient accumulation
                    loss = loss / self.accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accum_steps == 0:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                _, z_i = self.model(view_1)
                _, z_j = self.model(view_2)
                loss = self.criterion(z_i, z_j)
                loss = loss / self.accum_steps

                loss.backward()

                if (batch_idx + 1) % self.accum_steps == 0:
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * self.accum_steps  # Undo scaling for logging
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item() * self.accum_steps:.4f}"})

        # Handle remaining gradients if num_batches not divisible by accum_steps
        if num_batches % self.accum_steps != 0:
            if self.use_amp:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

        return total_loss / max(num_batches, 1)
