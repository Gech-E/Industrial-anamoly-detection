"""
SimCLR Trainer.
Handles the full contrastive learning training loop with logging,
checkpointing, mixed precision, and learning rate scheduling.
"""

import os
import time
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.simclr import SimCLRModel
from src.training.losses import NTXentLoss
from src.utils.utils import save_checkpoint, format_time

logger = logging.getLogger(__name__)


class SimCLRTrainer:
    """
    Trainer for SimCLR contrastive pretraining.
    
    Training pipeline:
    1. For each batch of N images, generate 2N augmented views
    2. Pass all views through encoder + projection head
    3. Compute NT-Xent loss between positive pairs
    4. Update model parameters via backpropagation
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
        self.warmup_epochs = training_cfg.get("warmup_epochs", 10)
        self.total_epochs = training_cfg.get("epochs", 100)
        
        # Mixed precision
        self.use_amp = training_cfg.get("use_amp", True) and device.type == "cuda"
        self.scaler = GradScaler(device.type) if self.use_amp else None
        
        # Gradient clipping
        self.grad_clip = training_cfg.get("gradient_clip_max_norm", 1.0)
        
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
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        
        logger.info(
            f"SimCLR Trainer initialized | "
            f"AMP={'ON' if self.use_amp else 'OFF'} | "
            f"Device={device} | "
            f"Epochs={self.total_epochs}"
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
                # Linear warmup
                return step / max(warmup_steps, 1)
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
    
    def train(self, train_dataloader, category: str) -> dict:
        """
        Run the full SimCLR training loop.
        
        Args:
            train_dataloader: DataLoader yielding (view_1, view_2, label, idx)
            category: MVTec category name (for saving checkpoints)
            
        Returns:
            Training history dict with loss values per epoch
        """
        self._create_scheduler(len(train_dataloader))
        
        history = {"epoch_losses": [], "learning_rates": []}
        
        logger.info(f"Starting SimCLR training for category: {category}")
        logger.info(f"Epochs: {self.total_epochs} | Batch size: {self.config['training']['batch_size']}")
        
        start_time = time.time()
        
        for epoch in range(self.total_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_one_epoch(train_dataloader, epoch)
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["epoch_losses"].append(epoch_loss)
            history["learning_rates"].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar(f"{category}/train_loss", epoch_loss, epoch)
            self.writer.add_scalar(f"{category}/learning_rate", current_lr, epoch)
            
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (self.total_epochs - epoch - 1)
            
            logger.info(
                f"Epoch [{epoch+1}/{self.total_epochs}] | "
                f"Loss: {epoch_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(eta)}"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0 or epoch_loss < self.best_loss:
                save_path = os.path.join(
                    self.checkpoint_dir,
                    f"{category}_epoch_{epoch+1}.pt"
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    save_path=save_path,
                    category=category,
                )
            
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
                )
        
        total_time = time.time() - start_time
        logger.info(
            f"Training complete for {category} | "
            f"Total time: {format_time(total_time)} | "
            f"Best loss: {self.best_loss:.4f}"
        )
        
        self.writer.close()
        return history
    
    def _train_one_epoch(self, dataloader, epoch: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{self.total_epochs}",
            leave=False,
        )
        
        for view_1, view_2, labels, indices in pbar:
            view_1 = view_1.to(self.device)
            view_2 = view_2.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(self.device.type):
                    _, z_i = self.model(view_1)
                    _, z_j = self.model(view_2)
                    loss = self.criterion(z_i, z_j)
                
                self.scaler.scale(loss).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, z_i = self.model(view_1)
                _, z_j = self.model(view_2)
                loss = self.criterion(z_i, z_j)
                
                loss.backward()
                
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / max(num_batches, 1)
