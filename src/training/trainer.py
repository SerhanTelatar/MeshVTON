"""
Training Loop for DiffFit-3D.

Main trainer with mixed precision, gradient accumulation, EMA,
checkpoint management, and wandb logging.
"""

import os
import time
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.models.tryon_pipeline import TryOnPipeline
from src.training.losses import TryOnLoss
from src.training.ema import EMAModel
from src.training.lr_scheduler import create_lr_scheduler


class Trainer:
    """
    Main training loop for the TryOn pipeline.

    Args:
        pipeline: TryOnPipeline model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration dict.
    """

    def __init__(self, pipeline: TryOnPipeline, train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None, config: Optional[dict] = None):
        self.config = config or {}
        self.pipeline = pipeline
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training settings
        training_cfg = self.config.get("training", {})
        self.epochs = training_cfg.get("epochs", 100)
        self.grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 1.0)
        self.mixed_precision = training_cfg.get("mixed_precision", "fp16")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)

        # Freeze backbone — only ControlNet3D is trainable
        if hasattr(self.pipeline, "freeze_backbone"):
            self.pipeline.freeze_backbone()

        # Optimizer — only trainable params (ControlNet3D)
        opt_cfg = training_cfg.get("optimizer", {})
        trainable_params = (
            self.pipeline.get_trainable_params()
            if hasattr(self.pipeline, "get_trainable_params")
            else [p for p in pipeline.parameters() if p.requires_grad]
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt_cfg.get("lr", 1e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # LR Scheduler
        sched_cfg = training_cfg.get("lr_scheduler", {})
        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            scheduler_type=sched_cfg.get("type", "cosine_with_warmup"),
            warmup_steps=sched_cfg.get("warmup_steps", 1000),
            total_steps=total_steps,
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )

        # Loss
        loss_cfg = self.config.get("losses", {})
        self.criterion = TryOnLoss(
            l1_weight=loss_cfg.get("l1_weight", 1.0),
            perceptual_weight=loss_cfg.get("perceptual_weight", 0.5),
            lpips_weight=loss_cfg.get("lpips_weight", 1.0),
        )

        # EMA
        ema_cfg = training_cfg.get("ema", {})
        self.ema = None
        if ema_cfg.get("enabled", True):
            self.ema = EMAModel(
                pipeline, decay=ema_cfg.get("decay", 0.9999),
                update_after_step=ema_cfg.get("update_after_step", 100),
                update_every=ema_cfg.get("update_every", 10),
            )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.mixed_precision != "no")

        # Checkpointing
        ckpt_cfg = training_cfg.get("checkpoint", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_every_n_steps = ckpt_cfg.get("save_every_n_steps", 5000)
        self.save_every_n_epochs = ckpt_cfg.get("save_every_n_epochs", 5)
        self.keep_last_n = ckpt_cfg.get("keep_last_n", 3)

        # Logging
        log_cfg = training_cfg.get("logging", {})
        self.log_every = log_cfg.get("log_every_n_steps", 50)
        self.use_wandb = log_cfg.get("use_wandb", False)
        self.wandb_run = None

        # State
        self.global_step = 0
        self.current_epoch = 0

    def train(self):
        """Run the full training loop."""
        if self.use_wandb:
            self._init_wandb()

        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}, Mixed precision: {self.mixed_precision}")
        print(f"Grad accumulation: {self.grad_accum_steps}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            train_loss = self._train_epoch(epoch)

            if self.val_loader is not None:
                val_loss = self._validate(epoch)
            else:
                val_loss = None

            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")

            self._log_epoch(epoch, train_loss, val_loss)

        self.save_checkpoint("final")
        print("Training complete!")

    def _train_epoch(self, epoch: int) -> float:
        self.pipeline.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(enabled=self.mixed_precision != "no",
                          dtype=torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16):
                outputs = self.pipeline(
                    person_image=batch["person_image"],
                    garment_image=batch["garment_image"],
                    agnostic_mask=batch["agnostic_image"],
                    pose_map=batch["pose_map"],
                    densepose_map=batch.get("densepose_map"),
                    conditioning_3d=batch.get("conditioning_3d"),
                )
                loss = outputs["loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                self.global_step += 1

                if self.ema is not None:
                    self.ema.update(self.global_step)

                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}",
                              "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

            if self.use_wandb and self.global_step % self.log_every == 0:
                self._log_step(loss.item() * self.grad_accum_steps)

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.pipeline.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = self.pipeline(
                person_image=batch["person_image"],
                garment_image=batch["garment_image"],
                agnostic_mask=batch["agnostic_image"],
                pose_map=batch["pose_map"],
                densepose_map=batch.get("densepose_map"),
                conditioning_3d=batch.get("conditioning_3d"),
            )
            total_loss += outputs["loss"].item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, name: str):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{name}.pt"
        state = {
            "model": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")
        self._cleanup_old_checkpoints()

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.pipeline.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.current_epoch = state["epoch"]
        if self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
        print(f"Checkpoint loaded from: {path}")

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(self.save_dir.glob("step_*.pt"), key=os.path.getmtime)
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            old.unlink()

    def _init_wandb(self):
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.get("training", {}).get("logging", {}).get("wandb_project", "difffit-3d"),
                config=self.config,
            )
        except ImportError:
            self.use_wandb = False

    def _log_step(self, loss: float):
        if self.wandb_run:
            import wandb
            wandb.log({"train/loss": loss, "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/step": self.global_step})

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float]):
        msg = f"Epoch {epoch + 1}: train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.4f}"
        print(msg)
        if self.wandb_run:
            import wandb
            log = {"epoch": epoch + 1, "epoch/train_loss": train_loss}
            if val_loss is not None:
                log["epoch/val_loss"] = val_loss
            wandb.log(log)
