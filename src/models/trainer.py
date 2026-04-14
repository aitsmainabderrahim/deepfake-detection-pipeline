"""
src/models/trainer.py — Full training pipeline with AMP, scheduling, early stopping.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.losses import CombinedLoss
from src.utils.metrics import compute_metrics, AverageMeter

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model: nn.Module, cfg):
        self.model  = model
        self.cfg    = cfg
        self.device = torch.device(
            cfg.project.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        Path(cfg.paths.checkpoints).mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.logs).mkdir(parents=True, exist_ok=True)

        self.loss_fn = CombinedLoss(
            primary=cfg.training.loss.primary,
            aux_weight=cfg.training.loss.aux_weight,
            gamma=cfg.training.loss.focal_gamma,
            alpha=cfg.training.loss.focal_alpha,
            smoothing=cfg.training.loss.label_smoothing,
        )

        # Separate LR: backbone gets much lower LR (already pretrained)
        backbone_params = list(self.model.backbone.parameters())
        head_params     = [p for n, p in self.model.named_parameters()
                           if "backbone" not in n]
        self.optimizer  = AdamW([
            {"params": backbone_params,
             "lr": cfg.training.optimizer.lr * cfg.training.optimizer.backbone_lr_multiplier},
            {"params": head_params, "lr": cfg.training.optimizer.lr},
        ], weight_decay=cfg.training.optimizer.weight_decay)

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cfg.training.scheduler.T_0,
            T_mult=cfg.training.scheduler.T_mult,
            eta_min=cfg.training.scheduler.eta_min,
        )

        self.use_amp = cfg.project.mixed_precision and self.device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)
        self.writer  = SummaryWriter(log_dir=cfg.paths.logs)

        self.best_auc     = 0.0
        self.best_epoch   = 0
        self.patience_ctr = 0

        logger.info(f"[Trainer] device={self.device} | AMP={self.use_amp}")

    # ── Public ───────────────────────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        cfg           = self.cfg
        epochs        = cfg.training.epochs
        freeze_epochs = cfg.model.freeze_backbone_epochs

        if freeze_epochs > 0:
            self.model.freeze_backbone(True)
            logger.info("[Trainer] Backbone frozen for initial epochs.")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            if epoch == freeze_epochs + 1:
                self.model.freeze_backbone(False)
                logger.info("[Trainer] Backbone unfrozen.")

            train_m = self._run_epoch(train_loader, train=True)
            val_m   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            elapsed = time.time() - t0
            self._log(epoch, train_m, val_m, elapsed)
            self._maybe_checkpoint(val_m["auc"], epoch)

            if self._should_stop():
                logger.info(f"[Trainer] Early stop at epoch {epoch}.")
                break

        self.writer.close()
        logger.info(f"[Trainer] Done. Best AUC={self.best_auc:.4f} @ epoch {self.best_epoch}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"[Trainer] Loaded checkpoint: {path}")

    # ── Epoch ────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train() if train else self.model.eval()
        loss_m = AverageMeter()
        probs_all, labels_all = [], []
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                loss, probs, labels = self._step(batch, train)
                loss_m.update(loss, len(labels))
                probs_all.extend(probs.tolist())
                labels_all.extend(labels.tolist())

        m = compute_metrics(np.array(labels_all), np.array(probs_all),
                            self.cfg.evaluation.threshold)
        m["loss"] = loss_m.avg
        return m

    def _step(self, batch, train: bool):
        # Unpack batch: (x, [feat,] labels)
        if len(batch) == 2:
            x, labels = batch; feat = None
        else:
            x, feat, labels = batch

        x      = x.to(self.device)
        labels = labels.to(self.device)
        feat   = feat.to(self.device) if feat is not None else None

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            logit, aux = self.model(x, feat)
            loss       = self.loss_fn(logit, labels, aux)

        if train:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.cfg.training.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(logit).squeeze().cpu().numpy()
        return loss.item(), probs, labels.cpu().numpy()

    # ── Logging & Checkpointing ───────────────────────────────────────────────

    def _log(self, epoch, tm, vm, elapsed):
        logger.info(
            f"Epoch {epoch:3d} ({elapsed:.0f}s) | "
            f"Train loss={tm['loss']:.4f} auc={tm['auc']:.4f} | "
            f"Val   loss={vm['loss']:.4f} auc={vm['auc']:.4f} f1={vm['f1']:.4f}"
        )
        for split, m in [("train", tm), ("val", vm)]:
            for k, v in m.items():
                self.writer.add_scalar(f"{split}/{k}", v, epoch)
        self.writer.add_scalar("lr", self.optimizer.param_groups[-1]["lr"], epoch)

    def _maybe_checkpoint(self, val_auc: float, epoch: int):
        if val_auc > self.best_auc:
            self.best_auc   = val_auc
            self.best_epoch = epoch
            self.patience_ctr = 0
            path = Path(self.cfg.paths.checkpoints) / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_auc": self.best_auc,
            }, path)
            logger.info(f"[Trainer] ✓ Saved (AUC={val_auc:.4f})")
        else:
            self.patience_ctr += 1

    def _should_stop(self) -> bool:
        es = self.cfg.training.early_stopping
        return es.enabled and self.patience_ctr >= es.patience
