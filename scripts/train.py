#!/usr/bin/env python
"""
scripts/train.py — Training entry point.

Usage:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/default.yaml --device cuda
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, logging, random
import numpy as np, torch

from src.utils.config import load_config
from src.data.dataset import build_dataloaders
from src.data.augmentation import get_train_transform, get_val_transform
from src.features.frequency import fft_tensor
from src.features.texture import texture_tensor
from src.models.deepfake_model import build_model
from src.models.trainer import Trainer

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("logs/train.log", mode="a")])
logger = logging.getLogger(__name__)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def feature_fn(img_rgb):
    """Combine FFT and texture features → (10, H, W) tensor."""
    fft = fft_tensor(img_rgb, size=224)
    tex = texture_tensor(img_rgb, size=224)
    return torch.cat([fft, tex], dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume",     default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:     cfg.project.device       = args.device
    if args.epochs:     cfg.training.epochs      = args.epochs
    if args.batch_size: cfg.training.batch_size  = args.batch_size

    set_seed(cfg.project.seed)

    train_tf = get_train_transform(cfg.data.image_size)
    val_tf   = get_val_transform(cfg.data.image_size)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=cfg.paths.faces_dir,
        transform_train=train_tf, transform_val=val_tf,
        feature_fn=feature_fn,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    model   = build_model(cfg)
    trainer = Trainer(model, cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit(train_loader, val_loader)

    # Final test evaluation
    import os
    best = os.path.join(cfg.paths.checkpoints, "best_model.pth")
    if os.path.exists(best):
        trainer.load_checkpoint(best)
    from src.utils.metrics import print_metrics
    m = trainer._run_epoch(test_loader, train=False)
    print_metrics(m, "Test (Final)")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Save remaining scripts as separate files via the next create_file call
# ─────────────────────────────────────────────────────────────────────────────
