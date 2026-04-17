"""
src/data/dataset.py — DeepFake face image dataset with optional feature maps.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import logging

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DeepFakeDataset(Dataset):
    """
    Loads face crops from:
      root/real/*.png
      root/fake/*.png
    or a CSV with columns: [path, label, split]

    Each item returns (img_tensor, label) or (img_tensor, feat_tensor, label).
    """

    def __init__(
        self,
        root:        str,
        transform:   Optional[Callable] = None,
        feature_fn:  Optional[Callable] = None,   # img_np → feat_tensor
        split:       str = "train",
        return_path: bool = False,
    ):
        self.transform   = transform
        self.feature_fn  = feature_fn
        self.return_path = return_path
        self.samples: List[Tuple[Path, int]] = []

        root = Path(root)
        if root.suffix == ".csv":
            self._from_csv(root, split)
        else:
            self._from_dir(root)

        logger.info(
            f"[Dataset/{split}] {len(self)} samples | "
            f"real={sum(1 for _,l in self.samples if l==0)} | "
            f"fake={sum(1 for _,l in self.samples if l==1)}"
        )

    def _from_dir(self, root: Path):
        for label, sub in [(0, "real"), (1, "fake")]:
            d = root / sub
            if not d.exists():
                raise FileNotFoundError(f"Missing: {d}")
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(d.glob(ext)):
                    self.samples.append((p, label))

    def _from_csv(self, csv: Path, split: str):
        df = pd.read_csv(csv)
        df = df[df["split"] == split]
        for _, row in df.iterrows():
            self.samples.append((Path(row["path"]), int(row["label"])))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Cannot read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Albumentations
        if self.transform:
            img = self.transform(image=img)["image"]

        # (H,W,3) → (3,H,W) float [0,1] with ImageNet norm
        mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
        std  = torch.tensor(IMAGENET_STD).view(3,1,1)
        t    = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        t    = (t - mean) / std

        label_t = torch.tensor(label, dtype=torch.float32)

        if self.feature_fn:
            raw  = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            feat = self.feature_fn(raw)
            if self.return_path:
                return t, feat, label_t, str(path)
            return t, feat, label_t

        if self.return_path:
            return t, label_t, str(path)
        return t, label_t

    def get_sampler(self) -> WeightedRandomSampler:
        labels  = [l for _, l in self.samples]
        counts  = np.bincount(labels)
        weights = 1.0 / counts
        sw      = torch.tensor([weights[l] for l in labels])
        return WeightedRandomSampler(sw, len(sw), replacement=True)


def build_dataloaders(data_root, transform_train, transform_val,
                      feature_fn=None, batch_size=32,
                      num_workers=4, pin_memory=True):
    """Returns (train_loader, val_loader, test_loader)."""
    kw = dict(feature_fn=feature_fn)
    train_ds = DeepFakeDataset(data_root, transform_train, split="train", **kw)
    val_ds   = DeepFakeDataset(data_root, transform_val,   split="val",   **kw)
    test_ds  = DeepFakeDataset(data_root, transform_val,   split="test",  **kw)

    train_loader = DataLoader(train_ds, batch_size, sampler=train_ds.get_sampler(),
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
