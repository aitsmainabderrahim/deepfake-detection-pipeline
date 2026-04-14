"""
src/models/backbone.py

YOLOv8 Backbone Feature Extractor.

Loads a pretrained YOLOv8 model and extracts intermediate feature maps
from its CSPDarknet backbone for use as a classification backbone.

YOLOv8 internal structure (simplified):
  model.model[0..9]   = backbone layers (stem + C2f blocks)
  model.model[10..21] = neck + detection head (we IGNORE these)

We hook into the backbone at three scales:
  P3 — stride 8  — fine-grained spatial features
  P4 — stride 16 — mid-level features
  P5 — stride 32 — high-level semantic features

These are then globally pooled and concatenated → 1D feature vector.

Why YOLOv8 backbone?
  - Pretrained on COCO (1.5M images) → very strong spatial features
  - CSPDarknet captures both local texture and global structure
  - Fast: YOLOv8n backbone runs in <5ms on CPU
  - Feature maps at P3/P4/P5 encode different abstraction levels,
    which is ideal for detecting subtle manipulation artifacts
"""

import logging
from typing import List, Tuple, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class YOLOv8BackboneExtractor(nn.Module):
    """
    Extracts multi-scale features from a pretrained YOLOv8 backbone.

    The neck and detection head are discarded. Only backbone layers
    (indices 0–9 for yolov8n) are kept and fine-tuned.

    Args:
        weights:      Path to pretrained YOLOv8 .pt file.
        variant:      'yolov8n' | 'yolov8s' | 'yolov8m'
        freeze:       If True, freeze all backbone parameters initially.

    Output feature dimensions per variant:
        yolov8n: P3=128, P4=256, P5=512  → concat=896
        yolov8s: P3=256, P4=512, P5=1024 → concat=1792
        yolov8m: P3=384, P4=768, P5=768  → concat=1920

    Usage:
        bb = YOLOv8BackboneExtractor('weights/yolov8n.pt')
        feats = bb(x)   # → (B, 896) for yolov8n
    """

    # Known output channels for backbone P3/P4/P5 per variant
    FEAT_DIMS: Dict[str, Tuple[int,int,int]] = {
        "yolov8n": (128, 256, 512),
        "yolov8s": (256, 512, 1024),
        "yolov8m": (384, 768, 768),
        "yolov8l": (512, 512, 512),
        "yolov8x": (640, 640, 640),
    }

    def __init__(
        self,
        weights:  str  = "weights/yolov8n.pt",
        variant:  str  = "yolov8n",
        freeze:   bool = True,
    ):
        super().__init__()
        self.variant = variant

        # ── Load YOLOv8 and extract backbone only ────────────────────────────
        try:
            from ultralytics import YOLO
            yolo      = YOLO(weights)
            full_model = yolo.model   # nn.Module

            # YOLOv8 backbone = layers 0..9 (varies slightly by variant)
            # We extract them as a sequential module
            backbone_layers = list(full_model.model.children())

            # Backbone ends before the neck (SPPF is usually index 9)
            # We take indices 0–9 for nano/small, adjust for larger
            n_backbone = self._get_backbone_depth(variant)
            self.backbone = nn.Sequential(*backbone_layers[:n_backbone])

            logger.info(
                f"[Backbone] YOLOv8 {variant} backbone loaded "
                f"({n_backbone} layers) from {weights}"
            )
        except Exception as e:
            logger.warning(f"[Backbone] Failed to load YOLOv8: {e}. Using random init.")
            self.backbone = self._build_fallback_backbone()

        # ── Hooks to capture intermediate feature maps ───────────────────────
        self._feats: Dict[str, torch.Tensor] = {}
        self._register_hooks(variant)

        # ── Global average pool + projection ─────────────────────────────────
        p3_dim, p4_dim, p5_dim = self.FEAT_DIMS.get(variant, (128, 256, 512))
        self.out_dim = p3_dim + p4_dim + p5_dim

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(self.out_dim)

        if freeze:
            self.freeze(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized RGB tensor.

        Returns:
            (B, out_dim) — concatenated multi-scale features.
        """
        self._feats.clear()
        _ = self.backbone(x)   # Forward pass triggers hooks

        parts = []
        for key in ["p3", "p4", "p5"]:
            if key in self._feats:
                f = self.pool(self._feats[key]).flatten(1)   # (B, C)
                parts.append(f)

        if not parts:
            raise RuntimeError("No feature maps captured. Check hook registration.")

        out = torch.cat(parts, dim=-1)   # (B, out_dim)
        return self.norm(out)

    def freeze(self, freeze: bool = True):
        """Freeze or unfreeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"[Backbone] Parameters {status}.")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_backbone_depth(self, variant: str) -> int:
        """Number of backbone layers to keep (before neck begins)."""
        return {"yolov8n": 10, "yolov8s": 10, "yolov8m": 10,
                "yolov8l": 10, "yolov8x": 10}.get(variant, 10)

    def _register_hooks(self, variant: str):
        """
        Register forward hooks on backbone layers to capture P3/P4/P5.

        Layer indices for feature capture (yolov8n):
          Layer 4 → P3 (stride 8)
          Layer 6 → P4 (stride 16)
          Layer 9 → P5 (stride 32, after SPPF)
        """
        hook_layers = {"yolov8n": (4, 6, 9), "yolov8s": (4, 6, 9),
                       "yolov8m": (4, 6, 9)}.get(variant, (4, 6, 9))

        layers = list(self.backbone.children())
        names  = ["p3", "p4", "p5"]

        for name, idx in zip(names, hook_layers):
            if idx < len(layers):
                layer = layers[idx]
                n     = name   # Capture in closure
                layer.register_forward_hook(
                    lambda module, inp, out, _n=n: self._feats.update({_n: out})
                )

    def _build_fallback_backbone(self) -> nn.Sequential:
        """Minimal CNN backbone if YOLOv8 weights unavailable."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
        )
