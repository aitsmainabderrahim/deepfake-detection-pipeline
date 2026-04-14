"""
src/models/deepfake_model.py

Unified DeepFake Detection Model.

Combines:
  - YOLOv8BackboneExtractor  (Stage 2 backbone — pretrained CSPDarknet)
  - DeepFakeHead             (custom classification head + frequency branch)

This is the model trained and loaded for inference.

Full forward pass:
  x_rgb  (B,3,224,224) → YOLOv8Backbone → (B, backbone_dim)
                                                    ↓
  x_feat (B,10,224,224) → FrequencyBranch → (B, hidden_dim)
                                                    ↓
                              CrossAttention Fusion → Classifier → (B,1)

Inference:
  prob = sigmoid(logit) ∈ [0,1]
  pred = "FAKE" if prob >= threshold else "REAL"
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.backbone import YOLOv8BackboneExtractor
from src.models.classification_head import DeepFakeHead

logger = logging.getLogger(__name__)


class DeepFakeYOLOModel(nn.Module):
    """
    Full deepfake detector: YOLOv8 backbone + custom head.

    Args:
        yolo_weights:    Path to pretrained YOLOv8 .pt (for backbone).
        yolo_variant:    'yolov8n' | 'yolov8s' | 'yolov8m'
        feat_channels:   Auxiliary feature map channels (FFT + texture).
        hidden_dim:      Classification head hidden dimension.
        dropout:         Dropout rate.
        use_freq_branch: Enable FFT/SRM auxiliary branch.
        use_attention:   Enable cross-branch attention.
        use_aux_head:    Auxiliary loss during training.
        freeze_backbone: Start with backbone frozen.
        num_classes:     1 for binary (Real/Fake).
    """

    def __init__(
        self,
        yolo_weights:    str   = "weights/yolov8n.pt",
        yolo_variant:    str   = "yolov8n",
        feat_channels:   int   = 10,
        hidden_dim:      int   = 256,
        dropout:         float = 0.3,
        use_freq_branch: bool  = True,
        use_attention:   bool  = True,
        use_aux_head:    bool  = True,
        freeze_backbone: bool  = True,
        num_classes:     int   = 1,
    ):
        super().__init__()

        # ── Stage 2 Backbone ─────────────────────────────────────────────────
        self.backbone = YOLOv8BackboneExtractor(
            weights=yolo_weights,
            variant=yolo_variant,
            freeze=freeze_backbone,
        )
        backbone_dim = self.backbone.out_dim

        # ── Classification Head ───────────────────────────────────────────────
        self.head = DeepFakeHead(
            backbone_dim=backbone_dim,
            feat_channels=feat_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_freq=use_freq_branch,
            use_attention=use_attention,
            use_aux_head=use_aux_head,
            num_classes=num_classes,
        )

        params = self._count_params()
        logger.info(
            f"[Model] DeepFakeYOLO | backbone_dim={backbone_dim} | "
            f"total={params['total']:,} | trainable={params['trainable']:,}"
        )

    def forward(
        self,
        x_rgb:  torch.Tensor,
        x_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x_rgb:  (B, 3, 224, 224) — normalized face crop
            x_feat: (B, 10, 224, 224) — FFT + texture features (optional)

        Returns:
            (logit, aux_logit)
        """
        backbone_feat = self.backbone(x_rgb)        # (B, backbone_dim)
        logit, aux    = self.head(backbone_feat, x_feat)
        return logit, aux

    @torch.no_grad()
    def predict_proba(
        self,
        x_rgb:  torch.Tensor,
        x_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference-mode forward. Returns fake probability per sample.

        Returns:
            (B,) tensor of probabilities in [0, 1].
        """
        self.eval()
        logit, _ = self.forward(x_rgb, x_feat)
        return torch.sigmoid(logit).squeeze(-1)

    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the YOLOv8 backbone."""
        self.backbone.freeze(freeze)

    def _count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_model(cfg) -> DeepFakeYOLOModel:
    """Build model from config object."""
    m  = cfg.model
    f  = cfg.features
    feat_ch = (5 if f.use_fft else 0) + (5 if (f.use_srm or f.use_lbp or f.use_gradient) else 0)
    feat_ch = max(feat_ch, 1)   # At least 1 channel

    model = DeepFakeYOLOModel(
        yolo_weights=m.yolo_weights,
        yolo_variant=m.yolo_variant,
        feat_channels=feat_ch,
        dropout=m.dropout,
        use_freq_branch=m.use_freq_branch,
        use_attention=m.use_attention,
        use_aux_head=m.use_aux_head,
        freeze_backbone=True,
    )
    return model
