"""
src/models/classification_head.py

Custom DeepFake Classification Head.

Sits on top of the YOLOv8 backbone features and includes:
  1. Primary RGB path   — backbone features → MLP classifier
  2. Frequency branch   — FFT + SRM auxiliary channels → small CNN → MLP
  3. Cross-attention    — weights both branches dynamically
  4. Auxiliary head     — extra loss signal on frequency branch (training only)

Design rationale:
  The YOLOv8 backbone was trained for object detection, not forgery detection.
  The frequency branch gives the model explicit forensic signal:
    - FFT spectrum reveals GAN upsampling artifacts
    - SRM residuals reveal noise tampering
  The cross-attention gate learns to trust whichever branch is more
  informative for the current input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─── Frequency Branch CNN ────────────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Lightweight CNN that encodes FFT + texture feature maps.

    Input:  (B, C_feat, H, W)  e.g. C_feat=10 (5 FFT + 5 texture)
    Output: (B, out_dim)
    """

    def __init__(self, in_channels: int = 10, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 224 → 112
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            # 112 → 56
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            # 56 → 28
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
            # 28 → 14
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.norm(self.proj(h))


# ─── Cross-Branch Attention ───────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Soft attention gate: learns to weight RGB vs frequency branch.

    Given f_rgb (B, D) and f_freq (B, D), outputs a weighted sum.
    The weights are input-dependent, so the model learns to trust
    semantics vs forensics depending on the manipulation type.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, f_rgb: torch.Tensor, f_freq: torch.Tensor) -> torch.Tensor:
        w = self.gate(torch.cat([f_rgb, f_freq], dim=-1))  # (B, 2)
        return w[:, :1] * f_rgb + w[:, 1:] * f_freq


# ─── Classification Head ──────────────────────────────────────────────────────

class DeepFakeHead(nn.Module):
    """
    Full classification head sitting on top of YOLOv8 backbone.

    Args:
        backbone_dim:   Output dim of YOLOv8 backbone (e.g. 896 for yolov8n).
        feat_channels:  Input channels for frequency branch (FFT + texture).
        hidden_dim:     Internal MLP dimension.
        dropout:        Dropout rate.
        use_freq:       Enable frequency branch.
        use_attention:  Enable cross-attention fusion.
        use_aux_head:   Enable auxiliary loss head (training only).
        num_classes:    1 for binary classification.
    """

    def __init__(
        self,
        backbone_dim:  int  = 896,
        feat_channels: int  = 10,
        hidden_dim:    int  = 256,
        dropout:       float = 0.3,
        use_freq:      bool = True,
        use_attention: bool = True,
        use_aux_head:  bool = True,
        num_classes:   int  = 1,
    ):
        super().__init__()
        self.use_freq      = use_freq
        self.use_attention = use_attention
        self.use_aux_head  = use_aux_head

        # Project backbone features → hidden_dim
        self.rgb_proj = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Frequency branch
        if use_freq:
            self.freq_branch = FrequencyBranch(feat_channels, hidden_dim)
            if use_attention:
                self.fusion = CrossAttentionFusion(hidden_dim)
                fusion_dim  = hidden_dim
            else:
                fusion_dim  = hidden_dim * 2
        else:
            fusion_dim = hidden_dim

        # Main classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Auxiliary head (frequency branch supervision)
        if use_aux_head and use_freq:
            self.aux_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.aux_head = None

    def forward(
        self,
        backbone_feat: torch.Tensor,
        freq_feat:     Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            backbone_feat: (B, backbone_dim)   — from YOLOv8 backbone
            freq_feat:     (B, C, H, W)        — FFT + texture maps

        Returns:
            (logit, aux_logit)
              logit:     (B, 1) — main prediction
              aux_logit: (B, 1) — auxiliary prediction (None if not training / disabled)
        """
        # RGB path
        f_rgb = self.rgb_proj(backbone_feat)   # (B, hidden_dim)

        aux_logit = None

        if self.use_freq and freq_feat is not None:
            f_freq = self.freq_branch(freq_feat)   # (B, hidden_dim)

            # Auxiliary supervision on frequency branch
            if self.aux_head is not None and self.training:
                aux_logit = self.aux_head(f_freq)

            # Fuse branches
            if self.use_attention:
                fused = self.fusion(f_rgb, f_freq)
            else:
                fused = torch.cat([f_rgb, f_freq], dim=-1)
        else:
            fused = f_rgb

        logit = self.classifier(fused)   # (B, 1)
        return logit, aux_logit
