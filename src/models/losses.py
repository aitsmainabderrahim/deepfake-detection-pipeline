"""
src/models/losses.py — Focal Loss + Label Smoothing BCE + Combined Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss: FL(p) = -α(1-p)^γ log(p). Focuses on hard examples."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t   = targets.view_as(logits).float()
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p   = torch.sigmoid(logits)
        pt  = p * t + (1 - p) * (1 - t)
        at  = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (at * (1 - pt) ** self.gamma * bce).mean()


class LabelSmoothBCE(nn.Module):
    """BCE with label smoothing — prevents overconfidence."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.view_as(logits).float()
        t = t * (1 - self.eps) + 0.5 * self.eps
        return F.binary_cross_entropy_with_logits(logits, t)


class CombinedLoss(nn.Module):
    """
    Primary loss + optional auxiliary head loss.
    total = L_primary(logit, y) + aux_weight * L_aux(aux_logit, y)
    """
    def __init__(
        self,
        primary:    str   = "focal",
        aux_weight: float = 0.3,
        gamma:      float = 2.0,
        alpha:      float = 0.75,
        smoothing:  float = 0.1,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.primary_fn = FocalLoss(gamma, alpha) if primary == "focal" \
                          else LabelSmoothBCE(smoothing)
        self.aux_fn     = LabelSmoothBCE(smoothing)

    def forward(
        self,
        logit:     torch.Tensor,
        targets:   torch.Tensor,
        aux_logit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.primary_fn(logit, targets)
        if aux_logit is not None and self.aux_weight > 0:
            loss = loss + self.aux_weight * self.aux_fn(aux_logit, targets)
        return loss
