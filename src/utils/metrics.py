"""src/utils/metrics.py — Evaluation metrics."""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              roc_curve)
from typing import Dict


def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "acc":       float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc":       float(roc_auc_score(labels, probs)),
        "ap":        float(average_precision_score(labels, probs)),
        "eer":       float(_eer(labels, probs)),
    }


def _eer(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def print_metrics(m: Dict[str, float], split: str = "Test"):
    print(f"\n{'═'*42}\n  {split} Metrics\n{'─'*42}")
    for k, v in m.items():
        print(f"  {k:<18} {v:.4f}")
    print(f"{'═'*42}\n")


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
