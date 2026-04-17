"""
src/features/texture.py

Forensic texture features for deepfake detection.
  - SRM noise residuals (3 high-pass kernels from steganalysis)
  - LBP (Local Binary Patterns) — micro-texture descriptor
  - Gradient magnitude map

Combined: 5-channel texture feature map (SRM×3 + LBP×1 + Gradient×1).
"""

import cv2
import numpy as np
import torch

# SRM filter kernels (Fridrich & Kodovsky, 2012)
SRM_KERNELS = [
    np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
              dtype=np.float32) / 4.0,
    np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
              dtype=np.float32) / 12.0,
    np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],
              dtype=np.float32) / 12.0,
]


def compute_srm(img: np.ndarray) -> np.ndarray:
    """
    Apply 3 SRM kernels to the luminance channel.
    Returns (3, H, W) float32 in [-1, 1].
    """
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    out  = []
    for k in SRM_KERNELS:
        r   = cv2.filter2D(gray, -1, k)
        sig = r.std()
        r   = np.clip(r, -3*sig, 3*sig) / (3*sig + 1e-8)
        out.append(r)
    return np.stack(out, axis=0).astype(np.float32)


def compute_lbp(img: np.ndarray) -> np.ndarray:
    """LBP map (H×W float32 normalized [0,1])."""
    try:
        from skimage.feature import local_binary_pattern
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lbp  = local_binary_pattern(gray.astype(np.float64), 8, 1, "uniform")
        lbp  = lbp.astype(np.float32)
        lbp  = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)
        return lbp
    except ImportError:
        # Fallback: simple local variance
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.boxFilter(gray, -1, (3,3))
        var  = cv2.boxFilter(gray**2, -1, (3,3)) - mean**2
        var  = (var - var.min()) / (var.max() - var.min() + 1e-8)
        return var.astype(np.float32)


def compute_gradient(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude (H×W float32 normalized [0,1])."""
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx**2 + gy**2)
    mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.astype(np.float32)


def compute_texture_feature_map(img: np.ndarray, size: int = 224) -> np.ndarray:
    """
    5-channel texture feature map:
      ch0-2 = SRM residuals × 3
      ch3   = LBP map
      ch4   = Gradient magnitude

    Returns: (5, size, size) float32
    """
    img_u8 = img.astype(np.uint8)

    srm  = compute_srm(img_u8)   # (3, H, W)
    lbp  = compute_lbp(img_u8)   # (H, W)
    grad = compute_gradient(img_u8) # (H, W)

    srm_r  = np.stack([cv2.resize(srm[i], (size, size)) for i in range(3)])
    lbp_r  = cv2.resize(lbp,  (size, size))[np.newaxis]
    grad_r = cv2.resize(grad, (size, size))[np.newaxis]

    return np.concatenate([srm_r, lbp_r, grad_r], axis=0).astype(np.float32)


def texture_tensor(img: np.ndarray, size: int = 224) -> torch.Tensor:
    """Returns texture feature map as torch.Tensor (5, H, W)."""
    return torch.from_numpy(compute_texture_feature_map(img, size))
