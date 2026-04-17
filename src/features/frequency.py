"""
src/features/frequency.py

High-frequency analysis via FFT.
GAN-generated faces leave periodic artifacts in the frequency domain
(checkerboard patterns from transposed convolutions).

Returns multi-channel feature maps that feed the auxiliary branch.
"""

import cv2
import numpy as np
import torch


def compute_fft_spectrum(img: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """Log-magnitude FFT spectrum (H×W float32, normalized [0,1])."""
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    f    = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag  = np.log1p(np.abs(f)) if log_scale else np.abs(f)
    mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.astype(np.float32)


def compute_high_pass(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """High-pass residual = image - gaussian_blur(image). Normalized [0,1]."""
    f   = img.astype(np.float32)
    res = f - cv2.GaussianBlur(f, (ksize, ksize), 0)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res.astype(np.float32)


def compute_fft_feature_map(img: np.ndarray, size: int = 224) -> np.ndarray:
    """
    5-channel FFT feature map:
      ch0 = grayscale FFT spectrum
      ch1-3 = per-channel RGB FFT spectra
      ch4 = high-pass residual (grayscale)

    Returns: (5, size, size) float32
    """
    img_u8 = img.astype(np.uint8)
    channels = []

    # Grayscale spectrum
    gs = compute_fft_spectrum(img_u8)
    channels.append(cv2.resize(gs, (size, size)))

    # Per-channel RGB spectra (convert BGR → RGB first)
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
    for c in range(3):
        sp = compute_fft_spectrum(rgb[:, :, c])
        channels.append(cv2.resize(sp, (size, size)))

    # High-pass residual
    hp = compute_high_pass(img_u8)
    if hp.ndim == 3:
        hp = cv2.cvtColor(hp, cv2.COLOR_BGR2GRAY)
    channels.append(cv2.resize(hp, (size, size)))

    return np.stack(channels, axis=0).astype(np.float32)


def fft_tensor(img: np.ndarray, size: int = 224) -> torch.Tensor:
    """Returns FFT feature map as torch.Tensor (5, H, W)."""
    return torch.from_numpy(compute_fft_feature_map(img, size))
