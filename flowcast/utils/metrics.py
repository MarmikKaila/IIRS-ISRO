"""Image-space reconstruction / forecast metrics: MSE, PSNR, SSIM."""
import numpy as np
from skimage.metrics import structural_similarity


def _data_range(a, b, data_range):
    if data_range is not None:
        return float(data_range)
    rng = max(float(a.max()), float(b.max())) - min(float(a.min()), float(b.min()))
    return rng if rng > 1e-8 else 1.0


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b, data_range=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    err = mse(a, b)
    if err <= 0.0:
        return float("inf")
    rng = _data_range(a, b, data_range)
    return float(10.0 * np.log10((rng ** 2) / err))


def ssim(a, b, data_range=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = _data_range(a, b, data_range)
    return float(structural_similarity(a, b, data_range=rng))


def all_metrics(a, b, data_range=None):
    """Return {'mse', 'psnr', 'ssim'} for a single image pair."""
    return {
        "mse": mse(a, b),
        "psnr": psnr(a, b, data_range),
        "ssim": ssim(a, b, data_range),
    }
