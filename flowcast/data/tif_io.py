"""Read INSAT-3S float32 TIFFs, parse timestamps, resize, normalize.

Filename pattern: ``3SIMG_<DDMMMYYYY>_<HHMM>_L1C_SGP_V01R00_IMG_<VIS|WV>.tif``
e.g. ``3SIMG_16DEC2025_1130_L1C_SGP_V01R00_IMG_VIS.tif``. Times are UTC.
"""
import datetime as dt
import glob
import os
import re

import numpy as np
import tifffile
from skimage.transform import resize as _sk_resize

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
     "SEP", "OCT", "NOV", "DEC"])}

_FNAME_RE = re.compile(r"3SIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})_")


# ── timestamps ──────────────────────────────────────────────────────────────
def parse_timestamp(filename):
    """Extract the UTC datetime encoded in an INSAT filename."""
    m = _FNAME_RE.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"cannot parse timestamp from {filename!r}")
    day, mon, year, hh, mm = m.groups()
    return dt.datetime(int(year), _MONTHS[mon], int(day), int(hh), int(mm),
                       tzinfo=dt.timezone.utc)


def list_files(source_dir, file_glob):
    """Return channel TIFF paths sorted by their UTC timestamp."""
    paths = glob.glob(os.path.join(source_dir, file_glob))
    paths = [p for p in paths if not p.endswith(".aux.xml")]
    return sorted(paths, key=parse_timestamp)


# ── raster IO ───────────────────────────────────────────────────────────────
def read_tif(path):
    """Load a single-band TIFF as a finite float32 2-D array."""
    arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    if arr.ndim == 3:                       # collapse any stray band axis
        arr = arr[..., 0]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def resize_image(arr, height, width):
    """Bilinear resize with anti-aliasing, preserving physical value range."""
    return _sk_resize(arr, (height, width), order=1, mode="reflect",
                      anti_aliasing=True, preserve_range=True).astype(np.float32)


# ── normalization (physical units <-> [-1, 1]) ──────────────────────────────
def normalize(arr, low, high):
    arr = np.clip(arr, low, high)
    return ((arr - low) / (high - low) * 2.0 - 1.0).astype(np.float32)


def denormalize(arr, low, high):
    arr = np.clip(arr, -1.0, 1.0)
    return ((arr + 1.0) * 0.5 * (high - low) + low).astype(np.float32)


def compute_norm_stats(paths, low_pct=1.0, high_pct=99.0,
                       sample=200, pixels_per_image=20000, seed=0):
    """Robust global [low_pct, high_pct] percentiles over a sample of frames."""
    rng = np.random.default_rng(seed)
    paths = list(paths)
    if len(paths) > sample:
        idx = sorted(rng.choice(len(paths), sample, replace=False))
        paths = [paths[i] for i in idx]
    pool = []
    for p in paths:
        flat = read_tif(p).ravel()
        if flat.size > pixels_per_image:
            flat = flat[rng.choice(flat.size, pixels_per_image, replace=False)]
        pool.append(flat)
    pool = np.concatenate(pool)
    return float(np.percentile(pool, low_pct)), float(np.percentile(pool, high_pct))


# ── SD-VAE tensor adapters ──────────────────────────────────────────────────
def to_vae_input(norm_arr):
    """[-1,1] (H,W) array -> (1,3,H,W) float32 tensor (grey replicated to RGB)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(norm_arr, dtype=np.float32))
    return t[None, None].repeat(1, 3, 1, 1)


def from_vae_output(tensor):
    """VAE-decoded (B,3,H,W) or (3,H,W) tensor -> single-channel (H,W) array."""
    if hasattr(tensor, "detach"):
        arr = tensor.detach().float().cpu().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.mean(axis=0).astype(np.float32)
