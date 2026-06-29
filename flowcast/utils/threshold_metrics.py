"""Threshold-based meteorological metrics for FlowCast forecasts.

Implements (per the FlowCast paper, section 4.1.2):
  - CRPS (Gaussian approximation from ensemble mean and std)
  - CSI, HSS, FAR from a 2x2 contingency table at a given threshold
  - Pooled CSI / FSS over 16x16 neighbourhoods
  - "-M" mean across thresholds

INSAT-appropriate thresholds (set in config; sensible defaults below):
  - VIS (albedo %, 0..~35) : [5, 10, 15, 20, 25, 30]
  - WV  (brightness temp K) : [220, 230, 240, 250, 260] (cold-cloud signal)
"""
import math
from typing import Iterable

import numpy as np


# ── continuous metric ───────────────────────────────────────────────────────
def crps_gaussian(ensemble, gt, data_range=None):
    """CRPS using the Gaussian closed-form (paper Eq. 1).

    ensemble : (N, H, W) array of N ensemble members.
    gt       : (H, W) ground truth.
    data_range : optional scalar to normalise CRPS (paper divides by 255 for SEVIR).
    Returns the mean CRPS over all pixels.
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mu = ens.mean(axis=0)
    sigma = ens.std(axis=0, ddof=0).clip(min=1e-8)
    z = (gt - mu) / sigma
    norm_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    norm_pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    crps = sigma * (z * (2.0 * norm_cdf - 1.0)
                    + 2.0 * norm_pdf - 1.0 / math.sqrt(math.pi))
    val = float(np.mean(crps))
    if data_range is not None and data_range > 0:
        val /= float(data_range)
    return val


# ── contingency-table primitives ────────────────────────────────────────────
def _confusion(pred, gt, thr):
    p = (pred > thr)
    g = (gt > thr)
    H = int(np.sum(p & g))
    F = int(np.sum(p & ~g))
    M = int(np.sum(~p & g))
    C = int(np.sum(~p & ~g))
    return H, F, M, C


def csi(H, F, M):
    return H / (H + M + F) if (H + M + F) > 0 else 0.0


def far(H, F):
    return F / (H + F) if (H + F) > 0 else 0.0


def hss(H, F, M, C):
    denom = (H + M) * (M + C) + (H + F) * (F + C)
    return 2.0 * (H * C - M * F) / denom if denom > 0 else 0.0


# ── pooled + spatial metrics ────────────────────────────────────────────────
def _maxpool(arr, k=16):
    """Non-overlapping max-pool over k x k blocks (truncates edges)."""
    H, W = arr.shape
    h2 = (H // k) * k
    w2 = (W // k) * k
    a = arr[:h2, :w2]
    a = a.reshape(h2 // k, k, w2 // k, k).max(axis=(1, 3))
    return a


def csi_p16(pred, gt, thr, k=16):
    p = _maxpool((pred > thr).astype(np.uint8), k)
    g = _maxpool((gt > thr).astype(np.uint8), k)
    H = int(np.sum(p & g))
    F = int(np.sum(p & ~g))
    M = int(np.sum(~p & g))
    return csi(H, F, M)


def _fraction(mask, k):
    """Fraction of pixels above threshold inside each non-overlapping k x k block."""
    H, W = mask.shape
    h2 = (H // k) * k
    w2 = (W // k) * k
    a = mask[:h2, :w2].astype(np.float64)
    a = a.reshape(h2 // k, k, w2 // k, k).mean(axis=(1, 3))
    return a


def fss_p16(pred, gt, thr, k=16):
    """Fractions Skill Score (paper Eq. 5) over k x k neighbourhoods."""
    Sf = _fraction(pred > thr, k)
    So = _fraction(gt > thr, k)
    num = float(np.sum((Sf - So) ** 2))
    den = float(np.sum(Sf ** 2) + np.sum(So ** 2))
    return 1.0 - num / max(den, 1e-12)


# ── aggregated "-M" metrics per frame ───────────────────────────────────────
def categorical_M(pred, gt, thresholds: Iterable[float]):
    """Returns dict with CSI-M, HSS-M, FAR-M (means across thresholds) plus
    CSI-P16-M and FSS-P16-M (pooled spatial variants)."""
    csis, hsss, fars, csi_ps, fsss = [], [], [], [], []
    for thr in thresholds:
        H, F, M, C = _confusion(pred, gt, thr)
        csis.append(csi(H, F, M))
        hsss.append(hss(H, F, M, C))
        fars.append(far(H, F))
        csi_ps.append(csi_p16(pred, gt, thr))
        fsss.append(fss_p16(pred, gt, thr))
    return {
        "CSI_M": float(np.mean(csis)),
        "HSS_M": float(np.mean(hsss)),
        "FAR_M": float(np.mean(fars)),
        "CSI_P16_M": float(np.mean(csi_ps)),
        "FSS_P16_M": float(np.mean(fsss)),
    }


def categorical_per_threshold(pred, gt, thresholds: Iterable[float]):
    """Per-threshold CSI/HSS/FAR, useful for extreme-event tables."""
    out = {}
    for thr in thresholds:
        H, F, M, C = _confusion(pred, gt, thr)
        out[float(thr)] = {
            "CSI": csi(H, F, M),
            "HSS": hss(H, F, M, C),
            "FAR": far(H, F),
        }
    return out


# ── sensible defaults ──────────────────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "vis": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],     # albedo % — DO NOT CHANGE
    # WV thresholds are on the INSAT WV TIFF native scale ([0, ~0.29] range
    # observed empirically), not raw Kelvin BT. Picked from upper-tail
    # percentiles of the WV dataset: ~P55, P72, P89, P97, P99.5.
    "wv":  [0.20, 0.225, 0.25, 0.265, 0.28],
}
