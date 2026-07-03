"""Persistence and climatology baselines for INGARSS 2026 evaluation.

Both forecasters return a dict with the same schema as src/forecast.py:
    {method, date, channel, summary, by_horizon, per_frame}

Usage (via eval_all_dates.py — not run directly):
    from src.baselines import persistence_forecast, climatology_forecast
"""
import datetime as dt
import os
from collections import defaultdict

import numpy as np

from .data import tif_io
from .utils import metrics
from .utils.config import resolve
from .utils.solar import is_daytime

try:
    from flowcast.utils.threshold_metrics import DEFAULT_THRESHOLDS, categorical_M
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False
    DEFAULT_THRESHOLDS = {}


# ── shared helpers ───────────────────────────────────────────────────────────

def _day_flags(slots, dn):
    if not dn.get("enabled", False):
        return [True] * len(slots)
    return [is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                       dn["elevation_threshold_deg"])
            for ts in slots]


def _night_fill(cfg, all_rows):
    """Mean of real night frames (VIS dark fill, same logic as src/forecast.py)."""
    dn = cfg["day_night"]
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")
    night_rows = [r for r in all_rows if r.get("is_day", True) is False]
    sample = night_rows[::max(1, len(night_rows) // 30)][:30]
    acc = np.zeros((H, W), np.float64)
    for r in sample:
        acc += tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
    return (acc / max(len(sample), 1)).astype(np.float32)


def _score(forecast_imgs, slots, by_utc, day_flags, cfg):
    """Score forecast_imgs against GT TIFFs; return (per_frame, summary, by_horizon)."""
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")
    dn = cfg["day_night"]
    channel = cfg["channel"]
    channel_thr = DEFAULT_THRESHOLDS.get(channel, []) if _HAS_CAT else []

    per_frame = []
    for i, ts in enumerate(slots):
        r = by_utc.get(ts)
        if r is None or (dn.get("enabled", False) and not day_flags[i]):
            continue
        gt = tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        m = metrics.all_metrics(forecast_imgs[i], gt, data_range=drange)
        if _HAS_CAT and channel_thr:
            m.update(categorical_M(forecast_imgs[i], gt, channel_thr))
        per_frame.append({"step": i, "utc": ts.isoformat(), **m})

    metric_keys = [k for k in (per_frame[0] if per_frame else {})
                   if k not in ("step", "utc")]
    summary = {k: float(np.mean([p[k] for p in per_frame if k in p]))
               for k in metric_keys} if per_frame else {}

    by_horizon = {}
    for label, lo, hi in [("0-11", 0, 11), ("12-23", 12, 23),
                           ("24-35", 24, 35), ("36-47", 36, 47)]:
        bucket = [p for p in per_frame if lo <= p["step"] <= hi]
        if bucket:
            by_horizon[label] = {
                k: float(np.mean([p[k] for p in bucket if k in p]))
                for k in metric_keys
            }

    return per_frame, summary, by_horizon


# ── persistence ──────────────────────────────────────────────────────────────

def persistence_forecast(cfg, date, all_rows):
    """Repeat the last observed TIFF frame for all 48 forecast slots.

    Loads the TIFF directly (no VAE encode/decode) so the baseline is
    VAE-artifact-free and GPU-independent.
    """
    cadence = cfg["sequence"]["cadence_minutes"]
    horizon = cfg["forecast"]["horizon"]
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]
    src_dir = resolve(cfg, "source_dir")

    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step for i in range(horizon)]
    by_utc = {r["utc"]: r for r in all_rows}
    day_flags = _day_flags(slots, dn)

    prior = [r for r in all_rows if r["utc"] < day_start]
    if not prior:
        raise ValueError(f"[persistence] no frames before {date}")
    last_img = tif_io.resize_image(
        tif_io.read_tif(os.path.join(src_dir, prior[-1]["filename"])), H, W)

    forecast_imgs = [last_img.copy() for _ in range(horizon)]

    if dn.get("enabled", False):
        fill = _night_fill(cfg, all_rows)
        for i in range(horizon):
            if not day_flags[i]:
                forecast_imgs[i] = fill

    per_frame, summary, by_horizon = _score(
        forecast_imgs, slots, by_utc, day_flags, cfg)

    return {
        "method": "persistence",
        "date": date.strftime("%Y%m%d"),
        "channel": cfg["channel"],
        "summary": summary,
        "by_horizon": by_horizon,
        "per_frame": per_frame,
    }


# ── climatology ──────────────────────────────────────────────────────────────

def climatology_forecast(cfg, date, all_rows, train_rows,
                         cache_dir=None):
    """Per-slot (30-min slot index 0-47) mean of training frames as forecast.

    slot_idx = hour * 2 + minute // 30  (0..47, repeats daily).

    Args:
        cache_dir: if given, slot means are saved/loaded as
                   <cache_dir>/climatology_slot_means.npy to avoid recomputing.
    """
    cadence = cfg["sequence"]["cadence_minutes"]
    horizon = cfg["forecast"]["horizon"]
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]
    src_dir = resolve(cfg, "source_dir")

    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step for i in range(horizon)]
    by_utc = {r["utc"]: r for r in all_rows}
    day_flags = _day_flags(slots, dn)

    # ── build / load per-slot means ────────────────────────────────────────
    cache_path = (os.path.join(cache_dir, "climatology_slot_means.npy")
                  if cache_dir else None)

    if cache_path and os.path.exists(cache_path):
        slot_means = np.load(cache_path)            # (48, H, W)
        print(f"[climatology] loaded slot means from {cache_path}")
    else:
        print(f"[climatology] building slot means from {len(train_rows)} train frames…")
        slot_accums = defaultdict(
            lambda: {"acc": np.zeros((H, W), np.float64), "n": 0})
        for j, r in enumerate(train_rows):
            ts = r["utc"]
            idx = ts.hour * 2 + ts.minute // 30
            img = tif_io.resize_image(
                tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
            slot_accums[idx]["acc"] += img.astype(np.float64)
            slot_accums[idx]["n"] += 1
            if (j + 1) % 200 == 0:
                print(f"  …{j + 1}/{len(train_rows)} frames loaded")

        # global fallback for slots with no training data
        total_acc = sum(s["acc"] for s in slot_accums.values())
        total_n = sum(s["n"] for s in slot_accums.values())
        global_mean = (total_acc / max(total_n, 1)).astype(np.float32)

        slot_means = np.stack([
            (slot_accums[i]["acc"] / slot_accums[i]["n"]).astype(np.float32)
            if slot_accums[i]["n"] > 0 else global_mean
            for i in range(48)
        ])                                          # (48, H, W)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, slot_means)
            print(f"[climatology] saved slot means -> {cache_path}")

    # ── assemble forecast ──────────────────────────────────────────────────
    forecast_imgs = []
    for ts in slots:
        idx = ts.hour * 2 + ts.minute // 30
        forecast_imgs.append(slot_means[idx].copy())

    if dn.get("enabled", False):
        fill = _night_fill(cfg, all_rows)
        for i in range(horizon):
            if not day_flags[i]:
                forecast_imgs[i] = fill

    per_frame, summary, by_horizon = _score(
        forecast_imgs, slots, by_utc, day_flags, cfg)

    return {
        "method": "climatology",
        "date": date.strftime("%Y%m%d"),
        "channel": cfg["channel"],
        "summary": summary,
        "by_horizon": by_horizon,
        "per_frame": per_frame,
    }
