"""Build ConvLSTM training windows from cached latents.

A *window* is ``input_len`` consecutive latents (the RNN input) followed by
``target_len`` consecutive latents (the prediction target). Windows are only
formed inside *contiguous runs* — stretches of frames exactly one cadence step
apart — so the model never learns a jump across a missing slot.

When ``day_night.enabled`` is true (VIS), each frame is augmented with a
**solar-zenith-angle** channel (normalised to ~[-1, 1]) so the model knows
where the sun is, and night frames are **zero-weighted in the loss** so they
stop dominating training.
"""
import csv
import datetime as dt
import os

import torch
from torch.utils.data import Dataset


# ── manifest ────────────────────────────────────────────────────────────────
def read_manifest(path):
    """Read the Stage-1 manifest CSV, parsing timestamps and the is_day flag."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["utc"] = dt.datetime.fromisoformat(r["utc"])
            if r.get("is_day", "") != "":
                r["is_day"] = str(r["is_day"]).strip().lower() in ("true", "1", "yes")
            rows.append(r)
    rows.sort(key=lambda r: r["utc"])
    return rows


def temporal_split(rows, val_fraction, test_fraction):
    """Chronological train/val/test split — no shuffling across boundaries."""
    n = len(rows)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    return (rows[:n_train],
            rows[n_train:n_train + n_val],
            rows[n_train + n_val:])


# ── contiguous runs & windows ───────────────────────────────────────────────
def contiguous_runs(rows, cadence_minutes=30, daytime_only=False):
    """Split time-sorted rows into runs with no missing slots."""
    if daytime_only:
        rows = [r for r in rows if r.get("is_day", True)]
    step = dt.timedelta(minutes=cadence_minutes)
    runs, cur = [], []
    for r in rows:
        if cur and (r["utc"] - cur[-1]["utc"]) != step:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def make_windows(runs, input_len, target_len=1):
    """List of (run, start_index) windows of length input_len + target_len."""
    need = input_len + target_len
    windows = []
    for run in runs:
        for s in range(len(run) - need + 1):
            windows.append((run, s))
    return windows


# ── dataset ─────────────────────────────────────────────────────────────────
class LatentSequenceDataset(Dataset):
    """Yields ``(x, future_cond, y_lat, y_weight)`` per window.

    x           : (T, latent_ch + cond_ch, H, W) — input latents + SZA channel
                  if cond_ch > 0.
    future_cond : (target_len, cond_ch, H, W) — SZA per target frame; empty
                  tensor when cond_ch == 0.
    y_lat       : (target_len, latent_ch, H, W) — target latents.
    y_weight    : (target_len,) — per-target-frame loss weight (1.0 day,
                  0.0 night when mask_night is on; all 1.0 otherwise).
    """

    def __init__(self, windows, input_len, target_len, repo_root,
                 cond_ch=0, mask_night=False, latent_h=36, latent_w=32):
        self.windows = windows
        self.input_len = input_len
        self.target_len = target_len
        self.repo_root = repo_root
        self.cond_ch = cond_ch
        self.mask_night = mask_night
        self.H = latent_h
        self.W = latent_w
        self._cache = {}

    def _load(self, latent_path):
        if latent_path not in self._cache:
            full = (latent_path if os.path.isabs(latent_path)
                    else os.path.join(self.repo_root, latent_path))
            self._cache[latent_path] = torch.load(full, map_location="cpu").float()
        return self._cache[latent_path]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        run, s = self.windows[i]
        n = self.input_len + self.target_len
        latents = [self._load(run[s + k]["latent_path"]) for k in range(n)]
        x_lat = torch.stack(latents[:self.input_len], 0)        # (T, 4, H, W)
        y_lat = torch.stack(latents[self.input_len:], 0)        # (target, 4, H, W)

        if self.cond_ch > 0:
            sza_in = torch.tensor(
                [run[s + k]["sza_norm"] for k in range(self.input_len)],
                dtype=torch.float32)
            sza_fu = torch.tensor(
                [run[s + self.input_len + k]["sza_norm"]
                 for k in range(self.target_len)], dtype=torch.float32)
            sza_in = sza_in.view(-1, 1, 1, 1).expand(-1, 1, self.H, self.W)
            sza_fu = sza_fu.view(-1, 1, 1, 1).expand(-1, 1, self.H, self.W)
            x = torch.cat([x_lat, sza_in], dim=1)
            future_cond = sza_fu.contiguous()
        else:
            x = x_lat
            future_cond = torch.zeros(self.target_len, 0, self.H, self.W)

        if self.mask_night:
            y_w = torch.tensor(
                [float(run[s + self.input_len + k].get("is_day", True))
                 for k in range(self.target_len)], dtype=torch.float32)
        else:
            y_w = torch.ones(self.target_len, dtype=torch.float32)

        return x, future_cond, y_lat, y_w


def build_datasets(cfg):
    """Build train/val/test LatentSequenceDatasets from a channel config.

    Returns (train_ds, val_ds, test_ds). Train uses target_len = unroll_steps;
    val/test use target_len = 1. When day_night.enabled, every row is stamped
    with a normalised solar elevation (``sza_norm`` in [-1, 1]) so the dataset
    can emit it as a conditioning channel.
    """
    from ..utils.config import resolve
    from ..utils.solar import solar_elevation_deg

    rows = read_manifest(resolve(cfg, "manifest_path"))
    cadence = cfg["sequence"]["cadence_minutes"]
    input_len = cfg["sequence"]["input_len"]
    unroll = int(cfg["train"].get("unroll_steps", 1))

    dn = cfg.get("day_night", {})
    cond_ch = 1 if dn.get("enabled", False) else 0
    mask_night = bool(dn.get("enabled", False))

    # Both channels train on the full diurnal cycle. VIS uses SZA conditioning
    # and a night-masked loss so dark frames stop dominating; WV has neither.
    if cond_ch:
        lat, lon = dn["scene_lat"], dn["scene_lon"]
        for r in rows:
            elev = solar_elevation_deg(r["utc"], lat, lon)
            r["sza_norm"] = max(-1.0, min(1.0, elev / 90.0))

    tr_rows, va_rows, te_rows = temporal_split(
        rows, cfg["train"]["val_fraction"], cfg["train"]["test_fraction"])

    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8

    def ds(split_rows, target_len):
        runs = contiguous_runs(split_rows, cadence, daytime_only=False)
        windows = make_windows(runs, input_len, target_len)
        return LatentSequenceDataset(windows, input_len, target_len,
                                     cfg["_repo_root"], cond_ch, mask_night,
                                     latent_h, latent_w)

    return ds(tr_rows, unroll), ds(va_rows, 1), ds(te_rows, 1)
