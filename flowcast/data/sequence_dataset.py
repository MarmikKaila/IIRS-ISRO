"""FlowCast-style sequence dataset: lag=13 past + lead=12 future windows.

Reuses the contiguous-runs / temporal-split logic from the original
src/data/sequence_dataset.py but with paper defaults (input_len=13,
joint_steps=12) and VIS day/night handling preserved (we still filter to
contiguous daytime runs for VIS so the seeded model never has to bridge the
night gap).

Each window is a 25-frame slice of cached latents:
    x_past    : (T_past, 4, H, W)         T_past = input_len, default 13
    y_future  : (T_lead, 4, H, W)         T_lead = joint_steps, default 12
    sza_past  : (T_past, 1, H, W)         optional, VIS only
    sza_lead  : (T_lead, 1, H, W)         optional, VIS only
    weight    : (T_lead,)                 1.0 day / 0.0 night, VIS only
"""
import csv
import datetime as dt
import os

import torch
from torch.utils.data import Dataset


# ── manifest ────────────────────────────────────────────────────────────────
def read_manifest(path):
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
    n = len(rows)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


def contiguous_runs(rows, cadence_minutes=30, daytime_only=False):
    """Time-sorted rows -> runs with no missing slots. ``daytime_only`` is the
    VIS path (drops night rows first so the daytime run is itself contiguous)."""
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


def make_windows(runs, input_len, target_len):
    need = input_len + target_len
    out = []
    for run in runs:
        for s in range(len(run) - need + 1):
            out.append((run, s))
    return out


# ── dataset ─────────────────────────────────────────────────────────────────
class FlowCastLatentDataset(Dataset):
    """Yields ``(past_lat, sza_past, future_lat, sza_future, weight)``.

    ``sza_past`` / ``sza_future`` are empty tensors when ``cond_ch == 0`` (WV).
    ``weight`` is all-ones unless ``mask_night`` is on (VIS).
    """

    def __init__(self, windows, input_len, target_len, repo_root,
                 cond_ch=0, mask_night=False, latent_h=36, latent_w=32):
        self.windows = windows
        self.T_past = input_len
        self.T_lead = target_len
        self.repo_root = repo_root
        self.cond_ch = cond_ch
        self.mask_night = mask_night
        self.H = latent_h
        self.W = latent_w
        self._cache = {}

    def _load(self, rel):
        if rel not in self._cache:
            full = rel if os.path.isabs(rel) else os.path.join(self.repo_root, rel)
            self._cache[rel] = torch.load(full, map_location="cpu").float()
        return self._cache[rel]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        run, s = self.windows[i]
        T = self.T_past + self.T_lead
        latents = [self._load(run[s + k]["latent_path"]) for k in range(T)]
        past = torch.stack(latents[:self.T_past], 0)   # (T_past, 4, H, W)
        fut  = torch.stack(latents[self.T_past:], 0)   # (T_lead, 4, H, W)

        if self.cond_ch > 0:
            sza_p = torch.tensor(
                [run[s + k]["sza_norm"] for k in range(self.T_past)],
                dtype=torch.float32)
            sza_f = torch.tensor(
                [run[s + self.T_past + k]["sza_norm"]
                 for k in range(self.T_lead)], dtype=torch.float32)
            sza_p = sza_p.view(-1, 1, 1, 1).expand(-1, 1, self.H, self.W).contiguous()
            sza_f = sza_f.view(-1, 1, 1, 1).expand(-1, 1, self.H, self.W).contiguous()
        else:
            sza_p = torch.zeros(self.T_past, 0, self.H, self.W)
            sza_f = torch.zeros(self.T_lead, 0, self.H, self.W)

        if self.mask_night:
            w = torch.tensor(
                [float(run[s + self.T_past + k].get("is_day", True))
                 for k in range(self.T_lead)], dtype=torch.float32)
        else:
            w = torch.ones(self.T_lead, dtype=torch.float32)

        return past, sza_p, fut, sza_f, w


def build_datasets(cfg):
    """Train / val / test datasets keyed off ``cfg["sequence"]`` and
    ``cfg["flow"]``. Reuses the channel manifest produced by Phase 1 encode.
    """
    from ..utils.config import resolve
    from ..utils.solar import solar_elevation_deg

    rows = read_manifest(resolve(cfg, "manifest_path"))
    cadence = cfg["sequence"]["cadence_minutes"]
    target_len = int(cfg["flow"].get("joint_steps", 12))

    dn = cfg.get("day_night", {})
    cond_ch_dataset = 1 if dn.get("enabled", False) else 0
    mask_night = bool(dn.get("enabled", False))
    daytime_only = bool(dn.get("daytime_runs_only", dn.get("enabled", False)))

    if cond_ch_dataset:
        lat, lon = dn["scene_lat"], dn["scene_lon"]
        for r in rows:
            elev = solar_elevation_deg(r["utc"], lat, lon)
            r["sza_norm"] = max(-1.0, min(1.0, elev / 90.0))

    tr_rows, va_rows, te_rows = temporal_split(
        rows,
        cfg["train"]["val_fraction"],
        cfg["train"]["test_fraction"])

    # ── Auto-shrink input_len if the requested 25-frame window (paper's 13+12)
    # doesn't fit in the contiguous runs. VIS daytime runs are ~22 frames
    # (sunrise-sunset at 30-min cadence), so we walk input_len down until the
    # train split has at least 10 windows. Mutates cfg so build_model picks up
    # the same value the dataset is using.
    requested = int(cfg["sequence"].get("input_len", 13))
    tr_runs = contiguous_runs(tr_rows, cadence, daytime_only=daytime_only)
    input_len = None
    for cand in range(requested, 0, -1):
        if len(make_windows(tr_runs, cand, target_len)) >= 10:
            input_len = cand
            break
    if input_len is None:
        raise SystemExit(
            f"[fc-dataset] cannot form a single training window with "
            f"target_len={target_len} even at input_len=1. "
            f"Lower flow.joint_steps or check the manifest.")
    if input_len != requested:
        print(f"[fc-dataset] auto-shrunk input_len {requested} -> {input_len} "
              f"(contiguous {'daytime ' if daytime_only else ''}runs are too "
              f"short for the requested window)")
        cfg["sequence"]["input_len"] = input_len

    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8

    def ds(split):
        runs = contiguous_runs(split, cadence, daytime_only=daytime_only)
        windows = make_windows(runs, input_len, target_len)
        return FlowCastLatentDataset(
            windows, input_len, target_len, cfg["_repo_root"],
            cond_ch_dataset, mask_night, latent_h, latent_w)

    return ds(tr_rows), ds(va_rows), ds(te_rows)
