"""Stage 3 (flow) — train the latent conditional flow-matching model.

Run:  python -m src.train_flow --config vis/config.yaml
      python -m src.train_flow --config wv/config.yaml

Reuses the cached SD-VAE latents and (for VIS) the SZA conditioning channel
from Stage 1. Single-step training target — autoregression happens at
forecast time. Best weights -> ``<ch>/checkpoints/best_flow.pt``.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.sequence_dataset import (LatentSequenceDataset, contiguous_runs,
                                    make_windows, read_manifest,
                                    temporal_split)
from .models.flow_model import build_flow_model, cfm_loss
from .utils.config import load_config, resolve
from .utils.device import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_flow_datasets(cfg):
    """Like sequence_dataset.build_datasets but with target_len always = 1."""
    from .utils.solar import solar_elevation_deg

    rows = read_manifest(resolve(cfg, "manifest_path"))
    cadence = cfg["sequence"]["cadence_minutes"]
    input_len = cfg["sequence"]["input_len"]
    dn = cfg.get("day_night", {})
    cond_ch_dataset = 1 if dn.get("enabled", False) else 0
    mask_night = bool(dn.get("enabled", False))

    if cond_ch_dataset:
        lat, lon = dn["scene_lat"], dn["scene_lon"]
        for r in rows:
            elev = solar_elevation_deg(r["utc"], lat, lon)
            r["sza_norm"] = max(-1.0, min(1.0, elev / 90.0))

    tr_rows, va_rows, te_rows = temporal_split(
        rows, cfg["train"]["val_fraction"], cfg["train"]["test_fraction"])

    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8

    def ds(split):
        runs = contiguous_runs(split, cadence, daytime_only=False)
        windows = make_windows(runs, input_len, target_len=1)
        return LatentSequenceDataset(windows, input_len, 1, cfg["_repo_root"],
                                     cond_ch_dataset, mask_night,
                                     latent_h, latent_w)

    return ds(tr_rows), ds(va_rows), ds(te_rows)


def make_cond(x, future_cond):
    """Pack past latents (+ SZA) and target SZA into one (B, cond_ch, H, W)."""
    B, T, C, H, W = x.shape
    cond = x.reshape(B, T * C, H, W)
    if future_cond.numel() > 0:
        cond = torch.cat([cond, future_cond[:, 0]], dim=1)   # add target SZA
    return cond


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, future_cond, y, w in loader:
            x = x.to(device)
            future_cond = future_cond.to(device)
            y = y.to(device)
            w = w.to(device)
            cond = make_cond(x, future_cond)
            target = y[:, 0]                                   # (B, 4, H, W)
            loss = cfm_loss(model, target, cond, weights=w[:, 0])
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser(description="Stage 3 (flow): train CFM.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    f = cfg["flow"]
    set_seed(int(f.get("seed", 42)))
    device = get_device()

    train_ds, val_ds, test_ds = build_flow_datasets(cfg)
    print(f"[flow-train] {cfg['channel']}: windows "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("[flow-train] empty split — check manifest / sequence")

    train_loader = DataLoader(train_ds, batch_size=f["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=f["batch_size"])

    # determine cond_ch from one batch
    sample_x, sample_fc, _, _ = next(iter(train_loader))
    cond_ch = make_cond(sample_x, sample_fc).shape[1]
    print(f"[flow-train] cond_ch into model = {cond_ch}")

    model = build_flow_model(cfg, cond_ch=cond_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[flow-train] flow model params: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=f["lr"],
                                 weight_decay=f.get("weight_decay", 0.0))

    ckpt_dir = resolve(cfg, "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_flow.pt")
    epochs = f["epochs"]
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}

    best_val, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        marker = ""
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(),
                        "config": config_snapshot,
                        "epoch": epoch, "val_loss": va,
                        "cond_ch": cond_ch}, best_path)
            marker = "  <- best"
        print(f"[flow-train] epoch {epoch:3d}/{epochs}  "
              f"train={tr:.5f}  val={va:.5f}{marker}")

    with open(os.path.join(ckpt_dir, "flow_history.json"), "w") as fh:
        json.dump({"history": history, "best_val": best_val}, fh, indent=2)
    print(f"[flow-train] best val={best_val:.5f}  -> {best_path}")


if __name__ == "__main__":
    main()
