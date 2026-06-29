"""diffcast Phase 2a — sanity check: VAE roundtrip PSNR/SSIM gate.

Run:  python -m diffcast.sanity_check --config vis/config_diff.yaml [--num 30]

Loads the fine-tuned full VAE (via cfg["vae"]["full_ckpt"]) and reports
mean reconstruction PSNR/SSIM over a set of val frames. The decision gate
is ``vae_train.target_psnr`` (default 30 dB) — if the PSNR is below that,
go back to train_vae for more steps before moving on to encode_latents.

Mirrors flowcast/sanity_check.py but reads INSAT TIFFs directly (no
pre-cached latents needed) and uses ``encode_mode`` for deterministic
inference.
"""
import argparse
import json
import os

import numpy as np
import torch

from flowcast.data import tif_io
from flowcast.data.sequence_dataset import read_manifest, temporal_split
from flowcast.utils import metrics, viz
from flowcast.utils.config import load_config, resolve
from flowcast.utils.device import get_device

from .models import vae as vae_mod


def main():
    ap = argparse.ArgumentParser(description="diffcast Phase 2a: VAE roundtrip sanity.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--split", choices=("train", "val", "test"), default="val",
                    help="which split to evaluate on (default: val)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    if low is None:
        raise SystemExit("[diffcast-sanity] norm stats missing — run train_vae first")
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")
    target_psnr = float(cfg.get("vae_train", {}).get("target_psnr", 30.0))

    rows = read_manifest(resolve(cfg, "manifest_path"))
    if cfg["day_night"]["enabled"]:
        rows = [r for r in rows if r.get("is_day", True) is not False]
    tr, va, te = temporal_split(
        rows, cfg["train"]["val_fraction"], cfg["train"]["test_fraction"])
    split = {"train": tr, "val": va, "test": te}[args.split]
    pool = split or rows
    pick_idx = np.linspace(0, len(pool) - 1, min(args.num, len(pool))).astype(int)
    picked = [pool[i] for i in pick_idx]
    print(f"[diffcast-sanity] {cfg['channel']}: scoring {len(picked)} frames "
          f"from the {args.split} split")

    vae = vae_mod.load_vae(cfg, device, trainable=False)

    preds, gts, titles, per_frame = [], [], [], []
    for r in picked:
        gt = tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        gt01 = vae_mod.standardize(gt, low, high)
        x = vae_mod.to_vae_input(gt01).to(device)
        z = vae_mod.encode_mode(vae, x)
        recon01 = vae_mod.from_vae_output(vae_mod.decode(vae, z))
        recon = vae_mod.destandardize(recon01, low, high)
        m = metrics.all_metrics(recon, gt, data_range=drange)
        preds.append(recon); gts.append(gt)
        ist_str = r["ist"][5:16] if "ist" in r else r["utc"][5:16]
        titles.append(f"{ist_str}\nPSNR {m['psnr']:.1f}")
        per_frame.append({"utc": r["utc"].isoformat()
                            if hasattr(r["utc"], "isoformat") else str(r["utc"]),
                          **m})

    summary = {k: float(np.mean([p[k] for p in per_frame]))
               for k in ("mse", "psnr", "ssim")}

    out_dir = resolve(cfg, "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(out_dir, "sanity_grid.png")
    json_path = os.path.join(out_dir, "sanity_metrics.json")
    viz.save_comparison_grid(preds, gts, grid_path, titles=titles,
                              vmin=low, vmax=high)
    with open(json_path, "w") as f:
        json.dump({"summary": summary,
                    "per_frame": per_frame,
                    "split": args.split,
                    "target_psnr": target_psnr,
                    "passed_gate": summary["psnr"] >= target_psnr},
                  f, indent=2)

    print(f"[diffcast-sanity] {cfg['channel']} {args.split}: "
          f"PSNR={summary['psnr']:.2f}  SSIM={summary['ssim']:.3f}")
    print(f"[diffcast-sanity] grid    -> {grid_path}")
    print(f"[diffcast-sanity] metrics -> {json_path}")
    if summary["psnr"] < target_psnr:
        print(f"[diffcast-sanity] GATE FAILED: PSNR {summary['psnr']:.2f} < "
              f"target {target_psnr:.2f} dB. Re-run train_vae with more steps "
              "before proceeding to encode_latents.")
    else:
        print(f"[diffcast-sanity] GATE PASSED (>= {target_psnr:.2f} dB). "
              "Safe to run encode_latents.")


if __name__ == "__main__":
    main()
