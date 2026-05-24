"""Stage 2 — decode cached latents and score them against ground truth.

Run:  python -m src.sanity_check --config vis/config.yaml [--num 20]

Catches the case where the pretrained SD-VAE is out-of-distribution on INSAT
imagery. Target reconstruction PSNR is ~30 dB; if it is far below, fine-tune
the VAE before training the ConvLSTM.
"""
import argparse
import json
import os

import numpy as np
import torch

from .data import tif_io
from .data.sequence_dataset import read_manifest
from .utils import metrics, viz
from .utils.config import load_config, resolve
from .utils.device import get_device
from .utils.sdvae import decode, load_vae, scaling_factor


def main():
    ap = argparse.ArgumentParser(description="Stage 2: VAE reconstruction sanity check.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int, default=20, help="frames to sample")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()
    vae = load_vae(cfg, device)
    sf = scaling_factor(cfg, vae)

    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    if low is None:
        raise SystemExit("[sanity] norm stats missing — run encode_latents first")
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")

    rows = read_manifest(resolve(cfg, "manifest_path"))
    # for VIS prefer daytime frames — reconstruction PSNR of dark frames is moot
    pool = [r for r in rows if r.get("is_day", True) is not False] or rows
    pick_idx = np.linspace(0, len(pool) - 1, min(args.num, len(pool))).astype(int)
    picked = [pool[i] for i in pick_idx]

    preds, gts, titles, per_frame = [], [], [], []
    for r in picked:
        latent = torch.load(os.path.join(repo, r["latent_path"]),
                             map_location="cpu").float()[None].to(device)
        recon_norm = tif_io.from_vae_output(decode(vae, latent, sf))
        recon = tif_io.denormalize(recon_norm, low, high)
        gt = tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)

        m = metrics.all_metrics(recon, gt, data_range=drange)
        preds.append(recon)
        gts.append(gt)
        titles.append(f"{r['ist'][5:16]}\nPSNR {m['psnr']:.1f}")
        per_frame.append({"utc": r["utc"].isoformat(), **m})

    summary = {k: float(np.mean([p[k] for p in per_frame]))
               for k in ("mse", "psnr", "ssim")}

    out_dir = resolve(cfg, "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    grid_path = os.path.join(out_dir, "sanity_grid.png")
    json_path = os.path.join(out_dir, "sanity_metrics.json")
    viz.save_comparison_grid(preds, gts, grid_path, titles=titles,
                             vmin=low, vmax=high)
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)

    print(f"[sanity] {cfg['channel']}: mean PSNR={summary['psnr']:.2f} dB  "
          f"SSIM={summary['ssim']:.3f}  MSE={summary['mse']:.4f}")
    print(f"[sanity] grid    -> {grid_path}")
    print(f"[sanity] metrics -> {json_path}")
    if summary["psnr"] < 25:
        print("[sanity] WARNING: PSNR < 25 dB — SD-VAE looks out-of-distribution; "
              "consider fine-tuning the VAE before Stage 3.")


if __name__ == "__main__":
    main()
