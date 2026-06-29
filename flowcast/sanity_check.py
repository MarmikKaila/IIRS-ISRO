"""FlowCast Stage 2 — decode cached FlowCast-VAE latents and score them.

Run:  python -m flowcast.sanity_check --config vis/config_fc.yaml [--num 20]

Decision gate (per plan): expect roundtrip PSNR >= 25 dB. The FlowCast VAE
was trained on SEVIR radar (single-channel like INSAT but different physics);
INSAT is out-of-distribution, so reconstruction quality is the first thing
to verify before sinking GPU time into the flow model.
"""
import argparse
import json
import os

import numpy as np
import torch

from .data import tif_io
from .data.sequence_dataset import read_manifest
from .utils import flowcast_vae, metrics, viz
from .utils.config import load_config, resolve
from .utils.device import get_device


def main():
    ap = argparse.ArgumentParser(description="FlowCast Stage 2: VAE roundtrip.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()
    vae = flowcast_vae.load_flowcast_vae(cfg, device)

    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    if low is None:
        raise SystemExit("[fc-sanity] norm stats missing — run encode_latents first")
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")

    rows = read_manifest(resolve(cfg, "manifest_path"))
    pool = [r for r in rows if r.get("is_day", True) is not False] or rows
    pick_idx = np.linspace(0, len(pool) - 1, min(args.num, len(pool))).astype(int)
    picked = [pool[i] for i in pick_idx]

    preds, gts, titles, per_frame = [], [], [], []
    for r in picked:
        latent = torch.load(os.path.join(repo, r["latent_path"]),
                             map_location="cpu").float()[None].to(device)
        recon01 = flowcast_vae.from_vae_output(flowcast_vae.decode(vae, latent))
        recon = flowcast_vae.destandardize(recon01, low, high)
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

    print(f"[fc-sanity] {cfg['channel']}: mean PSNR={summary['psnr']:.2f} dB  "
          f"SSIM={summary['ssim']:.3f}")
    print(f"[fc-sanity] grid    -> {grid_path}")
    print(f"[fc-sanity] metrics -> {json_path}")
    if summary["psnr"] < 20:
        print("[fc-sanity] WARNING: PSNR < 20 dB — FlowCast VAE is very far "
              "out-of-distribution on this channel; consider fine-tuning or "
              "falling back to the SD-VAE pipeline.")
    elif summary["psnr"] < 25:
        print("[fc-sanity] NOTE: PSNR < 25 dB — OOD VAE; results will inherit "
              "this reconstruction ceiling.")


if __name__ == "__main__":
    main()
