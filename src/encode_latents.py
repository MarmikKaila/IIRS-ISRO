"""Stage 1 — encode every TIFF into a cached SD-VAE latent.

Run:  python -m src.encode_latents --config vis/config.yaml
      python -m src.encode_latents --config wv/config.yaml

Writes one ``.pt`` per frame into ``<ch>/latents/`` plus ``<ch>/manifest.csv``.
"""
import argparse
import csv
import os

import torch
from tqdm import tqdm

from .data import tif_io
from .utils.config import load_config, resolve, save_config
from .utils.device import get_device
from .utils.sdvae import encode, load_vae, scaling_factor
from .utils.solar import is_daytime, utc_to_ist

MANIFEST_FIELDS = ["utc", "ist", "filename", "latent_path", "is_day"]


def main():
    ap = argparse.ArgumentParser(description="Stage 1: cache SD-VAE latents.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="encode only the first N files (debugging)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    files = tif_io.list_files(resolve(cfg, "source_dir"), cfg["file_glob"])
    if args.limit:
        files = files[:args.limit]
    print(f"[encode] channel={cfg['channel']}  files={len(files)}")
    if not files:
        raise SystemExit("[encode] no input files found — check source_dir/file_glob")

    # 1. normalization stats (computed once, cached into the config) ──────────
    if cfg["norm"]["low"] is None or cfg["norm"]["high"] is None:
        low, high = tif_io.compute_norm_stats(files)
        cfg["norm"]["low"], cfg["norm"]["high"] = low, high
        save_config(cfg)
        print(f"[encode] norm stats low={low:.4f} high={high:.4f} (saved to config)")
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]

    # 2. VAE ──────────────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    sf = scaling_factor(cfg, vae)

    latent_dir = resolve(cfg, "latent_dir")
    os.makedirs(latent_dir, exist_ok=True)
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]

    rows, batch_imgs, batch_paths = [], [], []

    def flush():
        if not batch_imgs:
            return
        latents = encode(vae, torch.cat(batch_imgs).to(device), sf).cpu()
        for k, abspath in enumerate(batch_paths):
            torch.save(latents[k].clone(), abspath)
        batch_imgs.clear()
        batch_paths.clear()

    # 3. encode pass ──────────────────────────────────────────────────────────
    for path in tqdm(files, desc=f"encode-{cfg['channel']}"):
        ts = tif_io.parse_timestamp(path)
        arr = tif_io.resize_image(tif_io.read_tif(path), H, W)
        img = tif_io.to_vae_input(tif_io.normalize(arr, low, high))

        name = f"{cfg['channel']}_{ts:%Y%m%d}_{ts:%H%M}.pt"
        rel = os.path.join(cfg["latent_dir"], name)
        batch_imgs.append(img)
        batch_paths.append(os.path.join(repo, rel))

        is_day = ""
        if dn["enabled"]:
            is_day = is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                                dn["elevation_threshold_deg"])
        rows.append({
            "utc": ts.replace(tzinfo=None).isoformat(),
            "ist": utc_to_ist(ts).replace(tzinfo=None).isoformat(),
            "filename": os.path.basename(path),
            "latent_path": rel,
            "is_day": is_day,
        })
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()

    # 4. manifest ──────────────────────────────────────────────────────────────
    rows.sort(key=lambda r: r["utc"])
    with open(resolve(cfg, "manifest_path"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_day = sum(1 for r in rows if r["is_day"] is True)
    print(f"[encode] cached {len(rows)} latents -> {latent_dir}")
    if dn["enabled"]:
        print(f"[encode] daytime frames: {n_day}/{len(rows)} "
              f"(night={len(rows) - n_day})")
    print(f"[encode] manifest -> {resolve(cfg, 'manifest_path')}")


if __name__ == "__main__":
    main()
