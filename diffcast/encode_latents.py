"""diffcast Phase 2b — encode all INSAT frames to new latents.

Run:  python -m diffcast.encode_latents --config vis/config_diff.yaml

Loads the fine-tuned full VAE (via cfg["vae"]["full_ckpt"]) and runs the
encoder on every TIFF in the channel's source_dir. Writes one ``.pt`` per
frame to ``<latent_dir>/`` and (re)writes the diffcast manifest at
``<manifest_path>``.

These latents will have DIFFERENT numerical values than flowcast's
``latents_fc/`` because the encoder is INSAT-fine-tuned; they live at a
completely separate path so flowcast's cached latents are untouched.
"""
import argparse
import csv
import os

import torch
from tqdm import tqdm

from flowcast.data import tif_io
from flowcast.utils.config import load_config, resolve, save_config
from flowcast.utils.device import get_device
from flowcast.utils.solar import is_daytime, utc_to_ist

from .models import vae as vae_mod

MANIFEST_FIELDS = ["utc", "ist", "filename", "latent_path", "is_day"]


def main():
    ap = argparse.ArgumentParser(description="diffcast Phase 2b: encode latents.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    if not cfg.get("vae", {}).get("full_ckpt"):
        print("[diffcast-encode] WARNING: cfg.vae.full_ckpt is not set — "
              "you will be encoding with the raw SEVIR VAE, not your "
              "fine-tuned one. Did you forget to uncomment after train_vae?")

    files = tif_io.list_files(resolve(cfg, "source_dir"), cfg["file_glob"])
    if args.limit:
        files = files[: args.limit]
    print(f"[diffcast-encode] channel={cfg['channel']}  files={len(files)}")

    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    if low is None or high is None:
        low, high = tif_io.compute_norm_stats(files)
        cfg["norm"]["low"], cfg["norm"]["high"] = low, high
        save_config(cfg)
        print(f"[diffcast-encode] norm low={low:.4f} high={high:.4f} (saved to config)")

    vae = vae_mod.load_vae(cfg, device, trainable=False)

    latent_dir = resolve(cfg, "latent_dir")
    os.makedirs(latent_dir, exist_ok=True)
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]

    rows, batch_imgs, batch_paths = [], [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.cat(batch_imgs).to(device)              # (B, 1, H, W) in [0, 1]
        z = vae_mod.encode_mode(vae, x).cpu()             # (B, 4, H/8, W/8)
        for k, abspath in enumerate(batch_paths):
            torch.save(z[k].clone(), abspath)
        batch_imgs.clear()
        batch_paths.clear()

    for path in tqdm(files, desc=f"diffcast-encode-{cfg['channel']}"):
        ts = tif_io.parse_timestamp(path)
        arr = tif_io.resize_image(tif_io.read_tif(path), H, W)
        x01 = vae_mod.standardize(arr, low, high)
        img = vae_mod.to_vae_input(x01)                   # (1, 1, H, W)

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

    rows.sort(key=lambda r: r["utc"])
    mpath = resolve(cfg, "manifest_path")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_day = sum(1 for r in rows if r["is_day"] is True)
    print(f"[diffcast-encode] cached {len(rows)} latents -> {latent_dir}")
    if dn["enabled"]:
        print(f"[diffcast-encode] daytime frames: {n_day}/{len(rows)}")
    print(f"[diffcast-encode] manifest -> {mpath}")


if __name__ == "__main__":
    main()
