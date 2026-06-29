"""FlowCast Stage 1 — encode INSAT TIFFs into FlowCast-VAE latents.

Run:  python -m flowcast.encode_latents --config vis/config_fc.yaml
      python -m flowcast.encode_latents --config wv/config_fc.yaml

Writes one ``.pt`` per frame under ``<ch>/latents_fc/`` and a manifest at
``<ch>/manifest_fc.csv``. The VAE is the paper's released SEVIR AutoencoderKL,
loaded frozen from b-rbmp/flowcast-cfm-sevir on HF Hub.

Standardisation: VIS / WV are clipped to (low, high) physical-unit percentiles
already stored in vis/config.yaml / wv/config.yaml; we map that range to
[0, 1] before encoding (matches FlowCast's normalize_dataset=true).
"""
import argparse
import csv
import os

import torch
from tqdm import tqdm

from .data import tif_io
from .utils import flowcast_vae
from .utils.config import load_config, resolve, save_config
from .utils.device import get_device
from .utils.solar import is_daytime, utc_to_ist

MANIFEST_FIELDS = ["utc", "ist", "filename", "latent_path", "is_day"]


def main():
    ap = argparse.ArgumentParser(description="FlowCast Stage 1: cache latents.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    files = tif_io.list_files(resolve(cfg, "source_dir"), cfg["file_glob"])
    if args.limit:
        files = files[: args.limit]
    print(f"[fc-encode] channel={cfg['channel']}  files={len(files)}")

    # 1. normalization stats — REUSE the SD-VAE pipeline's stats (saved in the
    #    main config). FlowCast only needs the range (low, high) -> [0, 1].
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    if low is None or high is None:
        low, high = tif_io.compute_norm_stats(files)
        cfg["norm"]["low"], cfg["norm"]["high"] = low, high
        save_config(cfg)
        print(f"[fc-encode] norm low={low:.4f} high={high:.4f} (saved to config)")

    # 2. VAE
    vae = flowcast_vae.load_flowcast_vae(cfg, device)

    latent_dir = resolve(cfg, "latent_dir")
    os.makedirs(latent_dir, exist_ok=True)
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]

    rows, batch_imgs, batch_paths = [], [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.cat(batch_imgs).to(device)             # (B, 1, H, W) in [0, 1]
        z = flowcast_vae.encode(vae, x).cpu()             # (B, 4, H/8, W/8)
        for k, abspath in enumerate(batch_paths):
            torch.save(z[k].clone(), abspath)
        batch_imgs.clear()
        batch_paths.clear()

    for path in tqdm(files, desc=f"fc-encode-{cfg['channel']}"):
        ts = tif_io.parse_timestamp(path)
        arr = tif_io.resize_image(tif_io.read_tif(path), H, W)
        x01 = flowcast_vae.standardize(arr, low, high)
        img = flowcast_vae.to_vae_input(x01)             # (1, 1, H, W)

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
    print(f"[fc-encode] cached {len(rows)} latents -> {latent_dir}")
    if dn["enabled"]:
        print(f"[fc-encode] daytime frames: {n_day}/{len(rows)}")
    print(f"[fc-encode] manifest -> {mpath}")


if __name__ == "__main__":
    main()
