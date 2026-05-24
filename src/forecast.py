"""Stage 4 — produce a 48-frame day forecast and decode it to images.

Run:  python -m src.forecast --config vis/config.yaml [--date YYYYMMDD]
      python -m src.forecast --config wv/config.yaml  [--date YYYYMMDD]

WV  : the ConvLSTM rolls out all 48 latents.
VIS : the ConvLSTM rolls out only the daytime slots; night slots (sun below the
      horizon) are filled with the mean of real VIS night frames. The two are
      assembled into the full 48-frame day.
"""
import argparse
import datetime as dt
import json
import os
from collections import Counter

import numpy as np
import torch

from .data import tif_io
from .data.sequence_dataset import read_manifest
from .models.convlstm import build_model
from .utils import metrics, viz
from .utils.config import load_config, resolve
from .utils.device import get_device
from .utils.sdvae import decode, load_vae, scaling_factor
from .utils.solar import is_daytime, solar_elevation_deg, utc_to_ist


def pick_forecast_date(rows, arg_date):
    """Forecast date: explicit --date, else the last day with >= 24 frames."""
    if arg_date:
        return dt.datetime.strptime(arg_date, "%Y%m%d").date()
    counts = Counter(r["utc"].date() for r in rows)
    for d in sorted(counts, reverse=True):
        if counts[d] >= 24:
            return d
    return max(counts)


def main():
    ap = argparse.ArgumentParser(description="Stage 4: 48-frame day forecast.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", default=None, help="forecast date YYYYMMDD (UTC)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    T = cfg["sequence"]["input_len"]
    cadence = cfg["sequence"]["cadence_minutes"]
    horizon = cfg["forecast"]["horizon"]
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]
    src_dir = resolve(cfg, "source_dir")

    rows = read_manifest(resolve(cfg, "manifest_path"))
    by_utc = {r["utc"]: r for r in rows}

    date = pick_forecast_date(rows, args.date)
    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step for i in range(horizon)]
    print(f"[forecast] channel={cfg['channel']}  date={date}  horizon={horizon}")

    # model + VAE ──────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(resolve(cfg, "checkpoint_dir"), "best.pt")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    vae = load_vae(cfg, device)
    sf = scaling_factor(cfg, vae)

    def load_latent(rel):
        return torch.load(os.path.join(repo, rel), map_location="cpu").float()

    @torch.no_grad()
    def decode_latents(latents):
        imgs = []
        for lat in latents:
            norm = tif_io.from_vae_output(decode(vae, lat[None].to(device), sf))
            imgs.append(tif_io.denormalize(norm, low, high))
        return imgs

    def slot_is_day(ts):
        if not dn["enabled"]:
            return True
        return is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                          dn["elevation_threshold_deg"])

    day_flags = [slot_is_day(ts) for ts in slots]

    # latent spatial dims and conditioning channels (VIS adds an SZA channel)
    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8
    cond_ch = int(getattr(model, "cond_ch", 0))

    def sza_plane(ts):
        elev = solar_elevation_deg(ts, dn["scene_lat"], dn["scene_lon"])
        v = max(-1.0, min(1.0, elev / 90.0))
        return torch.full((1, latent_h, latent_w), v, dtype=torch.float32)

    # ── seed with the last T real frames before the day, roll out all 48 ──────
    prior = [r for r in rows if r["utc"] < day_start]
    if len(prior) < T:
        raise SystemExit(f"[forecast] need {T} prior frames, have {len(prior)}")
    seed_rows = prior[-T:]
    seed_lat = torch.stack(
        [load_latent(r["latent_path"]) for r in seed_rows], dim=0)    # (T,4,H,W)

    if cond_ch:
        seed_sza = torch.stack([sza_plane(r["utc"]) for r in seed_rows], dim=0)
        x_in = torch.cat([seed_lat, seed_sza], dim=1)[None].to(device)  # (1,T,5,H,W)
        future_cond = torch.stack(
            [sza_plane(ts) for ts in slots], dim=0)[None].to(device)    # (1,horizon,1,H,W)
    else:
        x_in = seed_lat[None].to(device)
        future_cond = None

    with torch.no_grad():
        pred_latents = model(x_in, future_steps=horizon,
                             future_cond=future_cond)[0].cpu()         # (horizon,4,H,W)
    forecast_imgs = decode_latents(pred_latents)

    # ── VIS: overwrite night slots with a fixed dark frame ────────────────────
    if dn["enabled"]:
        night_rows = [r for r in rows if r.get("is_day", True) is False]
        sample = night_rows[::max(1, len(night_rows) // 30)][:30]
        acc = np.zeros((H, W), np.float64)
        for r in sample:
            acc += tif_io.resize_image(
                tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        night_fill = (acc / max(len(sample), 1)).astype(np.float32)
        n_night = 0
        for i in range(horizon):
            if not day_flags[i]:
                forecast_imgs[i] = night_fill
                n_night += 1
        print(f"[forecast] VIS: {horizon} slots predicted, "
              f"{n_night} night slots overwritten with dark fill")
    else:
        print(f"[forecast] WV: {horizon} slots predicted")

    # ── metrics vs ground truth (VIS: daytime slots only) ─────────────────────
    per_frame = []
    for i, ts in enumerate(slots):
        r = by_utc.get(ts)
        if r is None or (dn["enabled"] and not day_flags[i]):
            continue
        gt = tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        m = metrics.all_metrics(forecast_imgs[i], gt, data_range=drange)
        per_frame.append({"step": i, "utc": ts.isoformat(), **m})
    summary = ({k: float(np.mean([p[k] for p in per_frame]))
                for k in ("mse", "psnr", "ssim")} if per_frame else {})

    # per-horizon buckets so we can see how the rollout degrades over the day
    by_horizon = {}
    for label, lo, hi in [("0-11", 0, 11), ("12-23", 12, 23),
                          ("24-35", 24, 35), ("36-47", 36, 47)]:
        bucket = [p for p in per_frame if lo <= p["step"] <= hi]
        if bucket:
            by_horizon[label] = {
                k: float(np.mean([p[k] for p in bucket]))
                for k in ("mse", "psnr", "ssim")
            }

    # ── save outputs ──────────────────────────────────────────────────────────
    out_dir = resolve(cfg, "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    tag = date.strftime("%Y%m%d")
    stack = np.stack(forecast_imgs).astype(np.float32)
    np.save(os.path.join(out_dir, f"forecast_{tag}.npy"), stack)

    titles = [f"{utc_to_ist(ts):%H:%M} IST{'' if day_flags[i] else ' (night)'}"
              for i, ts in enumerate(slots)]
    viz.save_image_grid(forecast_imgs, os.path.join(out_dir, f"forecast_{tag}_grid.png"),
                        titles=titles, ncols=8, vmin=low, vmax=high)
    viz.save_gif(forecast_imgs, os.path.join(out_dir, f"forecast_{tag}.gif"),
                 vmin=low, vmax=high)

    # predicted-vs-truth comparison for available slots
    cmp_pred, cmp_gt, cmp_titles = [], [], []
    for i, ts in enumerate(slots):
        r = by_utc.get(ts)
        if r is None or (dn["enabled"] and not day_flags[i]):
            continue
        cmp_pred.append(forecast_imgs[i])
        cmp_gt.append(tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W))
        cmp_titles.append(f"{utc_to_ist(ts):%H:%M}")
    if cmp_pred:
        keep = np.linspace(0, len(cmp_pred) - 1, min(8, len(cmp_pred))).astype(int)
        viz.save_comparison_grid(
            [cmp_pred[k] for k in keep], [cmp_gt[k] for k in keep],
            os.path.join(out_dir, f"forecast_{tag}_compare.png"),
            titles=[cmp_titles[k] for k in keep], vmin=low, vmax=high)

    with open(os.path.join(out_dir, f"forecast_{tag}_metrics.json"), "w") as f:
        json.dump({"date": tag, "channel": cfg["channel"],
                   "summary": summary, "by_horizon": by_horizon,
                   "per_frame": per_frame}, f, indent=2)

    if summary:
        print(f"[forecast] vs ground truth ({len(per_frame)} frames): "
              f"PSNR={summary['psnr']:.2f} dB  SSIM={summary['ssim']:.3f}  "
              f"MSE={summary['mse']:.4f}")
        for label, m in by_horizon.items():
            print(f"[forecast]   steps {label:>6}: "
                  f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.3f}")
    print(f"[forecast] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
