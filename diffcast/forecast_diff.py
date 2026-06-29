"""diffcast Phase 5 — 48-frame forecast via Heun ODE sampling + VAE decode.

Run:  python -m diffcast.forecast_diff --config vis/config_diff.yaml [--samples 8]
      python -m diffcast.forecast_diff --config wv/config_diff.yaml  --samples 8

Mirrors flowcast/forecast_flow.py with two surgical changes:
  1. Sampler: Heun ODE over Karras sigma schedule (sampler_steps Euler+Heun
     stages per autoregressive block) instead of CFM's 10-step Euler.
  2. VAE: fine-tuned full VAE from diffcast.models.vae (cfg["vae"]["full_ckpt"])
     instead of frozen SEVIR.

Three-way ensemble aggregation (pixmean / medoid / latmean), VIS night-fill,
threshold metrics, and 3-mode output grids are identical to flowcast's pattern.
"""
import argparse
import datetime as dt
import json
import os
from collections import Counter

import numpy as np
import torch

from flowcast.data import tif_io
from flowcast.data.sequence_dataset import read_manifest
from flowcast.utils import metrics, viz
from flowcast.utils.config import load_config, resolve
from flowcast.utils.device import get_device
from flowcast.utils.solar import is_daytime, utc_to_ist
from flowcast.utils.threshold_metrics import (DEFAULT_THRESHOLDS,
                                                categorical_M, crps_gaussian)

from .models import vae as vae_mod
from .models.diffusion_model import build_diffusion_model
from .models.edm import karras_sigma_schedule


def pick_forecast_date(rows, arg_date):
    if arg_date:
        return dt.datetime.strptime(arg_date, "%Y%m%d").date()
    counts = Counter(r["utc"].date() for r in rows)
    for d in sorted(counts, reverse=True):
        if counts[d] >= 24:
            return d
    return max(counts)


def main():
    ap = argparse.ArgumentParser(
        description="diffcast Phase 5: 48-frame EDM diffusion forecast.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", default=None, help="forecast date YYYYMMDD UTC")
    ap.add_argument("--samples", type=int, default=8,
                    help="ensemble size (default 8)")
    ap.add_argument("--sampler-steps", type=int, default=None,
                    help="override diffusion.sampler_steps from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    # ── load checkpoint + mirror training-time dims (same pattern as flowcast) ─
    ckpt_path = os.path.join(resolve(cfg, "checkpoint_dir"), "best_diff.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    saved = ckpt.get("config") or {}
    if saved.get("diffusion"):
        for k in ("hidden", "n_blocks", "attn_heads", "dropout",
                   "joint_steps", "sigma_data", "P_mean", "P_std",
                   "sampler_steps", "sampler_rho", "sigma_min", "sigma_max"):
            if k in saved["diffusion"]:
                cfg["diffusion"][k] = saved["diffusion"][k]
    if saved.get("sequence") and "input_len" in saved["sequence"]:
        cfg["sequence"]["input_len"] = saved["sequence"]["input_len"]
    if "T_past" in ckpt:
        cfg["sequence"]["input_len"] = int(ckpt["T_past"])
    if "T_lead" in ckpt:
        cfg["diffusion"]["joint_steps"] = int(ckpt["T_lead"])

    horizon = int(cfg["forecast"].get("horizon", 48))
    T_lead = int(cfg["diffusion"]["joint_steps"])
    T_past = int(cfg["sequence"]["input_len"])
    cadence = cfg["sequence"]["cadence_minutes"]
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]
    sza_enabled = bool(dn.get("enabled", False))
    src_dir = resolve(cfg, "source_dir")
    sampler_steps = int(args.sampler_steps if args.sampler_steps
                         else cfg["diffusion"].get("sampler_steps", 18))

    rows = read_manifest(resolve(cfg, "manifest_path"))
    by_utc = {r["utc"]: r for r in rows}

    date = pick_forecast_date(rows, args.date)
    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step for i in range(horizon)]
    print(f"[diffcast-fc] channel={cfg['channel']}  date={date}  "
          f"horizon={horizon}  T_past={T_past}  T_lead={T_lead}  "
          f"samples={args.samples}  sampler_steps={sampler_steps}")
    print(f"[diffcast-fc] mirroring ckpt dims: "
          f"hidden={cfg['diffusion']['hidden']}  "
          f"n_blocks={cfg['diffusion']['n_blocks']}")

    # ── build model + load weights ──────────────────────────────────────────
    latent_h = H // 8
    latent_w = W // 8
    model = build_diffusion_model(cfg, latent_ch=4,
                                    latent_h=latent_h, latent_w=latent_w).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # ── load fine-tuned full VAE (encoder needed for seed encode + decoder for output)
    vae = vae_mod.load_vae(cfg, device, trainable=False)

    def load_lat(rel):
        return torch.load(os.path.join(repo, rel), map_location="cpu").float()

    def slot_is_day(ts):
        if not sza_enabled: return True
        return is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                           dn["elevation_threshold_deg"])
    day_flags = [slot_is_day(ts) for ts in slots]

    # ── seed window: last T_past real frames before day_start ────────────────
    prior = [r for r in rows if r["utc"] < day_start]
    if len(prior) < T_past:
        raise SystemExit(f"[diffcast-fc] need {T_past} prior frames, have {len(prior)}")
    seed = torch.stack([load_lat(r["latent_path"]) for r in prior[-T_past:]], dim=0)
    seed = seed[None].to(device)                            # (1, T_past, 4, Hl, Wl)

    # ── autoregressive rollout ──────────────────────────────────────────────
    n_blocks = (horizon + T_lead - 1) // T_lead
    sigmas = karras_sigma_schedule(
        sampler_steps,
        sigma_min=float(cfg["diffusion"].get("sigma_min", 0.002)),
        sigma_max=float(cfg["diffusion"].get("sigma_max", 80.0)),
        rho=float(cfg["diffusion"].get("sampler_rho", 7.0)),
        device=device, dtype=seed.dtype,
    )

    @torch.no_grad()
    def one_rollout():
        past = seed.clone()
        out = []
        for b in range(n_blocks):
            fut = model.sample(past, sigmas)              # (1, T_lead, 4, Hl, Wl)
            out.append(fut)
            past = torch.cat([past, fut], dim=1)[:, -T_past:]
        full = torch.cat(out, dim=1)
        return full[:, :horizon]

    @torch.no_grad()
    def decode_block(lat_btchw, chunk=4):
        """(1, horizon, 4, Hl, Wl) -> [physical images] of length horizon."""
        B, T, C, Hl, Wl = lat_btchw.shape
        x = lat_btchw.reshape(B * T, C, Hl, Wl).to(device)
        parts = []
        for i in range(0, x.shape[0], chunk):
            out01 = vae_mod.decode(vae, x[i:i + chunk]).cpu().numpy()
            parts.append(out01)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        recon01 = np.concatenate(parts, axis=0)            # (T, 1, H, W)
        return [vae_mod.destandardize(recon01[i, 0], low, high)
                for i in range(T)]

    # ── run N rollouts, keep both decoded images AND latents per sample ─────
    sample_imgs = []
    sample_lats = []
    for s in range(args.samples):
        pred_lat = one_rollout()
        imgs = decode_block(pred_lat)
        sample_imgs.append(imgs)
        sample_lats.append(pred_lat.detach().cpu())
        del pred_lat
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[diffcast-fc] sample {s + 1}/{args.samples} done")

    # ── 3-way aggregation: pixmean / medoid / latmean ────────────────────────
    ensemble_stack = np.stack([np.stack(s) for s in sample_imgs])  # (N, T, H, W)
    pixmean_imgs = list(ensemble_stack.mean(axis=0).astype(np.float32))
    if args.samples == 1:
        modes = {"pixmean": pixmean_imgs}
        medoid_idx = 0
    else:
        pm_flat = ensemble_stack.mean(axis=0).reshape(-1)
        flat = ensemble_stack.reshape(args.samples, -1)
        dists = ((flat - pm_flat[None]) ** 2).sum(axis=1)
        medoid_idx = int(np.argmin(dists))
        medoid_imgs = list(np.asarray(sample_imgs[medoid_idx], dtype=np.float32))
        lat_mean = torch.stack(sample_lats).mean(dim=0)
        latmean_imgs = decode_block(lat_mean)
        modes = {"pixmean": pixmean_imgs,
                 "medoid":  medoid_imgs,
                 "latmean": latmean_imgs}
        print(f"[diffcast-fc] 3-way aggregation: pixmean / medoid (s={medoid_idx}) / latmean")

    # ── VIS night fill (applied to every mode + ensemble_stack) ─────────────
    if sza_enabled:
        night_rows = [r for r in rows if r.get("is_day", True) is False]
        smp = night_rows[::max(1, len(night_rows) // 30)][:30]
        acc = np.zeros((H, W), np.float64)
        for r in smp:
            acc += tif_io.resize_image(
                tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        night_fill = (acc / max(len(smp), 1)).astype(np.float32)
        n_night = 0
        for i in range(horizon):
            if not day_flags[i]:
                for imgs in modes.values():
                    imgs[i] = night_fill
                ensemble_stack[:, i] = night_fill[None]
                n_night += 1
        print(f"[diffcast-fc] VIS: {horizon - n_night} predicted + "
              f"{n_night} dark fill")
    else:
        print(f"[diffcast-fc] WV: {horizon} slots predicted")

    # ── metrics vs ground truth (per mode; CRPS once on full ensemble) ──────
    channel_thr = DEFAULT_THRESHOLDS.get(cfg["channel"], [])
    gt_cache = {}
    for i, ts in enumerate(slots):
        r = by_utc.get(ts)
        if r is None or (sza_enabled and not day_flags[i]):
            continue
        gt_cache[i] = tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)

    summary_keys = ["mse", "psnr", "ssim",
                     "CSI_M", "HSS_M", "FAR_M", "CSI_P16_M", "FSS_P16_M"]

    def _score(imgs):
        per = []
        for i, gt in gt_cache.items():
            pix = metrics.all_metrics(imgs[i], gt, data_range=drange)
            cat = categorical_M(imgs[i], gt, channel_thr) if channel_thr else {}
            per.append({"step": i, "utc": slots[i].isoformat(), **pix, **cat})
        smry = {k: float(np.mean([p[k] for p in per if k in p]))
                for k in summary_keys
                if any(k in p for p in per)}
        return per, smry

    per_frame_by_mode = {}
    summary_by_mode = {}
    for name, imgs in modes.items():
        per_frame_by_mode[name], summary_by_mode[name] = _score(imgs)

    crps_per_frame = []
    for i, gt in gt_cache.items():
        crps_per_frame.append({"step": i, "utc": slots[i].isoformat(),
                                "CRPS": crps_gaussian(ensemble_stack[:, i], gt,
                                                       data_range=drange)})
    crps_mean = (float(np.mean([p["CRPS"] for p in crps_per_frame]))
                 if crps_per_frame else None)

    # ── save outputs ─────────────────────────────────────────────────────────
    out_dir = resolve(cfg, "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    base_tag = date.strftime("%Y%m%d")
    if args.samples > 1:
        base_tag = f"{base_tag}_ens{args.samples}"

    titles = [f"{utc_to_ist(ts):%H:%M} IST{'' if day_flags[i] else ' (night)'}"
              for i, ts in enumerate(slots)]

    for name, imgs in modes.items():
        suffix = f"_{name}" if args.samples > 1 else ""
        np.save(os.path.join(out_dir, f"diff_forecast_{base_tag}{suffix}.npy"),
                np.stack(imgs).astype(np.float32))
        viz.save_image_grid(
            imgs, os.path.join(out_dir, f"diff_forecast_{base_tag}{suffix}_grid.png"),
            titles=titles, ncols=8, vmin=low, vmax=high)
        viz.save_gif(imgs,
                       os.path.join(out_dir, f"diff_forecast_{base_tag}{suffix}.gif"),
                       vmin=low, vmax=high)
        cmp_pred, cmp_gt, cmp_titles = [], [], []
        for i in gt_cache:
            cmp_pred.append(imgs[i])
            cmp_gt.append(gt_cache[i])
            cmp_titles.append(f"{utc_to_ist(slots[i]):%H:%M}")
        if cmp_pred:
            keep = np.linspace(0, len(cmp_pred) - 1, min(8, len(cmp_pred))).astype(int)
            viz.save_comparison_grid(
                [cmp_pred[k] for k in keep], [cmp_gt[k] for k in keep],
                os.path.join(out_dir, f"diff_forecast_{base_tag}{suffix}_compare.png"),
                titles=[cmp_titles[k] for k in keep], vmin=low, vmax=high)

    with open(os.path.join(out_dir, f"diff_forecast_{base_tag}_metrics.json"), "w") as fh:
        json.dump({"date": base_tag, "channel": cfg["channel"],
                    "samples": args.samples, "medoid_idx": medoid_idx,
                    "sampler_steps": sampler_steps,
                    "thresholds": channel_thr,
                    "summary_by_mode": summary_by_mode,
                    "crps_mean": crps_mean,
                    "per_frame_by_mode": per_frame_by_mode,
                    "crps_per_frame": crps_per_frame}, fh, indent=2)

    n_scored = len(gt_cache)
    print(f"[diffcast-fc] vs GT ({n_scored} frames):")
    for name, smry in summary_by_mode.items():
        if not smry:
            continue
        bits = "  ".join(f"{k}={v:.3f}" for k, v in smry.items()
                          if k in ("psnr", "ssim", "CSI_M"))
        print(f"  [{name:>7}] {bits}")
    if crps_mean is not None:
        print(f"  [   crps] mean={crps_mean:.4f} (ensemble of {args.samples})")
    print(f"[diffcast-fc] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
