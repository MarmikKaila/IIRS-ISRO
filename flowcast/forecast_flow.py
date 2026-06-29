"""FlowCast Stage 4 — 48-frame forecast via 4 autoregressive blocks of 12.

Run:  python -m flowcast.forecast_flow --config vis/config_fc.yaml [--samples 8]
      python -m flowcast.forecast_flow --config wv/config_fc.yaml  --samples 8

Sampling: paper's Algorithm 2 — for each ensemble member, run the Earthformer
backbone with 10 Euler steps to generate 12 future latents, slide the past
window forward by 12, repeat for 4 blocks to cover the day. Latents are then
decoded with the same frozen FlowCast VAE, and the per-pixel ensemble mean
is reported alongside the threshold-based meteorological metrics.
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
from .models.flow_model import build_model
from .utils import flowcast_vae, metrics, viz
from .utils.config import load_config, resolve
from .utils.device import get_device
from .utils.solar import is_daytime, utc_to_ist
from .utils.threshold_metrics import (DEFAULT_THRESHOLDS, categorical_M,
                                        crps_gaussian)


def pick_forecast_date(rows, arg_date):
    if arg_date:
        return dt.datetime.strptime(arg_date, "%Y%m%d").date()
    counts = Counter(r["utc"].date() for r in rows)
    for d in sorted(counts, reverse=True):
        if counts[d] >= 24:
            return d
    return max(counts)


def main():
    ap = argparse.ArgumentParser(description="FlowCast Stage 4: 48-frame day forecast.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", default=None, help="forecast date YYYYMMDD UTC")
    ap.add_argument("--samples", type=int, default=8,
                    help="ensemble size (paper default 8)")
    ap.add_argument("--ode-steps", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    # ── load checkpoint FIRST so we can mirror the training-time dims ────────
    ckpt_path = os.path.join(resolve(cfg, "checkpoint_dir"), "best_flow.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Override architectural and sequence params from the saved snapshot. The
    # training run may have auto-shrunk input_len (short VIS daytime runs) and
    # auto-capped hidden / n_blocks / batch_size for the T4. Forecast must
    # rebuild the model with the same dims the saved weights were trained at.
    saved = ckpt.get("config") or {}
    if saved.get("flow"):
        for k in ("hidden", "n_blocks", "attn_heads", "dropout", "t_dim",
                   "joint_steps", "sigma"):
            if k in saved["flow"]:
                cfg["flow"][k] = saved["flow"][k]
    if saved.get("sequence") and "input_len" in saved["sequence"]:
        cfg["sequence"]["input_len"] = saved["sequence"]["input_len"]
    if "T_past" in ckpt:
        cfg["sequence"]["input_len"] = int(ckpt["T_past"])
    if "T_lead" in ckpt:
        cfg["flow"]["joint_steps"] = int(ckpt["T_lead"])

    horizon = int(cfg["forecast"].get("horizon", 48))
    T_lead = int(cfg["flow"]["joint_steps"])
    T_past = int(cfg["sequence"]["input_len"])
    cadence = cfg["sequence"]["cadence_minutes"]
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]
    sza_enabled = bool(dn.get("enabled", False))
    src_dir = resolve(cfg, "source_dir")

    rows = read_manifest(resolve(cfg, "manifest_path"))
    by_utc = {r["utc"]: r for r in rows}

    date = pick_forecast_date(rows, args.date)
    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step for i in range(horizon)]
    print(f"[fc-forecast] channel={cfg['channel']}  date={date}  "
          f"horizon={horizon}  T_past={T_past}  T_lead={T_lead}  "
          f"samples={args.samples}")
    print(f"[fc-forecast] mirroring checkpoint dims: "
          f"hidden={cfg['flow']['hidden']}  n_blocks={cfg['flow']['n_blocks']}")

    # ── build model with the mirrored config + load weights ─────────────────
    latent_h = H // 8
    latent_w = W // 8
    model = build_model(cfg, latent_ch=4, latent_h=latent_h, latent_w=latent_w).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    vae = flowcast_vae.load_flowcast_vae(cfg, device)

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
        raise SystemExit(f"[fc-forecast] need {T_past} prior frames, have {len(prior)}")
    seed = torch.stack([load_lat(r["latent_path"]) for r in prior[-T_past:]], dim=0)
    seed = seed[None].to(device)                # (1, T_past, 4, H, W)

    # ── autoregressive rollout: 4 blocks of T_lead to cover ``horizon`` ─────
    n_blocks = (horizon + T_lead - 1) // T_lead

    @torch.no_grad()
    def one_rollout():
        past = seed.clone()
        out = []
        for b in range(n_blocks):
            fut = model.sample(past, steps=args.ode_steps)        # (1, T_lead, 4, H, W)
            out.append(fut)
            # Slide window: keep the most recent T_past frames from (past + fut).
            # Works whether T_past >= T_lead (mix of real + predicted) or
            # T_past <= T_lead (window becomes entirely predicted).
            past = torch.cat([past, fut], dim=1)[:, -T_past:]
        full = torch.cat(out, dim=1)                                # (1, n_blocks*T_lead, ...)
        return full[:, :horizon]                                     # trim to exact horizon

    @torch.no_grad()
    def decode_block(lat_btchw, chunk=4):
        # lat: (1, horizon, 4, H, W) -> physical images (horizon, H_img, W_img).
        # Decode in small chunks so the VAE decoder + 48 frames don't blow VRAM.
        B, T, C, Hl, Wl = lat_btchw.shape
        x = lat_btchw.reshape(B * T, C, Hl, Wl).to(device)
        parts = []
        for i in range(0, x.shape[0], chunk):
            out01 = flowcast_vae.decode(vae, x[i:i + chunk]).cpu().numpy()
            parts.append(out01)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        recon01 = np.concatenate(parts, axis=0)                     # (T, 1, H, W)
        return [flowcast_vae.destandardize(recon01[i, 0], low, high)
                for i in range(T)]

    # ── run N rollouts; keep both decoded images AND latents per sample ─────
    sample_imgs = []
    sample_lats = []
    for s in range(args.samples):
        pred_lat = one_rollout()                  # (1, horizon, 4, H_lat, W_lat)
        imgs = decode_block(pred_lat)
        sample_imgs.append(imgs)
        sample_lats.append(pred_lat.detach().cpu())
        del pred_lat
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[fc-forecast] sample {s + 1}/{args.samples} done")

    # ── 3-way aggregation: pixmean / medoid / latmean ────────────────────────
    ensemble_stack = np.stack([np.stack(s) for s in sample_imgs])  # (N, T, H, W)
    pixmean_imgs = list(ensemble_stack.mean(axis=0).astype(np.float32))
    if args.samples == 1:
        modes = {"pixmean": pixmean_imgs}
        medoid_idx = 0
    else:
        # Medoid: sample closest to ensemble pixel-mean by L2 (sharp single trajectory).
        pm_flat = ensemble_stack.mean(axis=0).reshape(-1)
        flat = ensemble_stack.reshape(args.samples, -1)
        dists = ((flat - pm_flat[None]) ** 2).sum(axis=1)
        medoid_idx = int(np.argmin(dists))
        medoid_imgs = list(np.asarray(sample_imgs[medoid_idx], dtype=np.float32))
        # Latent-space mean → decode once. Often sharper than pixel-mean because
        # the decoder hallucinates texture from a single averaged latent.
        lat_mean = torch.stack(sample_lats).mean(dim=0)               # (1, T, 4, Hl, Wl)
        latmean_imgs = decode_block(lat_mean)
        modes = {"pixmean": pixmean_imgs,
                 "medoid":  medoid_imgs,
                 "latmean": latmean_imgs}
        print(f"[fc-forecast] 3-way aggregation: pixmean / medoid (s={medoid_idx}) / latmean")

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
        print(f"[fc-forecast] VIS: {horizon - n_night} predicted + "
              f"{n_night} dark fill")
    else:
        print(f"[fc-forecast] WV: {horizon} slots predicted")

    # ── metrics vs ground truth (per mode; CRPS is ensemble-wide, computed once)
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

    # CRPS once on the ensemble (independent of aggregation choice)
    crps_per_frame = []
    for i, gt in gt_cache.items():
        crps_per_frame.append({"step": i, "utc": slots[i].isoformat(),
                                "CRPS": crps_gaussian(ensemble_stack[:, i], gt,
                                                       data_range=drange)})
    crps_mean = (float(np.mean([p["CRPS"] for p in crps_per_frame]))
                 if crps_per_frame else None)

    # ── save outputs (one .npy + grid + gif + compare per mode) ─────────────
    out_dir = resolve(cfg, "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    base_tag = date.strftime("%Y%m%d")
    if args.samples > 1:
        base_tag = f"{base_tag}_ens{args.samples}"

    titles = [f"{utc_to_ist(ts):%H:%M} IST{'' if day_flags[i] else ' (night)'}"
              for i, ts in enumerate(slots)]

    for name, imgs in modes.items():
        suffix = f"_{name}" if args.samples > 1 else ""
        np.save(os.path.join(out_dir, f"flow_forecast_{base_tag}{suffix}.npy"),
                np.stack(imgs).astype(np.float32))
        viz.save_image_grid(
            imgs, os.path.join(out_dir, f"flow_forecast_{base_tag}{suffix}_grid.png"),
            titles=titles, ncols=8, vmin=low, vmax=high)
        viz.save_gif(imgs,
                       os.path.join(out_dir, f"flow_forecast_{base_tag}{suffix}.gif"),
                       vmin=low, vmax=high)
        # Side-by-side vs GT for this mode
        cmp_pred, cmp_gt, cmp_titles = [], [], []
        for i in gt_cache:
            cmp_pred.append(imgs[i])
            cmp_gt.append(gt_cache[i])
            cmp_titles.append(f"{utc_to_ist(slots[i]):%H:%M}")
        if cmp_pred:
            keep = np.linspace(0, len(cmp_pred) - 1, min(8, len(cmp_pred))).astype(int)
            viz.save_comparison_grid(
                [cmp_pred[k] for k in keep], [cmp_gt[k] for k in keep],
                os.path.join(out_dir, f"flow_forecast_{base_tag}{suffix}_compare.png"),
                titles=[cmp_titles[k] for k in keep], vmin=low, vmax=high)

    with open(os.path.join(out_dir, f"flow_forecast_{base_tag}_metrics.json"), "w") as fh:
        json.dump({"date": base_tag, "channel": cfg["channel"],
                    "samples": args.samples, "medoid_idx": medoid_idx,
                    "thresholds": channel_thr,
                    "summary_by_mode": summary_by_mode,
                    "crps_mean": crps_mean,
                    "per_frame_by_mode": per_frame_by_mode,
                    "crps_per_frame": crps_per_frame}, fh, indent=2)

    n_scored = len(gt_cache)
    print(f"[fc-forecast] vs GT ({n_scored} frames):")
    for name, smry in summary_by_mode.items():
        if not smry:
            continue
        bits = "  ".join(f"{k}={v:.3f}" for k, v in smry.items()
                          if k in ("psnr", "ssim", "CSI_M"))
        print(f"  [{name:>7}] {bits}")
    if crps_mean is not None:
        print(f"  [   crps] mean={crps_mean:.4f} (ensemble of {args.samples})")
    print(f"[fc-forecast] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
