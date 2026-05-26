"""Stage 4 (flow) — 48-frame day forecast using the latent flow-matching model.

Run:  python -m src.forecast_flow --config vis/config.yaml [--date YYYYMMDD]
      python -m src.forecast_flow --config wv/config.yaml  [--date YYYYMMDD]

For each of the 48 slots, sample a next latent from the flow model conditioned
on the past T latents (and the SZA channel on VIS). Slide the window, repeat
48 times, then decode each predicted latent through the SD-VAE. For VIS, night
slots are overwritten with the dark fill (mean of real VIS night frames).
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
from .models.flow_model import build_flow_model
from .models.flow_model import sample as flow_sample
from .utils import metrics, viz
from .utils.config import load_config, resolve
from .utils.device import get_device
from .utils.sdvae import decode, load_vae, scaling_factor
from .utils.solar import is_daytime, solar_elevation_deg, utc_to_ist


def pick_forecast_date(rows, arg_date):
    if arg_date:
        return dt.datetime.strptime(arg_date, "%Y%m%d").date()
    counts = Counter(r["utc"].date() for r in rows)
    for d in sorted(counts, reverse=True):
        if counts[d] >= 24:
            return d
    return max(counts)


def main():
    ap = argparse.ArgumentParser(description="Stage 4 (flow): 48-frame forecast.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--date", default=None, help="forecast date YYYYMMDD (UTC)")
    ap.add_argument("--samples", type=int, default=1,
                    help="number of independent rollouts to ensemble in pixel "
                         "space (default 1 = single sample)")
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
    sza_enabled = bool(dn.get("enabled", False))
    src_dir = resolve(cfg, "source_dir")

    rows = read_manifest(resolve(cfg, "manifest_path"))
    by_utc = {r["utc"]: r for r in rows}

    date = pick_forecast_date(rows, args.date)
    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step for i in range(horizon)]
    print(f"[flow-forecast] channel={cfg['channel']}  date={date}  "
          f"horizon={horizon}")

    # ── load checkpoint & build model with the right cond_ch ─────────────────
    ckpt_path = os.path.join(resolve(cfg, "checkpoint_dir"), "best_flow.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cond_ch = int(ckpt["cond_ch"])
    # joint_steps is needed for the sanity check below + the rollout loop
    joint_steps = int(ckpt.get("joint_steps",
                               cfg["flow"].get("joint_steps", 4)))
    model = build_flow_model(cfg, cond_ch=cond_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    expected = T * (5 if sza_enabled else 4) + (joint_steps if sza_enabled else 0)
    if cond_ch != expected:
        raise SystemExit(f"[flow-forecast] cond_ch mismatch — checkpoint {cond_ch} "
                         f"vs config expected {expected} "
                         f"(joint_steps={joint_steps})")

    vae = load_vae(cfg, device)
    sf = scaling_factor(cfg, vae)

    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8

    def sza_plane(ts):
        if not sza_enabled:
            return None
        elev = solar_elevation_deg(ts, dn["scene_lat"], dn["scene_lon"])
        v = max(-1.0, min(1.0, elev / 90.0))
        return torch.full((1, latent_h, latent_w), v, dtype=torch.float32)

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
        if not sza_enabled:
            return True
        return is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                          dn["elevation_threshold_deg"])
    day_flags = [slot_is_day(ts) for ts in slots]

    # ── seed window: T real frames before the forecast day ───────────────────
    prior = [r for r in rows if r["utc"] < day_start]
    if len(prior) < T:
        raise SystemExit(f"[flow-forecast] need {T} prior frames, "
                         f"have {len(prior)}")
    seed_lat = [load_latent(r["latent_path"]) for r in prior[-T:]]
    seed_ts = [r["utc"] for r in prior[-T:]]

    ode_steps = int(cfg["flow"]["ode_steps"])
    n_samples = max(1, int(args.samples))
    # joint_steps was read above (during the sanity check) — same value used here

    def one_rollout():
        """Single autoregressive rollout from the shared seed; new noise each time.

        With joint_steps=K the model samples a block of K next latents per call;
        we slide the window forward by K and continue until ``horizon`` frames
        are produced. K=1 reduces to the original one-at-a-time loop.
        """
        past_lat = list(seed_lat)
        past_ts = list(seed_ts)
        out = []
        block_start = 0
        while block_start < horizon:
            K = min(joint_steps, horizon - block_start)
            block_slots = slots[block_start:block_start + K]
            parts = []
            for k in range(T):
                parts.append(past_lat[k])
                if sza_enabled:
                    parts.append(sza_plane(past_ts[k]))
            if sza_enabled:
                # K target SZA planes — pad with the last slot's SZA if K < joint_steps
                for k in range(joint_steps):
                    if k < K:
                        parts.append(sza_plane(block_slots[k]))
                    else:
                        parts.append(sza_plane(block_slots[-1]))
            cond = torch.cat(parts, dim=0)[None].to(device)
            if cond.shape[1] != cond_ch:
                raise SystemExit(f"[flow-forecast] built cond_ch {cond.shape[1]} "
                                 f"!= model cond_ch {cond_ch}")
            # output channels = 4 * joint_steps; reshape into K frames
            block = flow_sample(model, cond,
                                (1, 4 * joint_steps, latent_h, latent_w),
                                steps=ode_steps, device=device)[0].cpu()
            block = block.view(joint_steps, 4, latent_h, latent_w)
            for k in range(K):
                nxt = block[k]
                out.append(nxt)
                past_lat = past_lat[1:] + [nxt]
                past_ts = past_ts[1:] + [block_slots[k]]
            block_start += K
        return out

    # ── run N independent rollouts, average DECODED pixels per slot ──────────
    sample_imgs = []
    for s in range(n_samples):
        pred_latents = one_rollout()
        sample_imgs.append(decode_latents(pred_latents))
        if n_samples > 1:
            print(f"[flow-forecast] sample {s + 1}/{n_samples} done")

    if n_samples == 1:
        forecast_imgs = sample_imgs[0]
    else:
        # ensemble in pixel space — never average latents (collapses to mean)
        stack = np.stack([np.stack(s) for s in sample_imgs])   # (N, horizon, H, W)
        forecast_imgs = list(stack.mean(axis=0).astype(np.float32))
        print(f"[flow-forecast] pixel-space ensemble over {n_samples} samples")

    # ── VIS night fill ───────────────────────────────────────────────────────
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
                forecast_imgs[i] = night_fill
                n_night += 1
        print(f"[flow-forecast] VIS: {horizon - n_night} predicted + "
              f"{n_night} dark fill")
    else:
        print(f"[flow-forecast] WV: {horizon} slots predicted")

    # ── metrics vs ground truth (VIS daytime only) ───────────────────────────
    per_frame = []
    for i, ts in enumerate(slots):
        r = by_utc.get(ts)
        if r is None or (sza_enabled and not day_flags[i]):
            continue
        gt = tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        m = metrics.all_metrics(forecast_imgs[i], gt, data_range=drange)
        per_frame.append({"step": i, "utc": ts.isoformat(), **m})
    summary = ({k: float(np.mean([p[k] for p in per_frame]))
                for k in ("mse", "psnr", "ssim")} if per_frame else {})
    by_horizon = {}
    for lbl, lo, hi in [("0-11", 0, 11), ("12-23", 12, 23),
                        ("24-35", 24, 35), ("36-47", 36, 47)]:
        b = [p for p in per_frame if lo <= p["step"] <= hi]
        if b:
            by_horizon[lbl] = {k: float(np.mean([p[k] for p in b]))
                               for k in ("mse", "psnr", "ssim")}

    # ── save outputs ─────────────────────────────────────────────────────────
    out_dir = resolve(cfg, "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    tag = date.strftime("%Y%m%d")
    if n_samples > 1:
        tag = f"{tag}_ens{n_samples}"            # keep single-sample run intact
    np.save(os.path.join(out_dir, f"flow_forecast_{tag}.npy"),
            np.stack(forecast_imgs).astype(np.float32))
    titles = [f"{utc_to_ist(ts):%H:%M} IST{'' if day_flags[i] else ' (night)'}"
              for i, ts in enumerate(slots)]
    viz.save_image_grid(
        forecast_imgs, os.path.join(out_dir, f"flow_forecast_{tag}_grid.png"),
        titles=titles, ncols=8, vmin=low, vmax=high)
    viz.save_gif(
        forecast_imgs, os.path.join(out_dir, f"flow_forecast_{tag}.gif"),
        vmin=low, vmax=high)

    cmp_pred, cmp_gt, cmp_titles = [], [], []
    for i, ts in enumerate(slots):
        r = by_utc.get(ts)
        if r is None or (sza_enabled and not day_flags[i]):
            continue
        cmp_pred.append(forecast_imgs[i])
        cmp_gt.append(tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W))
        cmp_titles.append(f"{utc_to_ist(ts):%H:%M}")
    if cmp_pred:
        keep = np.linspace(0, len(cmp_pred) - 1, min(8, len(cmp_pred))).astype(int)
        viz.save_comparison_grid(
            [cmp_pred[k] for k in keep], [cmp_gt[k] for k in keep],
            os.path.join(out_dir, f"flow_forecast_{tag}_compare.png"),
            titles=[cmp_titles[k] for k in keep], vmin=low, vmax=high)

    with open(os.path.join(out_dir, f"flow_forecast_{tag}_metrics.json"), "w") as fh:
        json.dump({"date": tag, "channel": cfg["channel"],
                   "summary": summary, "by_horizon": by_horizon,
                   "per_frame": per_frame}, fh, indent=2)

    if summary:
        print(f"[flow-forecast] vs GT ({len(per_frame)} frames): "
              f"PSNR={summary['psnr']:.2f}  SSIM={summary['ssim']:.3f}  "
              f"MSE={summary['mse']:.4f}")
        for lbl, m in by_horizon.items():
            print(f"[flow-forecast]   steps {lbl:>6}: "
                  f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.3f}")
    print(f"[flow-forecast] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
