"""Unified multi-date evaluation: persistence, climatology, ConvLSTM, FlowCast.

Runs every method on every test-split date and aggregates mean ± std.
Outputs a formatted table to stdout and saves JSON + CSV for the paper.

Run:
    python -m src.eval_all_dates --config vis/config.yaml
    python -m src.eval_all_dates --config vis/config.yaml --config-fc vis/config_fc.yaml --samples 8
    python -m src.eval_all_dates --config wv/config.yaml  --config-fc wv/config_fc.yaml

Flags:
    --config       path to the SD-VAE / ConvLSTM channel config (required)
    --config-fc    path to the FlowCast channel config (optional; skip FC if absent)
    --samples      FlowCast ensemble size (default 1; use 8 for CRPS)
    --skip-existing  skip re-running a (method, date) pair if its per-date JSON exists
    --methods      comma-separated subset, e.g. "persistence,convlstm" (default: all)
"""
import argparse
import csv
import datetime as dt
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from .baselines import climatology_forecast, persistence_forecast
from .data import tif_io
from .data.sequence_dataset import read_manifest, temporal_split
from .models.convlstm import build_model
from .utils import metrics
from .utils.config import load_config, resolve
from .utils.device import get_device
from .utils.sdvae import decode, load_vae, scaling_factor
from .utils.solar import is_daytime, solar_elevation_deg, utc_to_ist

try:
    from flowcast.utils.threshold_metrics import DEFAULT_THRESHOLDS, categorical_M
    from flowcast.utils.config import load_config as fc_load_config
    from flowcast.utils.config import resolve as fc_resolve
    from flowcast.models.flow_model import build_model as fc_build_model
    from flowcast.utils import flowcast_vae
    from flowcast.data.sequence_dataset import read_manifest as fc_read_manifest
    from flowcast.data.sequence_dataset import temporal_split as fc_temporal_split
    _HAS_FC_DEPS = True
except ImportError:
    _HAS_FC_DEPS = False
    DEFAULT_THRESHOLDS = {}

try:
    from flowcast.utils.threshold_metrics import crps_gaussian
    _HAS_CRPS = True
except ImportError:
    _HAS_CRPS = False

# Metrics shown in the summary table (order matters for display)
DISPLAY_KEYS = ["psnr", "ssim", "CSI_M", "HSS_M", "FAR_M"]
CRPS_KEY = "crps_mean"


# ── date helpers ─────────────────────────────────────────────────────────────

def _test_dates(rows, val_frac, test_frac, min_frames=24):
    """Return unique UTC dates in the test split that have >= min_frames."""
    _, _, te_rows = temporal_split(rows, val_frac, test_frac)
    counts = Counter(r["utc"].date() for r in te_rows)
    return sorted(d for d, n in counts.items() if n >= min_frames)


def _slots(date, horizon, cadence):
    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step = dt.timedelta(minutes=cadence)
    return [day_start + i * step for i in range(horizon)]


def _day_flags(slots, dn):
    if not dn.get("enabled", False):
        return [True] * len(slots)
    return [is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                       dn["elevation_threshold_deg"])
            for ts in slots]


def _score(forecast_imgs, slots, by_utc, day_flags, cfg, channel_thr):
    """Return per_frame list with psnr/ssim/mse + categorical metrics."""
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    drange = high - low
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")
    dn = cfg["day_night"]

    per_frame = []
    for i, ts in enumerate(slots):
        r = by_utc.get(ts)
        if r is None or (dn.get("enabled", False) and not day_flags[i]):
            continue
        gt = tif_io.resize_image(
            tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        m = metrics.all_metrics(forecast_imgs[i], gt, data_range=drange)
        if channel_thr:
            m.update(categorical_M(forecast_imgs[i], gt, channel_thr))
        per_frame.append({"step": i, "utc": ts.isoformat(), **m})
    return per_frame


def _summarise(per_frame):
    if not per_frame:
        return {}
    keys = [k for k in per_frame[0] if k not in ("step", "utc")]
    return {k: float(np.mean([p[k] for p in per_frame if k in p]))
            for k in keys}


def _by_horizon(per_frame):
    by_h = {}
    for label, lo, hi in [("0-11", 0, 11), ("12-23", 12, 23),
                           ("24-35", 24, 35), ("36-47", 36, 47)]:
        bucket = [p for p in per_frame if lo <= p["step"] <= hi]
        if bucket:
            keys = [k for k in bucket[0] if k not in ("step", "utc")]
            by_h[label] = {k: float(np.mean([p[k] for p in bucket if k in p]))
                           for k in keys}
    return by_h


# ── ConvLSTM ─────────────────────────────────────────────────────────────────

def run_convlstm(cfg, date, all_rows, device):
    """Run the trained ConvLSTM (src/) forecaster on a single date."""
    repo = cfg["_repo_root"]
    T = cfg["sequence"]["input_len"]
    cadence = cfg["sequence"]["cadence_minutes"]
    horizon = cfg["forecast"]["horizon"]
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    H, W = cfg["resize_height"], cfg["resize_width"]
    dn = cfg["day_night"]
    src_dir = resolve(cfg, "source_dir")
    channel = cfg["channel"]
    channel_thr = DEFAULT_THRESHOLDS.get(channel, []) if _HAS_FC_DEPS else []

    ckpt_path = os.path.join(resolve(cfg, "checkpoint_dir"), "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[convlstm] checkpoint not found: {ckpt_path}\n"
            f"Run: python -m src.train --config {cfg['_path']}")

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    vae = load_vae(cfg, device)
    sf = scaling_factor(cfg, vae)

    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step_td = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step_td for i in range(horizon)]
    by_utc = {r["utc"]: r for r in all_rows}
    day_flags = _day_flags(slots, dn)

    latent_h = H // 8
    latent_w = W // 8
    cond_ch = int(getattr(model, "cond_ch", 0))

    def load_lat(rel):
        return torch.load(os.path.join(repo, rel),
                          map_location="cpu").float()

    def sza_plane(ts):
        elev = solar_elevation_deg(ts, dn["scene_lat"], dn["scene_lon"])
        v = max(-1.0, min(1.0, elev / 90.0))
        return torch.full((1, latent_h, latent_w), v, dtype=torch.float32)

    prior = [r for r in all_rows if r["utc"] < day_start]
    if len(prior) < T:
        raise ValueError(
            f"[convlstm] need {T} prior frames for {date}, have {len(prior)}")

    seed_rows = prior[-T:]
    seed_lat = torch.stack([load_lat(r["latent_path"]) for r in seed_rows])

    if cond_ch:
        sza_seed = torch.stack([sza_plane(r["utc"]) for r in seed_rows])
        x_in = torch.cat([seed_lat, sza_seed], dim=1)[None].to(device)
        future_cond = torch.stack([sza_plane(ts) for ts in slots])[None].to(device)
    else:
        x_in = seed_lat[None].to(device)
        future_cond = None

    with torch.no_grad():
        pred_lat = model(x_in, future_steps=horizon,
                         future_cond=future_cond)[0].cpu()   # (horizon,4,H,W)

    # decode
    def decode_lat(lat):
        out = decode(vae, lat[None].to(device), sf)
        return tif_io.denormalize(tif_io.from_vae_output(out), low, high)

    forecast_imgs = [decode_lat(pred_lat[i]) for i in range(horizon)]

    # VIS night fill
    if dn.get("enabled", False):
        night_rows = [r for r in all_rows if r.get("is_day", True) is False]
        samp = night_rows[::max(1, len(night_rows) // 30)][:30]
        acc = np.zeros((H, W), np.float64)
        for r in samp:
            acc += tif_io.resize_image(
                tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        fill = (acc / max(len(samp), 1)).astype(np.float32)
        for i in range(horizon):
            if not day_flags[i]:
                forecast_imgs[i] = fill

    per_frame = _score(forecast_imgs, slots, by_utc, day_flags, cfg, channel_thr)
    return {
        "method": "convlstm",
        "date": date.strftime("%Y%m%d"),
        "channel": channel,
        "summary": _summarise(per_frame),
        "by_horizon": _by_horizon(per_frame),
        "per_frame": per_frame,
        "crps_mean": None,
    }


# ── FlowCast ──────────────────────────────────────────────────────────────────

def run_flowcast(cfg_fc, date, all_rows_fc, device, samples=1):
    """Run the FlowCast CFM forecaster on a single date."""
    if not _HAS_FC_DEPS:
        raise ImportError("flowcast package not importable")

    repo = cfg_fc["_repo_root"]
    ckpt_path = os.path.join(fc_resolve(cfg_fc, "checkpoint_dir"), "best_flow.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[flowcast] checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    # mirror dims saved at training time (T4 auto-scaling may have shrunk them)
    saved = ckpt.get("config") or {}
    if saved.get("flow"):
        for k in ("hidden", "n_blocks", "attn_heads", "dropout",
                   "joint_steps", "sigma"):
            if k in saved["flow"]:
                cfg_fc["flow"][k] = saved["flow"][k]
    if "T_past" in ckpt:
        cfg_fc["sequence"]["input_len"] = int(ckpt["T_past"])
    if "T_lead" in ckpt:
        cfg_fc["flow"]["joint_steps"] = int(ckpt["T_lead"])

    horizon = int(cfg_fc["forecast"].get("horizon", 48))
    T_lead = int(cfg_fc["flow"]["joint_steps"])
    T_past = int(cfg_fc["sequence"]["input_len"])
    cadence = cfg_fc["sequence"]["cadence_minutes"]
    low, high = cfg_fc["norm"]["low"], cfg_fc["norm"]["high"]
    drange = high - low
    H, W = cfg_fc["resize_height"], cfg_fc["resize_width"]
    dn = cfg_fc["day_night"]
    src_dir = fc_resolve(cfg_fc, "source_dir")
    channel = cfg_fc["channel"]
    channel_thr = DEFAULT_THRESHOLDS.get(channel, [])

    day_start = dt.datetime.combine(date, dt.time(0, 0))
    step_td = dt.timedelta(minutes=cadence)
    slots = [day_start + i * step_td for i in range(horizon)]
    by_utc = {r["utc"]: r for r in all_rows_fc}
    day_flags = _day_flags(slots, dn)

    latent_h = H // 8
    latent_w = W // 8
    model = fc_build_model(cfg_fc, latent_ch=4,
                           latent_h=latent_h, latent_w=latent_w).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    vae = flowcast_vae.load_flowcast_vae(cfg_fc, device)

    def load_lat(rel):
        return torch.load(os.path.join(repo, rel),
                          map_location="cpu").float()

    prior = [r for r in all_rows_fc if r["utc"] < day_start]
    if len(prior) < T_past:
        raise ValueError(
            f"[flowcast] need {T_past} prior frames for {date}, have {len(prior)}")
    seed = torch.stack(
        [load_lat(r["latent_path"]) for r in prior[-T_past:]], dim=0)[None].to(device)

    n_blocks = (horizon + T_lead - 1) // T_lead
    ode_steps = int(cfg_fc["flow"].get("ode_steps", 10))

    @torch.no_grad()
    def one_rollout():
        past = seed.clone()
        out = []
        for _ in range(n_blocks):
            fut = model.sample(past, steps=ode_steps)
            out.append(fut)
            past = torch.cat([past, fut], dim=1)[:, -T_past:]
        return torch.cat(out, dim=1)[:, :horizon]

    @torch.no_grad()
    def decode_block(lat_btchw, chunk=4):
        B, T, C, Hl, Wl = lat_btchw.shape
        x = lat_btchw.reshape(B * T, C, Hl, Wl).to(device)
        parts = []
        for i in range(0, x.shape[0], chunk):
            out01 = flowcast_vae.decode(vae, x[i:i + chunk]).cpu().numpy()
            parts.append(out01)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        recon01 = np.concatenate(parts, axis=0)
        return [flowcast_vae.destandardize(recon01[i, 0], low, high)
                for i in range(T)]

    sample_imgs, sample_lats = [], []
    for s in range(samples):
        pred_lat = one_rollout()
        sample_imgs.append(decode_block(pred_lat))
        sample_lats.append(pred_lat.detach().cpu())
        del pred_lat
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # use medoid when samples > 1, single sample otherwise
    if samples == 1:
        forecast_imgs = sample_imgs[0]
    else:
        stack = np.stack([np.stack(s) for s in sample_imgs])   # (N,T,H,W)
        pm = stack.mean(axis=0).reshape(-1)
        flat = stack.reshape(samples, -1)
        medoid_idx = int(np.argmin(((flat - pm[None]) ** 2).sum(axis=1)))
        forecast_imgs = list(np.asarray(sample_imgs[medoid_idx], np.float32))

    # VIS night fill
    if dn.get("enabled", False):
        night_rows = [r for r in all_rows_fc if r.get("is_day", True) is False]
        samp = night_rows[::max(1, len(night_rows) // 30)][:30]
        acc = np.zeros((H, W), np.float64)
        for r in samp:
            acc += tif_io.resize_image(
                tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
        fill = (acc / max(len(samp), 1)).astype(np.float32)
        ens_stack = (np.stack([np.stack(s) for s in sample_imgs])
                     if samples > 1 else None)
        for i in range(horizon):
            if not day_flags[i]:
                forecast_imgs[i] = fill
                if ens_stack is not None:
                    ens_stack[:, i] = fill[None]

    # score medoid
    per_frame = _score(forecast_imgs, slots, by_utc, day_flags, cfg_fc, channel_thr)

    # CRPS (ensemble-wide, independent of aggregation)
    crps_mean = None
    if samples > 1 and _HAS_CRPS:
        ens_stack = np.stack([np.stack(s) for s in sample_imgs])   # (N,T,H,W)
        crps_per = []
        for i, ts in enumerate(slots):
            r = by_utc.get(ts)
            if r is None or (dn.get("enabled", False) and not day_flags[i]):
                continue
            gt = tif_io.resize_image(
                tif_io.read_tif(os.path.join(src_dir, r["filename"])), H, W)
            crps_per.append(crps_gaussian(ens_stack[:, i], gt, data_range=drange))
        crps_mean = float(np.mean(crps_per)) if crps_per else None

    return {
        "method": f"flowcast_s{samples}",
        "date": date.strftime("%Y%m%d"),
        "channel": channel,
        "summary": _summarise(per_frame),
        "by_horizon": _by_horizon(per_frame),
        "per_frame": per_frame,
        "crps_mean": crps_mean,
        "samples": samples,
    }


# ── aggregation + display ─────────────────────────────────────────────────────

def _aggregate(results_by_method):
    """Mean ± std across dates for each method."""
    agg = {}
    for method, date_results in results_by_method.items():
        all_keys = set()
        for r in date_results:
            all_keys.update(r["summary"].keys())
        agg[method] = {}
        for k in all_keys:
            vals = [r["summary"][k] for r in date_results if k in r["summary"]]
            if vals:
                agg[method][k] = {"mean": float(np.mean(vals)),
                                   "std": float(np.std(vals)),
                                   "n": len(vals)}
        # CRPS
        crps_vals = [r["crps_mean"] for r in date_results
                     if r.get("crps_mean") is not None]
        if crps_vals:
            agg[method]["crps_mean"] = {"mean": float(np.mean(crps_vals)),
                                         "std": float(np.std(crps_vals)),
                                         "n": len(crps_vals)}
    return agg


def _print_table(agg, dates, methods):
    show_keys = DISPLAY_KEYS + [CRPS_KEY]
    # header
    col_w = 14
    header = f"{'Method':<20}" + "".join(f"{k:>{col_w}}" for k in show_keys)
    print("\n" + "=" * len(header))
    print(f"{'EVALUATION RESULTS':^{len(header)}}")
    print(f"{'Dates: ' + ', '.join(str(d) for d in dates):^{len(header)}}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in methods:
        if m not in agg:
            continue
        row = f"{m:<20}"
        for k in show_keys:
            if k not in agg[m]:
                row += f"{'—':>{col_w}}"
            else:
                v = agg[m][k]
                row += f"{v['mean']:>{col_w - 5}.3f}±{v['std']:.2f}"
        print(row)
    print("=" * len(header) + "\n")


def _save_csv(agg, methods, out_path):
    show_keys = DISPLAY_KEYS + [CRPS_KEY]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"{k}_mean" for k in show_keys]
                   + [f"{k}_std" for k in show_keys])
        for m in methods:
            if m not in agg:
                continue
            row = [m]
            for k in show_keys:
                row.append(f"{agg[m][k]['mean']:.4f}" if k in agg[m] else "")
            for k in show_keys:
                row.append(f"{agg[m][k]['std']:.4f}" if k in agg[m] else "")
            w.writerow(row)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Multi-date evaluation: persistence / climatology / ConvLSTM / FlowCast")
    ap.add_argument("--config", required=True,
                    help="SD-VAE / ConvLSTM channel config (e.g. vis/config.yaml)")
    ap.add_argument("--config-fc", default=None,
                    help="FlowCast channel config (optional, e.g. vis/config_fc.yaml)")
    ap.add_argument("--samples", type=int, default=1,
                    help="FlowCast ensemble size (default 1; use 8 for CRPS)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip re-running if per-date JSON already exists")
    ap.add_argument("--methods", default=None,
                    help="Comma-separated subset of methods to run "
                         "(default: all available)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = resolve(cfg, "output_dir")
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    # ── manifests + splits ───────────────────────────────────────────────
    all_rows = read_manifest(resolve(cfg, "manifest_path"))
    tr_rows, va_rows, te_rows = temporal_split(
        all_rows,
        cfg["train"]["val_fraction"],
        cfg["train"]["test_fraction"])
    test_dates = _test_dates(all_rows,
                              cfg["train"]["val_fraction"],
                              cfg["train"]["test_fraction"])
    if not test_dates:
        raise SystemExit("[eval] no test dates found — check manifest and split fractions")

    channel = cfg["channel"]
    print(f"[eval] channel={channel}  test dates ({len(test_dates)}): "
          f"{[str(d) for d in test_dates]}")

    # ── optional FlowCast config ────────────────────────────────────────
    cfg_fc = None
    all_rows_fc = None
    if args.config_fc:
        if not _HAS_FC_DEPS:
            print("[eval] WARNING: flowcast deps not available; skipping FlowCast")
        else:
            cfg_fc = fc_load_config(args.config_fc)
            all_rows_fc = fc_read_manifest(fc_resolve(cfg_fc, "manifest_path"))

    # ── decide which methods to run ────────────────────────────────────
    all_methods = ["persistence", "climatology", "convlstm"]
    if cfg_fc is not None:
        all_methods.append(f"flowcast_s{args.samples}")

    if args.methods:
        requested = [m.strip() for m in args.methods.split(",")]
        # allow shorthand "flowcast" → "flowcast_s{samples}"
        methods = []
        for m in requested:
            if m == "flowcast":
                methods.append(f"flowcast_s{args.samples}")
            elif m in all_methods:
                methods.append(m)
            else:
                print(f"[eval] WARNING: unknown method '{m}', skipping")
        all_methods = methods

    # ── run evaluation ──────────────────────────────────────────────────
    results_by_method = defaultdict(list)
    clim_cache_dir = out_dir   # save slot means alongside other outputs

    for date in test_dates:
        print(f"\n[eval] ── date {date} ──────────────────────────────────")

        for method in all_methods:
            tag = f"{method}_{date.strftime('%Y%m%d')}"
            json_path = os.path.join(out_dir, f"eval_{tag}.json")

            if args.skip_existing and os.path.exists(json_path):
                print(f"  [{method}] skipping (already exists: {json_path})")
                result = json.load(open(json_path))
                results_by_method[method].append(result)
                continue

            try:
                if method == "persistence":
                    result = persistence_forecast(cfg, date, all_rows)
                elif method == "climatology":
                    result = climatology_forecast(cfg, date, all_rows,
                                                   tr_rows, clim_cache_dir)
                elif method == "convlstm":
                    result = run_convlstm(cfg, date, all_rows, device)
                elif method.startswith("flowcast_s"):
                    result = run_flowcast(cfg_fc, date, all_rows_fc, device,
                                          samples=args.samples)
                else:
                    print(f"  [{method}] unknown method, skipping")
                    continue
            except (FileNotFoundError, ValueError) as exc:
                print(f"  [{method}] ERROR: {exc}")
                continue

            # print one-line summary
            s = result.get("summary", {})
            bits = "  ".join(f"{k}={s[k]:.3f}" for k in DISPLAY_KEYS if k in s)
            crps_str = (f"  CRPS={result['crps_mean']:.4f}"
                        if result.get("crps_mean") is not None else "")
            print(f"  [{method}] {bits}{crps_str}")

            with open(json_path, "w") as fh:
                json.dump(result, fh, indent=2)
            results_by_method[method].append(result)

    if not results_by_method:
        raise SystemExit("[eval] no results produced — check configs and checkpoints")

    # ── aggregate + display ────────────────────────────────────────────
    run_methods = list(results_by_method.keys())
    agg = _aggregate(results_by_method)
    _print_table(agg, test_dates, run_methods)

    # ── save combined outputs ──────────────────────────────────────────
    summary_path = os.path.join(out_dir, "eval_all_dates.json")
    csv_path = os.path.join(out_dir, "eval_all_dates.csv")

    with open(summary_path, "w") as fh:
        json.dump({
            "channel": channel,
            "test_dates": [str(d) for d in test_dates],
            "methods": run_methods,
            "aggregated": agg,
            "per_date": {m: results_by_method[m] for m in run_methods},
        }, fh, indent=2)
    _save_csv(agg, run_methods, csv_path)

    print(f"[eval] results -> {summary_path}")
    print(f"[eval] CSV     -> {csv_path}")


if __name__ == "__main__":
    main()
