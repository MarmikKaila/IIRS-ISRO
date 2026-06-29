"""FlowCast — Strategy B prerequisite: build a (predicted_latent -> GT) dataset.

Run:  python -m flowcast.build_pred_dataset --config vis/config_fc.yaml \
          [--rollouts 4] [--ode-steps 10]

Runs the trained CFM forecaster on every training-split day whose full 48-slot
horizon and ``T_past``-frame seed window both lie inside the train split. For
each rollout (default 4 fresh seeds per day), saves the predicted latent at
every daytime slot, paired with metadata pointing back to the GT TIFF.

Hard leakage discipline: any candidate day with a single slot or seed frame in
the val/test split is skipped, and a final assertion crashes the run if any
saved pair references a non-train UTC.

Output layout::
    <repo>/<channel>/pred_pairs_fc/
        manifest.csv          # one row per (day, sample, slot)
        meta.json             # flow ckpt sha + config dims
        <YYYYMMDD>/sample_<k>/step_<NN>.pt   # {"lat": (4, Hl, Wl), "utc": str}
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import torch

from .data.sequence_dataset import read_manifest, temporal_split
from .models.flow_model import build_model
from .utils import flowcast_vae  # noqa: F401  (loaded for side effects later if needed)
from .utils.config import load_config, resolve
from .utils.device import get_device
from .utils.solar import is_daytime


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build (pred_latent, GT) dataset.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--rollouts", type=int, default=None,
                    help="override pair_dataset.rollouts_per_day from config")
    ap.add_argument("--ode-steps", type=int, default=None,
                    help="override pair_dataset.ode_steps from config")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    pd_cfg = cfg.get("pair_dataset", {})
    rollouts_per_day = int(args.rollouts if args.rollouts
                            else pd_cfg.get("rollouts_per_day", 4))
    ode_steps = int(args.ode_steps if args.ode_steps
                     else pd_cfg.get("ode_steps", 10))
    out_dir_rel = pd_cfg.get("output_dir", f"{cfg['channel']}/pred_pairs_fc")
    out_dir = os.path.join(repo, out_dir_rel)
    os.makedirs(out_dir, exist_ok=True)

    # ── load checkpoint + mirror training dims (same as forecast_flow.py) ───
    ckpt_path = os.path.join(resolve(cfg, "checkpoint_dir"), "best_flow.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
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
    H, W = cfg["resize_height"], cfg["resize_width"]
    latent_h = H // 8
    latent_w = W // 8
    dn = cfg["day_night"]
    sza_enabled = bool(dn.get("enabled", False))

    rows = read_manifest(resolve(cfg, "manifest_path"))
    by_utc = {r["utc"]: r for r in rows}

    tr_rows, va_rows, te_rows = temporal_split(
        rows, cfg["train"]["val_fraction"], cfg["train"]["test_fraction"])
    TRAIN_UTCS = {r["utc"] for r in tr_rows}
    VAL_UTCS = {r["utc"] for r in va_rows}
    TEST_UTCS = {r["utc"] for r in te_rows}
    print(f"[fc-pairs] split: train={len(tr_rows)} val={len(va_rows)} "
          f"test={len(te_rows)}")

    # ── build model + load weights ──────────────────────────────────────────
    model = build_model(cfg, latent_ch=4, latent_h=latent_h, latent_w=latent_w).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    flow_sha = file_sha256(ckpt_path)
    print(f"[fc-pairs] mirroring ckpt dims: hidden={cfg['flow']['hidden']}  "
          f"n_blocks={cfg['flow']['n_blocks']}  T_past={T_past} T_lead={T_lead}  "
          f"flow_ckpt_sha={flow_sha[:12]}")

    # ── pick candidate days: every day where ALL horizon slots that exist in
    #    the manifest are in train AND the T_past seed window is entirely in
    #    train. We iterate dates of train_rows to keep the scan bounded.
    step = dt.timedelta(minutes=cadence)
    train_dates = sorted({r["utc"].date() for r in tr_rows})
    candidates = []
    rejected = defaultdict(int)
    for d in train_dates:
        day_start = dt.datetime.combine(d, dt.time(0, 0))
        slots = [day_start + i * step for i in range(horizon)]
        # Check horizon slots
        leaky = False
        for ts in slots:
            r = by_utc.get(ts)
            if r is None:
                continue                       # missing slot is allowed (no GT)
            if r["utc"] not in TRAIN_UTCS:
                leaky = True
                break
        if leaky:
            rejected["horizon_leak"] += 1
            continue
        # Check seed window
        prior = [r for r in rows if r["utc"] < day_start]
        if len(prior) < T_past:
            rejected["seed_too_short"] += 1
            continue
        seed_rows = prior[-T_past:]
        if any(r["utc"] not in TRAIN_UTCS for r in seed_rows):
            rejected["seed_leak"] += 1
            continue
        candidates.append((d, day_start, slots, seed_rows))
    print(f"[fc-pairs] candidate days: {len(candidates)}  "
          f"(rejected: {dict(rejected)})")
    if not candidates:
        raise SystemExit("[fc-pairs] no train-only candidate days — check splits")

    # ── rollout helper (mirrors forecast_flow.one_rollout) ──────────────────
    n_blocks = (horizon + T_lead - 1) // T_lead

    def load_lat(rel):
        return torch.load(os.path.join(repo, rel), map_location="cpu").float()

    @torch.no_grad()
    def one_rollout(seed_lat):
        past = seed_lat.clone()
        out = []
        for _ in range(n_blocks):
            fut = model.sample(past, steps=ode_steps)
            out.append(fut)
            past = torch.cat([past, fut], dim=1)[:, -T_past:]
        full = torch.cat(out, dim=1)
        return full[:, :horizon]               # (1, horizon, 4, Hl, Wl)

    def slot_is_day(ts):
        if not sza_enabled:
            return True
        return is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                           dn["elevation_threshold_deg"])

    # ── run all candidates, save pairs ──────────────────────────────────────
    manifest_rows = []
    pair_utcs = set()
    seed_utcs_used = set()
    total_pairs = 0
    for di, (d, day_start, slots, seed_rows) in enumerate(candidates):
        seed = torch.stack([load_lat(r["latent_path"]) for r in seed_rows],
                            dim=0)[None].to(device)            # (1, T_past, 4, Hl, Wl)
        for r in seed_rows:
            seed_utcs_used.add(r["utc"])
        day_tag = d.strftime("%Y%m%d")
        for s in range(rollouts_per_day):
            torch.manual_seed(args.seed + di * 1000 + s)
            np.random.seed(args.seed + di * 1000 + s)
            pred = one_rollout(seed)                            # (1, horizon, 4, Hl, Wl)
            pred = pred[0].detach().cpu()                       # (horizon, 4, Hl, Wl)
            sample_dir = os.path.join(out_dir, day_tag, f"sample_{s}")
            os.makedirs(sample_dir, exist_ok=True)
            for i, ts in enumerate(slots):
                if not slot_is_day(ts):
                    continue                                    # skip night (no GT)
                r = by_utc.get(ts)
                if r is None:
                    continue
                if r["utc"] not in TRAIN_UTCS:                  # belt + suspenders
                    raise RuntimeError(
                        f"leakage detected: slot {ts} not in TRAIN_UTCS")
                pair_utcs.add(r["utc"])
                pt_path_rel = os.path.join(
                    out_dir_rel, day_tag, f"sample_{s}", f"step_{i:02d}.pt")
                pt_path_abs = os.path.join(repo, pt_path_rel)
                torch.save({"lat": pred[i].to(torch.float16),
                             "utc": r["utc"].isoformat()},
                            pt_path_abs)
                manifest_rows.append({
                    "utc": r["utc"].isoformat(),
                    "day": day_tag,
                    "sample": s,
                    "step": i,
                    "pair_path": pt_path_rel,
                    "tif_filename": r["filename"],
                })
                total_pairs += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[fc-pairs] day {day_tag}  rollouts={rollouts_per_day}  "
              f"pairs so far={total_pairs}")

    # ── hard leakage assertions ─────────────────────────────────────────────
    bad = pair_utcs & (VAL_UTCS | TEST_UTCS)
    if bad:
        raise RuntimeError(
            f"[fc-pairs] LEAK: {len(bad)} pair UTCs are in val/test — aborting")
    seed_bad = seed_utcs_used & (VAL_UTCS | TEST_UTCS)
    if seed_bad:
        raise RuntimeError(
            f"[fc-pairs] LEAK: {len(seed_bad)} seed UTCs are in val/test — aborting")

    # ── write manifest + meta ───────────────────────────────────────────────
    man_path = os.path.join(out_dir, "manifest.csv")
    with open(man_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=[
            "utc", "day", "sample", "step", "pair_path", "tif_filename"])
        wcsv.writeheader()
        wcsv.writerows(manifest_rows)

    meta = {
        "channel": cfg["channel"],
        "n_pairs": total_pairs,
        "n_days": len(candidates),
        "rollouts_per_day": rollouts_per_day,
        "ode_steps": ode_steps,
        "T_past": T_past,
        "T_lead": T_lead,
        "horizon": horizon,
        "model_dims": {
            "hidden": cfg["flow"]["hidden"],
            "n_blocks": cfg["flow"]["n_blocks"],
            "attn_heads": cfg["flow"].get("attn_heads"),
            "sigma": cfg["flow"].get("sigma"),
        },
        "flow_ckpt_path": os.path.relpath(ckpt_path, repo),
        "flow_ckpt_sha": flow_sha,
        "norm_low": cfg["norm"]["low"],
        "norm_high": cfg["norm"]["high"],
        "resize_height": H,
        "resize_width": W,
        "seed_base": args.seed,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[fc-pairs] wrote {total_pairs} pairs across {len(candidates)} days "
          f"-> {out_dir}")
    print(f"[fc-pairs] manifest -> {man_path}")
    print(f"[fc-pairs] no train/val/test UTC leakage (asserts passed)")


if __name__ == "__main__":
    main()
