"""diffcast Phase 1 — full VAE fine-tune (encoder + decoder).

Run:  python -m diffcast.train_vae --config vis/config_diff.yaml

Starts from the SEVIR-pretrained AutoencoderKL, unfreezes EVERY parameter,
and trains on the channel's INSAT TIFFs with:

    loss = mse_w * MSE(recon, gt) + lpips_w * LPIPS(recon, gt) + kl_w * KL(q || N(0, I))

The KL term is tiny (default 1e-6) — its only job is to keep the latent
posterior from drifting arbitrarily far from a standard Gaussian, which
matters when the diffusion forecaster (Phase 4) samples in that space.

Success gate: roundtrip val PSNR >= ``vae_train.target_psnr`` (default 30 dB).
The fine-tuned full model is saved to
``<checkpoint_dir>/vae_full.pt`` and loaded later via the optional
``vae.full_ckpt`` config key.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from flowcast.data import tif_io
from flowcast.data.sequence_dataset import read_manifest, temporal_split
from flowcast.utils import metrics, viz
from flowcast.utils.config import load_config, resolve, save_config
from flowcast.utils.device import get_device
from flowcast.utils.solar import is_daytime, utc_to_ist

from .models import vae as vae_mod


# ── data ────────────────────────────────────────────────────────────────────
class FullVAEDataset(torch.utils.data.Dataset):
    """Yields (gt_image_01,) — the VAE encoder produces its own latent."""

    def __init__(self, rows, src_dir, H, W, low, high):
        self.rows = rows
        self.src_dir = src_dir
        self.H, self.W = H, W
        self.low, self.high = low, high

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        path = os.path.join(self.src_dir, r["filename"])
        gt = tif_io.resize_image(tif_io.read_tif(path), self.H, self.W)
        gt01 = vae_mod.standardize(gt, self.low, self.high)
        return torch.from_numpy(gt01)[None].float()       # (1, H, W)


# ── lr ──────────────────────────────────────────────────────────────────────
def cosine_warmup(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01):
    warmup_steps = max(1, int(warmup_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, lr_lambda)


# ── eval ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(vae, loader, device, low, high):
    vae.eval()
    drange = high - low
    psnrs, ssims = [], []
    for gt01 in loader:
        gt01 = gt01.to(device)
        z = vae.encode(gt01).latent_dist.mode()           # deterministic at eval
        recon01 = vae.decode(z).sample.clamp(0.0, 1.0)
        for j in range(recon01.shape[0]):
            r = vae_mod.destandardize(
                recon01[j, 0].float().cpu().numpy(), low, high)
            g = vae_mod.destandardize(
                gt01[j, 0].float().cpu().numpy(), low, high)
            m = metrics.all_metrics(r, g, data_range=drange)
            psnrs.append(m["psnr"]); ssims.append(m["ssim"])
    return float(np.mean(psnrs)), float(np.mean(ssims))


@torch.no_grad()
def save_progress(vae, samples, out_path, device, low, high):
    """4 fixed val frames: decoded (top) vs ground truth (bottom)."""
    vae.eval()
    preds, gts = [], []
    for gt01 in samples:
        gt01 = gt01.to(device)
        z = vae.encode(gt01).latent_dist.mode()
        recon01 = vae.decode(z).sample.clamp(0.0, 1.0)
        preds.append(vae_mod.destandardize(
            recon01[0, 0].float().cpu().numpy(), low, high))
        gts.append(vae_mod.destandardize(
            gt01[0, 0].float().cpu().numpy(), low, high))
    viz.save_comparison_grid(
        preds, gts, out_path,
        titles=[f"val[{i}]" for i in range(len(preds))],
        vmin=low, vmax=high,
        row_labels=("decoded", "ground truth"))


# ── compute norm stats if config doesn't have them ─────────────────────────
def _ensure_norm_stats(cfg):
    if cfg["norm"]["low"] is None or cfg["norm"]["high"] is None:
        files = tif_io.list_files(resolve(cfg, "source_dir"), cfg["file_glob"])
        low, high = tif_io.compute_norm_stats(files)
        cfg["norm"]["low"] = low
        cfg["norm"]["high"] = high
        save_config(cfg)
        print(f"[diffcast-vae] computed norm stats: "
              f"low={low:.4f} high={high:.4f}  (saved to {cfg['_path']})")


def _ensure_manifest(cfg):
    """If the diffcast manifest doesn't yet exist, build it now from the TIFFs.
    Same schema as flowcast's manifest — but with diffcast paths (latents may
    not exist yet; the dataset for Phase 1 only needs filename + utc + is_day).
    """
    import csv as csvmod
    mpath = resolve(cfg, "manifest_path")
    if os.path.exists(mpath):
        return
    files = tif_io.list_files(resolve(cfg, "source_dir"), cfg["file_glob"])
    dn = cfg["day_night"]
    rows = []
    for path in files:
        ts = tif_io.parse_timestamp(path)
        is_day = ""
        if dn["enabled"]:
            is_day = is_daytime(ts, dn["scene_lat"], dn["scene_lon"],
                                 dn["elevation_threshold_deg"])
        # latent_path placeholder — Phase 2's encode_latents will populate
        # actual files at vis/latents_diff/<channel>_<date>_<time>.pt.
        name = f"{cfg['channel']}_{ts:%Y%m%d}_{ts:%H%M}.pt"
        rel = os.path.join(cfg["latent_dir"], name)
        rows.append({
            "utc": ts.replace(tzinfo=None).isoformat(),
            "ist": utc_to_ist(ts).replace(tzinfo=None).isoformat(),
            "filename": os.path.basename(path),
            "latent_path": rel,
            "is_day": is_day,
        })
    rows.sort(key=lambda r: r["utc"])
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=["utc", "ist", "filename",
                                              "latent_path", "is_day"])
        w.writeheader()
        w.writerows(rows)
    print(f"[diffcast-vae] wrote manifest with {len(rows)} rows -> {mpath}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="diffcast Phase 1: full VAE fine-tune (encoder + decoder).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override vae_train.steps from config")
    ap.add_argument("--batch", type=int, default=None,
                    help="override vae_train.batch_size from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    ft = cfg.get("vae_train", {})
    total_steps = int(args.steps if args.steps else ft.get("steps", 15000))
    batch_size = int(args.batch if args.batch else ft.get("batch_size", 8))
    lr = float(ft.get("lr", 1e-5))
    wd = float(ft.get("weight_decay", 1e-4))
    mse_w = float(ft.get("mse_weight", 1.0))
    lpips_w = float(ft.get("lpips_weight", 0.5))
    kl_w = float(ft.get("kl_weight", 1e-6))
    warmup = int(ft.get("warmup_steps", 500))
    val_interval = int(ft.get("val_interval", 500))
    log_interval = int(ft.get("log_interval", 50))
    fp16 = bool(ft.get("fp16", True)) and device.type == "cuda"
    lpips_backbone = str(ft.get("lpips_backbone", "vgg"))
    target_psnr = float(ft.get("target_psnr", 30.0))

    # ── 1. norm stats + manifest (built once if absent) ────────────────────
    _ensure_norm_stats(cfg)
    _ensure_manifest(cfg)
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")

    # ── 2. dataset split (same as flowcast — temporal) ────────────────────
    rows = read_manifest(resolve(cfg, "manifest_path"))
    if cfg["day_night"]["enabled"]:
        rows = [r for r in rows if r.get("is_day", True) is not False]
    tr_rows, va_rows, te_rows = temporal_split(
        rows, cfg["train"]["val_fraction"], cfg["train"]["test_fraction"])
    print(f"[diffcast-vae] rows: train={len(tr_rows)} val={len(va_rows)} "
          f"test={len(te_rows)}  (day_night={cfg['day_night']['enabled']})")
    if len(tr_rows) == 0 or len(va_rows) == 0:
        raise SystemExit("[diffcast-vae] empty split — too few frames")

    train_ds = FullVAEDataset(tr_rows, src_dir, H, W, low, high)
    val_ds = FullVAEDataset(va_rows, src_dir, H, W, low, high)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"))

    progress_samples = []
    for i in range(min(4, len(val_ds))):
        progress_samples.append(val_ds[i][None])          # (1, 1, H, W)

    # ── 3. VAE: load SEVIR + unfreeze EVERYTHING ────────────────────────────
    vae = vae_mod.load_vae(cfg, device, trainable=True)
    trainable = [p for p in vae.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[diffcast-vae] trainable params (full VAE): {n_train:,}")

    # ── 4. LPIPS perceptual loss (fp32 even when fp16 training) ────────────
    try:
        import lpips
    except ImportError:
        raise SystemExit("[diffcast-vae] missing dependency — pip install lpips==0.1.4")
    lpips_net = lpips.LPIPS(net=lpips_backbone).to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd,
                                   betas=(0.9, 0.999))
    scheduler = cosine_warmup(optimizer, total_steps, warmup)
    scaler = torch.amp.GradScaler("cuda") if fp16 else None

    ckpt_dir = resolve(cfg, "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_dir = resolve(cfg, "output_dir")
    prog_dir = os.path.join(out_dir, "vae_train_progress")
    os.makedirs(prog_dir, exist_ok=True)

    best_path = os.path.join(ckpt_dir, "vae_full.pt")
    last_path = os.path.join(ckpt_dir, "vae_full_last.pt")
    hist_path = os.path.join(ckpt_dir, "vae_full_history.json")

    def save_ckpt(path, step, val_psnr, val_ssim):
        torch.save({
            "model": vae.state_dict(),
            "step": step,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "config": {"lr": lr, "wd": wd,
                        "mse_weight": mse_w, "lpips_weight": lpips_w,
                        "kl_weight": kl_w, "lpips_backbone": lpips_backbone,
                        "batch_size": batch_size, "warmup_steps": warmup,
                        "loss": f"{mse_w}*mse + {lpips_w}*lpips_{lpips_backbone} + {kl_w}*kl",
                        "norm_low": low, "norm_high": high},
        }, path)

    base_psnr, base_ssim = evaluate(vae, val_loader, device, low, high)
    print(f"[diffcast-vae] baseline val: psnr={base_psnr:.2f}  ssim={base_ssim:.3f}")
    save_progress(vae, progress_samples,
                   os.path.join(prog_dir, "step_00000_baseline.png"),
                   device, low, high)

    # ── 5. train loop ──────────────────────────────────────────────────────
    best_psnr = -float("inf")
    history = []
    step = 0
    t0 = time.time()

    def train_iter(loader):
        while True:
            for batch in loader:
                yield batch

    iterator = train_iter(train_loader)

    while step < total_steps:
        gt01 = next(iterator).to(device, non_blocking=True)
        vae.train()
        optimizer.zero_grad(set_to_none=True)

        if fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                posterior = vae.encode(gt01).latent_dist
                z = posterior.sample()
                recon01 = vae.decode(z).sample
                mse = F.mse_loss(recon01, gt01)
                kl = posterior.kl().mean()
            # LPIPS in fp32 for stability on small VAE batches.
            recon_fp32 = recon01.float().clamp(0.0, 1.0)
            r3 = (recon_fp32 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            g3 = (gt01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lp = lpips_net(r3, g3).mean()
            loss = mse_w * mse + lpips_w * lp + kl_w * kl
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            posterior = vae.encode(gt01).latent_dist
            z = posterior.sample()
            recon01 = vae.decode(z).sample
            mse = F.mse_loss(recon01, gt01)
            kl = posterior.kl().mean()
            r3 = (recon01.clamp(0.0, 1.0) * 2.0 - 1.0).repeat(1, 3, 1, 1)
            g3 = (gt01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lp = lpips_net(r3, g3).mean()
            loss = mse_w * mse + lpips_w * lp + kl_w * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        scheduler.step()
        step += 1

        if step % log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"[diffcast-vae] step {step:6d}/{total_steps}  "
                  f"loss={float(loss):.4f}  mse={float(mse):.5f}  "
                  f"lpips={float(lp):.4f}  kl={float(kl):.4f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

        if step % val_interval == 0 or step == total_steps:
            val_psnr, val_ssim = evaluate(vae, val_loader, device, low, high)
            history.append({"step": step, "val_psnr": val_psnr,
                             "val_ssim": val_ssim,
                             "loss": float(loss), "mse": float(mse),
                             "lpips": float(lp), "kl": float(kl)})
            marker = ""
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_ckpt(best_path, step, val_psnr, val_ssim)
                marker = "  <- best"
            save_ckpt(last_path, step, val_psnr, val_ssim)
            save_progress(vae, progress_samples,
                           os.path.join(prog_dir, f"step_{step:06d}.png"),
                           device, low, high)
            print(f"[diffcast-vae] val @ step {step}: psnr={val_psnr:.2f}  "
                  f"ssim={val_ssim:.3f}{marker}")

    with open(hist_path, "w") as fh:
        json.dump({"baseline": {"val_psnr": base_psnr, "val_ssim": base_ssim},
                    "history": history,
                    "best_val_psnr": best_psnr,
                    "target_psnr": target_psnr,
                    "passed_gate": best_psnr >= target_psnr},
                  fh, indent=2)
    print(f"[diffcast-vae] best val psnr={best_psnr:.2f} (baseline {base_psnr:.2f})")
    if best_psnr < target_psnr:
        print(f"[diffcast-vae] WARNING: best val PSNR {best_psnr:.2f} < target "
              f"{target_psnr:.2f} dB. Consider extending --steps and re-running.")
    else:
        print(f"[diffcast-vae] gate PASSED (>= {target_psnr:.2f} dB).")
    print(f"[diffcast-vae] checkpoint -> {best_path}")
    print(f"[diffcast-vae] enable via vae.full_ckpt: {cfg['checkpoint_dir']}/vae_full.pt "
          f"in {args.config}")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    main()
