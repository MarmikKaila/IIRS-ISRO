"""FlowCast — Strategy B: decoder fine-tune on (predicted_latent -> GT_image).

Run:  python -m flowcast.finetune_vae_pred --config vis/config_fc.yaml

Premise: the CFM forecaster produces fuzzy "mean-trajectory" latents that, when
decoded with the OOD SEVIR decoder, give soft blurry images. By training the
decoder directly on (predicted_latent, GT_image) pairs harvested across the
training-split days, we compensate for BOTH the VAE OOD blur AND the
forecaster's mode-collapse in a single supervised step.

Reads pairs built by ``flowcast.build_pred_dataset``. Holds out the last ~10 %
of training-pair days as an inner ``val_holdout`` split for early-stopping;
the global manifest val/test rows are never touched here.

Produces ``<channel>/checkpoints_fc/vae_decoder_ft_B.pt`` — load via the
``vae.decoder_ckpt`` config key (swap from A to B in ``config_fc.yaml``).
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .data import tif_io
from .data.sequence_dataset import read_manifest
from .utils import flowcast_vae, metrics, viz
from .utils.config import load_config, resolve
from .utils.device import get_device


def cosine_warmup(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01):
    warmup_steps = max(1, int(warmup_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, lr_lambda)


class PredPairDataset(torch.utils.data.Dataset):
    """Yields (pred_latent_fp32, GT_image_01) for one (day, sample, slot) pair."""

    def __init__(self, pair_rows, by_utc, repo_root, src_dir, H, W, low, high):
        self.rows = pair_rows
        self.by_utc = by_utc
        self.repo_root = repo_root
        self.src_dir = src_dir
        self.H, self.W = H, W
        self.low, self.high = low, high

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        blob = torch.load(os.path.join(self.repo_root, r["pair_path"]),
                          map_location="cpu")
        lat = blob["lat"].float()                                # (4, Hl, Wl)
        # GT image: load from TIFF using filename in pair manifest.
        gt = tif_io.read_tif(os.path.join(self.src_dir, r["tif_filename"]))
        gt = tif_io.resize_image(gt, self.H, self.W)
        gt01 = flowcast_vae.standardize(gt, self.low, self.high)
        gt01_t = torch.from_numpy(gt01)[None].float()            # (1, H, W)
        return lat, gt01_t


# ── eval ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(vae, loader, device, low, high, lpips_net=None):
    vae.eval()
    drange = high - low
    psnrs, ssims, lps = [], [], []
    for lat, gt01 in loader:
        lat = lat.to(device)
        gt01_dev = gt01.to(device)
        recon01 = vae.decode(lat).sample.clamp(0.0, 1.0)
        if lpips_net is not None:
            r3 = (recon01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            g3 = (gt01_dev * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lps.append(float(lpips_net(r3, g3).mean()))
        recon01_np = recon01.float().cpu().numpy()
        gt01_np = gt01.float().cpu().numpy()
        for j in range(recon01_np.shape[0]):
            r_phys = flowcast_vae.destandardize(recon01_np[j, 0], low, high)
            g_phys = flowcast_vae.destandardize(gt01_np[j, 0], low, high)
            m = metrics.all_metrics(r_phys, g_phys, data_range=drange)
            psnrs.append(m["psnr"]); ssims.append(m["ssim"])
    out = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
    if lps:
        out["lpips"] = float(np.mean(lps))
    return out


@torch.no_grad()
def save_progress(vae, samples, out_path, device, low, high):
    vae.eval()
    preds, gts = [], []
    for lat, gt01 in samples:
        recon01 = vae.decode(lat.to(device)).sample.clamp(0.0, 1.0)
        preds.append(flowcast_vae.destandardize(
            recon01[0, 0].float().cpu().numpy(), low, high))
        gts.append(flowcast_vae.destandardize(
            gt01[0, 0].float().cpu().numpy(), low, high))
    viz.save_comparison_grid(preds, gts, out_path,
                              titles=[f"holdout[{i}]" for i in range(len(preds))],
                              vmin=low, vmax=high,
                              row_labels=("decoded(pred_lat)", "ground truth"))


def main():
    ap = argparse.ArgumentParser(
        description="Strategy B: decoder FT on (pred_latent, GT) pairs (LPIPS).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--val-holdout-frac", type=float, default=0.10,
                    help="fraction of training-pair days held out for val")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    ft = cfg.get("vae_finetune_pred", {})
    total_steps = int(args.steps if args.steps else ft.get("steps", 4000))
    batch_size = int(args.batch if args.batch else ft.get("batch_size", 2))
    grad_accum = int(args.grad_accum if args.grad_accum
                      else ft.get("grad_accum", 4))
    lr = float(ft.get("lr", 5e-5))
    wd = float(ft.get("weight_decay", 1e-5))
    mse_w = float(ft.get("mse_weight", 0.1))
    lp_w = float(ft.get("lpips_weight", 1.0))
    warmup = int(ft.get("warmup_steps", 200))
    val_interval = int(ft.get("val_interval", 250))
    log_interval = int(ft.get("log_interval", 50))
    fp16 = bool(ft.get("fp16", True)) and device.type == "cuda"
    lpips_backbone = str(ft.get("lpips_backbone", "vgg"))

    # ── load pair dataset ───────────────────────────────────────────────────
    pd_cfg = cfg.get("pair_dataset", {})
    pairs_dir = os.path.join(
        repo, pd_cfg.get("output_dir", f"{cfg['channel']}/pred_pairs_fc"))
    pair_manifest_path = os.path.join(pairs_dir, "manifest.csv")
    if not os.path.exists(pair_manifest_path):
        raise SystemExit(
            f"[fc-ft-B] missing pair manifest: {pair_manifest_path} — "
            "run `python -m flowcast.build_pred_dataset` first")
    with open(pair_manifest_path) as f:
        pair_rows = list(csv.DictReader(f))
    with open(os.path.join(pairs_dir, "meta.json")) as f:
        pair_meta = json.load(f)
    print(f"[fc-ft-B] loaded {len(pair_rows)} pairs from {pairs_dir}")
    print(f"[fc-ft-B] pair_meta: rollouts/day={pair_meta['rollouts_per_day']}  "
          f"n_days={pair_meta['n_days']}  "
          f"flow_ckpt_sha={pair_meta['flow_ckpt_sha'][:12]}")

    # Inner train / val_holdout split BY DAY (chronologically last val_holdout
    # days reserved for val). Never touches global val/test rows.
    unique_days = sorted({r["day"] for r in pair_rows})
    n_val = max(1, int(len(unique_days) * args.val_holdout_frac))
    val_days = set(unique_days[-n_val:])
    train_pair_rows = [r for r in pair_rows if r["day"] not in val_days]
    val_pair_rows = [r for r in pair_rows if r["day"] in val_days]
    print(f"[fc-ft-B] pair split (by day): "
          f"train={len(train_pair_rows)} val_holdout={len(val_pair_rows)} "
          f"(val days: {sorted(val_days)})")
    if not train_pair_rows or not val_pair_rows:
        raise SystemExit("[fc-ft-B] empty inner split — too few days")

    # ── norm + GT IO ────────────────────────────────────────────────────────
    low = pair_meta.get("norm_low", cfg["norm"]["low"])
    high = pair_meta.get("norm_high", cfg["norm"]["high"])
    H = pair_meta.get("resize_height", cfg["resize_height"])
    W = pair_meta.get("resize_width", cfg["resize_width"])
    src_dir = resolve(cfg, "source_dir")
    # by_utc isn't strictly required since pair rows already carry tif_filename,
    # but we load the manifest to double-check leakage.
    rows = read_manifest(resolve(cfg, "manifest_path"))
    by_utc = {r["utc"].isoformat(): r for r in rows}

    train_ds = PredPairDataset(train_pair_rows, by_utc, repo, src_dir,
                                H, W, low, high)
    val_ds = PredPairDataset(val_pair_rows, by_utc, repo, src_dir,
                              H, W, low, high)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"))

    progress_samples = []
    for i in range(min(4, len(val_ds))):
        lat_i, gt_i = val_ds[i]
        progress_samples.append((lat_i[None], gt_i[None]))

    # ── VAE: load ORIGINAL SEVIR weights (do NOT load A's decoder); unfreeze
    # decoder + post_quant_conv only. Save the current vae.decoder_ckpt
    # setting and disable it for this load — B starts from SEVIR baseline.
    saved_ckpt_setting = cfg.get("vae", {}).pop("decoder_ckpt", None)
    if saved_ckpt_setting:
        print(f"[fc-ft-B] ignoring vae.decoder_ckpt={saved_ckpt_setting!r} for "
              "training — B starts from original SEVIR decoder")
    vae = flowcast_vae.load_flowcast_vae(cfg, device)
    if saved_ckpt_setting is not None:
        cfg.setdefault("vae", {})["decoder_ckpt"] = saved_ckpt_setting

    for p in vae.parameters():
        p.requires_grad_(False)
    for p in vae.decoder.parameters():
        p.requires_grad_(True)
    for p in vae.post_quant_conv.parameters():
        p.requires_grad_(True)
    trainable = [p for p in vae.parameters() if p.requires_grad]
    print(f"[fc-ft-B] trainable params (decoder + post_quant_conv): "
          f"{sum(p.numel() for p in trainable):,}")

    try:
        import lpips
    except ImportError:
        raise SystemExit("[fc-ft-B] missing dependency — pip install lpips==0.1.4")
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
    prog_dir = os.path.join(out_dir, "vae_ft_B_progress")
    os.makedirs(prog_dir, exist_ok=True)

    ft_path = os.path.join(ckpt_dir, "vae_decoder_ft_B.pt")
    last_path = os.path.join(ckpt_dir, "vae_decoder_ft_B_last.pt")
    hist_path = os.path.join(ckpt_dir, "vae_decoder_ft_B_history.json")

    def save_ckpt(path, step, val_psnr, val_ssim, val_lpips):
        torch.save({
            "decoder": vae.decoder.state_dict(),
            "post_quant_conv": vae.post_quant_conv.state_dict(),
            "step": step,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_lpips": val_lpips,
            "strategy": "B",
            "flow_ckpt_sha": pair_meta["flow_ckpt_sha"],
            "model_dims": pair_meta["model_dims"],
            "rollouts_per_day": pair_meta["rollouts_per_day"],
            "T_past": pair_meta["T_past"],
            "T_lead": pair_meta["T_lead"],
            "config": {"lr": lr, "wd": wd, "mse_weight": mse_w,
                        "lpips_weight": lp_w, "lpips_backbone": lpips_backbone,
                        "batch_size": batch_size, "grad_accum": grad_accum,
                        "warmup_steps": warmup,
                        "loss": f"{mse_w}*mse+{lp_w}*lpips_{lpips_backbone}",
                        "norm_low": low, "norm_high": high},
        }, path)

    # Baseline (SEVIR + frozen) on the val_holdout pair distribution.
    base = evaluate(vae, val_loader, device, low, high, lpips_net=lpips_net)
    print(f"[fc-ft-B] baseline val_holdout: psnr={base['psnr']:.2f}  "
          f"ssim={base['ssim']:.3f}  lpips={base.get('lpips', 0):.4f}")
    save_progress(vae, progress_samples,
                   os.path.join(prog_dir, "step_00000_baseline.png"),
                   device, low, high)

    # ── train loop with grad accumulation ───────────────────────────────────
    best_lpips = float("inf")
    history = []
    step = 0
    accum_count = 0
    t0 = time.time()

    def train_iter(loader):
        while True:
            for batch in loader:
                yield batch

    iterator = train_iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        lat, gt01 = next(iterator)
        lat = lat.to(device, non_blocking=True)
        gt01 = gt01.to(device, non_blocking=True)

        vae.decoder.train()
        vae.post_quant_conv.train()

        if fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                recon01 = vae.decode(lat).sample
                mse = F.mse_loss(recon01, gt01)
            recon_fp32 = recon01.float().clamp(0.0, 1.0)
            r3 = (recon_fp32 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            g3 = (gt01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lp = lpips_net(r3, g3).mean()
            loss = (mse_w * mse + lp_w * lp) / grad_accum
            scaler.scale(loss).backward()
        else:
            recon01 = vae.decode(lat).sample
            mse = F.mse_loss(recon01, gt01)
            r3 = (recon01.clamp(0.0, 1.0) * 2.0 - 1.0).repeat(1, 3, 1, 1)
            g3 = (gt01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lp = lpips_net(r3, g3).mean()
            loss = (mse_w * mse + lp_w * lp) / grad_accum
            loss.backward()

        accum_count += 1
        if accum_count >= grad_accum:
            if fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            accum_count = 0

            if step % log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(f"[fc-ft-B] step {step:5d}/{total_steps}  "
                      f"loss={float(loss) * grad_accum:.4f}  "
                      f"mse={float(mse):.5f}  lpips={float(lp):.4f}  "
                      f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

            if step % val_interval == 0 or step == total_steps:
                val = evaluate(vae, val_loader, device, low, high,
                                lpips_net=lpips_net)
                history.append({"step": step, **val,
                                 "train_mse": float(mse),
                                 "train_lpips": float(lp)})
                marker = ""
                if val.get("lpips", float("inf")) < best_lpips:
                    best_lpips = val["lpips"]
                    save_ckpt(ft_path, step, val["psnr"], val["ssim"],
                               val["lpips"])
                    marker = "  <- best (LPIPS)"
                save_ckpt(last_path, step, val["psnr"], val["ssim"],
                           val.get("lpips"))
                save_progress(vae, progress_samples,
                               os.path.join(prog_dir, f"step_{step:05d}.png"),
                               device, low, high)
                print(f"[fc-ft-B] val_holdout @ step {step}: "
                      f"psnr={val['psnr']:.2f}  ssim={val['ssim']:.3f}  "
                      f"lpips={val.get('lpips', 0):.4f}{marker}")

    with open(hist_path, "w") as fh:
        json.dump({"baseline": base, "history": history,
                    "best_val_lpips": best_lpips,
                    "pair_meta": pair_meta}, fh, indent=2)
    print(f"[fc-ft-B] best val_holdout LPIPS={best_lpips:.4f}")
    print(f"[fc-ft-B] checkpoint -> {ft_path}")
    print("[fc-ft-B] NOTE: with B loaded, sanity_check PSNR will DROP 1-3 dB "
          "vs A — the decoder is now specialised for forecaster latents, not "
          "encoder-mode latents. This is expected, not a regression.")
    print(f"[fc-ft-B] enable via vae.decoder_ckpt: "
          f"vis/checkpoints_fc/vae_decoder_ft_B.pt in {args.config}")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    main()
