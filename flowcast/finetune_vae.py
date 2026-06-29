"""FlowCast — Strategy A: decoder fine-tune on real (encode -> decode) pairs.

Run:  python -m flowcast.finetune_vae --config vis/config_fc.yaml

Premise: the frozen ``b-rbmp/flowcast-cfm-sevir`` AutoencoderKL is trained on
SEVIR radar — out-of-distribution for INSAT VIS. The encoder is left untouched
(so the cached latents at ``vis/latents_fc/`` and the trained CFM forecaster
stay valid) and only the decoder + post_quant_conv are fine-tuned to
reconstruct INSAT-domain detail with a 0.1*MSE + 1.0*LPIPS objective.

Produces ``<channel>/checkpoints_fc/vae_decoder_ft_A.pt`` — load via the
``vae.decoder_ckpt`` config key.
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

from .data import tif_io
from .data.sequence_dataset import read_manifest, temporal_split
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


# ── data ────────────────────────────────────────────────────────────────────
class RealPairDataset(torch.utils.data.Dataset):
    """Yields (cached_latent, GT_image_01) for VIS day frames.

    Latents are loaded from ``latent_path`` in the manifest (already encoded by
    ``encode_latents.py``). GT images are read from the original TIFF, resized,
    and standardized to [0, 1] via ``flowcast_vae.standardize``.
    """

    def __init__(self, rows, repo_root, src_dir, H, W, low, high):
        self.rows = rows
        self.repo_root = repo_root
        self.src_dir = src_dir
        self.H, self.W = H, W
        self.low, self.high = low, high

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        lat = torch.load(os.path.join(self.repo_root, r["latent_path"]),
                         map_location="cpu").float()                  # (4, Hl, Wl)
        gt = tif_io.read_tif(os.path.join(self.src_dir, r["filename"]))
        gt = tif_io.resize_image(gt, self.H, self.W)
        gt01 = flowcast_vae.standardize(gt, self.low, self.high)
        gt01_t = torch.from_numpy(gt01)[None].float()                  # (1, H, W)
        return lat, gt01_t


# ── eval ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(vae, loader, device, low, high):
    vae.eval()
    drange = high - low
    psnrs, ssims = [], []
    for lat, gt01 in loader:
        lat = lat.to(device)
        recon01 = vae.decode(lat).sample.clamp(0.0, 1.0)
        recon01 = recon01.float().cpu().numpy()
        gt01_np = gt01.float().cpu().numpy()
        for j in range(recon01.shape[0]):
            r_phys = flowcast_vae.destandardize(recon01[j, 0], low, high)
            g_phys = flowcast_vae.destandardize(gt01_np[j, 0], low, high)
            m = metrics.all_metrics(r_phys, g_phys, data_range=drange)
            psnrs.append(m["psnr"]); ssims.append(m["ssim"])
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ── progress viz ────────────────────────────────────────────────────────────
@torch.no_grad()
def save_progress(vae, samples, out_path, device, low, high):
    """4 fixed frames: top row = baseline reconstruction (before this step),
    bottom row = current decoder output. ``samples`` is a list of (lat, gt01)
    tensors pre-loaded in CPU memory."""
    vae.eval()
    preds, gts = [], []
    for lat, gt01 in samples:
        recon01 = vae.decode(lat.to(device)).sample.clamp(0.0, 1.0)
        preds.append(flowcast_vae.destandardize(
            recon01[0, 0].float().cpu().numpy(), low, high))
        gts.append(flowcast_vae.destandardize(
            gt01[0, 0].float().cpu().numpy(), low, high))
    viz.save_comparison_grid(preds, gts, out_path,
                              titles=[f"val[{i}]" for i in range(len(preds))],
                              vmin=low, vmax=high,
                              row_labels=("decoded", "ground truth"))


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Strategy A: decoder FT on real encode-decode pairs (LPIPS).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override vae_finetune.steps from config")
    ap.add_argument("--batch", type=int, default=None,
                    help="override vae_finetune.batch_size from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = cfg["_repo_root"]
    device = get_device()

    ft = cfg.get("vae_finetune", {})
    total_steps = int(args.steps if args.steps else ft.get("steps", 4000))
    batch_size = int(args.batch if args.batch else ft.get("batch_size", 8))
    lr = float(ft.get("lr", 1e-5))
    wd = float(ft.get("weight_decay", 1e-4))
    mse_w = float(ft.get("mse_weight", 0.1))
    lp_w = float(ft.get("lpips_weight", 1.0))
    warmup = int(ft.get("warmup_steps", 200))
    val_interval = int(ft.get("val_interval", 250))
    log_interval = int(ft.get("log_interval", 50))
    fp16 = bool(ft.get("fp16", True)) and device.type == "cuda"
    lpips_backbone = str(ft.get("lpips_backbone", "vgg"))

    # ── data ────────────────────────────────────────────────────────────────
    low, high = cfg["norm"]["low"], cfg["norm"]["high"]
    if low is None:
        raise SystemExit("[fc-ft-A] norm stats missing — run encode_latents first")
    H, W = cfg["resize_height"], cfg["resize_width"]
    src_dir = resolve(cfg, "source_dir")

    rows = read_manifest(resolve(cfg, "manifest_path"))
    day_rows = [r for r in rows if r.get("is_day", True) is not False]
    # Only keep rows whose cached latent exists on disk.
    day_rows = [r for r in day_rows
                if os.path.exists(os.path.join(repo, r["latent_path"]))]

    tr_rows, va_rows, te_rows = temporal_split(
        day_rows, cfg["train"]["val_fraction"], cfg["train"]["test_fraction"])
    print(f"[fc-ft-A] day-only rows: train={len(tr_rows)} val={len(va_rows)} "
          f"test={len(te_rows)}")
    if len(tr_rows) == 0 or len(va_rows) == 0:
        raise SystemExit("[fc-ft-A] empty split — manifest too small")

    train_ds = RealPairDataset(tr_rows, repo, src_dir, H, W, low, high)
    val_ds = RealPairDataset(va_rows, repo, src_dir, H, W, low, high)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"))

    # Cache 4 fixed val samples for progress viz (loaded once, reused).
    progress_samples = []
    for i in range(min(4, len(val_ds))):
        lat_i, gt_i = val_ds[i]
        progress_samples.append((lat_i[None], gt_i[None]))

    # ── VAE: unfreeze decoder + post_quant_conv only ────────────────────────
    vae = flowcast_vae.load_flowcast_vae(cfg, device)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in vae.decoder.parameters():
        p.requires_grad_(True)
    for p in vae.post_quant_conv.parameters():
        p.requires_grad_(True)
    trainable = [p for p in vae.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[fc-ft-A] trainable params (decoder + post_quant_conv): {n_train:,}")

    # ── LPIPS perceptual loss ───────────────────────────────────────────────
    try:
        import lpips
    except ImportError:
        raise SystemExit("[fc-ft-A] missing dependency — pip install lpips==0.1.4")
    lpips_net = lpips.LPIPS(net=lpips_backbone).to(device).eval()
    for p in lpips_net.parameters():
        p.requires_grad_(False)

    # ── optimizer / scheduler / fp16 ────────────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd,
                                   betas=(0.9, 0.999))
    scheduler = cosine_warmup(optimizer, total_steps, warmup)
    scaler = torch.amp.GradScaler("cuda") if fp16 else None

    ckpt_dir = resolve(cfg, "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_dir = resolve(cfg, "output_dir")
    prog_dir = os.path.join(out_dir, "vae_ft_A_progress")
    os.makedirs(prog_dir, exist_ok=True)

    ft_path = os.path.join(ckpt_dir, "vae_decoder_ft_A.pt")
    last_path = os.path.join(ckpt_dir, "vae_decoder_ft_A_last.pt")
    hist_path = os.path.join(ckpt_dir, "vae_decoder_ft_A_history.json")

    def save_ckpt(path, step, val_psnr, val_ssim):
        torch.save({
            "decoder": vae.decoder.state_dict(),
            "post_quant_conv": vae.post_quant_conv.state_dict(),
            "step": step,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "strategy": "A",
            "config": {"lr": lr, "wd": wd, "mse_weight": mse_w,
                        "lpips_weight": lp_w, "lpips_backbone": lpips_backbone,
                        "batch_size": batch_size, "warmup_steps": warmup,
                        "loss": f"{mse_w}*mse+{lp_w}*lpips_{lpips_backbone}",
                        "norm_low": low, "norm_high": high},
        }, path)

    # Baseline (SEVIR decoder) val score before any training.
    base_psnr, base_ssim = evaluate(vae, val_loader, device, low, high)
    print(f"[fc-ft-A] baseline val: psnr={base_psnr:.2f}  ssim={base_ssim:.3f}")
    save_progress(vae, progress_samples,
                   os.path.join(prog_dir, "step_00000_baseline.png"),
                   device, low, high)

    # ── train loop ──────────────────────────────────────────────────────────
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
        lat, gt01 = next(iterator)
        lat = lat.to(device, non_blocking=True)
        gt01 = gt01.to(device, non_blocking=True)

        vae.decoder.train()
        vae.post_quant_conv.train()
        optimizer.zero_grad(set_to_none=True)

        if fp16:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                recon01 = vae.decode(lat).sample  # bypass @no_grad wrapper
                mse = F.mse_loss(recon01, gt01)
            # LPIPS in fp32 for stability.
            recon_fp32 = recon01.float().clamp(0.0, 1.0)
            r3 = (recon_fp32 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            g3 = (gt01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lp = lpips_net(r3, g3).mean()
            loss = mse_w * mse + lp_w * lp
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            recon01 = vae.decode(lat).sample
            mse = F.mse_loss(recon01, gt01)
            r3 = (recon01.clamp(0.0, 1.0) * 2.0 - 1.0).repeat(1, 3, 1, 1)
            g3 = (gt01 * 2.0 - 1.0).repeat(1, 3, 1, 1)
            lp = lpips_net(r3, g3).mean()
            loss = mse_w * mse + lp_w * lp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        scheduler.step()
        step += 1

        if step % log_interval == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"[fc-ft-A] step {step:5d}/{total_steps}  "
                  f"loss={float(loss):.4f}  mse={float(mse):.5f}  "
                  f"lpips={float(lp):.4f}  lr={lr_now:.2e}  "
                  f"({elapsed:.0f}s)")

        if step % val_interval == 0 or step == total_steps:
            val_psnr, val_ssim = evaluate(vae, val_loader, device, low, high)
            history.append({"step": step, "val_psnr": val_psnr,
                             "val_ssim": val_ssim,
                             "loss": float(loss), "mse": float(mse),
                             "lpips": float(lp)})
            marker = ""
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_ckpt(ft_path, step, val_psnr, val_ssim)
                marker = "  <- best"
            save_ckpt(last_path, step, val_psnr, val_ssim)
            save_progress(vae, progress_samples,
                           os.path.join(prog_dir, f"step_{step:05d}.png"),
                           device, low, high)
            print(f"[fc-ft-A] val @ step {step}: psnr={val_psnr:.2f}  "
                  f"ssim={val_ssim:.3f}{marker}")

    with open(hist_path, "w") as fh:
        json.dump({"baseline": {"val_psnr": base_psnr, "val_ssim": base_ssim},
                    "history": history, "best_val_psnr": best_psnr}, fh, indent=2)
    print(f"[fc-ft-A] best val psnr={best_psnr:.2f} (baseline {base_psnr:.2f})")
    print(f"[fc-ft-A] checkpoint -> {ft_path}")
    print(f"[fc-ft-A] enable via vae.decoder_ckpt: vis/checkpoints_fc/"
          f"vae_decoder_ft_A.pt in {args.config}")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    main()
