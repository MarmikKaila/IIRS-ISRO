"""diffcast Phase 4 — train the EDM latent diffusion forecaster.

Run:  python -m diffcast.train_diffusion --config vis/config_diff.yaml

Mirrors flowcast/train_flow.py's scaffolding (EMA, cosine warmup, fp16
autocast, gradient checkpointing) but plugs in:
  - the EDM-preconditioned Earthformer-UNet from diffcast.models.diffusion_model
  - the EDM loss (sample sigma log-normal, predict denoised, weight by sigma)
  - diffcast latents at vis/latents_diff/ via the existing FlowCastLatentDataset

The dataset reuses flowcast's ``build_datasets`` — we alias
``cfg["flow"] = cfg["diffusion"]`` at runtime so the existing helper reads
the diffusion T_lead from the right place, without editing flowcast.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from flowcast.data.sequence_dataset import build_datasets
from flowcast.utils.config import load_config, resolve
from flowcast.utils.device import get_device

from .models.diffusion_model import build_diffusion_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── EMA (mirrors flowcast/train_flow.py::EMA) ───────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(),
                                                     alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.shadow.items()}


class _ema_swap:
    def __init__(self, model, ema):
        self.model = model; self.ema = ema; self.backup = None

    def __enter__(self):
        self.backup = {n: p.detach().clone()
                       for n, p in self.model.named_parameters()
                       if p.requires_grad and n in self.ema.shadow}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ema.shadow and p.requires_grad:
                    p.data.copy_(self.ema.shadow[n])
        return self.model

    def __exit__(self, *exc):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.backup:
                    p.data.copy_(self.backup[n])


def cosine_warmup(optimizer, total_steps, warmup_frac=0.01, min_lr_ratio=0.01):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, device, optimizer=None, scheduler=None,
              scaler=None, ema=None, fp16=True):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    use_amp = bool(fp16 and device.type == "cuda")
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for past, _sza_p, fut, _sza_f, _w in loader:
            past = past.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = model.edm_loss(past, fut)
            else:
                loss = model.edm_loss(past, fut)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
            total += float(loss.item()) * past.size(0)
            count += past.size(0)
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser(description="diffcast Phase 4: train EDM diffusion.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Alias diffusion config under "flow" so flowcast's build_datasets reads
    # joint_steps / batch_size / etc. from the right place. NO file change to
    # flowcast — pure runtime dict mutation on the cfg we loaded.
    cfg["flow"] = cfg["diffusion"]

    d = cfg["diffusion"]
    set_seed(int(d.get("seed", 42)))
    device = get_device()

    # T4 auto-cap (same idea as flowcast/train_flow.py:148-162)
    if device.type == "cuda":
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_gb < 24.0:
            new_hidden = min(int(d.get("hidden", 128)), 128)
            new_bs = min(int(d.get("batch_size", 2)), 2)
            new_depth = min(int(d.get("n_blocks", 2)), 2)
            if (new_hidden, new_bs, new_depth) != (
                    int(d.get("hidden")), int(d.get("batch_size")),
                    int(d.get("n_blocks"))):
                print(f"[diffcast-train] Detected {gpu_gb:.1f}GB GPU; capping "
                      f"hidden->{new_hidden}, n_blocks->{new_depth}, "
                      f"batch_size->{new_bs} for memory")
            d["hidden"] = new_hidden
            d["n_blocks"] = new_depth
            d["batch_size"] = new_bs
            # keep cfg["flow"] aliased to the same dict so build_datasets sees the cap
            cfg["flow"] = d

    train_ds, val_ds, test_ds = build_datasets(cfg)
    print(f"[diffcast-train] {cfg['channel']}: windows "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("[diffcast-train] empty split — check manifest_diff.csv")

    batch_size = int(d.get("batch_size", 2))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0,
                             pin_memory=True)

    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8
    model = build_diffusion_model(cfg, latent_ch=4,
                                    latent_h=latent_h, latent_w=latent_w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diffcast-train] diffusion model params: {n_params:,}")

    weight_decay = float(d.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=float(d.get("lr", 5e-4)),
                                   weight_decay=weight_decay,
                                   betas=(0.9, 0.999))
    epochs = int(d.get("epochs", 100))
    total_steps = epochs * max(1, len(train_loader))
    scheduler = cosine_warmup(optimizer, total_steps,
                                warmup_frac=float(d.get("warmup_frac", 0.01)))

    ema_decay = float(d.get("ema_decay", 0.999))
    ema = EMA(model, decay=ema_decay)

    fp16 = bool(d.get("fp16", True))
    scaler = (torch.amp.GradScaler("cuda")
              if (fp16 and device.type == "cuda") else None)

    print(f"[diffcast-train] AdamW wd={weight_decay}  cosine warmup="
          f"{d.get('warmup_frac', 0.01)}  EMA={ema_decay}  "
          f"FP16={fp16}  sigma_data={d.get('sigma_data', 0.5)}  "
          f"T_past={cfg['sequence']['input_len']} T_lead={d['joint_steps']}")

    ckpt_dir = resolve(cfg, "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_diff.pt")
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}

    best_val, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, device,
                        optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, ema=ema, fp16=fp16)
        with _ema_swap(model, ema):
            va = run_epoch(model, val_loader, device, fp16=fp16)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                         "lr": optimizer.param_groups[0]["lr"]})
        marker = ""
        if va < best_val:
            best_val = va
            torch.save({"model": ema.state_dict(),
                        "config": config_snapshot,
                        "epoch": epoch, "val_loss": va,
                        "T_past": int(cfg["sequence"]["input_len"]),
                        "T_lead": int(d["joint_steps"])}, best_path)
            marker = "  <- best (EMA saved)"
        print(f"[diffcast-train] epoch {epoch:3d}/{epochs}  "
              f"train={tr:.5f}  val={va:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}{marker}")

    with open(os.path.join(ckpt_dir, "diff_history.json"), "w") as fh:
        json.dump({"history": history, "best_val": best_val}, fh, indent=2)
    print(f"[diffcast-train] best val={best_val:.5f}  -> {best_path}")


if __name__ == "__main__":
    main()
