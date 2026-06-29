"""FlowCast Stage 3 — train the latent Conditional Flow Matching model.

Run:  python -m flowcast.train_flow --config vis/config_fc.yaml
      python -m flowcast.train_flow --config wv/config_fc.yaml

Implements (compact T4-friendly):
  - AdamW + cosine LR + linear warmup
  - EMA shadow weights (decay 0.999)
  - I-CFM loss with sigma = 0.01 (paper)
  - FP16 autocast + GradScaler
  - Best EMA-weights checkpoint at ``<ch>/checkpoints_fc/best_flow.pt``
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

from .data.sequence_dataset import build_datasets
from .models.flow_model import build_model
from .utils.config import load_config, resolve
from .utils.device import get_device


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── EMA ─────────────────────────────────────────────────────────────────────
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


# ── cosine warmup ──────────────────────────────────────────────────────────
def cosine_warmup(optimizer, total_steps, warmup_frac=0.01, min_lr_ratio=0.01):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return LambdaLR(optimizer, lr_lambda)


# ── epoch ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer=None, scheduler=None,
               scaler=None, ema=None, mask_night=False, fp16=True):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    use_amp = bool(fp16 and device.type == "cuda")
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for past, sza_p, fut, sza_f, w in loader:
            past = past.to(device, non_blocking=True)
            fut = fut.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True) if mask_night else None

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = model.cfm_loss(past, fut, weights=w)
            else:
                loss = model.cfm_loss(past, fut, weights=w)

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


class _null_autocast:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="FlowCast Stage 3: train CFM.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    f = cfg["flow"]
    set_seed(int(f.get("seed", 42)))
    device = get_device()

    # Auto-fit for low-VRAM GPUs (T4 ≈ 15GB). The paper used 4xH100 (80GB each),
    # so the bare hidden=192 + depth=4 + batch=4 spec OOMs on T4. We cap the
    # heaviest knobs here so the model still trains paper-style without a
    # config edit on Drive. (Gradient checkpointing in the U-Net does the rest.)
    if device.type == "cuda":
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_gb < 24.0:
            new_hidden = min(int(f.get("hidden", 128)), 128)
            new_bs = min(int(f.get("batch_size", 2)), 2)
            new_depth = min(int(f.get("n_blocks", 2)), 2)
            if (new_hidden, new_bs, new_depth) != (
                    int(f.get("hidden")), int(f.get("batch_size")),
                    int(f.get("n_blocks"))):
                print(f"[fc-train] Detected {gpu_gb:.1f}GB GPU; capping flow "
                      f"hidden->{new_hidden}, n_blocks->{new_depth}, "
                      f"batch_size->{new_bs} for memory")
            f["hidden"] = new_hidden
            f["n_blocks"] = new_depth
            f["batch_size"] = new_bs

    train_ds, val_ds, test_ds = build_datasets(cfg)
    print(f"[fc-train] {cfg['channel']}: windows "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("[fc-train] empty split — check manifest / sequence lengths")

    batch_size = int(f.get("batch_size", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0,
                             pin_memory=True)

    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8
    model = build_model(cfg, latent_ch=4, latent_h=latent_h, latent_w=latent_w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fc-train] flow model params: {n_params:,}")

    weight_decay = float(f.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(f.get("lr", 5e-4)),
                                   weight_decay=weight_decay, betas=(0.9, 0.999))
    epochs = int(f.get("epochs", 100))
    total_steps = epochs * max(1, len(train_loader))
    scheduler = cosine_warmup(optimizer, total_steps,
                                warmup_frac=float(f.get("warmup_frac", 0.01)))

    ema_decay = float(f.get("ema_decay", 0.999))
    ema = EMA(model, decay=ema_decay)

    fp16 = bool(f.get("fp16", True))
    scaler = torch.amp.GradScaler("cuda") if (fp16 and device.type == "cuda") else None

    mask_night = bool(cfg.get("day_night", {}).get("enabled", False))
    print(f"[fc-train] AdamW wd={weight_decay}  cosine warmup="
          f"{f.get('warmup_frac', 0.01)}  EMA={ema_decay}  "
          f"FP16={fp16}  sigma={f.get('sigma', 0.01)}  "
          f"T_past={cfg['sequence']['input_len']} T_lead={f['joint_steps']}")

    ckpt_dir = resolve(cfg, "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_flow.pt")
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}

    best_val, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, device,
                        optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, ema=ema, mask_night=mask_night, fp16=fp16)
        with _ema_swap(model, ema):
            va = run_epoch(model, val_loader, device, mask_night=mask_night,
                           fp16=fp16)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                         "lr": optimizer.param_groups[0]["lr"]})
        marker = ""
        if va < best_val:
            best_val = va
            torch.save({"model": ema.state_dict(),
                        "config": config_snapshot,
                        "epoch": epoch, "val_loss": va,
                        "T_past": int(cfg["sequence"]["input_len"]),
                        "T_lead": int(f["joint_steps"])}, best_path)
            marker = "  <- best (EMA saved)"
        print(f"[fc-train] epoch {epoch:3d}/{epochs}  "
              f"train={tr:.5f}  val={va:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}{marker}")

    with open(os.path.join(ckpt_dir, "flow_history.json"), "w") as fh:
        json.dump({"history": history, "best_val": best_val}, fh, indent=2)
    print(f"[fc-train] best val={best_val:.5f}  -> {best_path}")


if __name__ == "__main__":
    main()
