"""Stage 3 (flow) — train the latent conditional flow-matching model.

Run:  python -m src.train_flow --config vis/config.yaml
      python -m src.train_flow --config wv/config.yaml

This trainer ships four agent-validated improvements on top of the original:
1. EMA shadow weights (decay 0.999) — the saved checkpoint is the EMA copy.
2. AdamW + weight_decay (replaces plain Adam).
3. Cosine LR with linear warmup.
4. Short autoregressive unroll to fix train/inference exposure bias — for a
   fraction of batches the model's own one-step prediction (detached) is
   substituted into the context and the loss is computed on the *next* step.

All four are gated by config keys with safe defaults; absence preserves the
old behaviour. Reuses cached SD-VAE latents and the SZA conditioning channel
from Stage 1. Best weights -> ``<ch>/checkpoints/best_flow.pt``.
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

from .data.sequence_dataset import (LatentSequenceDataset, contiguous_runs,
                                    make_windows, read_manifest,
                                    temporal_split)
from .models.flow_model import build_flow_model, cfm_loss
from .models.flow_model import sample as flow_sample
from .utils.config import load_config, resolve
from .utils.device import get_device


# ── reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── EMA shadow weights ──────────────────────────────────────────────────────
class EMA:
    """Tracks an exponential moving average of trainable model parameters."""

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
    """Context manager: load EMA params into the model; restore on exit."""

    def __init__(self, model, ema):
        self.model = model
        self.ema = ema
        self.backup = None

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


# ── cosine LR with linear warmup ────────────────────────────────────────────
def cosine_warmup_scheduler(optimizer, total_steps, warmup_frac=0.05,
                            min_lr_ratio=0.01):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ── data: split datasets, with optional 2-step targets for unroll training ──
def build_flow_datasets(cfg):
    """Train uses target_len = flow.unroll_steps (default 1); val/test = 1."""
    from .utils.solar import solar_elevation_deg

    rows = read_manifest(resolve(cfg, "manifest_path"))
    cadence = cfg["sequence"]["cadence_minutes"]
    input_len = cfg["sequence"]["input_len"]
    dn = cfg.get("day_night", {})
    cond_ch_dataset = 1 if dn.get("enabled", False) else 0
    mask_night = bool(dn.get("enabled", False))

    if cond_ch_dataset:
        lat, lon = dn["scene_lat"], dn["scene_lon"]
        for r in rows:
            elev = solar_elevation_deg(r["utc"], lat, lon)
            r["sza_norm"] = max(-1.0, min(1.0, elev / 90.0))

    tr_rows, va_rows, te_rows = temporal_split(
        rows, cfg["train"]["val_fraction"], cfg["train"]["test_fraction"])

    latent_h = cfg["resize_height"] // 8
    latent_w = cfg["resize_width"] // 8

    # joint_steps > 1 -> Phase C (predict K next frames jointly).
    # Default 4 enables Phase C automatically without a config edit.
    joint_steps = int(cfg["flow"].get("joint_steps", 4))
    if joint_steps > 1:
        train_target_len = joint_steps
    else:
        train_target_len = int(cfg["flow"].get("unroll_steps", 2))

    def ds(split, target_len):
        runs = contiguous_runs(split, cadence, daytime_only=False)
        windows = make_windows(runs, input_len, target_len=target_len)
        return LatentSequenceDataset(windows, input_len, target_len,
                                     cfg["_repo_root"], cond_ch_dataset,
                                     mask_night, latent_h, latent_w)

    # val/test target_len matches train when joint_steps > 1 so the val loss
    # is on the same K-frame task the model is trained for.
    val_target_len = joint_steps if joint_steps > 1 else 1
    return (ds(tr_rows, train_target_len),
            ds(va_rows, val_target_len),
            ds(te_rows, val_target_len))


# ── building the conditioning tensor ────────────────────────────────────────
def _cond_from_parts(past_latents, past_sza, target_sza):
    """Interleave per-step latent+SZA channels into a single (B, cond_ch, H, W)."""
    B, T = past_latents.shape[:2]
    H, W = past_latents.shape[-2:]
    if past_sza is not None:
        x_step = torch.cat([past_latents, past_sza], dim=2)   # (B, T, 5, H, W)
        cond = x_step.reshape(B, T * x_step.shape[2], H, W)
    else:
        cond = past_latents.reshape(B, T * past_latents.shape[2], H, W)
    if target_sza is not None:
        cond = torch.cat([cond, target_sza], dim=1)
    return cond


def make_cond(x, future_cond):
    """Original single-step cond: past T (lat+SZA interleaved) + target SZA."""
    B, T, C, H, W = x.shape
    cond = x.reshape(B, T * C, H, W)
    if future_cond.numel() > 0:
        cond = torch.cat([cond, future_cond[:, 0]], dim=1)
    return cond


# ── one epoch ───────────────────────────────────────────────────────────────
def _build_joint_cond(past_latents, past_sza, future_cond, K):
    """Cond for joint K-frame prediction: past T (lat+SZA interleaved) + K future SZAs."""
    B, T = past_latents.shape[:2]
    has_sza = past_sza is not None
    parts = []
    for ti in range(T):
        parts.append(past_latents[:, ti])
        if has_sza:
            parts.append(past_sza[:, ti])
    if has_sza:
        for ki in range(K):
            parts.append(future_cond[:, ki])
    return torch.cat(parts, dim=1)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, ema=None,
              ode_steps=10, unroll_prob=0.0, joint_steps=1):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, future_cond, y, w in loader:
            x = x.to(device)
            future_cond = future_cond.to(device)
            y = y.to(device)
            w = w.to(device)

            B, T, C_step, H, W = x.shape
            has_sza = (future_cond.numel() > 0)
            past_latents = x[:, :, :4]
            past_sza = x[:, :, 4:] if has_sza else None

            if joint_steps > 1 and y.shape[1] >= joint_steps:
                # ── Phase C: K-frame joint target ────────────────────────────
                K = joint_steps
                target = y[:, :K].reshape(B, K * 4, H, W)         # concat in channels
                cond = _build_joint_cond(past_latents, past_sza, future_cond, K)
                block_w = w[:, :K].mean(dim=1)                    # (B,)
                loss = cfm_loss(model, target, cond, weights=block_w)
            else:
                # ── Phase A.5 single-frame target (optionally with unroll) ───
                target_sza_0 = future_cond[:, 0] if has_sza else None
                do_unroll = (training and unroll_prob > 0.0
                             and y.shape[1] >= 2
                             and torch.rand(()).item() < unroll_prob)
                if do_unroll:
                    cond0 = _cond_from_parts(past_latents, past_sza, target_sza_0)
                    with torch.no_grad():
                        z_pred = flow_sample(model, cond0,
                                             y[:, 0].shape, steps=ode_steps,
                                             device=device)
                    new_past_lat = torch.cat(
                        [past_latents[:, 1:], z_pred.unsqueeze(1)], dim=1)
                    if has_sza:
                        new_past_sza = torch.cat(
                            [past_sza[:, 1:], target_sza_0.unsqueeze(1)], dim=1)
                        target_sza_1 = future_cond[:, 1]
                    else:
                        new_past_sza, target_sza_1 = None, None
                    cond1 = _cond_from_parts(new_past_lat, new_past_sza,
                                             target_sza_1)
                    loss = cfm_loss(model, y[:, 1], cond1, weights=w[:, 1])
                else:
                    cond = _cond_from_parts(past_latents, past_sza, target_sza_0)
                    loss = cfm_loss(model, y[:, 0], cond, weights=w[:, 0])

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Stage 3 (flow): train CFM.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    f = cfg["flow"]
    set_seed(int(f.get("seed", 42)))
    device = get_device()

    train_ds, val_ds, test_ds = build_flow_datasets(cfg)
    print(f"[flow-train] {cfg['channel']}: windows "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("[flow-train] empty split — check manifest / sequence")

    batch_size = int(f["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # determine cond_ch from one batch (joint or single)
    sample_x, sample_fc, _, _ = next(iter(train_loader))
    joint_steps_peek = int(f.get("joint_steps", 4))
    if joint_steps_peek > 1:
        past_lat_pk = sample_x[:, :, :4]
        past_sza_pk = sample_x[:, :, 4:] if sample_fc.numel() > 0 else None
        cond_ch = _build_joint_cond(past_lat_pk, past_sza_pk,
                                    sample_fc, joint_steps_peek).shape[1]
    else:
        cond_ch = make_cond(sample_x, sample_fc).shape[1]
    print(f"[flow-train] cond_ch into model = {cond_ch}  "
          f"(joint_steps={joint_steps_peek})")

    model = build_flow_model(cfg, cond_ch=cond_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[flow-train] flow model params: {n_params:,}")

    # ── AdamW + cosine LR with warmup ────────────────────────────────────────
    weight_decay = float(f.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(f["lr"]),
                                  weight_decay=weight_decay)

    epochs = int(f["epochs"])
    total_steps = epochs * max(1, len(train_loader))
    warmup_frac = float(f.get("warmup_frac", 0.05))
    scheduler = cosine_warmup_scheduler(optimizer, total_steps,
                                        warmup_frac=warmup_frac)

    # ── EMA shadow weights ───────────────────────────────────────────────────
    ema_decay = float(f.get("ema_decay", 0.999))
    ema = EMA(model, decay=ema_decay)

    unroll_steps = int(f.get("unroll_steps", 2))
    unroll_prob = float(f.get("unroll_prob", 0.3))
    ode_steps = int(f.get("ode_steps", 10))
    joint_steps = int(f.get("joint_steps", 4))
    if joint_steps > 1:
        print(f"[flow-train] AdamW wd={weight_decay}  cosine warmup={warmup_frac}  "
              f"EMA decay={ema_decay}  joint_steps={joint_steps} "
              f"(Phase C — unroll disabled when joint > 1)")
    else:
        print(f"[flow-train] AdamW wd={weight_decay}  cosine warmup={warmup_frac}  "
              f"EMA decay={ema_decay}  "
              f"unroll_steps={unroll_steps}  unroll_prob={unroll_prob}")

    ckpt_dir = resolve(cfg, "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_flow.pt")
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}

    best_val, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, device,
                       optimizer=optimizer, scheduler=scheduler, ema=ema,
                       ode_steps=ode_steps, unroll_prob=unroll_prob,
                       joint_steps=joint_steps)
        # validate using EMA weights (a temporary swap, restored on exit)
        with _ema_swap(model, ema):
            va = run_epoch(model, val_loader, device, joint_steps=joint_steps)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                        "lr": optimizer.param_groups[0]["lr"]})

        marker = ""
        if va < best_val:
            best_val = va
            # save EMA weights as "model" so forecast picks them up unchanged.
            # raw weights are intentionally NOT saved — halves the Drive write.
            torch.save({"model": ema.state_dict(),
                        "config": config_snapshot,
                        "epoch": epoch, "val_loss": va,
                        "cond_ch": cond_ch,
                        "joint_steps": joint_steps}, best_path)
            marker = "  <- best (EMA saved)"
        print(f"[flow-train] epoch {epoch:3d}/{epochs}  "
              f"train={tr:.5f}  val={va:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}{marker}")

    with open(os.path.join(ckpt_dir, "flow_history.json"), "w") as fh:
        json.dump({"history": history, "best_val": best_val}, fh, indent=2)
    print(f"[flow-train] best val={best_val:.5f}  -> {best_path}")


if __name__ == "__main__":
    main()
