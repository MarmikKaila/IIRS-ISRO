"""Stage 3 — train the ConvLSTM RNN on cached latents.

Run:  python -m src.train --config vis/config.yaml
      python -m src.train --config wv/config.yaml

VIS trains on daytime-only sequences (config day_night.enabled = true); WV
trains on full-day sequences. Best weights -> ``<ch>/checkpoints/best.pt``.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.sequence_dataset import build_datasets
from .models.convlstm import build_model
from .utils.config import load_config, resolve
from .utils.device import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, tf_ratio=0.0):
    """One pass over a loader. Pass an optimizer to train; omit it to evaluate.

    Loss is a per-target-frame weighted MSE — night frames get weight 0 when
    the dataset has ``mask_night`` enabled (VIS), so dark frames stop
    dominating training.
    """
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, future_cond, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            future_cond = future_cond.to(device) if future_cond.numel() > 0 else None
            pred = model(x, future_steps=y.shape[1],
                         future_cond=future_cond,
                         teacher=(y if training else None),
                         tf_ratio=(tf_ratio if training else 0.0))
            # weighted MSE: average MSE per target frame, then weight by w
            per_frame_mse = (pred - y).pow(2).mean(dim=(2, 3, 4))   # (B, target)
            denom = w.sum().clamp(min=1e-6)
            loss = (per_frame_mse * w).sum() / denom
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser(description="Stage 3: train the ConvLSTM.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = cfg["train"]
    set_seed(tr["seed"])
    device = get_device()

    train_ds, val_ds, test_ds = build_datasets(cfg)
    print(f"[train] {cfg['channel']}: windows "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("[train] empty split — check manifest / sequence settings")

    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=tr["batch_size"])

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] ConvLSTM parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"],
                                 weight_decay=tr.get("weight_decay", 0.0))

    ckpt_dir = resolve(cfg, "checkpoint_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")
    epochs = tr["epochs"]
    config_snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}

    best_val, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        # scheduled sampling: full teacher forcing -> free-running over 70% of run
        if tr.get("scheduled_sampling", True):
            tf_ratio = max(0.0, 1.0 - epoch / (0.7 * epochs))
        else:
            tf_ratio = 1.0
        train_loss = run_epoch(model, train_loader, device, optimizer, tf_ratio)
        val_loss = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "tf_ratio": tf_ratio})
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": config_snapshot,
                        "epoch": epoch, "val_loss": val_loss}, best_path)
            marker = "  <- best"
        print(f"[train] epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"tf={tf_ratio:.2f}{marker}")

    # final test with the best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_loss = run_epoch(model, test_loader, device)
    print(f"[train] best val={best_val:.5f}  test={test_loss:.5f}")
    print(f"[train] checkpoint -> {best_path}")

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump({"history": history, "best_val": best_val,
                   "test_loss": test_loss}, f, indent=2)


if __name__ == "__main__":
    main()
