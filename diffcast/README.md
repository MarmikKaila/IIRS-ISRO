# DiffCast Pipeline — INSAT-3S VIS + WV

EDM latent diffusion forecaster (Karras et al. 2022) on INSAT-3S imagery.
Unlike FlowCast (which uses a frozen, out-of-distribution SEVIR VAE),
DiffCast **fully fine-tunes** the VAE (encoder + decoder) on INSAT data first,
producing in-distribution latents before training the diffusion forecaster.

The result is sharper decoded images at inference without any post-hoc
decoder fine-tune step (no Strategy A / B needed).

---

## Prerequisites

### Hardware

Same requirements as FlowCast. Recommended: A100/L4 (40 GB).
Phase 1 (full VAE fine-tune) is the most memory- and time-intensive step;
the diffusion model is intentionally smaller (`hidden=128`) than the FlowCast
backbone to fit on T4.

| Phase | T4 time | A100 time |
|-------|---------|-----------|
| Phase 1 — VAE fine-tune (15k steps) | 3–6 h | 1–2 h |
| Phase 1 extended (25k steps) | 6–10 h | 2–4 h |
| Phase 2 — sanity + encode | ~15 min | ~5 min |
| Phase 4 — diffusion train (100 epochs) | 8–12 h | 2–4 h |
| Phase 5 — forecast ×8 | ~5 min | ~2 min |

### Google Drive layout

Identical to FlowCast — see `flowcast/README.md`.  
Both pipelines coexist in the same repo; they write to separate directories
(`*_diff` vs `*_fc`) and never share latents or checkpoints.

```
MyDrive/
  ISRO/
    ISRO A.1/
      vis/
        config_fc.yaml      ← flowcast config (shared norm stats)
        config_diff.yaml    ← diffcast config (this pipeline)
      wv/
        config_fc.yaml
        config_diff.yaml
      sample-images/
        Jan26_167972/
        Jan26_168372/
```

> **Important:** run FlowCast Stage 1 (`encode_latents`) on VIS first.
> `train_vae` auto-computes norm stats if they are missing, but having
> FlowCast's stats already in `vis/config_fc.yaml` confirms the sensor range
> is consistent across both pipelines.

---

## Key Difference vs FlowCast

| | FlowCast | DiffCast |
|--|---------|---------|
| VAE | Frozen SEVIR (OOD for INSAT) | Full fine-tune on INSAT |
| Forecaster loss | I-CFM (continuous flow) | EDM (denoising diffusion) |
| Sampler at inference | 10-step Euler ODE | 18-step Heun ODE |
| Blur reduction | Strategy A + B post-hoc | Built-in via in-dist VAE |
| Latent dir | `vis/latents_fc/` | `vis/latents_diff/` |
| Checkpoint | `best_flow.pt` | `best_diff.pt` |
| VAE gate | PSNR ≥ 25 dB | PSNR ≥ 30 dB |

---

## Cell Execution Order

Run cells top-to-bottom. After a Colab disconnect, run **Setup** then
**Session Status** before anything else.

---

## VIS Channel — Phase-by-phase guide

### Setup (run after every fresh Colab session)

Same as FlowCast: mount Drive, `cd` to project root, install deps.
`lpips==0.1.4` is installed here (needed for VAE fine-tune perceptual loss).

### Session status / resume check

Prints all existing DiffCast checkpoints and outputs so you know exactly
which phases are done and where to resume.

---

### Phase 1 — Full VAE fine-tune (~3–6 h on T4)

```bash
python -m diffcast.train_vae --config vis/config_diff.yaml
```

Starts from the SEVIR-pretrained `AutoencoderKL`, unfreezes **every** parameter
(encoder + decoder + quant layers), and trains with:

```
loss = 1.0 × MSE(recon, gt)
     + 0.5 × LPIPS-VGG(recon, gt)
     + 1e-6 × KL(q ∥ N(0, I))
```

The tiny KL weight keeps the latent posterior near-Gaussian so the EDM
diffusion model can sample from it stably. Without the KL term, the encoder
would drift and the diffusion model would fail to learn the latent distribution.

**What happens automatically:**
- If `norm.low / norm.high` are `null` in the config, norm stats are computed
  from the TIFFs and saved back into `vis/config_diff.yaml`.
- If `vis/manifest_diff.csv` doesn't exist, it is built from the TIFF files.

Best-by-val-PSNR model → `vis/checkpoints_diff/vae_full.pt`  
Last-step model → `vis/checkpoints_diff/vae_full_last.pt`  
History → `vis/checkpoints_diff/vae_full_history.json`  
Progress images (every 500 steps) → `vis/outputs_diff/vae_train_progress/`

#### VAE training history plot

The **VAE history plot cell** (in the notebook, added after this phase) reads
`vae_full_history.json` and plots val PSNR over steps alongside the baseline
(pre-fine-tune) PSNR. Run it at any point to check convergence.

#### Gate: PSNR ≥ 30 dB

The **gate + extension cell** reads the history JSON and prints:
- `GATE PASSED` if best val PSNR ≥ 30 dB → proceed to Phase 2
- `GATE FAILED` → run the extension command:

```bash
python -m diffcast.train_vae --config vis/config_diff.yaml --steps 25000
```

This resumes from scratch but overwrites the best checkpoint only on
improvement. If val PSNR is still improving (seen in the history plot), more
steps will push it past the gate. If it has plateaued below 30 dB:
- Try a higher LPIPS weight (1.0) or lower KL weight
- Use an A100 (larger batch sees more diverse INSAT frames per step)

**Resume after disconnect:** re-run Phase 1 from the top. The script does not
checkpoint mid-training, but `vae_full.pt` on Drive is always the best
checkpoint seen so far and is not lost.

---

### Phase 2 — Enable VAE + sanity check + encode latents (~15 min)

#### 2a — Enable the fine-tuned VAE in the config

The **enable cell** uncomments `full_ckpt: vis/checkpoints_diff/vae_full.pt`
in `vis/config_diff.yaml` so that all subsequent scripts load your fine-tuned
VAE instead of the SEVIR baseline.

> Run this cell **once**. If it runs again after the key is already uncommented,
> it is a no-op (the regex only matches the commented form).

#### 2b — Sanity check

```bash
python -m diffcast.sanity_check --config vis/config_diff.yaml --num 30
```

Scores 30 val frames and saves a reconstruction grid to
`vis/outputs_diff/sanity_grid.png`. The **gate cell** immediately after checks
`sanity_metrics.json` and asserts PSNR ≥ 30 dB before you can proceed.

#### 2c — Encode latents with the new VAE

```bash
python -m diffcast.encode_latents --config vis/config_diff.yaml
```

Writes a fresh set of latents to `vis/latents_diff/` using the fine-tuned
encoder. These are the latents the diffusion model trains on. They are
**completely separate** from the FlowCast latents in `vis/latents_fc/`.

---

### Phase 4 — Train EDM latent diffusion (~8–12 h on T4)

```bash
python -m diffcast.train_diffusion --config vis/config_diff.yaml
```

Same Earthformer-UNet backbone as FlowCast, wrapped with EDM preconditioning:
- Training: sample σ from a log-normal distribution (P_mean=-1.2, P_std=1.2),
  add σ-scaled noise to latents, predict denoised latents, weight loss by
  `1 / (σ² + σ_data²)`.
- Inference: Heun second-order ODE over a Karras σ schedule
  (`sigma_max=80 → sigma_min=0.002`, 18 steps, ρ=7).

Best EMA checkpoint → `vis/checkpoints_diff/best_diff.pt`  
History → `vis/checkpoints_diff/diff_history.json`

The **diffusion training history plot cell** reads this JSON and plots
train + val loss over epochs.

**Resume after disconnect:** same as Phase 1 — restart from epoch 1, best
checkpoint is preserved on Drive.

---

### Phase 5 — 48-frame forecast (8-sample ensemble)

```bash
python -m diffcast.forecast_diff --config vis/config_diff.yaml --samples 8
```

Heun ODE, 18 steps per block × 4 autoregressive blocks = 48 frames.
Same 3-way aggregation as FlowCast (pixmean / medoid / latmean).

VIS night slots are filled with the mean of real night frames; metrics are
daytime-only.

**Outputs** (`vis/outputs_diff/`):
```
diff_forecast_<date>_ens8_pixmean.npy / .gif / _grid.png / _compare.png
diff_forecast_<date>_ens8_medoid.npy  / .gif / _grid.png / _compare.png
diff_forecast_<date>_ens8_latmean.npy / .gif / _grid.png / _compare.png
diff_forecast_<date>_metrics.json     # PSNR/SSIM/CSI/HSS/CRPS
```

---

### Phase 6 — Compare DiffCast vs FlowCast

The comparison cell prints a side-by-side table across all metric files in
`vis/outputs_fc/` and `vis/outputs_diff/`. Because DiffCast has an
in-distribution VAE, expect:

- Medoid PSNR: +2–4 dB over FlowCast baseline
- SSIM: +0.05–0.10
- CRPS: lower (tighter ensemble spread, less mean-drift)

If DiffCast underperforms FlowCast, Phase 1 likely did not converge — re-run
with `--steps 25000`.

---

## WV Channel

Run the same pipeline with `wv/config_diff.yaml`:

```bash
python -m diffcast.train_vae      --config wv/config_diff.yaml
# [enable full_ckpt in wv/config_diff.yaml]
python -m diffcast.sanity_check   --config wv/config_diff.yaml --num 30
python -m diffcast.encode_latents --config wv/config_diff.yaml
python -m diffcast.train_diffusion --config wv/config_diff.yaml
python -m diffcast.forecast_diff  --config wv/config_diff.yaml --samples 8
```

Only run WV after VIS Phase 1 has passed its gate. WV has no day/night
handling; the full 48-slot horizon is predicted and scored.

Note: the SEVIR VAE is somewhat closer to WV physics (radar ↔ brightness temp)
than to VIS, so the Phase 1 baseline PSNR may already be 20–22 dB before
fine-tuning, and the gate (30 dB) may be reached faster.

---

## Key Files at a Glance

```
vis/config_diff.yaml
    norm.low, norm.high           auto-filled by train_vae (Phase 1)
    vae.full_ckpt                 uncommented by the enable cell (Phase 2a)

vis/manifest_diff.csv             auto-built by train_vae if absent

vis/latents_diff/                 in-distribution latents (Phase 2c output)

vis/checkpoints_diff/
    vae_full.pt                   best full-VAE checkpoint (Phase 1)
    vae_full_last.pt              last-step full-VAE checkpoint
    vae_full_history.json         step-by-step val PSNR + loss
    best_diff.pt                  best EMA diffusion model (Phase 4)
    diff_history.json             epoch-by-epoch train/val loss

vis/outputs_diff/
    sanity_grid.png               Phase 2b reconstruction grid
    sanity_metrics.json           Phase 2b PSNR/SSIM + gate flag
    vae_train_progress/           per-500-step progress PNGs (Phase 1)
    diff_forecast_<date>_*        Phase 5 outputs
```

---

## Troubleshooting

**`norm stats missing — run train_vae first`**
`train_vae` auto-computes them. If the config still shows `null`, the script
errored before saving. Delete `vis/manifest_diff.csv`, fix the error, and
re-run Phase 1 from scratch.

**VAE gate failed (best PSNR < 30 dB)**
Check the history plot. If PSNR is still improving, extend with `--steps 25000`.
If it has plateaued well below 30 dB, increase `vae_train.lpips_weight` to 1.0
in `vis/config_diff.yaml` and re-run.

**`sanity_check` PSNR high but Phase 5 outputs still blurry**
The diffusion model converged to the data mean. Check the Phase 4 loss history —
if val loss is much higher than train loss, the model overfits to a small number
of latent patterns. Add more TIFF data or reduce `diffusion.hidden` / `n_blocks`.

**`need N prior frames, have M` at forecast time**
The manifest doesn't have enough frames before the forecast date. Check that
`vis/manifest_diff.csv` was built from the full TIFF set and not truncated.

**OOM during Phase 4**
Set `diffusion.batch_size: 1` in `vis/config_diff.yaml`. The EDM model is
already small (`hidden=128`, `n_blocks=2`) so OOM during Phase 4 is rare.

**Phase 1 OOM on T4**
Set `vae_train.batch_size: 4` (from 8). The full-VAE fine-tune runs the
encoder forward pass which is heavier than the frozen encode path in FlowCast.

**DiffCast forecast worse than FlowCast**
Almost always a Phase 1 convergence issue. The diffusion model can only be as
good as the VAE it sits on. Re-run Phase 1 with `--steps 25000` and confirm
the gate is passed before training the diffusion model.
