# FlowCast Pipeline — INSAT-3S VIS + WV

Latent Conditional Flow Matching (I-CFM) forecaster based on
[FlowCast (ICLR 2026)](https://github.com/b-rbmp/flowcast-cfm-sevir).
Produces 48-frame (24 h) day-ahead nowcasts for INSAT-3S VIS (visible
reflectance) and WV (water-vapour brightness temperature) channels independently.

---

## Prerequisites

### Hardware

| Tier | Runtime | Stage 3 time |
|------|---------|-------------|
| Free Colab | T4 (15 GB) | 12–24 h over multiple sessions |
| Colab Pro | A100 (40 GB) | 4–6 h in one session |
| Recommended | L4 / A100 | < 6 h |

Set the runtime **before** running any cell:
**Runtime → Change runtime type → T4 GPU** (or A100 if available).

### Google Drive layout

Upload or clone the entire repo to Drive, then verify the structure:

```
MyDrive/
  ISRO/
    ISRO A.1/            ← repo root  (edit PROJECT_DIR if path differs)
      flowcast/
      diffcast/
      vis/
        config_fc.yaml
      wv/
        config_fc.yaml
      sample-images/
        Jan26_167972/    ← VIS TIFFs  (3SIMG_26JAN2026_*_IMG_VIS.tif)
        Jan26_168372/    ← WV  TIFFs  (3SIMG_26JAN2026_*_IMG_WV.tif)
      requirements.txt
```

If your Drive path differs from `MyDrive/ISRO/ISRO A.1`, change `PROJECT_DIR`
in the **Setup** cell before running anything else.

### Dependencies (auto-installed by the Setup cell)

```
requirements.txt   (torch, diffusers, scikit-image, tifffile, etc.)
imagecodecs        (TIFF compression support)
huggingface_hub    (downloads FlowCast VAE from HF Hub)
lpips==0.1.4       (for Strategy A + B blur reduction)
```

The FlowCast VAE (`b-rbmp/flowcast-cfm-sevir`) is downloaded from Hugging Face
on the first `encode_latents` run and cached to `~/.cache/huggingface/`.
It requires internet access; make sure Colab has network access enabled.

---

## Cell Execution Order

Run every cell **top to bottom**. Never skip a section.
After a Colab disconnect, run the **Setup** cell and the **Session Status** cell
first to see exactly which stages are already done.

---

## VIS Channel — Stage-by-stage guide

### Setup (run after every fresh Colab session)

Mounts Drive, `cd`s into the project root, and installs all dependencies.
`torch.cuda.is_available()` should print `True`.

### Session status / resume check

Prints which checkpoints and outputs already exist so you know where to
continue after a disconnect. Nothing is run or changed by this cell.

### Stage 1 — Encode TIFFs to latents (~10 min on T4)

```bash
python -m flowcast.encode_latents --config vis/config_fc.yaml
```

- Reads every `*_IMG_VIS.tif` from `sample-images/Jan26_167972/`.
- Computes 1st/99th-percentile normalisation stats on the first run and writes
  them back to `vis/config_fc.yaml` (the `norm.low` / `norm.high` keys).
- Encodes each frame with the frozen SEVIR FlowCast VAE and saves one
  `vis/latents_fc/<channel>_<date>_<time>.pt` per frame.
- Writes `vis/manifest_fc.csv` (UTC, IST, filename, latent path, is_day flag).

**Resume:** safe to re-run; output is deterministic and latents are overwritten.

**Expected output:**
```
[fc-encode] channel=vis  files=1488
[fc-encode] norm low=... high=... (saved to config)
[fc-encode] cached 1488 latents -> vis/latents_fc
[fc-encode] manifest -> vis/manifest_fc.csv
```

### Stage 2 — VAE sanity check + gate (decision gate)

```bash
python -m flowcast.sanity_check --config vis/config_fc.yaml
```

Decodes 20 cached latents and compares them to their source TIFFs.
Saves a reconstruction grid to `vis/outputs_fc/sanity_grid.png` and metrics
to `vis/outputs_fc/sanity_metrics.json`.

**Gate (enforced by the gate-check cell immediately after):**

| Mean PSNR | Decision |
|-----------|---------|
| ≥ 25 dB | Proceed to Stage 3 |
| 20–25 dB | Proceed, but forecast is ceiling-bounded by VAE quality |
| < 20 dB  | Stop — FlowCast VAE is very OOD; fall back to the SD-VAE pipeline or run DiffCast |

### Stage 3 — Train the CFM flow model (12–24 h on T4)

```bash
python -m flowcast.train_flow --config vis/config_fc.yaml
```

Trains an Earthformer-UNet backbone with I-CFM loss (σ = 0.01, paper spec),
AdamW + cosine-warmup LR, EMA (decay 0.999), and FP16 autocast.

**T4 memory auto-scaling:** the script detects GPU VRAM. For GPUs < 24 GB it
automatically caps `hidden → 128`, `n_blocks → 2`, `batch_size → 2`. The paper
spec in `config_fc.yaml` (192 / 4 / 4) is for 4×H100; these auto-caps keep
training stable on T4 without any manual config edit.

Best EMA weights are saved to `vis/checkpoints_fc/best_flow.pt` every time
val loss improves. Epoch-by-epoch history is in `vis/checkpoints_fc/flow_history.json`.

**Training loss plot cell** (added to notebook): reads `flow_history.json` and
plots train + val CFM loss curves. Run it at any point during or after training
to judge convergence.

**Resume after Colab disconnect:**
- `train_flow.py` always starts from epoch 1 (no mid-epoch resume).
- The best checkpoint is on Drive and is not lost.
- Re-running Stage 3 will restart from epoch 1 but only overwrites the
  checkpoint if val loss improves further.
- Check the loss plot first: if the curves have flattened, the current
  `best_flow.pt` is already near-optimal; go straight to Stage 4.

### Stage 4 — 48-frame forecast

```bash
# single trajectory (quick)
python -m flowcast.forecast_flow --config vis/config_fc.yaml --samples 1

# 8-sample ensemble (full)
python -m flowcast.forecast_flow --config vis/config_fc.yaml --samples 8
```

Runs 4 autoregressive blocks of 12 frames = 48 slots (24 h).
With `--samples 8`, computes 3 aggregations:

| Mode | Description |
|------|-------------|
| `pixmean` | Per-pixel average over 8 ensemble members (smoothest) |
| `medoid`  | Member closest to pixmean by L2 — sharpest single trajectory |
| `latmean` | Average in latent space, decoded once — often sharpest |

VIS night slots (solar elevation < 0°) are filled with the mean of real night
frames. Metrics (PSNR, SSIM, CSI, HSS, FAR, CRPS) are computed on **daytime
slots only**.

**Outputs** (`vis/outputs_fc/`):
```
flow_forecast_<date>_ens8_pixmean.npy / .gif / _grid.png / _compare.png
flow_forecast_<date>_ens8_medoid.npy  / .gif / _grid.png / _compare.png
flow_forecast_<date>_ens8_latmean.npy / .gif / _grid.png / _compare.png
flow_forecast_<date>_metrics.json
```

---

## Blur Reduction — Strategy A and B

The frozen SEVIR VAE is out-of-distribution (OOD) for INSAT, causing soft,
blurry decoded images. These two optional stages sharpen the forecast output
by fine-tuning **only the decoder** (encoder and the trained CFM model are
never changed).

### Strategy A — Decoder fine-tune on real encode-decode pairs (~45 min on T4)

```bash
python -m flowcast.finetune_vae --config vis/config_fc.yaml
```

Freezes the encoder; fine-tunes decoder + `post_quant_conv` with:
```
loss = 0.1 × MSE(decoded, gt) + 1.0 × LPIPS-VGG(decoded, gt)
```

Trains for 4,000 steps (override with `--steps N` if needed).
Saves best-by-val-PSNR to `vis/checkpoints_fc/vae_decoder_ft_A.pt`.
Progress images are written every 250 steps to
`vis/outputs_fc/vae_ft_A_progress/` — the display cell shows them.

**Expected gain:** roundtrip PSNR +6 to +10 dB over the frozen SEVIR baseline.

After training, the **enable-A cell** uncomments `decoder_ckpt` in
`vis/config_fc.yaml`, re-runs `sanity_check`, and re-runs the forecast.
Inspect the 3-way comparison table to see PSNR / SSIM / CSI changes.

### Strategy B — Decoder fine-tune on (predicted latent → GT) pairs (~3 h on T4)

Strategy B compensates for **both** VAE OOD blur and CFM mode-averaging in
one supervised step.

**Phase 4 — build pair dataset (~80 min):**
```bash
python -m flowcast.build_pred_dataset --config vis/config_fc.yaml
```
Runs the trained CFM on every training-split day, harvesting predicted latents
paired with ground-truth TIFFs. Output at `vis/pred_pairs_fc/`.
Hard leakage check: no val/test frame is ever included.

**Phase 5 — train Strategy-B decoder (~100 min):**
```bash
python -m flowcast.finetune_vae_pred --config vis/config_fc.yaml
```
Trains the decoder directly on (pred_latent → GT) pairs using the same
`0.1·MSE + 1.0·LPIPS` objective. Saves to
`vis/checkpoints_fc/vae_decoder_ft_B.pt`.

**Phase 6 — swap config to B and validate:**
The swap cell changes `decoder_ckpt` from `_ft_A.pt` → `_ft_B.pt`, then
re-runs sanity_check + forecast. **Important:**

> sanity_check PSNR will **drop** 1–3 dB vs Strategy A. This is expected and
> not a regression. The decoder is now specialised for forecaster latents,
> not encoder-mode latents. Watch the **forecast metrics** (medoid PSNR/SSIM/
> LPIPS/CSI) — those should improve vs both baseline and Strategy A.

---

## WV Channel

Run the same pipeline with `wv/config_fc.yaml`:

- No day/night handling (WV signal is continuous 24 h).
- Blur reduction (Strategy A + B) follows the same pattern — the **WV blur
  reduction section** of the notebook mirrors the VIS section exactly.
- Only run WV after VIS Stage 2 has cleared the gate.

Expected WV sanity PSNR: slightly higher than VIS because the SEVIR VAE
(trained on radar) is closer to WV physics than VIS physics.

---

## Expected Timing on T4

| Task | Time |
|------|------|
| Stage 1 — encode | ~10 min |
| Stage 2 — sanity check | ~2 min |
| Stage 3 — train (100 epochs) | 12–24 h |
| Stage 4 — forecast ×1 | ~3 min |
| Stage 4 — forecast ×8 | ~15 min |
| Strategy A FT | ~45 min |
| build_pred_dataset | ~80 min |
| Strategy B FT | ~100 min |

Colab free-tier sessions time out after ~12 h of GPU use. For Stage 3, either:
- Use Colab Pro / background execution
- Run in two sessions: first 50 epochs, check loss plot, resume (restart from
  epoch 1 but best checkpoint is preserved)

---

## Key Files at a Glance

```
vis/config_fc.yaml                       # config; norm stats auto-filled after Stage 1
vis/latents_fc/                          # cached latents (Stage 1 output)
vis/manifest_fc.csv                      # frame index (UTC, IST, latent path, is_day)
vis/checkpoints_fc/
    best_flow.pt                         # best EMA CFM weights
    flow_history.json                    # train/val loss per epoch
    vae_decoder_ft_A.pt                  # Strategy A decoder
    vae_decoder_ft_A_history.json
    vae_decoder_ft_B.pt                  # Strategy B decoder
    vae_decoder_ft_B_history.json
vis/outputs_fc/
    sanity_grid.png                      # Stage 2 reconstruction grid
    sanity_metrics.json                  # Stage 2 PSNR/SSIM
    flow_forecast_<date>_ens8_*.npy/gif/png
    flow_forecast_<date>_metrics.json    # PSNR/SSIM/CSI/HSS/CRPS per mode
    vae_ft_A_progress/                   # Step-by-step progress images (Strategy A)
    vae_ft_B_progress/                   # Step-by-step progress images (Strategy B)
```

---

## Troubleshooting

**`norm stats missing — run encode_latents first`**
Stage 2/3/4 was run before Stage 1. Run `encode_latents` first; it fills the
`norm.low` / `norm.high` fields in the config.

**`empty split — check manifest / sequence lengths`**
Too few manifest rows for the sliding-window dataset. Check that
`vis/manifest_fc.csv` exists and has > 40 rows, and that `sequence.input_len`
+ `flow.joint_steps` ≤ the shortest contiguous run length.

**OOM during Stage 3**
The auto-scaling should handle T4. If you still OOM, manually add
`--batch 1` to the training cell.

**Sanity PSNR stuck below 20 dB even after Strategy A**
Try `--steps 8000`. Also confirm that `vae.decoder_ckpt` in the config points
to `_ft_A.pt` (not a stale path). If PSNR is still < 20 dB the SEVIR VAE is
very OOD for this channel — switch to the DiffCast pipeline.

**Strategy B PSNR dropped vs A — is something wrong?**
No. The decoder is now specialised for forecaster latents, not encoder-mode
latents. Check the forecast metrics table; if medoid PSNR/SSIM improved vs A,
Strategy B is working correctly.

**Colab notebook disconnected mid-training**
Re-run Setup → Session Status → Stage 3. The best checkpoint is safe on Drive.
The training will restart from epoch 1, but only overwrites the checkpoint on
improvement. Run fewer epochs and check the loss plot to decide when to stop.
