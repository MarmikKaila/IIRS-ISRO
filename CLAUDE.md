# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Latent-space **48-frame day-ahead nowcasting** of INSAT-3S satellite imagery.
Each image is compressed to an SD-VAE latent; a ConvLSTM RNN predicts future
latents autoregressively; predicted latents are decoded back to images. The
**VIS** (visible reflectance) and **WV** (water-vapour brightness temperature)
channels are processed **independently** with the same code.

## Layout

```
src/
  data/tif_io.py            # read float32 TIFF, parse timestamp, resize, normalize
  data/sequence_dataset.py  # contiguous-run windows + train/val/test datasets
  utils/config.py           # YAML config load/resolve/save
  utils/device.py           # CUDA -> MPS -> CPU auto-detect
  utils/solar.py            # solar elevation -> day/night (VIS), UTC<->IST
  utils/sdvae.py            # pretrained SD-VAE load + encode/decode
  utils/metrics.py          # MSE / PSNR / SSIM
  utils/viz.py              # image grids, comparison grids, GIFs
  models/convlstm.py        # ConvLSTM cell + LatentForecaster (residual rollout)
  encode_latents.py         # Stage 1
  sanity_check.py           # Stage 2
  train.py                  # Stage 3
  forecast.py               # Stage 4
vis/  wv/                   # per-channel: config.yaml + generated latents/, manifest.csv, checkpoints/, outputs/
sample-images/              # raw data: Jan26_167972 (VIS), Jan26_168372 (WV)
notebooks/colab_pipeline.ipynb
```

## Pipeline — run order (per channel)

Scripts are Python modules; run from the repo root with `-m`:

```bash
python -m src.encode_latents --config vis/config.yaml   # Stage 1: cache latents
python -m src.sanity_check   --config vis/config.yaml   # Stage 2: VAE PSNR check
python -m src.train          --config vis/config.yaml   # Stage 3: train ConvLSTM
python -m src.forecast       --config vis/config.yaml   # Stage 4: 48-frame forecast
```

Swap `vis/config.yaml` for `wv/config.yaml` for the WV channel. The notebook
runs all eight invocations end to end.

## Key facts

- **Data:** INSAT-3S L1C TIFFs, 888×968 single-band float32, 30-min cadence
  (48 slots/day). Filenames are **UTC**: `3SIMG_<DDMMMYYYY>_<HHMM>_..._IMG_<VIS|WV>.tif`.
- **Preprocessing:** resize to 256×288 (W×H), normalize to [-1,1] with robust
  1st/99th-percentile stats (auto-computed, saved into `config.yaml`).
- **SD-VAE:** pretrained `stabilityai/sd-vae-ft-mse`, frozen. Latent is 4×36×32;
  grey input is replicated to 3 channels.
- **VIS day/night:** the ConvLSTM trains on the **full 48-frame diurnal cycle**
  (night frames included) so it can be seeded across the night and forecast a
  day ahead. At forecast time, night slots (solar elevation, `utils/solar.py`)
  are overwritten with the mean of real VIS night frames; metrics use daytime
  slots only. WV has no day/night handling.
- **ConvLSTM:** residual prediction (`pred = prev + delta`); trained with
  scheduled sampling so it tolerates consuming its own predictions during rollout.

## Conventions

- Config is the single source of truth — every script takes `--config`.
- All config path keys are relative to the repo root (parent of the config file).
- Channels never share normalization stats, latents, checkpoints, or outputs.
- Timestamps stored naive-UTC in the manifest; convert with `utils.solar`.
- Compute target is a Colab/cloud GPU; device auto-detects (`utils.device`).

## Gotchas

- Run scripts with `python -m src.<name>` from the repo root — relative imports
  require the `src` package context.
- Stage 2 PSNR well below ~25 dB means the SD-VAE is out-of-distribution; the
  decision is to fine-tune the VAE before trusting Stage 3/4.
- `sample-images.zip` is the full archive of the already-extracted folders —
  ignore it; use the extracted `sample-images/` directories.
