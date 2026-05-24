# INSAT VIS & WV — Latent-Space 48-Frame Nowcasting

Forecast a full day (**48 frames**, 30-minute cadence) of INSAT-3S satellite
imagery for the **VIS** (visible) and **WV** (water-vapour) channels.

Each image is encoded into a compact **SD-VAE latent**; a **ConvLSTM** RNN is
trained and rolled out in latent space to predict future latents; predicted
latents are decoded back to images with the SD-VAE decoder.

```
.tif → resize+normalize → SD-VAE encoder → cached latent .pt        [Stage 1]
cached .pt → SD-VAE decoder → PSNR/SSIM vs ground truth             [Stage 2]
latent sequences → ConvLSTM RNN (train)                             [Stage 3]
seed latents → RNN rollout → 48 latents → SD-VAE decoder → images   [Stage 4]
```

## Dataset

| Channel | Folder | Files |
|---|---|---|
| VIS | `sample-images/Jan26_167972/` | ~1,509 TIFFs |
| WV  | `sample-images/Jan26_168372/` | ~1,532 TIFFs |

INSAT-3S L1C, 888×968 single-band float32, 30-min cadence. Filename times are
UTC; VIS day/night uses IST (= UTC + 5:30).

## Quick start

```bash
pip install -r requirements.txt

# per channel: vis/config.yaml or wv/config.yaml
python -m src.encode_latents --config vis/config.yaml   # 1 — cache latents
python -m src.sanity_check   --config vis/config.yaml   # 2 — VAE PSNR check
python -m src.train          --config vis/config.yaml   # 3 — train ConvLSTM
python -m src.forecast       --config vis/config.yaml   # 4 — 48-frame forecast
```

Run scripts from the repo root with `python -m`. Or open
[`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb) on a Colab
GPU to run both channels end to end.

## Project structure

- `src/` — shared, channel-agnostic code (parametrised by `--config`)
- `vis/`, `wv/` — self-contained per-channel folders: `config.yaml` plus the
  generated `latents/`, `manifest.csv`, `checkpoints/`, `outputs/`
- `notebooks/colab_pipeline.ipynb` — drives all stages for both channels

## VIS day/night handling

The VIS channel is dark at night, so ~20 of the 48 daily slots carry no usable
signal. Night frames are detected from **solar elevation** (`src/utils/solar.py`).
The ConvLSTM trains and predicts on **daytime-only** sequences; night slots in
the 48-frame forecast are filled with the mean of real VIS night frames. WV
(brightness temperature) is usable 24 h, so all 48 slots are modelled.

See [`CLAUDE.md`](CLAUDE.md) for implementation details and conventions.
