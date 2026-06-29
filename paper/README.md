# Paper — `ISRO A.1` write-up

LaTeX source for the conference-style paper covering the full project:
**flowcast** (CFM + frozen encoder + decoder-only FT) and **diffcast**
(EDM diffusion + full VAE FT) on INSAT-3S VIS and WV.

## Files

- `main.tex` — the paper (single-column article, ~10–12 pages compiled)
- `references.bib` — bibliography (BibTeX, all citations)

## How to compile

### Easiest — Overleaf

1. Make a new project on https://overleaf.com
2. Upload `main.tex` and `references.bib`
3. Click **Recompile**. Overleaf handles the LaTeX/BibTeX cycle automatically.

### Local LaTeX (TeX Live / MacTeX)

```bash
cd paper/
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

This produces `main.pdf`. Two final passes are needed for all cross-references
(figures, sections, citations) to resolve.

## What's covered in the paper

1. **Introduction & motivation** — INSAT-3S nowcasting, why latent generative models
2. **Related work** — FlowCast paper, EDM, Earthformer, LPIPS, SEVIR, perceptual losses, meteorological metrics
3. **Dataset** — INSAT-3S Jan 2026, VIS + WV, 30 days, manifest splits
4. **Methodology**
   - Common 4-stage pipeline (encode → train → roll out → decode)
   - SEVIR-pretrained AutoencoderKL — baseline VAE roundtrip 27.27 dB
   - FlowCast (CFM, frozen encoder)
   - Strategy A — decoder FT on real (encode, decode) pairs + LPIPS
   - Strategy B — decoder FT on (predicted-latent, GT) pairs + LPIPS
   - DiffCast (full VAE FT + EDM diffusion)
   - **Two documented failure fixes** during DiffCast training
       - FP16 KL-gradient overflow → set `kl_weight = 0`
       - FP16 decoder activation overflow on T4 → switch to FP32
   - Three-way ensemble aggregation (pixmean / medoid / latmean) — exposes
     perception-distortion tradeoff
   - Evaluation: PSNR, SSIM, MSE, CSI_M, HSS_M, FAR_M, CSI_P16_M, FSS_P16_M, CRPS
   - **Threshold calibration fix for WV** — Kelvin thresholds → data-scale
     percentile thresholds (CSI went from 0.000 to 0.204)
5. **Results**
   - VIS — FlowCast vs FlowCast(N=32) vs DiffCast head-to-head table
   - WV — FlowCast metrics with corrected thresholds
   - VAE roundtrip results across configurations
6. **Failure analysis of DiffCast** — encoder representation collapse:
     full encoder FT on 560 frames lost the geographic prior even while
     improving pixel-wise metrics
7. **Discussion**
   - Perception-distortion tradeoff (Blau & Michaeli 2018) revisited
   - Why WV is easier than VIS for this VAE
   - The threshold calibration trap — caveats for re-using published metric configs
   - Mode-collapse evidence: N=8 vs N=32 ensemble produces flat metrics
   - Limitations: single test date, T4 cap, geographic-prior recovery not attempted
8. **Conclusion** — FlowCast as production pipeline; DiffCast as ablation that
   justifies the frozen-encoder design choice on small datasets
9. **References** — 21 citations

## Where the numbers in the paper come from

| Paper claim | Source |
|---|---|
| FlowCast VIS pixmean PSNR 12.31, CSI 0.063 | `vis/outputs_fc/flow_forecast_20260115_ens8_metrics.json` |
| FlowCast VIS N=32 pixmean PSNR 12.21, CSI 0.061 | `vis/outputs_fc/flow_forecast_20260115_ens32_metrics.json` |
| FlowCast WV pixmean CSI 0.204, SSIM 0.720 | `wv/outputs_fc/flow_forecast_20260115_ens8_metrics.json` |
| DiffCast VIS pixmean PSNR 15.25, CSI 0.012, CRPS 0.123 | `vis/outputs_diff/diff_forecast_20260115_ens8_metrics.json` |
| VAE baseline 27.27 dB | `vis/outputs_diff/sanity_metrics.json` (first run) and the train_vae baseline log |
| VAE full-FT best 27.56 dB | `vis/checkpoints_diff/vae_full_history.json` |
| WV native data range [0.007, 0.288] | diagnostic cell over 200 frames, summarised in Table 3 |
| Recalibrated WV thresholds [0.20, 0.225, 0.25, 0.265, 0.28] | `flowcast/utils/threshold_metrics.py` (committed change) |

## Replacing the figure placeholders

The current `main.tex` has two `\fbox{...}` blocks where real figure PNGs
should go (the DiffCast vs FlowCast visual comparison and the per-frame
metrics curves). To swap in real images:

1. Copy the actual PNGs into `paper/figures/`:
   ```bash
   mkdir -p paper/figures
   cp ../vis/outputs_fc/flow_forecast_20260115_ens8_medoid_grid.png paper/figures/flowcast_medoid.png
   cp ../vis/outputs_diff/diff_forecast_20260115_ens8_medoid_grid.png paper/figures/diffcast_medoid.png
   cp ../vis/outputs_fc/per_frame_metrics_comparison.png paper/figures/per_frame_curves.png
   ```
2. In `main.tex`, replace each `\fbox{\parbox{...}{...}}` line with
   `\includegraphics[width=0.45\textwidth]{figures/flowcast_medoid.png}` etc.

The captions already reference the exact source file paths so the change
is mechanical.

## Conference template variants

The current template is the generic `article` class for portability. Switch
to IEEE/NeurIPS by:

- **IEEEtran**: replace `\documentclass[11pt,a4paper]{article}` with
  `\documentclass[conference]{IEEEtran}` and add `\IEEEauthorblockN{}` /
  `\IEEEauthorblockA{}` formatting for the author.
- **NeurIPS**: download `neurips_2024.sty` and replace the preamble with
  `\documentclass{article}\usepackage{neurips_2024}`.

Both leave the body content unchanged.

## Suggested next edits before submission

1. Run an actual A100 training of the paper-spec model (hidden=192, n_blocks=4)
   if compute allows, and add a third row to Table 4 documenting that result.
2. Run forecasts on 3–5 additional dates and report mean ± std in the
   results tables — strongest single improvement to the rigour.
3. Add a persistence baseline column (predict at $t+k$ = image at $t$) to
   the comparison tables.
4. Replace `\fbox` placeholders with real PNGs (instructions above).
5. Author affiliation block — update to the correct institution/department.
