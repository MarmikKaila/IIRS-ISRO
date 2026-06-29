"""Full-trainable AutoencoderKL for diffcast.

Same architecture as flowcast.utils.flowcast_vae (we import from the same
SEVIR pretrained checkpoint), but ALL parameters are unfrozen for end-to-end
fine-tuning on INSAT data — both encoder and decoder learn to map the INSAT
domain into and out of a 4-channel latent space.

The fine-tuned full VAE is loaded via ``cfg["vae"]["full_ckpt"]`` once
Phase 1 produces it. Without that key, this loader returns the SEVIR
pretrained weights (useful only as a starting point for training).
"""
import os

import numpy as np
import torch
from diffusers.models.autoencoders import AutoencoderKL
from huggingface_hub import hf_hub_download

HF_REPO = "b-rbmp/flowcast-cfm-sevir"
HF_FILENAME = "saved_models/sevir/autoencoder/models/early_stopping_model.pt"


def _build_autoencoder_kl():
    """Architecture mirrors flowcast/utils/flowcast_vae.py::_build_autoencoder_kl
    so a SEVIR-pretrained state_dict loads cleanly."""
    return AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=32,
        layers_per_block=2,
    )


def load_vae(cfg, device, trainable=True):
    """Load the AutoencoderKL with SEVIR weights, optionally with our
    fine-tuned overlay, optionally fully trainable.

    Reads ``cfg["vae"]``:
      hf_id       : HF repo (default b-rbmp/flowcast-cfm-sevir)
      hf_filename : path inside repo
      full_ckpt   : optional path (repo-root-relative) to OUR fine-tuned
                    full VAE; if set, overrides SEVIR weights end-to-end.
    """
    v = cfg.get("vae", {})
    repo = v.get("hf_id", HF_REPO)
    fname = v.get("hf_filename", HF_FILENAME)
    ckpt_path = hf_hub_download(repo_id=repo, filename=fname)

    model = _build_autoencoder_kl()
    blob = torch.load(ckpt_path, map_location=device)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    cleaned = {(k[7:] if k.startswith("module.") else k): val
               for k, val in sd.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[diffcast-vae] init: missing={len(missing)} unexpected={len(unexpected)}")

    full_path = v.get("full_ckpt")
    if full_path:
        abs_path = (full_path if os.path.isabs(full_path)
                    else os.path.join(cfg["_repo_root"], full_path))
        ft = torch.load(abs_path, map_location=device)
        model.load_state_dict(ft["model"], strict=True)
        psnr = ft.get("val_psnr")
        psnr_str = f"{psnr:.2f}" if psnr is not None else "n/a"
        print(f"[diffcast-vae] loaded fine-tuned full VAE from {full_path} "
              f"(step={ft.get('step')}, val_psnr={psnr_str})")

    model.to(device)
    for p in model.parameters():
        p.requires_grad_(trainable)
    if not trainable:
        model.eval()
    return model


# ── physical units <-> [0, 1] (mirrors flowcast convention, so cached
#    latents are interchangeable in scale if not in numerical values) ──────
def standardize(x_phys, lo, hi):
    """Physical units -> [0, 1] (clipped to [lo, hi])."""
    x = np.clip(x_phys.astype("float32"), lo, hi)
    return ((x - lo) / max(hi - lo, 1e-8)).astype("float32")


def destandardize(x01, lo, hi):
    """[0, 1] -> physical units."""
    return (np.clip(x01.astype("float32"), 0.0, 1.0) * (hi - lo) + lo).astype("float32")


def to_vae_input(x01_hw):
    """(H, W) float in [0, 1] -> (1, 1, H, W) tensor (single-channel native)."""
    t = torch.from_numpy(np.ascontiguousarray(x01_hw, dtype="float32"))
    return t[None, None]


def from_vae_output(t_b1hw):
    """(B, 1, H, W) tensor -> (H, W) float32 numpy."""
    if hasattr(t_b1hw, "detach"):
        a = t_b1hw.detach().float().cpu().numpy()
    else:
        a = np.asarray(t_b1hw, dtype="float32")
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a[0]
    return a.astype("float32")


@torch.no_grad()
def encode_mode(vae, x_b1hw):
    """Encode (B, 1, H, W) in [0, 1] to latent (B, 4, H/8, W/8) using ``mode``.
    For inference / latent caching: deterministic, no sampling."""
    return vae.encode(x_b1hw).latent_dist.mode()


@torch.no_grad()
def decode(vae, z_bchw):
    """Decode (B, 4, H/8, W/8) latents to (B, 1, H, W) ~[0, 1]."""
    return vae.decode(z_bchw).sample
