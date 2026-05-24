"""Pretrained SD-VAE loading and latent encode/decode helpers."""
import torch
from diffusers import AutoencoderKL


def load_vae(cfg, device):
    """Load the frozen pretrained SD-VAE onto the given device."""
    vae = AutoencoderKL.from_pretrained(cfg["vae_model"])
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def scaling_factor(cfg, vae):
    """Latent scaling factor — from the VAE config, else the channel config."""
    sf = getattr(vae.config, "scaling_factor", None)
    return float(sf if sf else cfg.get("latent_scaling_factor", 0.18215))


@torch.no_grad()
def encode(vae, img_bchw, sf):
    """Encode (B,3,H,W) images in [-1,1] to scaled latents (B,4,H/8,W/8)."""
    return vae.encode(img_bchw).latent_dist.mean * sf


@torch.no_grad()
def decode(vae, latent_bchw, sf):
    """Decode scaled latents back to (B,3,H,W) images in [-1,1]."""
    return vae.decode(latent_bchw / sf).sample
