"""FlowCast pretrained VAE loader (b-rbmp/flowcast-cfm-sevir on HF Hub).

Loads the AutoencoderKL the paper trained on SEVIR radar (single-channel,
block_out_channels=[128,256,512,512], latent_ch=4, f=8) using the diffusers
AutoencoderKL class and the paper's released state-dict. Used FROZEN — no
fine-tuning on INSAT. Single-channel native (no RGB replication).

This is closer to the FlowCast paper's architecture than `stabilityai/sd-vae-ft-mse`
without the multi-day from-scratch training cost. INSAT data is out-of-
distribution for this VAE; sanity check before training the flow model.
"""
import os

import torch
from diffusers.models.autoencoders import AutoencoderKL
from huggingface_hub import hf_hub_download

HF_REPO = "b-rbmp/flowcast-cfm-sevir"
HF_FILENAME = "saved_models/sevir/autoencoder/models/early_stopping_model.pt"


def _build_autoencoder_kl():
    """AutoencoderKL constructor args copied verbatim from FlowCast's
    experiments/sevir/autoencoder/autoencoder_kl_config.yaml."""
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


def load_flowcast_vae(cfg, device):
    """Load FlowCast's pretrained AutoencoderKL, frozen.

    Reads optional overrides from ``cfg["vae"]``:
        hf_id        : HF repo id (default b-rbmp/flowcast-cfm-sevir)
        hf_filename  : path inside repo (default the SEVIR autoencoder ckpt)
    """
    v = cfg.get("vae", {})
    repo = v.get("hf_id", HF_REPO)
    fname = v.get("hf_filename", HF_FILENAME)
    ckpt_path = hf_hub_download(repo_id=repo, filename=fname)

    model = _build_autoencoder_kl()
    blob = torch.load(ckpt_path, map_location=device)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    # strip 'module.' prefix from DDP-saved weights (matches FlowCast's load code)
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        # surface a short warning but don't fail; SEVIR-trained weights may have
        # tiny key differences across diffusers versions
        print(f"[flowcast-vae] load: missing={len(missing)} unexpected={len(unexpected)}")

    # Optional INSAT fine-tuned decoder overlay (Strategy A or B). Strict=True
    # on purpose: a typo silently reverting to the SEVIR decoder would be a
    # painful debugging trap.
    ft_path = v.get("decoder_ckpt")
    if ft_path:
        abs_path = (ft_path if os.path.isabs(ft_path)
                    else os.path.join(cfg["_repo_root"], ft_path))
        ft = torch.load(abs_path, map_location=device)
        model.decoder.load_state_dict(ft["decoder"], strict=True)
        model.post_quant_conv.load_state_dict(ft["post_quant_conv"], strict=True)
        psnr_str = (f"{ft['val_psnr']:.2f}" if ft.get("val_psnr") is not None
                    else "n/a")
        print(f"[flowcast-vae] loaded fine-tuned decoder "
              f"({ft.get('strategy', '?')}) from {ft_path} "
              f"(step={ft.get('step')}, val_psnr={psnr_str})")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# Paper note: the SEVIR pipeline standardises pixels into [0, 1] before
# encoding (normalize_dataset=true). For INSAT we map the channel's robust
# (1st, 99th) percentile range to [0, 1] so the VAE sees inputs in the same
# scale it was trained on. Inverse mapping is applied after decode.

def standardize(x_phys, lo, hi):
    """Physical units -> [0, 1] (clipped), matching FlowCast's normalize_dataset."""
    import numpy as np
    x = np.clip(x_phys.astype("float32"), lo, hi)
    return (x - lo) / max(hi - lo, 1e-8)


def destandardize(x01, lo, hi):
    """[0, 1] -> physical units."""
    import numpy as np
    return (np.clip(x01.astype("float32"), 0.0, 1.0) * (hi - lo) + lo).astype("float32")


@torch.no_grad()
def encode(vae, x_b1hw):
    """Encode (B, 1, H, W) in [0, 1] to latent (B, 4, H/8, W/8) using ``mode``.

    Matches FlowCast's caching path (``encoded_obj.latent_dist.mode()``).
    """
    return vae.encode(x_b1hw).latent_dist.mode()


@torch.no_grad()
def decode(vae, z_bchw):
    """Decode (B, 4, H/8, W/8) latents to (B, 1, H, W) images approximately in [0, 1]."""
    return vae.decode(z_bchw).sample


def to_vae_input(x01_hw):
    """(H, W) float in [0,1] -> (1, 1, H, W) tensor (single-channel native)."""
    import numpy as np
    t = torch.from_numpy(np.ascontiguousarray(x01_hw, dtype="float32"))
    return t[None, None]


def from_vae_output(t_b1hw):
    """(B, 1, H, W) or (1, H, W) tensor -> (H, W) float32 numpy."""
    import numpy as np
    if hasattr(t_b1hw, "detach"):
        a = t_b1hw.detach().float().cpu().numpy()
    else:
        a = np.asarray(t_b1hw, dtype="float32")
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a[0]
    return a.astype("float32")
