"""Latent diffusion forecaster — EDM preconditioning wrapped around the
existing flowcast Earthformer-UNet.

The same backbone flowcast uses for Conditional Flow Matching is reused as
the denoising network. Only the I/O preconditioning (Karras Table 1) and
the loss/sampler change.

Conditioning:
  - past_latents : (B, T_past, C, H, W) — passed unchanged
  - noisy_future : (B, T_lead, C, H, W) — scaled by c_in
  - sigma        : (B,) — fed via c_noise to the backbone's time embedding

A mask channel (1 on past, 0 on future) is concatenated to the channel
axis just like the flowcast CFM model does, so the backbone sees the same
input layout.
"""
import torch
import torch.nn as nn

from flowcast.models.earthformer_unet import EarthformerUNet
from .edm import (
    edm_preconditioning,
    sample_sigma_train,
    edm_loss_weight,
)


class LatentEDMForecaster(nn.Module):
    """Wraps an EarthformerUNet backbone with EDM preconditioning + sampler.

    Predicts the CLEAN future latents (B, T_lead, C, H, W) given past
    latents, a noisy future at noise level sigma, and sigma itself.
    """

    def __init__(self, backbone, T_past, T_lead, latent_ch=4,
                 sigma_data=0.5, P_mean=-1.2, P_std=1.2):
        super().__init__()
        self.backbone = backbone
        self.T_past = T_past
        self.T_lead = T_lead
        self.latent_ch = latent_ch
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std

    def _assemble(self, z_past, z_future):
        """(B, T_past, C, H, W) + (B, T_lead, C, H, W) -> (B, T_total, C+1, H, W).

        Concatenates along time, adds a mask channel (1 on past, 0 on future).
        """
        B = z_past.shape[0]
        H, W = z_past.shape[-2:]
        Tp, Tl = self.T_past, self.T_lead
        full_lat = torch.cat([z_past, z_future], dim=1)
        mask = torch.zeros(B, Tp + Tl, 1, H, W,
                            device=z_past.device, dtype=z_past.dtype)
        mask[:, :Tp] = 1.0
        return torch.cat([full_lat, mask], dim=2)         # (B, T, C+1, H, W)

    def denoise(self, z_past, z_noisy, sigma):
        """EDM-preconditioned denoising: returns predicted CLEAN future."""
        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, self.sigma_data)
        x = self._assemble(z_past, (c_in * z_noisy).to(z_past.dtype))
        # Backbone returns (B, T_total, C, H, W); keep only future slice.
        f_all = self.backbone(x, c_noise.to(z_past.dtype))
        f_future = f_all[:, self.T_past:]                 # (B, T_lead, C, H, W)
        return (c_skip * z_noisy + c_out * f_future).to(z_past.dtype)

    def edm_loss(self, z_past, z_future, weights=None):
        """Training step. weights is ignored (no per-frame masking yet)."""
        B = z_future.shape[0]
        sigma = sample_sigma_train(B, self.P_mean, self.P_std,
                                    z_future.device, z_future.dtype)
        noise = torch.randn_like(z_future) * sigma.view(-1, 1, 1, 1, 1)
        z_noisy = z_future + noise
        z_pred = self.denoise(z_past, z_noisy, sigma)
        w = edm_loss_weight(sigma, self.sigma_data)
        return (w * (z_pred - z_future) ** 2).mean()

    @torch.no_grad()
    def sample(self, z_past, sigmas):
        """Heun ODE sampler (paper Algorithm 1) over a Karras schedule.

        z_past : (B, T_past, C, H, W)
        sigmas : (N+1,) tensor of sigma values, last entry == 0.
        Returns: (B, T_lead, C, H, W) clean predicted future latents.
        """
        B = z_past.shape[0]
        H, W = z_past.shape[-2:]
        z = torch.randn(B, self.T_lead, self.latent_ch, H, W,
                         device=z_past.device, dtype=z_past.dtype) * sigmas[0]
        for i in range(len(sigmas) - 1):
            sigma_i = sigmas[i].expand(B)
            sigma_next = sigmas[i + 1].expand(B)
            denoised = self.denoise(z_past, z, sigma_i)
            d = (z - denoised) / sigma_i.view(-1, 1, 1, 1, 1)
            dt = (sigma_next - sigma_i).view(-1, 1, 1, 1, 1)
            z_next = z + d * dt
            # Heun second-order correction (skip if sigma_next == 0).
            if float(sigma_next.max()) > 0.0:
                denoised_next = self.denoise(z_past, z_next, sigma_next)
                d_next = (z_next - denoised_next) / sigma_next.view(-1, 1, 1, 1, 1)
                z = z + 0.5 * (d + d_next) * dt
            else:
                z = z_next
        return z


def build_diffusion_model(cfg, latent_ch, latent_h, latent_w):
    """Read cfg["diffusion"] and assemble LatentEDMForecaster."""
    d = cfg["diffusion"]
    T_past = int(cfg["sequence"]["input_len"])
    T_lead = int(d["joint_steps"])
    # Build the SAME EarthformerUNet flowcast uses for CFM. The backbone's
    # T axis covers T_past + T_lead, with one extra "mask" channel concat'd.
    backbone = EarthformerUNet(
        in_channels=latent_ch + 1,
        out_channels=latent_ch,
        hidden=int(d.get("hidden", 128)),
        T=T_past + T_lead,
        H=latent_h,
        W=latent_w,
        depth=int(d.get("n_blocks", 2)),
        num_heads=int(d.get("attn_heads", 4)),
        t_dim=int(d.get("hidden", 128)) * 4,
        dropout=float(d.get("dropout", 0.1)),
    )
    return LatentEDMForecaster(
        backbone, T_past=T_past, T_lead=T_lead, latent_ch=latent_ch,
        sigma_data=float(d.get("sigma_data", 0.5)),
        P_mean=float(d.get("P_mean", -1.2)),
        P_std=float(d.get("P_std", 1.2)),
    )
