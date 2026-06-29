"""FlowCast latent CFM model — Earthformer-UNet wrapper + I-CFM loss + sampler.

Implements the paper's Algorithms 1 (training) and 2 (sampling):
    z_t = (1 - t) * z0 + t * z1 + sigma * eps         (I-CFM, sigma = 0.01)
    u_t = z1 - z0                                      (target velocity)
    loss = || v_theta(Z_t, t, Z_past) - u_t ||^2

The "Observation Masking" of the paper is implemented by concatenating a
mask channel (1.0 on past frames, 0.0 on future frames) to the per-timestep
feature before the Earthformer-UNet projection.

Sampling: Euler integration of dz/dt = v(z, t, Z_past) from 0 -> 1 with S
steps. Inputs to ``sample()`` are the encoded past latents; the function
returns the predicted future latents (still in latent space).
"""
import torch
import torch.nn as nn

from .earthformer_unet import EarthformerUNet, build_earthformer


class LatentFlowCast(nn.Module):
    """End-to-end wrapper that:
       - holds the Earthformer-UNet
       - assembles (past, z_t_future, mask) into one (B, T, C_in, H, W) tensor
       - exposes train_step(z_past, z_future) -> loss and sample(z_past) -> z_pred
    """

    def __init__(self, backbone: EarthformerUNet, T_past: int, T_lead: int,
                 latent_ch: int = 4, sigma: float = 0.01):
        super().__init__()
        self.backbone = backbone
        self.T_past = T_past
        self.T_lead = T_lead
        self.latent_ch = latent_ch
        self.sigma = sigma
        # mask channel is the (T_past+T_lead)-th dimension along the channel axis
        # (we feed latent_ch + 1 channels into the backbone)

    def _assemble(self, z_past, z_future_t):
        """(B, T_past, C, H, W), (B, T_lead, C, H, W) -> (B, T_total, C+1, H, W)."""
        B = z_past.shape[0]
        H, W = z_past.shape[-2:]
        Tp, Tl = self.T_past, self.T_lead
        full_lat = torch.cat([z_past, z_future_t], dim=1)            # (B, Tp+Tl, C, H, W)
        mask = torch.zeros(B, Tp + Tl, 1, H, W, device=z_past.device,
                            dtype=z_past.dtype)
        mask[:, :Tp] = 1.0
        x = torch.cat([full_lat, mask], dim=2)                        # (B, T, C+1, H, W)
        return x

    def forward(self, z_past, z_future_t, t):
        """Run the backbone and return the predicted velocity for *future* frames only."""
        x = self._assemble(z_past, z_future_t)
        v_all = self.backbone(x, t)                                   # (B, T, C, H, W)
        return v_all[:, self.T_past:]                                  # (B, T_lead, C, H, W)

    # ── I-CFM loss ─────────────────────────────────────────────────────────
    def cfm_loss(self, z_past, z_future, weights=None):
        B = z_future.shape[0]
        device = z_future.device
        z0 = torch.randn_like(z_future)
        eps = torch.randn_like(z_future)
        t = torch.rand(B, device=device)
        t_e = t[:, None, None, None, None]                            # (B, 1, 1, 1, 1)
        z_t = (1.0 - t_e) * z0 + t_e * z_future + self.sigma * eps
        u_t = z_future - z0
        v_pred = self.forward(z_past, z_t, t)
        # per-frame MSE
        per = (v_pred - u_t).pow(2).mean(dim=(2, 3, 4))                # (B, T_lead)
        if weights is None:
            return per.mean()
        w = weights.to(per.device)
        return (per * w).sum() / w.sum().clamp(min=1e-6)

    # ── Euler ODE sampler ──────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, z_past, steps=10):
        """Sample a future block conditioned on z_past. Returns (B, T_lead, C, H, W)."""
        B = z_past.shape[0]
        H, W = z_past.shape[-2:]
        z = torch.randn(B, self.T_lead, self.latent_ch, H, W,
                         device=z_past.device, dtype=z_past.dtype)
        dt = 1.0 / max(int(steps), 1)
        for i in range(int(steps)):
            t = torch.full((B,), i * dt, device=z_past.device, dtype=z_past.dtype)
            v = self.forward(z_past, z, t)
            z = z + v * dt
        return z


def build_model(cfg, latent_ch=4, latent_h=36, latent_w=32):
    """Read cfg and assemble a LatentFlowCast ready for training/inference."""
    T_past = int(cfg["sequence"].get("input_len", 13))
    T_lead = int(cfg["flow"].get("joint_steps", 12))
    sigma = float(cfg["flow"].get("sigma", 0.01))
    backbone = build_earthformer(
        cfg, in_channels=latent_ch + 1, out_channels=latent_ch,
        T=T_past + T_lead, H=latent_h, W=latent_w)
    return LatentFlowCast(backbone, T_past=T_past, T_lead=T_lead,
                          latent_ch=latent_ch, sigma=sigma)
