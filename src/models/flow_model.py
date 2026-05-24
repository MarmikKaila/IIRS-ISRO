"""Conditional Flow Matching for next-latent prediction.

The model learns a velocity field ``v(z_t, t, cond)`` that transports Gaussian
noise ``z_0 ~ N(0, 1)`` to the next-frame latent ``z_1`` given a conditioning
tensor (past T latents, plus per-frame SZA on VIS, plus the target SZA).
Sampling integrates ``z'(t) = v(z(t), t, cond)`` with Euler steps; ~10 steps
suffice. This is FlowCast in spirit — adapted to run on a Colab T4 over the
existing SD-VAE latent cache (no per-dataset VAE retrain).
"""
import math

import torch
import torch.nn as nn


# ── time embedding ──────────────────────────────────────────────────────────
def sinusoidal_embed(t, dim):
    """Standard transformer-style positional embedding for the flow time."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ── residual block with FiLM time conditioning ──────────────────────────────
class FlowResBlock(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="reflect")
        self.t_proj = nn.Linear(t_dim, 2 * ch)         # FiLM scale + shift
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.t_proj(self.act(t_emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        return x + self.conv2(self.act(h))


# ── flow model ──────────────────────────────────────────────────────────────
class LatentFlowMatcher(nn.Module):
    """v(z_t, t, condition) -> velocity prediction in latent space."""

    def __init__(self, latent_ch=4, cond_ch=41, hidden=96,
                 n_blocks=4, t_dim=128):
        super().__init__()
        self.latent_ch = latent_ch
        self.cond_ch = cond_ch
        self.t_dim = t_dim
        self.proj_in = nn.Conv2d(latent_ch + cond_ch, hidden, 3, padding=1,
                                 padding_mode="reflect")
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim))
        self.blocks = nn.ModuleList(
            [FlowResBlock(hidden, t_dim) for _ in range(n_blocks)])
        self.norm_out = nn.GroupNorm(8, hidden)
        self.conv_out = nn.Conv2d(hidden, latent_ch, 3, padding=1,
                                  padding_mode="reflect")
        self.act = nn.SiLU()

    def forward(self, z_t, t, cond):
        """z_t: (B, latent_ch, H, W); t: (B,); cond: (B, cond_ch, H, W)."""
        t_emb = self.t_mlp(sinusoidal_embed(t, self.t_dim))
        x = self.proj_in(torch.cat([z_t, cond], dim=1))
        for blk in self.blocks:
            x = blk(x, t_emb)
        return self.conv_out(self.act(self.norm_out(x)))


# ── conditional flow-matching loss ──────────────────────────────────────────
def cfm_loss(model, z1, cond, weights=None):
    """Conditional Flow Matching loss with optional per-sample weighting.

    z1      : (B, C, H, W) target latent.
    cond    : (B, cond_ch, H, W) conditioning.
    weights : optional (B,) per-sample loss weights — used to zero out night
              samples for VIS so they stop dominating the loss.
    """
    B = z1.shape[0]
    z0 = torch.randn_like(z1)
    t = torch.rand(B, device=z1.device)
    t_e = t[:, None, None, None]
    z_t = (1.0 - t_e) * z0 + t_e * z1
    v_target = z1 - z0
    v_pred = model(z_t, t, cond)
    per = (v_pred - v_target).pow(2).mean(dim=(1, 2, 3))   # (B,)
    if weights is None:
        return per.mean()
    return (per * weights).sum() / weights.sum().clamp(min=1e-6)


# ── Euler ODE sampler ───────────────────────────────────────────────────────
@torch.no_grad()
def sample(model, cond, latent_shape, steps=10, device=None):
    """Sample a next-latent by integrating z' = v(z, t, cond) from 0 to 1."""
    device = device or cond.device
    z = torch.randn(latent_shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0],), i * dt, device=device)
        v = model(z, t, cond)
        z = z + dt * v
    return z


def build_flow_model(cfg, cond_ch):
    """Construct LatentFlowMatcher from the ``flow`` config block."""
    f = cfg["flow"]
    return LatentFlowMatcher(
        latent_ch=4,
        cond_ch=cond_ch,
        hidden=f["hidden"],
        n_blocks=f["n_blocks"],
        t_dim=f["t_dim"],
    )
