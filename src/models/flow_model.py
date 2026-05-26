"""Conditional Flow Matching for next-latent prediction — 2-level U-Net.

The model learns a velocity field ``v(z_t, t, cond)`` that transports Gaussian
noise to the next-frame latent given a conditioning tensor (past T latents,
plus per-frame SZA on VIS, plus the target SZA). Sampling integrates
``z'(t) = v(z(t), t, cond)`` with Euler steps; ~10 steps suffice.

Architecture (Phase B — agent-validated restructure):
    proj_in (4 + cond_ch -> h0)
      └─ enc0:  N ResBlocks @ 36x32, h0=96
              ↓ downsample (stride-2 3x3 conv)
      └─ enc1/bottleneck:  N ResBlocks @ 18x16, h1=192
                            +  one self-attention block
              ↑ upsample (nearest + 3x3 conv)
      └─ fuse (concat skip from enc0)
      └─ dec0:  N ResBlocks @ 36x32, h0=96
    head (h0 -> 4)

Skip connection and downsampling give the multi-scale receptive field the
old flat ResBlock stack lacked. Attention at 18x16 (288 tokens) is cheap and
helps long-range cloud-system consistency.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── time embedding ──────────────────────────────────────────────────────────
def sinusoidal_embed(t, dim):
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
        self.t_proj = nn.Linear(t_dim, 2 * ch)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.t_proj(self.act(t_emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        return x + self.conv2(self.act(h))


# ── self-attention at the bottleneck ────────────────────────────────────────
class SelfAttention2D(nn.Module):
    """Single-head spatial self-attention over a (B, C, H, W) map."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self._scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, C, H * W).transpose(1, 2)        # (B, HW, C)
        k = k.view(B, C, H * W)                        # (B, C, HW)
        v = v.view(B, C, H * W).transpose(1, 2)        # (B, HW, C)
        attn = torch.softmax((q @ k) * self._scale, dim=-1)
        out = (attn @ v).transpose(1, 2).view(B, C, H, W)
        return x + self.proj(out)


# ── down / up sampling helpers ──────────────────────────────────────────────
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1,
                              padding_mode="reflect")

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsample + 3x3 conv (more stable than transpose conv)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ── 2-level U-Net flow model ────────────────────────────────────────────────
class LatentFlowMatcher(nn.Module):
    """v(z_t, t, condition) -> velocity prediction (2-level U-Net).

    When ``joint_steps`` > 1, the model predicts a block of K next latents
    in one forward pass — the K frames are concatenated along the channel
    axis so the *effective* latent_ch the model sees is ``latent_ch * K``.
    This enforces temporal coherence at the loss level (Phase C).
    """

    def __init__(self, latent_ch=4, cond_ch=41, hidden=96,
                 n_blocks=2, t_dim=128, use_attn=True, joint_steps=1):
        super().__init__()
        self.latent_ch = latent_ch
        self.joint_steps = joint_steps
        self.out_ch = latent_ch * joint_steps          # K frames stacked in channels
        self.cond_ch = cond_ch
        self.t_dim = t_dim
        h0, h1 = hidden, hidden * 2

        self.proj_in = nn.Conv2d(self.out_ch + cond_ch, h0, 3, padding=1,
                                 padding_mode="reflect")
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim))

        # encoder level 0 (full resolution)
        self.enc0 = nn.ModuleList(
            [FlowResBlock(h0, t_dim) for _ in range(n_blocks)])
        self.down = Downsample(h0, h1)

        # encoder level 1 / bottleneck (half resolution)
        self.enc1 = nn.ModuleList(
            [FlowResBlock(h1, t_dim) for _ in range(n_blocks)])
        self.attn = SelfAttention2D(h1) if use_attn else nn.Identity()

        # decoder back to full resolution
        self.up = Upsample(h1, h0)
        self.fuse = nn.Conv2d(h0 + h0, h0, 3, padding=1, padding_mode="reflect")
        self.dec0 = nn.ModuleList(
            [FlowResBlock(h0, t_dim) for _ in range(n_blocks)])

        self.norm_out = nn.GroupNorm(8, h0)
        self.conv_out = nn.Conv2d(h0, self.out_ch, 3, padding=1,
                                  padding_mode="reflect")
        self.act = nn.SiLU()

    def forward(self, z_t, t, cond):
        t_emb = self.t_mlp(sinusoidal_embed(t, self.t_dim))
        x = self.proj_in(torch.cat([z_t, cond], dim=1))

        for blk in self.enc0:
            x = blk(x, t_emb)
        skip0 = x                                       # (B, h0, H, W)

        x = self.down(x)                                # (B, h1, H/2, W/2)
        for blk in self.enc1:
            x = blk(x, t_emb)
        x = self.attn(x)

        x = self.up(x)                                  # (B, h0, H, W)
        x = self.fuse(torch.cat([x, skip0], dim=1))     # skip
        for blk in self.dec0:
            x = blk(x, t_emb)

        return self.conv_out(self.act(self.norm_out(x)))


# ── conditional flow-matching loss ──────────────────────────────────────────
def cfm_loss(model, z1, cond, weights=None):
    """CFM loss with optional per-sample weighting (night-masking for VIS)."""
    B = z1.shape[0]
    z0 = torch.randn_like(z1)
    t = torch.rand(B, device=z1.device)
    t_e = t[:, None, None, None]
    z_t = (1.0 - t_e) * z0 + t_e * z1
    v_target = z1 - z0
    v_pred = model(z_t, t, cond)
    per = (v_pred - v_target).pow(2).mean(dim=(1, 2, 3))
    if weights is None:
        return per.mean()
    return (per * weights).sum() / weights.sum().clamp(min=1e-6)


# ── Euler ODE sampler ───────────────────────────────────────────────────────
@torch.no_grad()
def sample(model, cond, latent_shape, steps=10, device=None):
    device = device or cond.device
    z = torch.randn(latent_shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0],), i * dt, device=device)
        v = model(z, t, cond)
        z = z + dt * v
    return z


def build_flow_model(cfg, cond_ch):
    """Construct LatentFlowMatcher from the ``flow`` config block.

    Keys used here:
      hidden       = base channel count (h0); bottleneck uses 2 * hidden.
      n_blocks     = ResBlocks per U-Net level (enc0, enc1, dec0).
      use_attn     = 1-head self-attention at the bottleneck (default true).
      joint_steps  = number of next frames predicted jointly in one forward
                     pass. K > 1 enforces temporal coherence (Phase C).
    """
    f = cfg["flow"]
    return LatentFlowMatcher(
        latent_ch=4,
        cond_ch=cond_ch,
        hidden=f["hidden"],
        n_blocks=f["n_blocks"],
        t_dim=f["t_dim"],
        use_attn=bool(f.get("use_attn", True)),
        joint_steps=int(f.get("joint_steps", 4)),   # Phase C default
    )
