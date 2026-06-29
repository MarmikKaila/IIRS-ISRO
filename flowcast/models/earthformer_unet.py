"""Earthformer-UNet backbone for FlowCast Conditional Flow Matching.

Reimplements (compact, T4-friendly) the FlowCast paper's architecture
(section 3.2.2, Figure 1):
    proj_in (Conv3D-style: 1x3x3) on (T, C_in -> hidden)
    Spatiotemporal positional embedding (learned)
    Observation Masking is delegated to the caller — the model expects an
    input that already has past frames untouched and future frames replaced
    by the noisy interpolation z_t (FlowCast eq. line 6 of Algorithm 1).
    --- encoder stage 0 (full res) ---
    4 x [CuboidAttentionBlock + TimeFiLM]
    PatchMerge: H/2, W/2, C*2
    --- bottleneck stage 1 ---
    4 x [CuboidAttentionBlock + TimeFiLM]
    Upsample: nearest x2, channel /2
    --- decoder stage 0 (full res, skip-fused) ---
    fuse(concat(skip, up)) -> hidden
    4 x [CuboidAttentionBlock + TimeFiLM]
    head: pointwise 1x1 conv (hidden -> C_out)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .cuboid_attention import CuboidAttentionBlock


# ── time embedding ──────────────────────────────────────────────────────────
def sinusoidal_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeFiLM(nn.Module):
    """FiLM modulation: produces per-channel (scale, shift) from the CFM time."""

    def __init__(self, t_dim, ch):
        super().__init__()
        self.proj = nn.Linear(t_dim, 2 * ch)
        self.act = nn.SiLU()

    def forward(self, x_btchw, t_emb):
        s, b = self.proj(self.act(t_emb)).chunk(2, dim=1)
        s = s[:, None, :, None, None]
        b = b[:, None, :, None, None]
        return x_btchw * (1.0 + s) + b


# ── spatial down / up ──────────────────────────────────────────────────────
class PatchMerge(nn.Module):
    """Spatial /2, channel x2 — Swin-style PatchMerging applied per timestep."""

    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Linear(4 * ch, 2 * ch, bias=False)
        self.norm = nn.LayerNorm(4 * ch)

    def forward(self, x_btchw):
        B, T, C, H, W = x_btchw.shape
        x = x_btchw.permute(0, 1, 3, 4, 2)                  # (B, T, H, W, C)
        x = x.reshape(B, T, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, H // 2, W // 2, 4 * C)
        x = self.proj(self.norm(x))                          # (B, T, H/2, W/2, 2C)
        return x.permute(0, 1, 4, 2, 3).contiguous()


class NearestUpsample(nn.Module):
    """Nearest-neighbour spatial x2 + 3x3 conv that halves the channel count."""

    def __init__(self, ch):
        super().__init__()
        # 2D conv applied per-timestep
        self.conv = nn.Conv2d(ch, ch // 2, 3, padding=1, padding_mode="reflect")

    def forward(self, x_btchw):
        B, T, C, H, W = x_btchw.shape
        x = x_btchw.reshape(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x.reshape(B, T, -1, H * 2, W * 2)


# ── stage = N Cuboid blocks with FiLM time conditioning ────────────────────
class CuboidStage(nn.Module):
    """N cuboid blocks + FiLM time conditioning. With ``use_checkpoint=True``
    (default during training) each block is wrapped in
    ``torch.utils.checkpoint`` — activations are recomputed on backward,
    trading ~30 % more compute for a roughly 3x drop in peak memory. This
    is what makes the paper-spec hidden=192 / depth=4 stage fit on a T4."""

    def __init__(self, ch, T, H, W, depth, num_heads, t_dim, dropout,
                  use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            CuboidAttentionBlock(ch, T, H, W, num_heads=num_heads,
                                 dropout=dropout)
            for _ in range(depth)
        ])
        self.films = nn.ModuleList([TimeFiLM(t_dim, ch) for _ in range(depth)])

    def forward(self, x, t_emb):
        for blk, film in zip(self.blocks, self.films):
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            x = film(x, t_emb)
        return x


# ── Earthformer-UNet ───────────────────────────────────────────────────────
class EarthformerUNet(nn.Module):
    """2-level U-Net with Cuboid attention; predicts a velocity field of the
    same spatiotemporal shape as the input.

    in_channels  : latent_ch (default 4) + 1 mask channel = 5 by default.
    out_channels : latent_ch (default 4) — velocity prediction.
    """

    def __init__(self, in_channels=5, out_channels=4, hidden=192,
                 T=25, H=36, W=32, depth=4, num_heads=4, t_dim=192 * 4,
                 dropout=0.1):
        super().__init__()
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"H, W must be even for PatchMerge; got {H}x{W}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.T, self.H, self.W = T, H, W
        self.t_dim = t_dim

        # initial projection (per-timestep 3x3 conv)
        self.proj_in = nn.Conv2d(in_channels, hidden, 3, padding=1,
                                  padding_mode="reflect")
        # learned spatiotemporal positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, T, hidden, H, W))
        # time MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim // 4, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim))
        self._t_emb_in = t_dim // 4

        # encoder stage 0 (full res)
        self.enc0 = CuboidStage(hidden, T, H, W, depth=depth,
                                 num_heads=num_heads, t_dim=t_dim, dropout=dropout)
        self.down = PatchMerge(hidden)
        # bottleneck stage 1 (half res, 2x channels)
        self.enc1 = CuboidStage(hidden * 2, T, H // 2, W // 2,
                                 depth=depth, num_heads=num_heads,
                                 t_dim=t_dim, dropout=dropout)
        # up + skip-fuse + decoder
        self.up = NearestUpsample(hidden * 2)
        self.fuse = nn.Conv2d(hidden * 2, hidden, 3, padding=1,
                               padding_mode="reflect")
        self.dec0 = CuboidStage(hidden, T, H, W, depth=depth,
                                 num_heads=num_heads, t_dim=t_dim,
                                 dropout=dropout)

        self.norm_out = nn.GroupNorm(8, hidden)
        self.conv_out = nn.Conv2d(hidden, out_channels, 1)

        nn.init.normal_(self.pos_emb, std=0.02)

    def _proj_in_per_t(self, x_btchw):
        B, T, C, H, W = x_btchw.shape
        x = x_btchw.reshape(B * T, C, H, W)
        x = self.proj_in(x)
        return x.reshape(B, T, -1, H, W)

    def _fuse_per_t(self, x_btchw):
        B, T, C, H, W = x_btchw.shape
        x = x_btchw.reshape(B * T, C, H, W)
        x = self.fuse(x)
        return x.reshape(B, T, -1, H, W)

    def _head(self, x_btchw):
        B, T, C, H, W = x_btchw.shape
        x = x_btchw.reshape(B * T, C, H, W)
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x.reshape(B, T, -1, H, W)

    def forward(self, x_btchw, t):
        """x_btchw: (B, T, in_channels, H, W); t: (B,) in [0,1]."""
        t_emb = self.t_mlp(sinusoidal_embed(t, self._t_emb_in))

        # initial projection + positional embedding
        x = self._proj_in_per_t(x_btchw)
        x = x + self.pos_emb

        # encoder
        skip = self.enc0(x, t_emb)
        x = self.down(skip)
        x = self.enc1(x, t_emb)

        # decoder
        x = self.up(x)                       # back to (B, T, hidden, H, W)
        x = torch.cat([x, skip], dim=2)      # channel-axis cat
        x = self._fuse_per_t(x)
        x = self.dec0(x, t_emb)

        return self._head(x)                 # (B, T, out_channels, H, W)


def build_earthformer(cfg, in_channels, out_channels, T, H, W):
    """Read flow config and return an EarthformerUNet."""
    f = cfg["flow"]
    return EarthformerUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden=int(f.get("hidden", 192)),
        T=T, H=H, W=W,
        depth=int(f.get("n_blocks", 4)),
        num_heads=int(f.get("attn_heads", 4)),
        t_dim=int(f.get("hidden", 192)) * 4,
        dropout=float(f.get("dropout", 0.1)),
    )
