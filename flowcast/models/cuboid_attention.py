"""Axial Cuboid Self-Attention block (FlowCast / Earthformer style).

Processes a spatiotemporal feature map (B, T, C, H, W) by applying
multi-head self-attention sequentially along the temporal, height, and
width axes — the "axial" pattern from Earthformer (Gao et al., 2022)
referenced in FlowCast section 3.2.2. Each pass keeps the other axes as
batch dimensions, which keeps the attention matrix tractable (max size
~max(T, H, W) instead of T*H*W).

This is a faithful but compact reimplementation aimed at fitting on a
single T4. It uses relative positional embeddings per axis and a standard
GroupNorm + GELU FFN with residual connections.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _relpos_table(n, head_dim):
    """Learnable relative-position bias table of shape (2n-1, head_dim)."""
    return nn.Parameter(torch.zeros(2 * n - 1, head_dim))


def _relpos_index(n, device):
    """Indices into the (2n-1) relpos table for an n x n pairwise grid."""
    coords = torch.arange(n, device=device)
    rel = coords[None, :] - coords[:, None] + (n - 1)        # (n, n)
    return rel


class AxialMHSA(nn.Module):
    """Multi-head self-attention along a single axis with relative-pos bias.

    Input  : (B, L, C)  L = axis length, C = channel.
    """

    def __init__(self, dim, num_heads=4, axis_len=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.relpos = _relpos_table(axis_len, num_heads)  # bias per head, not per head_dim
        self.axis_len = axis_len

    def forward(self, x):
        B, L, C = x.shape
        if L != self.axis_len:
            # rebuild relpos lazily if a different axis length appears (rare)
            self.relpos = nn.Parameter(torch.zeros(
                2 * L - 1, self.num_heads, device=x.device, dtype=x.dtype))
            self.axis_len = L
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                           # (3, B, h, L, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale              # (B, h, L, L)
        idx = _relpos_index(L, x.device)
        bias = self.relpos[idx].permute(2, 0, 1)                   # (h, L, L)
        attn = attn + bias[None]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)           # (B, L, C)
        return self.proj_drop(self.proj(out))


class FFN(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.1):
        super().__init__()
        hidden = dim * mult
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class CuboidAttentionBlock(nn.Module):
    """One Cuboid Self-Attention block: axial attention over T, H, W + FFN.

    Input  : (B, T, C, H, W)
    Output : same shape; residuals applied after each axial pass.
    """

    def __init__(self, dim, T, H, W, num_heads=4, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = AxialMHSA(dim, num_heads, axis_len=T, dropout=dropout)
        self.norm_h = nn.LayerNorm(dim)
        self.attn_h = AxialMHSA(dim, num_heads, axis_len=H, dropout=dropout)
        self.norm_w = nn.LayerNorm(dim)
        self.attn_w = AxialMHSA(dim, num_heads, axis_len=W, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mult=ffn_mult, dropout=dropout)

    @staticmethod
    def _along_t(x):
        B, T, C, H, W = x.shape
        return x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C), (B, H, W, T, C)

    @staticmethod
    def _back_t(y, shape):
        B, H, W, T, C = shape
        return y.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()

    @staticmethod
    def _along_h(x):
        B, T, C, H, W = x.shape
        return x.permute(0, 1, 4, 3, 2).reshape(B * T * W, H, C), (B, T, W, H, C)

    @staticmethod
    def _back_h(y, shape):
        B, T, W, H, C = shape
        return y.reshape(B, T, W, H, C).permute(0, 1, 4, 3, 2).contiguous()

    @staticmethod
    def _along_w(x):
        B, T, C, H, W = x.shape
        return x.permute(0, 1, 3, 4, 2).reshape(B * T * H, W, C), (B, T, H, W, C)

    @staticmethod
    def _back_w(y, shape):
        B, T, H, W, C = shape
        return y.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

    def forward(self, x):
        # axial T
        y, sh = self._along_t(x)
        y = y + self.attn_t(self.norm_t(y))
        x = self._back_t(y, sh)
        # axial H
        y, sh = self._along_h(x)
        y = y + self.attn_h(self.norm_h(y))
        x = self._back_h(y, sh)
        # axial W
        y, sh = self._along_w(x)
        y = y + self.attn_w(self.norm_w(y))
        x = self._back_w(y, sh)
        # FFN (apply on channel dim)
        y, sh = self._along_t(x)        # reuse: any reshape works since FFN is pointwise
        y = y + self.ffn(self.norm_ffn(y))
        x = self._back_t(y, sh)
        return x
