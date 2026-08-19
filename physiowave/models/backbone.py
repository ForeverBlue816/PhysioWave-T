"""Factorized backbone over the compressed latent grid ``[B, K, S, D]``.

The sequence is never flattened back to ``(J + 1) * C * S``.  Each block

1. attends along the **time** axis ``S`` independently per latent slot
   (cost ``O(B K S^2 D)``), then
2. mixes the ``K`` latent slots with a light attention over the slot axis
   (cost ``O(B S K^2 D)``, and ``K`` is 4-32), then
3. applies a position-wise FFN.

Compared with full attention over ``K * S`` tokens this replaces an
``O((KS)^2)`` term with ``O(K S^2 + S K^2)``, which is what keeps long windows
affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BackboneConfig:
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    slot_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_rope: bool = True


class RotaryEmbedding(nn.Module):
    """Rotary position embedding applied to the temporal axis."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim, self.base = dim, base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, H, N, D] -> [B, H, N, D]``."""
        B, H, N, D = x.shape
        pos = torch.arange(N, device=x.device, dtype=x.dtype)
        inv = 1.0 / (self.base ** (torch.arange(0, D, 2, device=x.device, dtype=x.dtype) / D))
        freqs = pos[:, None] * inv[None, :]                # [N, D/2]
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
        xe, xo = x[..., 0::2], x[..., 1::2]
        out = torch.stack([xe * cos - xo * sin, xe * sin + xo * cos], dim=-1)
        return out.flatten(-2)


class MultiHeadAttention(nn.Module):
    """Standard MHA with optional RoPE, used for both factorized axes."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_rope: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} not divisible by num_heads={num_heads}"
        self.h, self.hd = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.hd) if use_rope else None

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[N, L, D] -> [N, L, D]``; ``key_padding_mask`` is ``[N, L]``, True = keep."""
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(N, 1, 1, L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(N, L, D)
        return self.drop(self.proj(out))


class FactorizedBlock(nn.Module):
    """Temporal attention -> slot mixing -> FFN, each pre-normed and residual."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        D = cfg.embed_dim
        self.n1 = nn.LayerNorm(D)
        self.time_attn = MultiHeadAttention(D, cfg.num_heads, cfg.dropout, cfg.use_rope)
        self.n2 = nn.LayerNorm(D)
        self.slot_attn = MultiHeadAttention(D, cfg.slot_heads, cfg.dropout, use_rope=False)
        self.n3 = nn.LayerNorm(D)
        hidden = int(D * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(hidden, D), nn.Dropout(cfg.dropout))

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, K, S, D] -> [B, K, S, D]``."""
        B, K, S, D = x.shape
        h = self.n1(x).reshape(B * K, S, D)
        tm = None
        if time_mask is not None:
            tm = time_mask.unsqueeze(1).expand(B, K, S).reshape(B * K, S)
        x = x + self.time_attn(h, tm).view(B, K, S, D)

        h = self.n2(x).permute(0, 2, 1, 3).reshape(B * S, K, D)
        x = x + self.slot_attn(h).view(B, S, K, D).permute(0, 2, 1, 3)

        return x + self.mlp(self.n3(x))


class FactorizedBackbone(nn.Module):
    """Stack of :class:`FactorizedBlock` with a learned temporal position embedding."""

    def __init__(self, cfg: BackboneConfig, max_patches: int = 512) -> None:
        super().__init__()
        self.cfg = cfg
        self.slot_embed = nn.Parameter(torch.randn(1, 64, 1, cfg.embed_dim) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, 1, max_patches, cfg.embed_dim) * 0.02)
        self.blocks = nn.ModuleList(FactorizedBlock(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 4, f"expected [B, K, S, D], got {tuple(x.shape)}"
        B, K, S, D = x.shape
        assert K <= self.slot_embed.shape[1], f"K={K} exceeds slot embedding capacity"
        assert S <= self.time_embed.shape[2], f"S={S} exceeds max_patches"
        x = x + self.slot_embed[:, :K] + self.time_embed[:, :, :S]
        for blk in self.blocks:
            x = blk(x, time_mask)
        return self.norm(x)
