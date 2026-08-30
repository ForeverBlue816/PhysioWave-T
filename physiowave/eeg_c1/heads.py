# -*- coding: utf-8 -*-
"""Downstream heads: what sits between a frozen encoder and a logit.

WHY THIS IS NOT ONE LINEAR LAYER
--------------------------------
EEGPT calls its downstream protocol "linear-probing" and describes the added
modules as "adaptive spatial filters (1x1 convolution) ... and a linear layer".
Their code is more than that, and the difference is most of the gap between
their published rows and a mean-pooled 1,538-parameter head:

    PhysioP300   chan_scale (a learnable per-electrode gain)
                 -> frozen encoder
                 -> Dropout(0.5)
                 -> LinearWithConstraint(2048, 16)   per time position
                 -> flatten over the 15 positions
                 -> LinearWithConstraint(240, 2)     ~33k trainable

    Sleep-EDFx   Conv1dWithConstraint(2, 13, 1)      2 derivations -> 13 named
                 -> frozen encoder                      10-20 electrodes
                 -> Dropout(0.5)
                 -> LinearWithConstraint(2048, 64)   per time position
                 -> + sinusoidal position, prepend a cls token
                 -> 4-layer transformer over the positions
                 -> LinearWithConstraint(64, 5)      ~330k trainable

So three things are doing work that a mean-and-a-linear cannot do at all:

1.  THE TIME AXIS SURVIVES. Both heads project per position and aggregate
    afterwards. A mean over positions is the one aggregation that cannot
    represent "this happened at 300 ms".
2.  THE INPUT MONTAGE IS ADAPTED, and it is trainable even when the encoder is
    not. For Sleep-EDFx this is the interesting one: Fpz-Cz and Pz-Oz are
    bipolar derivations whose channel-vocabulary rows no monopolar pretraining
    corpus ever touched, so the channel encoder contributes nothing on this
    dataset. A 1x1 mix onto thirteen named 10-20 electrodes gives it rows that
    pretraining actually trained -- it switches the C1 mechanism ON for a task
    where it is currently idle.
3.  MAX-NORM CONSTRAINED LINEARS, from EEGNet. On a few thousand labelled
    windows this is the regulariser that does the work; weight decay on a
    1,500-parameter head is close to a no-op.

None of this makes the probe less of a measurement of the encoder, as long as
the pretrained and the from-scratch arm get the SAME head. It does mean the
word "linear" is doing no work in either their protocol or ours, and the
parameter count belongs next to the number.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Constrained layers (EEGNet, via EEGPT's Modules/Network/utils.py)
# --------------------------------------------------------------------------- #
class MaxNormLinear(nn.Linear):
    """``nn.Linear`` whose output rows are renormalised to ``max_norm``.

    Applied in ``forward`` rather than after the optimiser step, which is where
    it belongs and where a module cannot reach. Doing it every forward -- eval
    included, as EEGPT does -- keeps train and eval reading the same weights;
    renormalising only in train mode would leave eval looking at whatever the
    last optimiser step produced, which is a different function.

    The projection is idempotent, so a forward that changes nothing costs a
    ``renorm`` and no more.
    """

    def __init__(self, in_features: int, out_features: int,
                 max_norm: float = 1.0, **kw) -> None:
        super().__init__(in_features, out_features, **kw)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm > 0:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0,
                                               maxnorm=self.max_norm))
        return super().forward(x)


class MaxNormConv1d(nn.Conv1d):
    """``nn.Conv1d`` under the same constraint, for the 1x1 spatial mix."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 max_norm: float = 1.0, **kw) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kw)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm > 0:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0,
                                               maxnorm=self.max_norm))
        return super().forward(x)


# --------------------------------------------------------------------------- #
# In front of the encoder
# --------------------------------------------------------------------------- #
class AdaptiveSpatialFilter(nn.Module):
    """Adapts the recorded montage to the one the encoder was trained on.

    ``scale``  a learnable gain per input electrode, EEGPT's ``chan_scale``.
               The montage is unchanged; what this absorbs is the per-electrode
               amplitude the encoder expects, which a fixed preprocessing
               z-score can only get right on average.
    ``mix``    a 1x1 convolution onto a NAMED set of output electrodes, EEGPT's
               ``chan_conv``. The names are the point: they are what the channel
               encoder looks up, so the mix decides which pretrained electrode
               identities this dataset gets to use.

    Initialised at identity where that is meaningful -- a gain of 1, and for a
    mix, each output drawing equally from the inputs -- so the first forward of
    a fine-tune is the forward the model would have done without the filter.
    """

    def __init__(self, kind: str, in_channels: int,
                 out_names: Optional[Sequence[str]] = None,
                 max_norm: float = 1.0) -> None:
        super().__init__()
        self.kind = kind
        self.in_channels = int(in_channels)
        if kind == "scale":
            # 1 + noise, not 1: identical rows have identical gradients, and a
            # per-electrode gain that starts perfectly tied stays tied for as
            # long as the inputs are symmetric. EEGPT seeds it the same way.
            self.gain = nn.Parameter(torch.ones(1, self.in_channels, 1)
                                     + 0.001 * torch.rand(1, self.in_channels, 1))
            self.out_names = None
            self.out_channels = self.in_channels
        elif kind == "mix":
            if not out_names:
                raise ValueError("spatial_filter='mix' needs the output electrode "
                                 "names; they are what the channel encoder reads")
            self.out_names = list(out_names)
            self.out_channels = len(self.out_names)
            self.mix = MaxNormConv1d(self.in_channels, self.out_channels, 1,
                                     max_norm=max_norm, bias=False)
            with torch.no_grad():
                self.mix.weight.fill_(1.0 / self.in_channels)
                self.mix.weight.add_(0.01 * torch.randn_like(self.mix.weight))
        else:
            raise ValueError(f"spatial_filter must be scale or mix, got {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "scale":
            return x * self.gain
        return self.mix(x)

    def extra_repr(self) -> str:
        return f"kind={self.kind}, {self.in_channels} -> {self.out_channels}"


# --------------------------------------------------------------------------- #
# After the encoder
# --------------------------------------------------------------------------- #
def sinusoidal_positions(n: int, dim: int, device, dtype) -> torch.Tensor:
    """``[1, n, dim]`` sin/cos positions, as EEGPT's head uses.

    Parameter-free on purpose: the head is the part that is allowed to be small,
    and a learned table over positions is one more thing that has to be fit from
    the same few thousand labelled windows.
    """
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    half = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * half / dim)
    out = torch.zeros(n, dim, device=device, dtype=torch.float32)
    out[:, 0::2] = torch.sin(pos * freq)
    out[:, 1::2] = torch.cos(pos * freq)[:, :out[:, 1::2].shape[1]]
    return out.unsqueeze(0).to(dtype)


class ChannelPool(nn.Module):
    """``[B, C, P, D] -> [B, P, D]``: how the electrodes are summarised.

    ``mean``  what EEGPT's encoder does for them -- their summary tokens are
              already channel-agnostic, so their head never sees an electrode
              axis. Ours does, and a mean is the honest default.
    ``attn``  a single learned query scores each electrode against the token
              itself, per position. It can express "read this from the parietal
              electrodes at this moment", which a mean cannot, and it is the one
              place this head goes past theirs rather than beside it.

              D parameters, deliberately: the query is dotted with the raw
              token and there is no key projection. A D-by-D key would add
              147k parameters to a head whose whole claim is that it is too
              small to be where the score comes from.
    """

    def __init__(self, kind: str, dim: int) -> None:
        super().__init__()
        self.kind = kind
        self.scale = dim ** -0.5
        if kind == "attn":
            self.query = nn.Parameter(torch.randn(dim) * 0.02)
        elif kind != "mean":
            raise ValueError(f"channel_pool must be mean or attn, got {kind!r}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.kind == "mean":
            return tokens.mean(dim=1)
        # [B, C, P, D] -> one score per electrode at each time position
        scores = (tokens * self.query).sum(-1) * self.scale              # [B, C, P]
        weights = scores.softmax(dim=1).unsqueeze(-1)                    # [B, C, P, 1]
        return (tokens * weights).sum(dim=1)


class TemporalHead(nn.Module):
    """``[B, C, P, D] -> [B, K]``, keeping the time axis until the last step.

    ``kind='flatten'`` is EEGPT's PhysioP300 head: project each position, then
    concatenate. Exact about *when*, and its parameter count grows with the
    number of positions, which is why they use it on a 2 s window and not on a
    30 s one.

    ``kind='attn'`` is their Sleep-EDFx head: project each position, add a
    position code, prepend a cls token and let a small transformer read the
    sequence. Constant in the number of positions and the only one of the two
    that can be handed a 60-position window.
    """

    def __init__(self, kind: str, embed_dim: int, n_positions: int,
                 num_classes: int, probe_dim: int = 16, depth: int = 4,
                 num_heads: int = 4, dropout: float = 0.0,
                 max_norm: float = 0.0, channel_pool: str = "mean",
                 norm: str = "rmsnorm", ffn: str = "swiglu",
                 qk_norm: bool = True) -> None:
        super().__init__()
        if kind not in ("flatten", "attn"):
            raise ValueError(f"head kind must be flatten or attn, got {kind!r}")
        self.kind = kind
        self.n_positions = int(n_positions)
        self.channel_pool = ChannelPool(channel_pool, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, int(probe_dim))
        if kind == "flatten":
            self.encoder = None
            self.cls_token = None
            self.out = MaxNormLinear(self.n_positions * int(probe_dim),
                                     num_classes, max_norm=max_norm)
        else:
            from transformer_modules import TransformerEncoder

            self.cls_token = nn.Parameter(torch.randn(1, 1, int(probe_dim)) * 0.001)
            self.encoder = TransformerEncoder(
                embed_dim=int(probe_dim), depth=int(depth), num_heads=int(num_heads),
                mlp_ratio=4.0, dropout=dropout, rope_dim=None,
                norm=norm, ffn=ffn, qk_norm=qk_norm)
            self.out = MaxNormLinear(int(probe_dim), num_classes, max_norm=max_norm)

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.channel_pool(tokens)                       # [B, P, D]
        h = self.proj(self.drop(self.norm(h)))              # [B, P, probe]
        if self.kind == "flatten":
            return h.flatten(1)
        cls = self.cls_token.expand(h.shape[0], -1, -1).to(h.dtype)
        h = h + sinusoidal_positions(h.shape[1], h.shape[2], h.device, h.dtype)
        return self.encoder(torch.cat([cls, h], dim=1))[:, 0]

    def forward(self, tokens: torch.Tensor):
        pooled = self.features(tokens)
        return self.out(pooled), pooled
