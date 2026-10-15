"""EEGPT and LaBraM as EEG encoders of the fusion trainer, and the channel adapter.

The point is a like-for-like single-modality comparison on SeizeIT2: the
same windows, split, head, schedule, model selection and event-level scorer
for the C1 EEG encoder and for two published EEG foundation models. The
fusion trainer only needs an encoder to offer ``embed_dim`` and
``encode(x, meta) -> [B, L, D]`` tokens; the two classes here wrap the
published backbones to that interface.

Backbones -- the published encoders, re-implemented with the checkpoints'
parameter names so the released weights load strictly:

    eegpt   EEGPT (Wang et al., NeurIPS 2024), the target encoder of
            ``eegpt_mcae_58chs_4s_large4E``: 64-sample patches at 256 Hz,
            each patch's channels plus 4 summary tokens through 8 blocks
            (512 wide); the encoder's output is the summary tokens,
            [B, patches, 4, 512] -> [B, 4*patches, 512]. Input in mV.
            Source: github.com/BINE022/EEGPT (Apache-2.0),
            downstream/Modules/models/EEGPT_mcae.py, ``EEGTransformer``.
    labram  LaBraM-Base (Jiang et al., ICLR 2024), ``labram-base.pth``'s
            student: 200-sample patches at 200 Hz through a temporal conv,
            electrode and time embeddings, 12 blocks (200 wide) with q/k
            LayerNorm and layer scale. Input in units of 0.1 mV.
            Source: github.com/935963004/LaBraM (MIT), modeling_finetune.py,
            ``NeuralTransformer``.

Montage. SeizeIT2 has two behind-the-ear channels that are no electrode of
either model's vocabulary (nor of C1's). EEGPT's own recipe for a two-channel
dataset (Sleep-EDF, downstream/finetune_EEGPT_SleepEDF.py) is a learned 1x1
convolution with a max-norm constraint from the recorded channels onto named
10-20 electrodes; :class:`ChannelAdapter` is that layer, and the same adapter
onto the same electrodes (default the temporal chains F7 T7 P7 / F8 T8 P8,
which every model knows and which put C1 on its E19_256 route) is given to
all three models, so none gets a montage advantage. The adapter always
trains, also when the backbone is frozen.

Amplitude. The stored windows are z-scored per channel (prep_common); the
adapter's output is z-scored per channel again and then scaled to the input
convention of each model's pretraining: C1 z-scored (scale 1), EEGPT in mV
and LaBraM in 0.1 mV for a channel of ``input_uv`` microvolts RMS (30,
typical behind-the-ear EEG; SeizeIT2's reference code accepts 13-150 uV).
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: EEGPT's channel vocabulary (EEGPT_mcae.py CHANNEL_DICT), in index order.
EEGPT_CHANNELS: Tuple[str, ...] = (
    "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2")

#: LaBraM's electrode list (utils.py standard_1020); position i+1 of pos_embed
#: (0 is the CLS token).
LABRAM_CHANNELS: Tuple[str, ...] = tuple(
    "FP1 FPZ FP2 AF9 AF7 AF5 AF3 AF1 AFZ AF2 AF4 AF6 AF8 AF10 "
    "F9 F7 F5 F3 F1 FZ F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 FT10 "
    "T9 T7 C5 C3 C1 CZ C2 C4 C6 T8 T10 TP9 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 TP10 "
    "P9 P7 P5 P3 P1 PZ P2 P4 P6 P8 P10 PO9 PO7 PO5 PO3 PO1 POZ PO2 PO4 PO6 PO8 PO10 "
    "O1 OZ O2 O9 CB1 CB2 IZ O10 T3 T5 T4 T6 M1 M2 A1 A2 "
    "CFC1 CFC2 CFC3 CFC4 CFC5 CFC6 CFC7 CFC8 CCP1 CCP2 CCP3 CCP4 CCP5 CCP6 CCP7 CCP8 "
    "T1 T2 FTT9h TTP7h TPP9h FTT10h TPP8h TPP10h".split()) + (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2")

DEFAULT_TARGETS: Tuple[str, ...] = ("F7", "T7", "P7", "F8", "T8", "P8")
ARCHS = ("c1", "eegpt", "labram")


# --------------------------------------------------------------------------- #
# Checkpoint reading
# --------------------------------------------------------------------------- #

_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
              "F64": torch.float64, "I64": torch.int64, "I32": torch.int32}


def read_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """A ``.safetensors`` file as a state dict, without the safetensors package."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        blob = fh.read()
    header.pop("__metadata__", None)
    out = {}
    for k, spec in header.items():
        a, b = spec["data_offsets"]
        dt = _ST_DTYPES[spec["dtype"]]
        t = torch.frombuffer(bytearray(blob[a:b]), dtype=dt) if b > a else torch.empty(0, dtype=dt)
        out[k] = t.reshape(spec["shape"]).clone()
    return out


def read_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """The flat state dict of a released checkpoint (.ckpt / .pth / .safetensors)."""
    if path.endswith(".safetensors"):
        return read_safetensors(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "module"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            return ck[key]
    return ck


def sub_state(sd: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    """The entries under the first prefix that has any, prefix removed."""
    for p in prefixes:
        got = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        if got:
            return got
    return {}


# --------------------------------------------------------------------------- #
# The channel adapter (EEGPT's Conv1dWithConstraint, max_norm=1)
# --------------------------------------------------------------------------- #

class ChannelAdapter(nn.Conv1d):
    """``[B, C_in, T] -> [B, C_out, T]``: a 1x1 convolution whose output
    filters are renormalised to L2 norm <= ``max_norm`` before every use."""

    def __init__(self, in_channels: int, out_channels: int, max_norm: float = 1.0):
        super().__init__(in_channels, out_channels, kernel_size=1)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.std(dim=-1, keepdim=True).clamp_min(eps)


# --------------------------------------------------------------------------- #
# EEGPT target encoder
# --------------------------------------------------------------------------- #

class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _EEGPTAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class _EEGPTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _EEGPTAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _EEGPTPatch(nn.Module):
    def __init__(self, patch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(1, dim, kernel_size=(1, patch), stride=(1, patch))

    def forward(self, x):                                    # [B, C, T] -> [B, N, C, D]
        return self.proj(x.unsqueeze(1)).transpose(1, 3)


class EEGPTBackbone(nn.Module):
    """EEGPT's ``EEGTransformer`` (the target encoder), large: 512 x 8, 4 summary tokens."""

    def __init__(self, patch: int = 64, dim: int = 512, depth: int = 8, heads: int = 8,
                 embed_num: int = 4):
        super().__init__()
        self.patch, self.embed_dim, self.embed_num = patch, dim, embed_num
        self.patch_embed = _EEGPTPatch(patch, dim)
        self.chan_embed = nn.Embedding(len(EEGPT_CHANNELS), dim)
        self.blocks = nn.ModuleList(_EEGPTBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for i, b in enumerate(self.blocks):                   # EEGPT's fix_init_weight
            b.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            b.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

    @staticmethod
    def channel_ids(names: Sequence[str]) -> List[int]:
        idx = {c: i for i, c in enumerate(EEGPT_CHANNELS)}
        out = []
        for n in names:
            k = n.upper().strip(".")
            if k not in idx:
                raise ValueError(f"EEGPT has no electrode {n!r}")
            out.append(idx[k])
        return out

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor) -> torch.Tensor:
        """``[B, C, T]`` (mV) -> summary tokens ``[B, N * embed_num, D]``."""
        x = self.patch_embed(x)                              # B, N, C, D
        B, N, C, D = x.shape
        x = x + self.chan_embed(chan_ids.long()).view(1, 1, C, D)
        x = x.flatten(0, 1)                                  # B*N, C, D
        x = torch.cat([x, self.summary_token.expand(x.shape[0], -1, -1).to(x.dtype)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, -self.embed_num:, :])             # B*N, E, D
        return x.reshape(B, N * self.embed_num, D)


# --------------------------------------------------------------------------- #
# LaBraM-Base
# --------------------------------------------------------------------------- #

class _LaBraMAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        hd = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.LayerNorm(hd, eps=1e-6)
        self.k_norm = nn.LayerNorm(hd, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q).type_as(v), self.k_norm(k).type_as(v)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, N, -1))


class _LaBraMBlock(nn.Module):
    def __init__(self, dim: int, heads: int, init_values: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _LaBraMAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, dim * 4)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        return x + self.gamma_2 * self.mlp(self.norm2(x))


class _TemporalConv(nn.Module):
    def __init__(self, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)

    def forward(self, x):                                    # [B, C, A, 200] -> [B, C*A, 200]
        B, C, A, T = x.shape
        x = x.reshape(B, 1, C * A, T)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))                # B, 8, C*A, 25
        return x.permute(0, 2, 3, 1).reshape(B, C * A, -1)   # (T C) order, as rearrange does


class LaBraMBackbone(nn.Module):
    """LaBraM-Base's ``NeuralTransformer`` (200 x 12, 10 heads), the pretrained final norm kept."""

    def __init__(self, dim: int = 200, depth: int = 12, heads: int = 10, patch: int = 200):
        super().__init__()
        self.embed_dim, self.patch = dim, patch
        self.patch_embed = _TemporalConv()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 16, dim))
        self.blocks = nn.ModuleList(_LaBraMBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        for p in (self.cls_token, self.pos_embed, self.time_embed):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for i, b in enumerate(self.blocks):
            b.attn.proj.weight.data.div_(math.sqrt(2.0 * (i + 1)))
            b.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (i + 1)))

    @staticmethod
    def channel_ids(names: Sequence[str]) -> List[int]:
        """Rows of ``pos_embed`` (1-based; 0 is CLS), as LaBraM's get_input_chans."""
        idx = {c.upper(): i for i, c in enumerate(LABRAM_CHANNELS)}
        out = []
        for n in names:
            k = n.upper()
            if k not in idx:
                raise ValueError(f"LaBraM has no electrode {n!r}")
            out.append(idx[k] + 1)
        return out

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor) -> torch.Tensor:
        """``[B, C, A, 200]`` (0.1 mV) -> patch tokens ``[B, C*A, D]``."""
        B, C, A, _ = x.shape
        x = self.patch_embed(x)
        pos = self.pos_embed[:, chan_ids.long()]             # 1, C, D
        pos = pos.unsqueeze(2).expand(B, C, A, -1).reshape(B, C * A, -1)
        time = self.time_embed[:, :A].unsqueeze(1).expand(B, C, A, -1).reshape(B, C * A, -1)
        x = x + pos + time
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        x = torch.cat([cls.to(x.dtype), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 1:]


# --------------------------------------------------------------------------- #
# The encoders the fusion trainer builds
# --------------------------------------------------------------------------- #

def _decay_for(key: str, prefix: str, depth: int, d: float) -> float:
    """Layer-wise decay for a backbone parameter named ``<prefix>blocks.<i>....``:
    block i at d**(depth - i), the final norm at 1, everything feeding block 0
    (patch embedding, electrode/time embeddings, tokens) at d**(depth + 1)."""
    m = re.match(re.escape(prefix) + r"blocks\.(\d+)\.", key)
    if m:
        return d ** (depth - int(m.group(1)))
    if key.startswith(prefix + "norm."):
        return 1.0
    return d ** (depth + 1)


class BaselineEEGEncoder(nn.Module):
    """Adapter -> per-channel z-score -> amplitude scale -> EEGPT or LaBraM."""

    def __init__(self, arch: str, in_channels: int, fs: float, window_samples: int,
                 targets: Sequence[str] = DEFAULT_TARGETS, input_uv: float = 30.0):
        super().__init__()
        if arch not in ("eegpt", "labram"):
            raise ValueError(arch)
        self.arch, self.fs, self.window_samples = arch, float(fs), int(window_samples)
        self.targets = [str(t) for t in targets]
        self.adapter = ChannelAdapter(in_channels, len(self.targets))
        if arch == "eegpt":
            if int(round(fs)) != 256:
                raise ValueError("EEGPT was pretrained at 256 Hz; the window is at %s Hz" % fs)
            self.backbone = EEGPTBackbone()
            if self.window_samples % self.backbone.patch:
                raise ValueError(f"{window_samples} samples is not a whole number of "
                                 f"{self.backbone.patch}-sample EEGPT patches")
            self.scale = float(input_uv) * 1e-3                 # mV
            ids = EEGPTBackbone.channel_ids(self.targets)
        else:
            self.backbone = LaBraMBackbone()
            self.resampled = int(round(self.window_samples * 200.0 / self.fs))
            if self.resampled % self.backbone.patch or self.resampled // self.backbone.patch > 16:
                raise ValueError(f"{self.resampled} samples at 200 Hz do not make 1-16 "
                                 "LaBraM patches")
            self.scale = float(input_uv) * 1e-2                 # 0.1 mV
            ids = LaBraMBackbone.channel_ids(self.targets)
        self.register_buffer("chan_ids", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.embed_dim = int(self.backbone.embed_dim)
        self.depth = len(self.backbone.blocks)

    def encode(self, x: torch.Tensor, meta=None) -> torch.Tensor:
        h = zscore(self.adapter(x.float())) * self.scale
        if self.arch == "labram":
            h = F.interpolate(h, size=self.resampled, mode="linear", align_corners=False)
            h = h.reshape(h.shape[0], h.shape[1], -1, self.backbone.patch)
        return self.backbone(h, self.chan_ids)

    # ------------------------------------------------------------------ #
    def load_pretrained(self, path: str) -> Dict[str, List[str]]:
        sd = read_checkpoint(path)
        prefixes = (("target_encoder.",) if self.arch == "eegpt"
                    else ("student.", ""))
        st = sub_state(sd, prefixes)
        if self.arch == "labram":
            # the pretraining-only parts: the mask token and the codebook head
            st = {k: v for k, v in st.items()
                  if k != "mask_token" and not k.startswith("lm_head.")}
        if not st:
            raise SystemExit(f"{path}: no {self.arch} encoder weights under {prefixes}")
        missing, unexpected = self.backbone.load_state_dict(st, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{path} does not fit the {self.arch} backbone: missing "
                             f"{list(missing)[:8]}, unexpected {list(unexpected)[:8]}")
        return {"taken": sorted({k.split(".")[0] for k in st}), "skipped": []}

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def layer_lr_scale(self, key: str, encoder_lr_scale: float, layer_decay: float) -> float:
        if not key.startswith("backbone."):
            return 1.0                                       # the adapter learns at lr
        f = float(encoder_lr_scale)
        if layer_decay != 1.0:
            f *= _decay_for(key, "backbone.", self.depth, float(layer_decay))
        return f


class AdaptedC1Encoder(nn.Module):
    """The same adapter in front of a C1 EEG encoder built for the target electrodes."""

    def __init__(self, inner: nn.Module, in_channels: int, targets: Sequence[str]):
        super().__init__()
        self.adapter = ChannelAdapter(in_channels, len(targets))
        self.inner = inner
        self.targets = [str(t) for t in targets]
        from .encoders import channel_meta
        self.inner_meta = channel_meta(self.targets)
        self.embed_dim = int(inner.embed_dim)
        self.depth = len(inner.shared_transformer.blocks)

    def encode(self, x: torch.Tensor, meta=None) -> torch.Tensor:
        return self.inner.encode(zscore(self.adapter(x.float())).to(x.dtype), self.inner_meta)

    def layer_lr_scale(self, key: str, encoder_lr_scale: float, layer_decay: float) -> float:
        if not key.startswith("inner."):
            return 1.0
        from physiowave.train.finetune_main import c1_lr_scale
        return c1_lr_scale(self.depth, encoder_lr_scale, layer_decay)(key[len("inner."):])


def build_baseline(arch: str, names: Sequence[str], fs: float, window_samples: int,
                   checkpoint: Optional[str], freeze: bool,
                   targets: Sequence[str] = DEFAULT_TARGETS, input_uv: float = 30.0
                   ) -> Tuple[nn.Module, dict]:
    enc = BaselineEEGEncoder(arch, len(names), fs, window_samples, targets, input_uv)
    report = {"modality": "eeg", "arch": arch, "route": None, "exported": checkpoint,
              "embed_dim": enc.embed_dim, "adapter": {"from": list(names), "to": enc.targets},
              "input_uv": float(input_uv), "taken": [], "skipped": [],
              "dropped_channels": [], "keep_index": None}
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise SystemExit(f"no {arch} checkpoint at {checkpoint}")
        report.update(enc.load_pretrained(checkpoint))
        report["summary"] = (f"{arch} loaded from {os.path.basename(checkpoint)}; adapter "
                             f"{len(names)} -> {'/'.join(enc.targets)} trains")
    else:
        report["summary"] = f"{arch} from scratch"
    if freeze:
        enc.freeze_backbone()
    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    report["trainable_params"] = int(n_tr)
    return enc, report


__all__ = ["ARCHS", "DEFAULT_TARGETS", "EEGPT_CHANNELS", "LABRAM_CHANNELS", "ChannelAdapter",
           "EEGPTBackbone", "LaBraMBackbone", "BaselineEEGEncoder", "AdaptedC1Encoder",
           "build_baseline", "read_checkpoint", "read_safetensors"]
