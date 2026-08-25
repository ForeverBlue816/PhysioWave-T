"""
Multi-route EEG pretrainer: four wavelet frontends, one Transformer.

The seven corpora do not share an electrode count or a sampling rate, and the
parts of the model that depend on those are exactly the parts that are
duplicated. Everything that does not is shared, because sharing it is the whole
point -- a separate model per corpus would be seven pretrainings, not one.

    per route (4)     SoftGateWaveletDecomp + dynamic ScaleFold
                      the wavelet filters are per-channel, so a frontend is
                      tied to its electrode count and cannot be shared
    per rate  (2)     PatchEmbed, reconstruction head
                      a 0.5 s patch is 128 samples at 256 Hz and 256 at 512,
                      so the conv kernel and the decoder width follow the rate
                      and nothing else -- two of each, not four
    shared    (1)     ChannelEncoder (C1, id), its projection and gate,
                      the 2-D position encoding, the mask token,
                      the RoPE Transformer encoder

The 2-D position encoding is sinusoidal and parameter-free -- it is generated
from (freq_size, time_size) at every call -- so one instance serves all four
token counts with nothing to resize and nothing to learn per route.

Routing is by ``route_id``, which the data carries. There is no learned gate
and this is not a mixture of experts: the recording's montage is known before
the model sees it.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn

from channel_embedding import CHANNEL_VOCAB, ChannelEncoder, vocab_sha256
from head_modules import ReconstructionHead
from transformer_modules import PatchEmbed, PositionEmbedding, TransformerEncoder
from wavelet_modules import ScaleFold, SoftGateWaveletDecomp

from .routes import ROUTES, Route


class WaveletFrontend(nn.Module):
    """One route's ``[B, C, T] -> [B, C, T]``: decompose, then fold the scales.

    The fold reduces the scale axis only. Its output still has one row per
    electrode, which is what lets a shared patcher treat a row as a channel on
    every route.
    """

    def __init__(self, route: Route, max_level: int = 3,
                 wave_kernel_size: int = 16, wavelet_names=None,
                 use_separate_channel: bool = True, wave_init_mode: str = "pad",
                 fold_synthesis: int = 3, fold_gamma: float = 0.1):
        super().__init__()
        self.route_id = route.route_id
        self.decomp = SoftGateWaveletDecomp(
            in_channels=route.n_channels,
            max_level=max_level,
            kernel_size=wave_kernel_size,
            wavelet_names=wavelet_names,
            use_separate_channel=use_separate_channel,
            init_mode=wave_init_mode,
            ffn_ratio=4.0, ffn_kernel_size=3, ffn_drop=0.1,
        )
        self.fold = ScaleFold(
            mode="dynamic",
            num_scales=max_level + 1,
            in_channels=route.n_channels,
            patch_len=route.patch_t,
            synthesis_kernel=fold_synthesis,
            gamma_init=fold_gamma,
        )

    def forward(self, x):
        return self.fold(self.decomp(x))

    @property
    def reg_loss(self):
        return self.fold.reg_loss

    @property
    def alpha_mean(self):
        return self.fold.alpha_mean


class MultiRouteEEGPretrainer(nn.Module):
    """Frequency-guided masked wavelet-patch reconstruction across four routes.

    ``forward`` takes one route's batch. A batch never mixes routes: the shapes
    differ, and mixing them would mean padding one route's tokens to another's
    length and then explaining the padding to the attention.
    """

    def __init__(self,
                 # -- the standard model, matching EEG/finetune_sleep.sh -------
                 max_level: int = 3,
                 wave_kernel_size: int = 16,
                 wavelet_names=None,
                 wave_init_mode: str = "pad",
                 use_separate_channel: bool = True,
                 fold_synthesis: int = 3,
                 fold_gamma: float = 0.1,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 norm: str = "rmsnorm",
                 ffn: str = "swiglu",
                 qk_norm: bool = True,
                 rope_dim=None,
                 masking_strategy: str = "frequency_guided",
                 importance_ratio: float = 0.6,
                 mask_ratio: float = 0.5,
                 # -- C1, exactly as the ablation defines it -------------------
                 channel_encoding: str = "id",
                 channel_injection: str = "token",
                 channel_embed_dim: int = 64,
                 channel_token_gate_init: float = 0.0,
                 channel_vocab_size: Optional[int] = None,
                 routes: Optional[Dict[str, Route]] = None):
        super().__init__()
        if wavelet_names is None:
            wavelet_names = ["sym4", "sym5", "db6", "sym8", "db8"]
        if channel_encoding not in ("none", "id"):
            raise ValueError(
                f"this pretrainer implements C0 and C1 only; "
                f"channel_encoding={channel_encoding!r} belongs to the ablation's "
                f"geometric rows, which are not part of the settled model.")
        if channel_injection not in ("none", "token"):
            raise ValueError(
                f"C1 injects at the token site only; channel_injection="
                f"{channel_injection!r} would bias the fold's scale logits, "
                f"which is C3/C4 and not what was settled on.")
        if (channel_encoding == "none") != (channel_injection == "none"):
            raise ValueError("channel_encoding and channel_injection must both "
                             "be 'none', or neither.")

        self.routes = dict(ROUTES if routes is None else routes)
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy
        self.importance_ratio = importance_ratio
        self.channel_encoding = channel_encoding
        self.channel_injection = channel_injection
        self.max_level = max_level

        # -- per route: the wavelet expert -------------------------------- #
        self.wavelet_frontends = nn.ModuleDict({
            rid: WaveletFrontend(
                route, max_level=max_level, wave_kernel_size=wave_kernel_size,
                wavelet_names=wavelet_names,
                use_separate_channel=use_separate_channel,
                wave_init_mode=wave_init_mode, fold_synthesis=fold_synthesis,
                fold_gamma=fold_gamma)
            for rid, route in self.routes.items()
        })

        # -- per rate: patcher and decoder -------------------------------- #
        rate_to_patch_t = {r.rate_key: r.patch_t for r in self.routes.values()}
        self.patch_embed_by_rate = nn.ModuleDict({
            rate: PatchEmbed(input_channels=1, patch_size=(1, patch_t),
                             embed_dim=embed_dim)
            for rate, patch_t in rate_to_patch_t.items()
        })
        self.reconstruction_heads = nn.ModuleDict({
            rate: ReconstructionHead(embed_dim=embed_dim, patch_dim=patch_t,
                                     dropout=dropout)
            for rate, patch_t in rate_to_patch_t.items()
        })
        self._rate_patch_t = dict(rate_to_patch_t)

        # -- shared -------------------------------------------------------- #
        self.channel_encoder = None
        if channel_encoding != "none":
            self.channel_encoder = ChannelEncoder(
                channel_encoding, channel_embed_dim,
                vocab_size=channel_vocab_size or len(CHANNEL_VOCAB))
            self.channel_to_token = nn.Linear(channel_embed_dim, embed_dim)
            # Zero gate, non-zero projection: the backbone is untouched at step
            # 0 and the gate still has gradient, so the branch can start
            # learning. Zeroing both would put a zero on each side of the
            # product and neither would ever move.
            self.channel_token_gate = nn.Parameter(
                torch.tensor(float(channel_token_gate_init)))

        self.pos_embed = PositionEmbedding(embed_dim=embed_dim, pos_type="2d")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.shared_transformer = TransformerEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, rope_dim=rope_dim,
            norm=norm, ffn=ffn, qk_norm=qk_norm)

        self.apply(self._init_weights)
        self.reset_channel_parameters()
        nn.init.normal_(self.mask_token, std=0.02)

    # -- initialisation ---------------------------------------------------- #
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_channel_parameters(self):
        """Re-initialise the channel modules after the generic apply().

        ``apply(_init_weights)`` walks every nn.Linear, including this module's
        projection and the encoder's, and would overwrite what ChannelEncoder
        set for itself. The gate is not touched here: it is meant to start at
        its configured value, which is zero.
        """
        if self.channel_encoder is None:
            return
        self.channel_encoder.reset_channel_parameters()
        with torch.no_grad():
            nn.init.normal_(self.channel_to_token.weight, std=0.02)
            nn.init.zeros_(self.channel_to_token.bias)

    # -- introspection ------------------------------------------------------ #
    def route(self, route_id: str) -> Route:
        if route_id not in self.routes:
            raise KeyError(f"unknown route_id {route_id!r}; "
                           f"have {sorted(self.routes)}")
        return self.routes[route_id]

    def channel_gate_value(self) -> Optional[float]:
        """``tanh(gate)`` -- how much of the channel code reaches the tokens."""
        if self.channel_encoder is None:
            return None
        return float(torch.tanh(self.channel_token_gate).detach())

    def parameter_report(self) -> Dict[str, int]:
        """Parameter counts by part, for the startup banner and the report."""
        def count(mod):
            return sum(p.numel() for p in mod.parameters())

        report = {
            "total": sum(p.numel() for p in self.parameters()),
            "shared_transformer": count(self.shared_transformer),
        }
        for rid, front in self.wavelet_frontends.items():
            report[f"wavelet_frontend.{rid}"] = count(front)
        for rate, pe in self.patch_embed_by_rate.items():
            report[f"patch_embed.{rate}"] = count(pe)
        for rate, head in self.reconstruction_heads.items():
            report[f"reconstruction_head.{rate}"] = count(head)
        if self.channel_encoder is not None:
            report["channel_encoder"] = count(self.channel_encoder)
            report["channel_to_token"] = count(self.channel_to_token)
            report["channel_token_gate"] = 1
        report["mask_token"] = self.mask_token.numel()
        return report

    def vocab_fingerprint(self) -> Dict[str, object]:
        return {"channel_vocab_size": (
                    0 if self.channel_encoder is None
                    else self.channel_encoder.vocab_size),
                "channel_vocab_sha256": vocab_sha256()}

    # -- pieces of the forward pass ----------------------------------------- #
    def _channel_code(self, channel_meta):
        if self.channel_encoder is None:
            return None
        if channel_meta is None:
            raise ValueError(
                "channel_encoding='id' needs channel metadata and got None. "
                "The HDF5 carries channel_ids; the loader must pass them.")
        return self.channel_encoder(channel_meta)

    def _inject_channel_tokens(self, tokens, code, n_channels, n_patches):
        """``[B, C*P, D]`` + one vector per channel, over that channel's P only.

        PatchEmbed emits the sequence channel-major: Conv2d over ``[B, 1, C, T]``
        gives ``[B, D, C, T/p]`` and ``flatten(2)`` walks C then P, so token
        ``c*P + p`` belongs to channel ``c``. Reshaping to ``[B, C, P, D]`` is
        therefore the semantic view, not a reinterpretation.
        """
        B, L, D = tokens.shape
        if n_channels * n_patches != L:
            raise RuntimeError(
                f"token count {L} is not C*P = {n_channels}*{n_patches}")
        if code.dim() == 2:
            code = code.unsqueeze(0)
        if code.shape[-2] != n_channels:
            raise ValueError(
                f"channel code has {code.shape[-2]} channels, the route has "
                f"{n_channels}")
        delta = torch.tanh(self.channel_token_gate) * self.channel_to_token(code)
        tokens = tokens.reshape(B, n_channels, n_patches, D) + delta.unsqueeze(-2)
        return tokens.reshape(B, L, D)

    @staticmethod
    def _zero_padded_channels(x, channel_meta):
        """Force padded channel rows to zero before anything reads them.

        The wavelet frontend is not channel-separable end to end -- its
        ChannelAggregationFFN mixes across the channel axis -- so whatever
        happens to sit in a padded row leaks into every real channel's
        representation. Preprocessing already writes zeros there, which makes
        this a no-op on real data; it is here so the guarantee holds for any
        input, and so that "a padded slot carries no measurement" is enforced by
        the model rather than assumed of the loader.
        """
        if channel_meta is None:
            return x
        valid = channel_meta.get("valid_channel_mask")
        if valid is None:
            return x
        v = valid.to(device=x.device, dtype=torch.bool).reshape(-1)
        if v.numel() != x.shape[-2] or bool(v.all()):
            return x
        return x * v.to(x.dtype).view(1, -1, 1)

    @staticmethod
    def _valid_token_mask(channel_meta, n_channels, n_patches, batch_size,
                          device):
        """``[B, C*P]`` or ``None``. A padded slot is padded for its whole window."""
        if channel_meta is None:
            return None
        valid = channel_meta.get("valid_channel_mask")
        if valid is None:
            return None
        valid = valid.to(device=device, dtype=torch.bool).reshape(-1)
        if valid.numel() != n_channels:
            raise ValueError(
                f"valid_channel_mask has {valid.numel()} entries, the route has "
                f"{n_channels} channels")
        return valid.repeat_interleave(n_patches).unsqueeze(0).expand(
            batch_size, -1)

    @staticmethod
    def patchify(spec, patch_t):
        """``[B, C, T] -> [B, C*P, patch_t]``, channel-major to match the tokens."""
        B, C, T = spec.shape
        P = T // patch_t
        return spec.reshape(B, C, P, patch_t).reshape(B, C * P, patch_t)

    @staticmethod
    def unpatchify(patches, n_channels, patch_t):
        """``[B, C*P, patch_t] -> [B, C, T]``. The exact inverse of patchify."""
        B, L, D = patches.shape
        if D != patch_t:
            raise ValueError(f"patch width {D} != {patch_t}")
        P = L // n_channels
        if n_channels * P != L:
            raise ValueError(f"{L} patches is not a whole number per channel")
        return patches.reshape(B, n_channels, P, patch_t).reshape(
            B, n_channels, P * patch_t)

    def _select_mask(self, tokens, mask_ratio, valid_tokens, generator):
        B, L, D = tokens.shape
        if valid_tokens is None:
            budget = int(L * mask_ratio)
        else:
            eligible = int(valid_tokens.sum(dim=1).min().item())
            budget = max(1, int(eligible * mask_ratio)) if eligible else 0
        if budget == 0:
            return torch.zeros(B, L, device=tokens.device, dtype=torch.bool)

        if generator is None:
            noise = torch.rand(B, L, device=tokens.device)
        else:
            noise = torch.rand(B, L, device=generator.device,
                               generator=generator).to(tokens.device)

        if self.masking_strategy == "frequency_guided":
            # Importance is the summed magnitude spectrum along the sequence --
            # the same statistic the legacy model masks by, kept so the
            # objective is the one the ablation was run under.
            # Along the sequence axis directly, so the input stays contiguous.
            #
            # The filter is for one specific warning from torch.fft's output
            # allocation, which some builds emit on every call -- that is once
            # per step per route for an entire run, and it drowns the log. It is
            # cosmetic: the transform's result is unaffected, which is why this
            # narrows to that one message rather than silencing the call.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="An output with one or more elements was "
                                      "resized", category=UserWarning)
                fft = torch.abs(torch.fft.rfft(tokens, dim=1))  # [B, L//2+1, D]
            importance = fft.sum(dim=2)
            importance = torch.nn.functional.interpolate(
                importance.unsqueeze(1), size=L, mode="linear",
                align_corners=True).squeeze(1)
            scores = (self.importance_ratio * importance
                      + (1 - self.importance_ratio) * noise)
        else:
            scores = noise

        if valid_tokens is not None:
            scores = scores.masked_fill(~valid_tokens, float("-inf"))
        idx = torch.topk(scores, budget, dim=1).indices
        mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    # -- forward ------------------------------------------------------------ #
    def encode(self, x, route_id, channel_meta=None):
        """``[B, C, T] -> [B, C*P, D]`` with no masking. What downstream uses."""
        route = self.route(route_id)
        x = self._zero_padded_channels(x, channel_meta)
        spec = self.wavelet_frontends[route_id](x)
        tokens = self.patch_embed_by_rate[route.rate_key](spec.unsqueeze(1))
        code = self._channel_code(channel_meta)
        if code is not None:
            tokens = self._inject_channel_tokens(
                tokens, code, route.n_channels, route.patches_per_channel)
        tokens = self.pos_embed(tokens, freq_size=route.n_channels,
                                time_size=route.patches_per_channel)
        return self.shared_transformer(tokens)

    def forward(self, x, route_id, channel_meta=None, mask_ratio=None,
                mask_generator=None):
        """One route's masked-reconstruction step.

        Returns a dict rather than a tuple: the trainer needs the validity mask
        and the fold's KL as well as the three tensors the loss is built from,
        and a five-tuple whose order has to be remembered is how those get
        silently swapped.
        """
        route = self.route(route_id)
        if x.shape[-2] != route.n_channels or x.shape[-1] != route.window_samples:
            raise ValueError(
                f"{route_id} expects [B, {route.n_channels}, "
                f"{route.window_samples}], got {tuple(x.shape)}")
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        C, P = route.n_channels, route.patches_per_channel
        rate = route.rate_key

        # 0. a padded slot holds no measurement; make that true of the tensor
        x = self._zero_padded_channels(x, channel_meta)

        # 1. route-specific wavelet expert, then the dynamic fold
        spec = self.wavelet_frontends[route_id](x)          # [B, C, T]

        # 2. rate-specific patcher
        tokens = self.patch_embed_by_rate[rate](spec.unsqueeze(1))   # [B, C*P, D]
        target = self.patchify(spec, route.patch_t)                  # [B, C*P, pt]

        # 3. shared C1 code, broadcast over each channel's own time patches
        code = self._channel_code(channel_meta)
        if code is not None:
            tokens = self._inject_channel_tokens(tokens, code, C, P)

        # 4. shared 2-D position encoding
        tokens = self.pos_embed(tokens, freq_size=C, time_size=P)

        # 5. mask, then the shared encoder
        valid_tokens = self._valid_token_mask(
            channel_meta, C, P, tokens.shape[0], tokens.device)
        mask = self._select_mask(tokens, mask_ratio, valid_tokens,
                                 mask_generator)
        masked = tokens.clone()
        masked[mask] = self.mask_token.expand_as(tokens)[mask]
        encoded = self.shared_transformer(masked)

        # 6. rate-specific decoder
        pred = self.reconstruction_heads[rate](encoded)              # [B, C*P, pt]

        return {
            "pred": pred,
            "target": target,
            "mask": mask,
            "valid_tokens": valid_tokens,
            "spec": spec,
            "tokens": tokens,
            "fold_reg": self.wavelet_frontends[route_id].reg_loss,
            "fold_alpha": self.wavelet_frontends[route_id].alpha_mean,
            "route_id": route_id,
        }


def masked_reconstruction_loss(out, fold_kl: float = 1e-3):
    """``masked patch MSE + fold_kl * ScaleFold KL``, plus the reported metrics.

    Only masked tokens enter the reconstruction term, and padded channel slots
    are never masked, so they never enter it either -- there is no second place
    validity has to be applied.
    """
    pred, target, mask = out["pred"], out["target"], out["mask"]
    sel = mask.unsqueeze(-1).expand_as(pred)
    n = int(mask.sum())
    if n == 0:
        zero = pred.sum() * 0.0
        return zero, {"loss_masked_mse": 0.0, "masked_mae": 0.0,
                      "masked_rmse": 0.0, "masked_corr": 0.0,
                      "actual_mask_ratio": 0.0, "loss_fold_kl": 0.0,
                      "loss_total": 0.0}

    p = pred[sel]
    t = target[sel]
    mse = torch.nn.functional.mse_loss(p, t)

    reg = out.get("fold_reg")
    kl = reg if reg is not None else pred.sum() * 0.0
    total = mse + fold_kl * kl

    with torch.no_grad():
        mae = (p - t).abs().mean()
        rmse = mse.sqrt()
        pc, tc = p - p.mean(), t - t.mean()
        denom = pc.norm() * tc.norm()
        corr = (pc @ tc / denom) if float(denom) > 0 else torch.zeros((),
                                                                     device=p.device)
        valid = out.get("valid_tokens")
        denom_tokens = (int(valid.sum()) if valid is not None
                        else mask.numel())
        ratio = n / max(1, denom_tokens)

    return total, {
        "loss_total": float(total.detach()),
        "loss_masked_mse": float(mse.detach()),
        "loss_fold_kl": float(kl.detach() if torch.is_tensor(kl) else kl),
        "masked_mae": float(mae),
        "masked_rmse": float(rmse),
        "masked_corr": float(corr),
        "actual_mask_ratio": float(ratio),
    }
