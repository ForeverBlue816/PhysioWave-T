"""
Channel metadata -> one code per channel, for the legacy BERTWaveletTransformer.
-------------------------------------------------------------------------------
Four modes, each a single change on the one before, so every row of the ablation
differs from its neighbour in one thing:

``none``    no code at all; the model is bit-for-bit the one without this file.
``id``      a learned embedding of the channel's name, looked up by integer id.
            This is the EEGPT-style channel token (their Eq. 11) and is the
            baseline the physical encoding has to beat.
``signed``  the derivation's geometry. A bipolar channel A-B is not a point: it
            is an ordered pair, and the sign of A-B against B-A is the entire
            difference between the two. Encoding only a midpoint throws that
            away, so midpoint and direction are carried separately.
``hybrid``  ``Norm(id + signed)`` -- whether a learned name and a physical
            derivation say different things.

What this deliberately is NOT:

* It is not reference-invariant, and nothing here should be described that way.
  A learned map of a signed derivation is not invariant to re-referencing; it
  merely stops being blind to it.
* It does not encode a montage topology. There are no neighbour graphs, no
  Laplacians and no CSD: on two channels those degenerate, and a topology term
  that cannot be evaluated is a claim rather than a component.
* It is not applied to the waveform. Adding a per-channel vector to raw EEG
  would put a DC offset on the signal that every downstream filter then has to
  model. The code conditions the fold's scale choice and the patch tokens.

The forward path takes numeric tensors only. Names are resolved to integer ids
once, in preprocessing, and travel in the HDF5; nothing here parses a string.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Reserved ids. Never reassign these -- a checkpoint's embedding table is
#: indexed by them, so a shifted vocabulary silently relabels every channel.
PAD_ID = 0
UNK_ID = 1
_RESERVED = ("<pad>", "<unk>")

#: The fixed channel vocabulary. Append only, never reorder or delete: the ids
#: are stored in every HDF5 and every checkpoint. Sleep-EDF needs two of these;
#: the rest are here so a second dataset does not force a vocabulary migration.
CHANNEL_VOCAB = list(_RESERVED) + [
    # Sleep-EDF Sleep Cassette, the two EEG derivations
    "Fpz-Cz", "Pz-Oz",
    # Common bipolar derivations, for whatever comes next
    "Fp1-F7", "F7-T3", "T3-T5", "T5-O1",
    "Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
    "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
    "Fz-Cz", "Cz-Pz",
    # -- monopolar electrodes, common-referenced ------------------------------
    # PhysioNet ERP-BCI (PhysioP300) records 58 of these. A monopolar name and
    # a bipolar one are different strings and get different ids on purpose:
    # "Cz" is an electrode, "Fz-Cz" is a difference of two, and a model that
    # conflated them would be told the same thing about incomparable signals.
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
]
CHANNEL_TO_ID = {name: i for i, name in enumerate(CHANNEL_VOCAB)}

#: Dimension of phi below. Named so the projections cannot drift out of step.
GEOM_DIM = 8


def channel_id(name: str) -> int:
    """Vocabulary id for a channel name, or UNK. Preprocessing calls this."""
    return CHANNEL_TO_ID.get(name, UNK_ID)


def spherical_basis(xyz: torch.Tensor) -> torch.Tensor:
    """``[..., 3] -> [..., 8]``: a fixed low-order basis on the unit sphere.

    ``[x, y, z, xy, xz, yz, x^2 - y^2, 3z^2 - 1]`` -- the three linear terms and
    the five quadratic ones, which up to scale are the real spherical harmonics
    of degree 1 and 2. Fixed rather than learned because with four electrodes
    there is nothing to fit a basis on, and because a fixed basis is the same
    function for a montage this model has never seen.

    The input is normalised to the sphere first, so the code describes a
    direction on the head and not the radius of whichever template supplied it.
    """
    if xyz.shape[-1] != 3:
        raise ValueError(f"expected [..., 3], got {tuple(xyz.shape)}")
    v = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = v.unbind(-1)
    return torch.stack(
        [x, y, z, x * y, x * z, y * z, x * x - y * y, 3.0 * z * z - 1.0], dim=-1
    )


class ChannelEncoder(nn.Module):
    """``channel metadata -> [C, Dc]`` or ``[B, C, Dc]``.

    Args:
        mode: ``none`` | ``id`` | ``signed`` | ``hybrid``.
        embed_dim: width of the code, ``Dc``. Projected to the backbone's width
            by whoever consumes it, not here -- the same code feeds two
            injection sites at different widths.
        norm: ``rmsnorm`` | ``layernorm``, applied to the output. Without it the
            two branches of ``hybrid`` are free to drift to different scales and
            the sum is dominated by whichever grew.
    """

    MODES = ("none", "id", "signed", "hybrid")

    def __init__(self, mode: str = "none", embed_dim: int = 64,
                 vocab_size: int | None = None, norm: str = "rmsnorm"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"channel_encoding must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.embed_dim = int(embed_dim)
        self.vocab_size = int(vocab_size or len(CHANNEL_VOCAB))

        if mode in ("id", "hybrid"):
            self.id_embed = nn.Embedding(self.vocab_size, self.embed_dim,
                                         padding_idx=PAD_ID)
        if mode in ("signed", "hybrid"):
            # Midpoint and direction get their own map. Sharing one would make
            # the code a function of phi(A) + phi(B) and phi(A) - phi(B) through
            # the same weights, i.e. of phi(A) and phi(B) separately, and the
            # split into "where" and "which way round" would carry no meaning.
            self.mid_proj = nn.Linear(GEOM_DIM, self.embed_dim, bias=False)
            self.dir_proj = nn.Linear(GEOM_DIM, self.embed_dim, bias=False)
            # Marks what kind of thing the code describes. A bipolar
            # derivation and a common-referenced electrode are not the same
            # measurement, and a model trained on both must be able to tell
            # which it is looking at.
            #
            # A monopolar channel has a position and no direction. Its two
            # endpoint indices are equal, so `dirn` is exactly zero and the
            # direction branch contributes nothing -- the degeneracy is a fact
            # about the montage, and the encoder states it rather than
            # inventing a reference electrode to subtract.
            self.bipolar_token = nn.Parameter(torch.zeros(self.embed_dim))
            self.monopolar_token = nn.Parameter(torch.zeros(self.embed_dim))

        if mode != "none":
            self.norm = (nn.RMSNorm(self.embed_dim) if norm == "rmsnorm"
                         and hasattr(nn, "RMSNorm") else nn.LayerNorm(self.embed_dim))

    # -- initialisation ---------------------------------------------------- #
    def reset_channel_parameters(self):
        """Initialise this module's own parameters.

        Separate from ``__init__`` because the owning model runs a generic
        ``apply(_init_weights)`` whose nn.Linear branch would otherwise
        overwrite these, the way it already did to ScaleFold's MLP.

        Note what is *not* zeroed: the projections and the embedding table are
        given real values. Only the gates that scale this module's contribution
        into the backbone are zero, and they live in the model. Zeroing both
        would leave the branch with zero gradient on both sides of the product
        and it would never start learning.
        """
        with torch.no_grad():
            if hasattr(self, "id_embed"):
                nn.init.normal_(self.id_embed.weight, std=0.02)
                self.id_embed.weight[PAD_ID].zero_()
            if hasattr(self, "mid_proj"):
                nn.init.normal_(self.mid_proj.weight, std=0.02)
                nn.init.normal_(self.dir_proj.weight, std=0.02)
                # Both markers start at zero and neither draws from the global
                # RNG, so adding the monopolar one left every bipolar model
                # numerically identical -- only its parameter count moved.
                self.bipolar_token.zero_()
                self.monopolar_token.zero_()

    # -- forward ----------------------------------------------------------- #
    def forward(self, meta: dict) -> torch.Tensor | None:
        """``meta`` -> ``[C, Dc]``, or ``None`` in ``none`` mode.

        Required keys, all numeric tensors:
            ``channel_ids``               ``[C]``    int64
            ``electrode_xyz``             ``[E, 3]`` float
            ``positive_electrode_index``  ``[C]``    int64
            ``negative_electrode_index``  ``[C]``    int64
        """
        if self.mode == "none":
            return None
        if meta is None:
            raise ValueError(
                f"channel_encoding={self.mode!r} needs channel metadata and got None. "
                "Rebuild the HDF5 with EEG/sleep_edf_finetune.py --stage split, or "
                "run with --channel_encoding none."
            )

        code = None
        if self.mode in ("id", "hybrid"):
            code = self.id_embed(meta["channel_ids"])                 # [C, Dc]

        if self.mode in ("signed", "hybrid"):
            phi = spherical_basis(meta["electrode_xyz"])              # [E, 8]
            pa = phi[meta["positive_electrode_index"]]                # [C, 8]
            pb = phi[meta["negative_electrode_index"]]                # [C, 8]
            # Symmetric in the endpoints; antisymmetric in the endpoints. Swap
            # A and B and mid is untouched while dirn is exactly negated, which
            # is the property the whole mode exists for.
            mid = 0.5 * (pa + pb)
            dirn = pa - pb
            # Equal endpoints mean a monopolar channel: mid is phi(A) and dirn
            # is zero. Marked as such, so "electrode Cz" and "derivation X-Cz"
            # do not arrive at the backbone wearing the same label.
            same = (meta["positive_electrode_index"]
                    == meta["negative_electrode_index"]).unsqueeze(-1)
            marker = torch.where(same, self.monopolar_token, self.bipolar_token)
            signed = self.mid_proj(mid) + self.dir_proj(dirn) + marker
            code = signed if code is None else code + signed

        code = self.norm(code)
        mask = meta.get("valid_channel_mask")
        if mask is not None:
            code = code * mask.to(code.dtype).unsqueeze(-1)
        return code

    def extra_repr(self):
        return f"mode={self.mode}, embed_dim={self.embed_dim}, vocab={self.vocab_size}"


def required_meta_keys(mode: str) -> tuple[str, ...]:
    """Which metadata a mode actually reads. Used to check an HDF5 up front."""
    if mode == "none":
        return ()
    if mode == "id":
        return ("channel_ids",)
    return ("channel_ids", "electrode_xyz",
            "positive_electrode_index", "negative_electrode_index")
