"""One C1 encoder per (task, modality), on the three pretrained checkpoints.

Each modality's encoder is the downstream class of that modality
(physiowave.<m>_c1.downstream), built for the task's montage: on a
pretraining route when the montage fits one (MPDB's 59 electrodes in
E64_256's slots, a chest ECG patch in L1_250's), otherwise with a fresh
frontend of its own. What pretraining supplied -- channel encoder, gate and
Transformer, plus the frontend on a route -- loads from the exported
encoder; the fresh parts train.

Two tasks share one encoder per modality: the second task's model is built
for ITS montage and then has its pretrained core replaced by the first
task's modules, so there is one Transformer per modality however many tasks
train on it, and only the montage adapters differ.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .prep_common import PATCH, ROOT

#: The pretrained core, shared across tasks; the names are the downstream
#: class's TRANSFERABLE prefixes as attributes.
SHARED = ("channel_encoder", "channel_to_token", "channel_token_gate", "shared_transformer")


def modality_class(modality: str):
    if modality == "eeg":
        from physiowave.eeg_c1.downstream import EEGC1Downstream as cls
    elif modality == "ecg":
        from physiowave.ecg_c1.downstream import ECGC1Downstream as cls
    elif modality == "emg":
        from physiowave.emg_c1.downstream import EMGC1Downstream as cls
    else:
        raise ValueError(modality)
    return cls


def choose_route(modality: str, names: Sequence[str], fs: float) -> Optional[str]:
    """The smallest pretraining route this montage sits inside, or None."""
    cls = modality_class(modality)
    fits = []
    for rid, route in cls.ROUTES_TABLE.items():
        if int(route.sampling_rate) != int(fs):
            continue
        slots = set(route.slots)
        if all(cls.normalize_name(n) in slots for n in names):
            fits.append((route.n_channels, rid))
    return min(fits)[1] if fits else None


def exported_encoder(checkpoint: str, modality: str, route_id: Optional[str],
                     cache_dir: str) -> str:
    """The path of an exported encoder for ``checkpoint`` on ``route_id``.

    A pretraining checkpoint (it carries an optimizer) is exported through
    scripts/export_eeg_pretrained_encoder.py into ``cache_dir``; an already
    exported file is returned as is when its route matches, and re-exported
    from its recorded source otherwise.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "optimizer" not in ck and "model_config" in ck:
        if (ck.get("route_id") or None) == (route_id or None):
            return checkpoint
        src = ck.get("source_checkpoint")
        if not src or not os.path.isfile(src):
            raise SystemExit(f"{checkpoint} was exported for route {ck.get('route_id')} and "
                             f"this montage needs {route_id}; its source checkpoint "
                             f"{src} is not here to re-export from")
        checkpoint = src
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{modality}_{route_id or 'routeless'}.pth")
    if os.path.isfile(out):
        return out
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "export_eeg_pretrained_encoder.py"),
           "--checkpoint", checkpoint, "--modality", modality, "--output", out]
    if route_id:
        cmd += ["--route", route_id]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"exporting {checkpoint} failed:\n{r.stdout}\n{r.stderr}")
    return out


def channel_meta(names: Sequence[str]) -> SimpleNamespace:
    """What the downstream forward reads: names, and no per-item mask."""
    return SimpleNamespace(channel_names=list(names), channel_mask=None)


def build_encoder(modality: str, names: Sequence[str], fs: float, window_samples: int,
                  checkpoint: Optional[str], cache_dir: str, freeze: bool,
                  dropout: float = 0.0) -> Tuple[nn.Module, dict]:
    """A downstream C1 model for this montage, its pretrained parts loaded.

    ``checkpoint`` None builds the same architecture from scratch (the
    control). The report says what was taken, what the montage could not use
    and which route, if any, carried the frontend over.
    """
    cls = modality_class(modality)
    route_id = choose_route(modality, names, fs)
    exported = (exported_encoder(checkpoint, modality, route_id, cache_dir)
                if checkpoint else None)
    mc = {}
    if exported:
        mc = dict(torch.load(exported, map_location="cpu", weights_only=False)
                  .get("model_config") or {})
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in mc.items()
              if k in accepted and k not in ("dropout", "freeze_encoder", "freeze_scope",
                                             "route_id", "num_classes", "channel_names",
                                             "in_channels", "window_samples",
                                             "sampling_rate", "patch_samples", "pool")}
    model = cls(in_channels=len(names), window_samples=int(window_samples),
                sampling_rate=float(fs), patch_samples=PATCH[modality], num_classes=1,
                channel_names=list(names), route_id=route_id, dropout=float(dropout),
                pool="mean", freeze_encoder=bool(freeze), freeze_scope="pretrained",
                **kwargs)
    report = {"modality": modality, "route": route_id, "exported": exported,
              "embed_dim": int(model.embed_dim), "taken": [], "skipped": []}
    if exported:
        rep = model.load_pretrained(exported, route_id=route_id)
        report.update({k: list(v) for k, v in rep.items()})
        report["summary"] = model.describe_transfer(rep)
    else:
        report["summary"] = "from scratch"
    return model, report


def share_core(primary: nn.Module, other: nn.Module) -> None:
    """Make ``other`` use ``primary``'s pretrained core; its adapters stay its own."""
    for name in SHARED:
        if hasattr(primary, name):
            setattr(other, name, getattr(primary, name))


def encoder_lr_scale(model_names_prefix: str, depth: int, encoder_lr_scale: float,
                     layer_decay: float):
    """Layer-wise decay for encoder parameters named ``<prefix>.<task>.<m>.<key>``."""
    from physiowave.train.finetune_main import c1_lr_scale
    inner = c1_lr_scale(depth, encoder_lr_scale, layer_decay)

    def scale(name: str) -> float:
        if not name.startswith(model_names_prefix + "."):
            return 1.0
        key = name.split(".", 3)[-1]           # encoders.<task>.<m>.<key>
        return inner(key)
    return scale


__all__ = ["SHARED", "build_encoder", "choose_route", "channel_meta", "exported_encoder",
           "share_core", "encoder_lr_scale", "modality_class"]
