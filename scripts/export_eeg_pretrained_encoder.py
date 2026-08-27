#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export one route's encoder from a pretraining checkpoint, for fine-tuning.

    python scripts/export_eeg_pretrained_encoder.py \
        --checkpoint best.pth --route E32_512 --output exported_E32_512.pth

A pretraining checkpoint holds four wavelet frontends, two patchers and two
reconstruction decoders. A downstream task uses one route and no decoder, so
carrying the rest means shipping several times the weights and inviting someone
to load a frontend built for a different electrode count.

What comes out:

    wavelet_frontend.*    the named route's expert only
    patch_embed.*         the patcher for that route's sampling rate
    channel_encoder.*     the C1 channel-name embedding, whole
    channel_to_token.*    its projection, and the gate
    shared_transformer.*  the encoder
    + the channel vocabulary, its hash, and the preprocessing spec

What does not:

    raw_reconstruction_heads.*  the second pretraining decoder, which predicts
                             the preprocessed EEG. Pretraining-only, like the
                             one below.
    reconstruction_heads.*   the pretraining decoder. It predicts folded
                             wavelet patches, which no downstream head wants,
                             and keeping it invites fine-tuning against the
                             pretext objective by accident.

The channel vocabulary travels with the weights because an embedding row means
whatever channel held that id when the row was learned. A fine-tuning run that
resolves names against a different vocabulary silently trains on relabelled
electrodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_embedding import vocab_payload                   # noqa: E402
from physiowave.eeg_c1.routes import ROUTES                    # noqa: E402

KEEP_PREFIXES = ("channel_encoder.", "channel_to_token.", "shared_transformer.",
                 "pos_embed.", "mask_token")
#: Documentation of what the allowlist above already excludes. Both decoders
#: are pretraining-only: the spec head predicts folded wavelet patches and the
#: raw head predicts preprocessed EEG, and fine-tuning needs neither.
DROP_PREFIXES = ("reconstruction_heads.", "raw_reconstruction_heads.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--route", required=True, choices=sorted(ROUTES))
    p.add_argument("--output", required=True)
    p.add_argument("--keep-mask-token", action="store_true",
                   help="keep the pretraining mask token (downstream ignores it)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model")
    if sd is None:
        print("ERROR: checkpoint has no 'model' state dict", file=sys.stderr)
        return 1

    route = ROUTES[args.route]
    front = f"wavelet_frontends.{args.route}."
    patch = f"patch_embed_by_rate.{route.rate_key}."

    out = {}
    for k, v in sd.items():
        if any(k.startswith(d) for d in DROP_PREFIXES):
            continue
        if k.startswith("wavelet_frontends."):
            if k.startswith(front):
                out["wavelet_frontend." + k[len(front):]] = v
            continue
        if k.startswith("patch_embed_by_rate."):
            if k.startswith(patch):
                out["patch_embed." + k[len(patch):]] = v
            continue
        if k.startswith("mask_token") and not args.keep_mask_token:
            continue
        if any(k.startswith(pref) for pref in KEEP_PREFIXES):
            out[k] = v

    if not any(k.startswith("wavelet_frontend.") for k in out):
        print(f"ERROR: the checkpoint has no frontend for route {args.route}",
              file=sys.stderr)
        return 1

    cfg = ck.get("config", {})
    payload = {
        "model": out,
        "route_id": args.route,
        "route": {"n_channels": route.n_channels,
                  "sampling_rate": route.sampling_rate,
                  "window_samples": route.window_samples,
                  "patch_size": list(route.patch_size),
                  "n_tokens": route.n_tokens,
                  "slots": list(route.slots)},
        "model_config": cfg.get("model", {}),
        "preprocessing_spec": {
            "window_seconds": route.window_seconds,
            "patch_seconds": route.patch_seconds,
            "target_sampling_rate": route.sampling_rate,
            "pipeline": ("units_uV -> detrend -> notch -> 0.5 Hz high-pass -> "
                         "resample_poly -> slot_map -> window -> zscore_clip"),
            "note": ("no 0.5-45 Hz band-pass; the encoder has seen the full "
                     "band up to Nyquist and expects input prepared the same "
                     "way"),
        },
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "epoch": ck.get("epoch"),
        "global_step": ck.get("global_step"),
        "best_val_loss_masked_mse": ck.get("best_val_loss_masked_mse"),
        **vocab_payload(),
    }
    recorded = ck.get("channel_vocab_sha256")
    if recorded and recorded != payload["channel_vocab_sha256"]:
        print(f"ERROR: the checkpoint was trained under channel vocabulary "
              f"{recorded[:16]} and this working tree has "
              f"{payload['channel_vocab_sha256'][:16]}. Exporting would attach "
              f"the wrong vocabulary to these weights.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    torch.save(payload, args.output)

    n_par = sum(v.numel() for v in out.values() if hasattr(v, "numel"))
    print(f"exported {args.route} from {args.checkpoint}")
    print(f"  {len(out)} tensors, {n_par:,} parameters -> {args.output}")
    print(f"  input shape  [B, {route.n_channels}, {route.window_samples}] "
          f"@ {route.sampling_rate} Hz")
    print(f"  tokens       {route.n_tokens}")
    print(f"  vocab sha    {payload['channel_vocab_sha256'][:16]}")
    print(f"  decoder      excluded (pretraining head)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
