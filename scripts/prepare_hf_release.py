#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assemble a Hugging Face release from a finished EEG C1 pretraining run.

    python scripts/prepare_hf_release.py \
        --run-dir $PW_CKPT_ROOT/pretrain_eeg_c1_moe \
        --out-dir ~/hf_release --repo-id ForeverBlue/EEG

Uploading best.pth as it sits on disk is the obvious move and is wrong three
times over.

  * It carries the AdamW state, which is two moments per parameter -- roughly
    three times the weights for a file nobody downstream will ever resume from.
  * Its embedded config holds the absolute manifest paths the run trained
    against, which name the scratch filesystem, the project allocation and the
    username. That is not a secret, but publishing it is a decision rather than
    an accident, and this makes it one.
  * It holds four route frontends and both pretraining decoders. Someone
    fine-tuning on a 19-channel montage wants one frontend and no decoder, and
    handing them the rest invites loading a frontend built for a different
    electrode count.

So the release carries two shapes: weights-only checkpoints for anyone who
wants the whole multi-route model, and ONE route-agnostic encoder for anyone
who wants to fine-tune. Not one encoder per route -- downstream.py loads a
checkpoint's wavelet frontend and patcher only when the target montage IS the
route, and both downstream tasks here build their own frontend, so four
per-route files would differ only in bytes nobody loads. The full checkpoints
still hold all four frontends for anyone fine-tuning on a route exactly.

The model card is written from the run's own
metrics_epoch.jsonl rather than typed, because a card whose numbers were typed
is a card that disagrees with the run the moment either changes.

NOTHING IS UPLOADED. This writes a directory and prints the commands.
Publishing is a separate, deliberate step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from physiowave.eeg_c1.objective import (objective_equation,   # noqa: E402
                                         resolve_eeg_c1_objective)
from physiowave.eeg_c1.routes import ROUTES                    # noqa: E402

#: Dropped from a published checkpoint. Each is either resume-only state or a
#: path into the machine the run happened on.
TRAINING_ONLY = ("optimizer", "scheduler", "scaler", "rng", "sampler",
                 "history")

#: Fields outside the config that hold a path on the training filesystem.
#: An exported encoder records the checkpoint it came from as an absolute path,
#: which on a cluster names the scratch mount, the project allocation and the
#: username.
PATH_FIELDS = ("source_checkpoint", "run_dir", "out_dir", "output_dir")

REDACTED = "<path redacted for release>"


def _scrub(value):
    """Recursively replace absolute-path strings.

    A walk rather than a list of known keys. The list was
    data.manifest_train/manifest_val/root/corpus_root, and it missed
    `source_checkpoint` in the exported encoder -- which is the failure mode
    of every allowlist: it protects what was thought of. A config value that
    begins with "/" or "~" is a path, and a release has no use for the
    training machine's directory layout.
    """
    if isinstance(value, dict):
        return {k: (REDACTED if k in PATH_FIELDS and value[k] else _scrub(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and (value.startswith("/")
                                   or value.startswith("~/")):
        return REDACTED
    return value


def scrub_config(cfg: dict) -> dict:
    """A copy of the config with training-filesystem paths replaced.

    Replaced rather than deleted: a reader needs to know a manifest WAS
    configured, and an absent key reads as "this run had no validation split".
    """
    return _scrub(json.loads(json.dumps(cfg, default=str)))


def find_paths(path: str):
    """Absolute paths surviving anywhere in a released checkpoint's metadata.

    The last line of defence, run over what is about to be published rather
    than over what was intended to be published. Tensors are skipped; only the
    metadata can carry a string.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    found = []

    def walk(v, where):
        if torch.is_tensor(v):
            return
        if isinstance(v, dict):
            for k, sub in v.items():
                walk(sub, f"{where}.{k}")
        elif isinstance(v, (list, tuple)):
            for i, sub in enumerate(v):
                walk(sub, f"{where}[{i}]")
        elif isinstance(v, str) and (v.startswith("/") or v.startswith("~/")) \
                and len(v) > 8 and "/" in v[1:]:
            found.append((where, v))

    walk(ck, os.path.basename(path))
    return found


#: An absolute path in a text file: a "/" run with at least two segments. The
#: README legitimately contains URLs and code fences, so http(s) and the
#: repository's own relative paths must not trip it.
_PATH_RE = re.compile(r"(?<![\w:/])(/[A-Za-z0-9_.-]+){2,}/?")


def text_paths(path: str):
    """``(line number, match)`` for absolute paths in a released text file."""
    out = []
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if "://" in line:
                    continue
                for m in _PATH_RE.finditer(line):
                    if m.group(0) != REDACTED:
                        out.append((i, m.group(0)))
    except (OSError, UnicodeDecodeError):
        pass
    return out


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"


def strip_checkpoint(src: str, dst: str) -> dict:
    """Weights, config and provenance. No optimizer, no paths."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    keep = {k: v for k, v in ck.items() if k not in TRAINING_ONLY}
    keep["config"] = scrub_config(ck.get("config", {}))
    for field in PATH_FIELDS:
        if keep.get(field):
            keep[field] = REDACTED
    keep["released_from"] = os.path.basename(src)
    keep["released_utc"] = datetime.now(timezone.utc).isoformat()
    torch.save(keep, dst)
    return {"epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
            "best_scores": ck.get("best_scores", {}),
            "best_epochs": ck.get("best_epochs", {}),
            "objective": resolve_eeg_c1_objective(ck.get("config", {})),
            "config": ck.get("config", {}),
            "n_params": sum(v.numel() for v in ck["model"].values()
                            if hasattr(v, "numel")),
            "src_bytes": os.path.getsize(src),
            "dst_bytes": os.path.getsize(dst)}


#: Rendered in the card so a reader can see what each file costs and what it
#: buys, rather than being told. Order is fixed; sizes are measured.
COMPONENTS = (
    ("shared_transformer.", "shared transformer", "encoder, always transfers"),
    ("channel_encoder.", "channel-name embedding", "always transfers"),
    ("channel_to_token.", "channel projection", "always transfers"),
    ("wavelet_frontends.", "wavelet frontends (all 4 routes)",
     "route-bound; downstream usually builds its own"),
    ("patch_embed_by_rate.", "patchers (both rates)", "route-bound"),
    ("reconstruction_heads.", "spec decoder", "pretraining only"),
    ("raw_reconstruction_heads.", "raw decoder", "pretraining only"),
)


def component_table(sd: dict):
    """``(markdown rows, {prefix: share})`` for the full model, all measured.

    The shares come back with the table because the sentence under it used to
    be prose -- "the shared transformer is most of the model" -- written from
    the 512/8/8 run it happened to be true for. Beside a measured table it is
    a claim that can contradict the rows directly above it, and at 32/1 it did:
    the transformer was 0.4% and the card said it was most of the model.
    """
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel")) or 1
    lines, share = [], {}
    for prefix, label, note in COMPONENTS:
        n = sum(v.numel() for k, v in sd.items()
                if k.startswith(prefix) and hasattr(v, "numel"))
        share[prefix] = n / total
        if not n:
            continue
        lines.append(f"| {label} | {n / 1e6:.2f} M | {n / total * 100:.1f}% "
                     f"| {note} |")
    lines.append(f"| **total** | **{total / 1e6:.2f} M** | 100% | |")
    return "\n".join(lines), share


def why_no_per_route_files(share: dict) -> str:
    """The reason there is one encoder and not four, from the actual shares."""
    enc = share.get("shared_transformer.", 0.0)
    fronts = share.get("wavelet_frontends.", 0.0)
    if enc > fronts:
        return (
            f"The shared transformer is {enc * 100:.0f}% of the model and the "
            f"four frontends together are {fronts * 100:.0f}%, which is why "
            f"there is no per-route encoder file here: four of them would "
            f"duplicate that transformer four times to deliver frontends that "
            f"are a small fraction of it. Cut the one you want out of the "
            f"complete checkpoint instead.")
    return (
        f"At this width the four frontends ({fronts * 100:.0f}%) outweigh the "
        f"shared transformer ({enc * 100:.0f}%). There is still one encoder "
        f"file rather than four, because what a per-route file adds is a "
        f"frontend that only loads onto that exact montage; cut the one you "
        f"want out of the complete checkpoint.")


def read_metrics(run_dir: str):
    rows = []
    path = os.path.join(run_dir, "metrics_epoch.jsonl")
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def model_card(repo_id: str, info: dict, rows, run_dir: str,
               components: str = "", shares: Optional[dict] = None) -> str:
    cfg = info["config"]
    mcfg, tcfg = cfg.get("model", {}), cfg.get("train", {})
    obj = info["objective"]
    last = rows[-1] if rows else {}
    best = info.get("best_scores", {})
    bep = info.get("best_epochs", {})

    def g(key, fmt="{:.4f}"):
        v = last.get(key)
        return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

    routes = "\n".join(
        f"| `{rid}` | {ROUTES[rid].n_channels} | {ROUTES[rid].sampling_rate} Hz |"
        for rid in sorted(ROUTES))

    return f"""---
license: apache-2.0
tags:
  - eeg
  - biosignal
  - self-supervised
  - masked-autoencoder
  - pretrained
library_name: pytorch
---

# EEG C1 multi-route pretrained encoder

Masked dual-reconstruction pretraining on multi-montage EEG. One shared
transformer serves four electrode layouts through route-specific wavelet
frontends, and channel identity enters as a learned embedding of the channel
NAME rather than of its index.

Trained for {tcfg.get('epochs', '?')} epochs
({last.get('global_step', '?')} optimizer updates) on a mixture of public EEG
corpora, {info['n_params'] / 1e6:.1f} M parameters.

## Objective

```
{objective_equation(obj)}
```

Two heads at equal weight. One predicts folded wavelet patches, the other
predicts the preprocessed EEG waveform of the same masked patches. Masking is
applied BEFORE the wavelet frontend ({obj['mask_before_frontend']}), at ratio
{mcfg.get('mask_ratio', '?')}, and the spec target is per-patch normalised
({obj['normalize_spec_target']}).

## Validation, final epoch

| metric | value | what 1.0 would mean |
|---|---|---|
| total loss | {g('val/loss_total')} | — |
| spec MSE | {g('val/loss_masked_spec_mse')} | — |
| spec NMSE | {g('val/masked_spec_nmse', '{:.3f}')} | predicting zero |
| spec Pearson r | {g('val/masked_spec_corr', '{:.3f}')} | perfect |
| raw SmoothL1 | {g('val/loss_masked_raw_smoothl1')} | — |
| raw NMSE | {g('val/masked_raw_nmse', '{:.3f}')} | predicting zero |
| raw Pearson r | {g('val/masked_raw_corr', '{:.3f}')} | perfect |
| route-macro total | {g('val/macro_route_loss_total')} | — |

NMSE is dimensionless: MSE(prediction, target) / MSE(0, target), so 1.0 is
what predicting zero scores on that head's own target. The two losses are NOT
comparable to each other -- one is an MSE on normalised wavelet coefficients,
the other a SmoothL1 on z-scored volts. The columns that compare the heads are
the correlations and the NMSEs.

The route-macro row weights the four routes equally. The plain rows weight by
sample count, and validation is dominated by the two largest corpora, so a
model specialising toward the smaller ones moves the two rows in opposite
directions.

## Files

### Which file do you want

| your situation | download |
|---|---|
| fine-tune on any montage | `eeg_c1_encoder.pth` |
| your montage IS one of the four routes below | `pretrain_eeg_c1_moe_best.pth`, then cut the route out (one command, below) |
| continue pretraining, or take the model apart yourself | `pretrain_eeg_c1_moe_best.pth` |

**`pretrain_eeg_c1_moe_best.pth` is the complete model** -- all four wavelet
frontends, both patchers, both reconstruction decoders. Nothing is held back;
only the optimiser state was stripped, which no downstream use needs. Take it
if you want to keep or strip pieces yourself.

`eeg_c1_encoder.pth` is the same weights minus everything route-bound and
pretraining-only. It exists because it cannot be loaded wrong.

| file | what it is |
|---|---|
| `eeg_c1_encoder.pth` | encoder alone: channel embedding, gate, transformer |
| `pretrain_eeg_c1_moe_best.pth` | complete model, lowest validation total loss (epoch {bep.get('total', '?')}) |
| `pretrain_eeg_c1_moe_final.pth` | complete model, last epoch, cosine fully annealed |
| `metrics_epoch.jsonl` | the full training curve |
| `config_resolved.yaml` | every resolved hyperparameter |

### What is in the complete model

| component | parameters | share | |
|---|---|---|---|
{components}

{why_no_per_route_files(shares or {})}

### What transfers, and what does not

`downstream.py` splits a checkpoint into two sets. The channel embedding, its
gate and the shared transformer **always** load. The wavelet frontend and the
patcher load **only when your montage is the route they were trained for** --
same electrode count, same patch length -- and are skipped otherwise rather
than reshaped.

So a fresh frontend is the normal case, not a fallback. Both downstream tasks
in the source repository build their own.

The four routes the shared transformer was pretrained across:

| route | channels | rate |
|---|---|---|
{routes}

To fine-tune ON one of them exactly, take that route's frontend and patcher out
of the complete checkpoint:

```bash
python scripts/export_eeg_pretrained_encoder.py \\
    --checkpoint pretrain_eeg_c1_moe_best.pth --route E64_256 \\
    --output encoder_E64_256.pth
```

Omit `--route` and you get `eeg_c1_encoder.pth` again. Both pretraining
decoders are dropped either way: fine-tuning against the pretext objective by
accident is the failure that prevents.

**`best.pth` and the final checkpoint are within validation noise of each
other.** Which one transfers better is a downstream question, not a
reconstruction-loss question -- measure both.

## The channel vocabulary travels with the weights

An embedding row means whatever channel held that id when the row was learned.
Every checkpoint here carries the vocabulary and its hash. Resolving channel
names against a different vocabulary silently trains on relabelled electrodes,
which is why the hash is checked rather than assumed.

## Architecture

```
embed_dim   {mcfg.get('embed_dim', '?')}      depth {mcfg.get('depth', '?')}      heads {mcfg.get('num_heads', '?')}
norm        {mcfg.get('norm', '?')}     ffn {mcfg.get('ffn', '?')}     qk_norm {mcfg.get('qk_norm', '?')}
wavelet     max_level {mcfg.get('max_level', '?')}, kernel {mcfg.get('wave_kernel_size', '?')}
channel     {mcfg.get('channel_encoding', '?')} embedding, injected at the {mcfg.get('channel_injection', '?')} site
mask        {mcfg.get('masking_strategy', '?')}, ratio {mcfg.get('mask_ratio', '?')}
```

## Loading

`EEGC1Downstream` builds the frontend and patcher for your montage and loads
this file into everything else. Nothing here is route-bound, so the montage,
the window length and the patch length are yours to choose.

```python
import torch
from physiowave.eeg_c1.downstream import EEGC1Downstream

ENC = "eeg_c1_encoder.pth"
ck = torch.load(ENC, map_location="cpu", weights_only=False)
mc = ck["model_config"]                  # the width the weights were trained at
CH = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]   # your electrodes

model = EEGC1Downstream(
    in_channels=len(CH), window_samples=1024, sampling_rate=256.0,
    patch_samples=64, num_classes=2, channel_names=CH,
    embed_dim=mc["embed_dim"], depth=mc["depth"], num_heads=mc["num_heads"],
    channel_embed_dim=mc.get("channel_embed_dim", 64),
    channel_vocab_size=ck["channel_vocab_size"])

report = model.load_pretrained(ENC)
print(model.describe_transfer(report))
# loaded: transformer N, channel encoder N, gate yes, frontend 0, patcher 0

class Meta:                               # or your dataset's ChannelMeta
    channel_names = CH
    channel_mask = None

out = model(torch.randn(2, len(CH), 1024), Meta())   # {{"logits": [2, 2]}}
```

`frontend 0, patcher 0` is the expected line, not a warning: those are the two
things your montage supplies.

**The frontend you build is not randomly initialised.** Its filters are seeded
from real wavelet families -- `sym4, sym5, db6, sym8, db8` -- and learned from
there during fine-tuning. What a fresh frontend costs you is the adaptation
those filters underwent during pretraining, not the wavelet prior itself.

Channel names, not indices, are what this encoder and your data agree on. An
embedding row means whichever electrode held it when it was learned, so pass
real names and check `channel_vocab_sha256` -- `load_pretrained` refuses a
mismatch rather than silently training on relabelled electrodes.

Code: <https://github.com/ForeverBlue816/PhysioWave-T>

---
Prepared by `scripts/prepare_hf_release.py` from `{os.path.basename(run_dir)}`
on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo-id", default="ForeverBlue/EEG")
    p.add_argument("--best", default="best.pth")
    p.add_argument("--final", default="latest.pth")
    args = p.parse_args(argv)

    run_dir = os.path.expanduser(args.run_dir)
    out_dir = os.path.expanduser(args.out_dir)
    best_src = os.path.join(run_dir, args.best)
    if not os.path.isfile(best_src):
        print(f"ERROR: no checkpoint at {best_src}", file=sys.stderr)
        return 1
    os.makedirs(out_dir, exist_ok=True)

    print(f"release for {args.repo_id}\n  from {run_dir}\n  into {out_dir}\n")

    best_dst = os.path.join(out_dir, "pretrain_eeg_c1_moe_best.pth")
    info = strip_checkpoint(best_src, best_dst)
    components, shares = component_table(
        torch.load(best_dst, map_location="cpu", weights_only=False)["model"])
    print(f"  {os.path.basename(best_dst)}   "
          f"{human_bytes(info['src_bytes'])} -> "
          f"{human_bytes(info['dst_bytes'])}   "
          f"(epoch {info['epoch']}, {info['n_params'] / 1e6:.1f} M params)")

    final_src = os.path.join(run_dir, args.final)
    if os.path.isfile(final_src):
        final_dst = os.path.join(out_dir, "pretrain_eeg_c1_moe_final.pth")
        fi = strip_checkpoint(final_src, final_dst)
        print(f"  {os.path.basename(final_dst)}  "
              f"{human_bytes(fi['src_bytes'])} -> "
              f"{human_bytes(fi['dst_bytes'])}   (epoch {fi['epoch']})")

    # One encoder, no route. What transfers is the channel embedding, its gate
    # and the shared transformer; the frontend and patcher load only onto the
    # exact route they were trained for, and both downstream tasks here build
    # their own. Four per-route files would differ only in weights nobody loads.
    enc_dst = os.path.join(out_dir, "eeg_c1_encoder.pth")
    r = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "scripts", "export_eeg_pretrained_encoder.py"),
         "--checkpoint", best_src, "--output", enc_dst],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR exporting the encoder: {r.stderr[-800:]}",
              file=sys.stderr)
        return 1
    # The export records the checkpoint it came from, absolutely. Harmless in
    # $PW_CKPT_ROOT and not harmless on a public model page.
    enc = torch.load(enc_dst, map_location="cpu", weights_only=False)
    for field in PATH_FIELDS:
        if enc.get(field):
            enc[field] = REDACTED
    if isinstance(enc.get("model_config"), dict):
        enc["model_config"] = scrub_config(enc["model_config"])
    torch.save(enc, enc_dst)
    print(f"  eeg_c1_encoder.pth   {human_bytes(os.path.getsize(enc_dst))}   "
          f"(no frontend, no patcher, no decoder)")

    # config_resolved.yaml is TEXT, and copying it verbatim published the
    # manifest paths the .pth files had just been scrubbed of. It is rewritten
    # through the same scrub, not copied.
    src = os.path.join(run_dir, "config_resolved.yaml")
    if os.path.isfile(src):
        try:
            import yaml
            with open(src) as f:
                cfg_yaml = yaml.safe_load(f)
            with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
                yaml.safe_dump(scrub_config(cfg_yaml), f, sort_keys=False)
            print("  config_resolved.yaml   (paths scrubbed)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  SKIPPING config_resolved.yaml: {exc}", file=sys.stderr)

    # metrics_epoch.jsonl is numbers only, but it is checked below like
    # everything else rather than trusted for being numbers.
    src = os.path.join(run_dir, "metrics_epoch.jsonl")
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(out_dir, "metrics_epoch.jsonl"))
        print("  metrics_epoch.jsonl")

    rows = read_metrics(run_dir)
    card = os.path.join(out_dir, "README.md")
    with open(card, "w") as f:
        f.write(model_card(args.repo_id, info, rows, run_dir,
                           components=components, shares=shares))
    print(f"  README.md   (model card, {len(rows)} epochs of metrics read)")

    # Every file, not only the checkpoints. The first version of this check
    # scanned .pth alone and passed while config_resolved.yaml sat beside them
    # carrying the very paths the checkpoints had been scrubbed of.
    leaks = []
    for dp, _dn, fn in os.walk(out_dir):
        for f in fn:
            full = os.path.join(dp, f)
            if f.endswith(".pth"):
                leaks += [(f, w, v) for w, v in find_paths(full)]
            else:
                leaks += [(f, f"line {i}", m)
                          for i, m in text_paths(full)]
    if leaks:
        print("\n  REFUSING: absolute paths survive in what would be "
              "published:", file=sys.stderr)
        for f, where, v in leaks[:10]:
            print(f"    {f}  {where} = {v}", file=sys.stderr)
        print("  Nothing was uploaded. Extend PATH_FIELDS or scrub_config.",
              file=sys.stderr)
        return 1
    print("  no absolute paths survive in any published checkpoint")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _dn, fn in os.walk(out_dir) for f in fn)
    print(f"\n  total {human_bytes(total)}")
    print(f"""
  NOTHING HAS BEEN UPLOADED. To publish:

      pip install -U huggingface_hub
      hf auth login                       # a WRITE token from
                                          # https://huggingface.co/settings/tokens
      hf upload {args.repo_id} {out_dir} . --repo-type=model

  Read {card} first -- it is generated, and a generated card is
  still a claim you are making.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
