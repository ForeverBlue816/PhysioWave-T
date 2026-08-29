#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""What a pretraining run has actually done so far, per route and per dataset.

    python scripts/eeg_c1_progress.py $PW_CKPT_ROOT/pretrain_eeg_c1_moe_n1

THE LOSS IS NOT THE SUMMARY. `val mse` is an average over validation BATCHES,
and the validation sweep is proportional to corpus size while the training
mixture is not: 96.8% of validation batches are TUEG or HBN, against 37% of
training steps. A model specialising toward the five small corpora -- which
balanced sampling shows it 20 to 50 times each -- therefore looks like a model
getting worse, and the per-route columns are what tell the two apart.

`masked_corr` rather than MSE per route: correlation is dimensionless, so it
stays comparable if anything about the target's scale changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROUTES = ("E19_256", "E32_512", "E64_256", "E128_512")
DATASETS = ("tueg", "hbn", "faced", "tdbrain", "m3cv", "physionet_mi", "hgd")


def rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt(v, width=8, places=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return " " * (width - 1) + "-"
    return f"{v:>{width}.{places}f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="the --output-dir of the run")
    p.add_argument("--by", choices=("route", "dataset"), default="route")
    p.add_argument("--metric", default="masked_corr")
    args = p.parse_args(argv)

    path = os.path.join(args.run_dir, "metrics_epoch.jsonl")
    if not os.path.isfile(path):
        print(f"ERROR: no {path}\n"
              f"  If the path looks like it is missing its root, PW_CKPT_ROOT\n"
              f"  was empty in your shell: source scripts/cineca_env.sh first.",
              file=sys.stderr)
        return 1

    data = rows(path)
    if not data:
        print(f"ERROR: {path} is empty -- no epoch has finished yet.",
              file=sys.stderr)
        return 1

    keys = ROUTES if args.by == "route" else DATASETS
    present = [k for k in keys
               if any(f"val/{args.by}/{k}/{args.metric}" in r for r in data)]

    head = f"{'ep':>3}{'train':>9}{'val':>9}{'corr':>8}"
    head += "".join(f"{k[:9]:>10}" for k in present)
    head += f"{'gate':>8}{'mins':>7}"
    print(head)
    print("-" * len(head))

    best_ep, best = None, float("inf")
    for r in data:
        v = r.get("val/loss_masked_mse")
        if v is not None and v < best:
            best, best_ep = v, r["epoch"]
        line = (f"{r['epoch']:>3}"
                f"{fmt(r.get('train/loss_masked_mse'), 9, 5)}"
                f"{fmt(v, 9, 5)}"
                f"{fmt(r.get('val/masked_corr'))}")
        for k in present:
            line += fmt(r.get(f"val/{args.by}/{k}/{args.metric}"), 10)
        secs = (r.get("train/epoch_seconds") or 0) + (r.get("val/val_seconds") or 0)
        line += fmt(r.get("train/channel_token_gate_tanh"), 8)
        line += f"{secs / 60:>7.1f}" if secs else "      -"
        print(line)

    print(f"\nbest val loss {best:.5f} at epoch {best_ep} "
          f"(best.pth holds it)")
    if len(data) >= 3:
        recent = [r.get("val/masked_corr") for r in data[-3:]]
        if all(x is not None for x in recent):
            drift = recent[-1] - recent[0]
            verdict = ("still improving" if drift > 0.005 else
                       "flat" if drift > -0.005 else "going backwards")
            print(f"val corr over the last 3 epochs: "
                  f"{recent[0]:.4f} -> {recent[-1]:.4f}  ({verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
