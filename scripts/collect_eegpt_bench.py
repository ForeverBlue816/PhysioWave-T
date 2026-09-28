#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Every EEGPT-benchmark run, in one table, next to EEGPT's published rows.

    python scripts/collect_eegpt_bench.py --root $PW_CKPT_ROOT/eegpt_bench

Reads <root>/<task>_f<fold>_<mode>/results.json -- the files
EEG/finetune_eegpt_bench.sh writes -- and prints the TEST metrics.

The reference rows are EEGPT's Table 4 (NeurIPS 2024), mean ± std. They are
NOT like-for-like with ours, and the table says so rather than leaving it to a
footnote: theirs are a frozen-encoder linear probe averaged over folds (nine
LOSO folds on BCIC, four on KaggleERN) and scored on the subjects that were
also their validation set. Ours is one fold, with a test set nothing selected
on, and every parameter fine-tuned. `ft` starts from the pretrained encoder;
`scratch` is the same model from random initialisation -- the control that
says whether pretraining did anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

#: EEGPT, Table 4 ("Ours"): (balanced acc, kappa, weighted F1 or AUROC).
EEGPT = {
    "bcic2a":    {"balanced_acc": 0.5846, "kappa": 0.4462, "weighted_f1": 0.5715},
    "bcic2b":    {"balanced_acc": 0.7212, "kappa": 0.4426, "auroc": 0.8059},
    "kaggleern": {"balanced_acc": 0.5837, "kappa": 0.1882, "auroc": 0.6621},
}
#: The third column EEGPT reports: weighted F1 for multi-class, AUROC for binary.
THIRD = {"bcic2a": "weighted_f1", "bcic2b": "auroc", "kaggleern": "auroc"}
MODE_ORDER = {"ft": 0, "scratch": 1}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--json", help="also write the table here")
    args = p.parse_args(argv)

    rows = []
    for path in glob.glob(os.path.join(args.root, "*", "results.json")):
        m = re.fullmatch(r"(bcic2a|bcic2b|kaggleern)_f(\d+)_(ft|scratch)(?:_(\w+))?",
                         os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        with open(path) as f:
            res = json.load(f)
        te = res.get("test")
        if not te:
            continue
        rows.append({"task": m.group(1), "fold": int(m.group(2)), "mode": m.group(3),
                     "tag": m.group(4) or "",
                     "best_epoch": res.get("best_epoch"),
                     "select_by": res.get("select_by"),
                     "trainable": res.get("trainable_params"),
                     **{k: te.get(k) for k in ("balanced_acc", "kappa",
                                               "weighted_f1", "auroc")}})
    if not rows:
        print(f"no finished runs under {args.root}")
        return 1

    rows.sort(key=lambda r: (r["task"], r["fold"], r["tag"], MODE_ORDER[r["mode"]]))
    print(f"{'task':10} {'fold':>4} {'mode[:tag]':14} {'BAC':>7} {'kappa':>7} "
          f"{'3rd':>7}  {'3rd is':12} {'trainable':>11}  best@ep")
    last = None
    for r in rows:
        if r["task"] != last:
            ref = EEGPT[r["task"]]
            third = THIRD[r["task"]]
            print(f"{r['task']:10} {'':>4} {'EEGPT':14} {ref['balanced_acc']:7.4f} "
                  f"{ref['kappa']:7.4f} {ref[third]:7.4f}  {third:12} "
                  f"{'probe':>11}  (Table 4, fold mean)")
            last = r["task"]
        third = THIRD[r["task"]]
        f = lambda v: f"{v:7.4f}" if isinstance(v, (int, float)) else f"{'n/a':>7}"  # noqa: E731
        label = r["mode"] + (f":{r['tag']}" if r["tag"] else "")
        print(f"{'':10} {r['fold']:>4} {label:14} {f(r['balanced_acc'])} "
              f"{f(r['kappa'])} {f(r[third])}  {third:12} "
              f"{r['trainable'] or 0:>11,}  {r['best_epoch']} ({r['select_by']})")

    # The comparison the benchmark exists for: did pretraining help?
    print()
    for task in sorted({r["task"] for r in rows}):
        key = "kappa" if task == "bcic2a" else "auroc"
        for fold in sorted({r["fold"] for r in rows if r["task"] == task}):
            here = [r for r in rows if r["task"] == task and r["fold"] == fold]
            scratch = [r for r in here if r["mode"] == "scratch"]
            if not scratch:
                continue
            base = scratch[0][key] or 0
            for r in here:
                if r["mode"] == "ft":
                    tag = f" [{r['tag']}]" if r["tag"] else ""
                    print(f"  {task} f{fold}{tag}: pretrained - scratch {key} "
                          f"= {(r[key] or 0) - base:+.4f}")

    print("\n  EEGPT's rows: frozen encoder + linear probe, mean over folds, scored on")
    print("  their validation subjects. Ours: one fold, test subjects select nothing.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"runs": rows, "eegpt_table4": EEGPT}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
