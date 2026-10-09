#!/usr/bin/env python
"""
Table of the ECG C1 downstream runs, with the paper's PhysioWave (v1) row.

    python scripts/collect_ecg_downstream.py --root $PW_CKPT_ROOT/ecg_downstream \\
        [--json summary.json] [--markdown summary.md]

Reads <root>/<task>_<mode>[_<tag>]/results.json, as ECG/finetune_ecg_c1.sh
writes them, and prints per task: fine-tuned, linear probe and from scratch,
the test metrics the paper reports for that task, and pretrained - scratch.

The v1 row is printed for reference and is NOT a like-for-like comparison:
v1 scored 4.1 s windows (not records), with min-max / z-score normalisation,
a fixed 0.3 multi-label threshold and its own random splits. Here the test
set is scored once, at the checkpoint validation chose, with per-class F1
thresholds also chosen on validation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

TASKS = ("ptbxl", "ptbxl_super", "cpsc2018", "chapman")
MODES = ("ft", "probe", "scratch")
MODE_NAME = {"ft": "C1 fine-tuned", "probe": "C1 linear probe",
             "scratch": "C1 from scratch"}

#: (key in results.json "test", column label). The first is the paper's metric.
COLUMNS = {
    "ptbxl": [("acc", "Acc"), ("balanced_acc", "BalAcc"), ("macro_f1", "F1-mac"),
              ("auroc", "AUROC")],
    "ptbxl_super": [("auroc", "AUROC-mac"), ("micro_auroc", "AUROC-mic"),
                    ("macro_auprc", "AUPRC"), ("macro_f1_tuned", "F1-mac")],
    "cpsc2018": [("micro_f1_tuned", "F1-mic"), ("macro_f1_tuned", "F1-mac"),
                 ("auroc", "AUROC-mac"), ("micro_auroc", "AUROC-mic")],
    "chapman": [("micro_f1_tuned", "F1-mic"), ("macro_f1_tuned", "F1-mac"),
                ("auroc", "AUROC-mac"), ("micro_auroc", "AUROC-mic")],
}
#: PhysioWave (v1), README "Benchmark Results" -- window-level, see above.
PAPER_V1 = {
    "ptbxl": {"acc": 0.731},
    "cpsc2018": {"micro_f1_tuned": 0.7709, "macro_f1_tuned": 0.6500,
                 "auroc": 0.9280, "micro_auroc": 0.9584},
    "chapman": {"micro_f1_tuned": 0.9462, "macro_f1_tuned": 0.9413,
                "auroc": 0.9930, "micro_auroc": 0.9949},
}
TITLE = {"ptbxl": "PTB-XL, 5 superclasses, single label (paper: accuracy)",
         "ptbxl_super": "PTB-XL superdiagnostic, multi-label (literature: macro AUROC)",
         "cpsc2018": "CPSC 2018, 9 classes, multi-label, per record (paper: F1-micro)",
         "chapman": "Chapman-Shaoxing, 4 rhythm classes, multi-label (paper: F1-micro)"}

DIR_RE = re.compile(rf"^({'|'.join(TASKS)})_({'|'.join(MODES)})(?:_(\w+))?$")


def fmt(v):
    return "   -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def collect(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(path))
        m = DIR_RE.match(name)
        if not m:
            continue
        task, mode, tag = m.group(1), m.group(2), m.group(3) or ""
        with open(path) as f:
            r = json.load(f)
        if "test" not in r:
            continue
        runs.setdefault(task, {}).setdefault(tag, {})[mode] = {
            "test": r["test"], "best_epoch": r.get("best_epoch"),
            "select_by": r.get("select_by"), "dir": os.path.dirname(path)}
    return runs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--markdown", default=None)
    args = p.parse_args(argv)
    runs = collect(args.root)
    if not runs:
        print(f"no finished runs under {args.root}")
        return 1
    md = ["# ECG C1 downstream", ""]
    for task in TASKS:
        if task not in runs:
            continue
        cols = COLUMNS[task]
        for tag, modes in sorted(runs[task].items()):
            head = TITLE[task] + (f"  [{tag}]" if tag else "")
            print("=" * 78)
            print(head)
            print("-" * 78)
            print(f"  {'':22s}" + "".join(f"{lab:>11s}" for _, lab in cols))
            md += [f"## {head}", "",
                   "| | " + " | ".join(lab for _, lab in cols) + " |",
                   "|---|" + "---|" * len(cols)]
            for mode in MODES:
                if mode in modes:
                    t = modes[mode]["test"]
                    vals = [t.get(k) for k, _ in cols]
                    extra = f"   (best epoch {modes[mode]['best_epoch']}, by val {modes[mode]['select_by']})"
                    print(f"  {MODE_NAME[mode]:22s}" + "".join(f"{fmt(v):>11s}" for v in vals) + extra)
                    md.append(f"| {MODE_NAME[mode]} | " + " | ".join(fmt(v) for v in vals) + " |")
            if task in PAPER_V1:
                ref = PAPER_V1[task]
                print(f"  {'PhysioWave v1 (paper)':22s}" + "".join(
                    f"{fmt(ref.get(k)):>11s}" for k, _ in cols) + "   (window-level, see header)")
                md.append("| PhysioWave v1 (paper)* | " + " | ".join(
                    fmt(ref.get(k)) for k, _ in cols) + " |")
            if "ft" in modes and "scratch" in modes:
                k, lab = cols[0]
                d = modes["ft"]["test"].get(k, float("nan")) - modes["scratch"]["test"].get(k, float("nan"))
                print(f"  pretrained - scratch  {lab} {d:+.4f}")
            md.append("")
    print("=" * 78)
    print("  * v1 scored 4.1 s windows with a fixed threshold and its own splits; this")
    print("    table scores records once, at the checkpoint validation chose, with F1")
    print("    thresholds chosen on validation. Indicative, not like-for-like.")
    md += ["\\* PhysioWave v1 scored 4.1 s windows with a fixed threshold and its own "
           "splits; indicative, not like-for-like."]
    if args.json:
        with open(args.json, "w") as f:
            json.dump(runs, f, indent=2, default=str)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
