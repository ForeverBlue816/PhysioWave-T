#!/usr/bin/env python
"""
The SeizeIT2 event-level table: our runs beside the published numbers.

    python scripts/collect_seizeit2_events.py --root $PW_CKPT_ROOT/fusion [--gate cross] \\
        [--markdown out.md] [--condition complete]

Reads <root>/*/events_<gate>.json (Fusion/seizeit2_events.py). Rows: each run
at threshold 0.5 under the dataset paper's scoring (means over recordings of
the any-overlap sensitivity and FA/h, epoch FA/h, the challenge score), the
pooled event sensitivity and FA/24h, the sensitivity reachable at 1 and 6.22
FA/h, and the pooled epoch AUROC / AUPRC; then the published rows.

THE PUBLISHED ROWS ARE NOT ALL LIKE-FOR-LIKE. The dataset paper's SVM and
ChronoNet (Sci. Data 2025, Table 2) use this split and this scoring on bte-EEG
alone. The Epilepsia Open 2026 validation reports per-patient means of a bte-EEG
+ ECG algorithm over its own adult cohort; it is indicative only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

#: (label, sens %, FA/h, AUROC, AUPRC, note)
PUBLISHED = [
    ("SVM, bte-EEG (Bhagubai et al., Sci. Data 2025)", 71.1, 11.0, 0.6995, 0.0810,
     "same split and scoring"),
    ("ChronoNet, bte-EEG (Bhagubai et al., Sci. Data 2025)", 84.2, 100.5, 0.7921, 0.1042,
     "same split and scoring"),
    ("bte-EEG + ECG algorithm (Swinnen et al., Epilepsia Open 2026)", 78.0, 6.22, None, None,
     "per-patient mean, own cohort and protocol"),
]


def fmt(v, nd=3):
    return "  -  " if v is None or v != v else f"{v:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate", default="cross")
    ap.add_argument("--condition", default=None, help="one condition; default all")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.root, "*", f"events_{args.gate}.json")))
    if not files:
        print(f"no events_{args.gate}.json under {args.root}")
        return 1
    head = (f"{'run / condition':44s} {'sens%':>6s} {'FA/h':>6s} {'FAep/h':>7s} {'score':>7s}"
            f" | {'pool sens':>9s} {'FA/24h':>7s} | {'s@1FA/h':>7s} {'s@6.2':>6s}"
            f" | {'AUROC':>6s} {'AUPRC':>6s}")
    md = ["# SeizeIT2, event-level, official split (test = sub-097..125)", "",
          f"RMS gate: {args.gate}. Threshold 0.5 unless stated.", "",
          "| run / condition | sens % | FA/h | FA epoch/h | score | pooled sens | FA/24h | "
          "sens @1 FA/h | sens @6.22 FA/h | AUROC | AUPRC |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    print(head)
    print("-" * len(head))
    for f in files:
        name = os.path.basename(os.path.dirname(f))
        ev = json.load(open(f))
        for cond, r in ev["conditions"].items():
            if args.condition and cond != args.condition:
                continue
            a = r["at_0.5"]
            o, e = a["official"], a["events"]
            row = [100 * o["sens_ovlp"], o["fa_ovlp_h"], o["fa_epoch_h"], o["score"],
                   e["sens"], e["fa_24h"], r["sens_at_fa_events"].get("1.0"),
                   r["sens_at_fa_events"].get("6.22"), r["auroc_epoch"], r["auprc_epoch"]]
            label = f"{name} / {cond}"
            print(f"{label[:44]:44s} {fmt(row[0], 1):>6s} {fmt(row[1], 2):>6s} "
                  f"{fmt(row[2], 1):>7s} {fmt(row[3], 1):>7s} | {fmt(row[4]):>9s} "
                  f"{fmt(row[5], 1):>7s} | {fmt(row[6]):>7s} {fmt(row[7]):>6s} | "
                  f"{fmt(row[8]):>6s} {fmt(row[9], 4):>6s}")
            md.append(f"| {label} | " + " | ".join(
                fmt(v, nd) for v, nd in zip(row, (1, 2, 1, 1, 3, 1, 3, 3, 3, 4))) + " |")
    print("-" * len(head))
    md.append("| **published** | | | | | | | | | | |")
    for label, sens, fa, auroc, auprc, note in PUBLISHED:
        print(f"{label[:44]:44s} {sens:6.1f} {fa:6.2f} {'':>7s} {'':>7s} | {'':>9s} {'':>7s} | "
              f"{'':>7s} {'':>6s} | {fmt(auroc):>6s} {fmt(auprc, 4):>6s}   ({note})")
        md.append(f"| {label} ({note}) | {sens:.1f} | {fa:.2f} | | | | | | | "
                  f"{fmt(auroc)} | {fmt(auprc, 4)} |")
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
