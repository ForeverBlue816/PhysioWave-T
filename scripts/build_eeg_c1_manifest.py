#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge per-dataset (and per-shard) manifests into the two the trainer reads.

    python scripts/build_eeg_c1_manifest.py --corpus-root $DATA_ROOT

    <corpus-root>/
        tueg/     manifest_train.0000.jsonl  manifest_train.0001.jsonl  ...
        faced/    manifest_train.jsonl       manifest_val.jsonl
        ...
        merged/   manifest_train.jsonl       manifest_val.jsonl     <- written

Preprocessing runs once per dataset, and TUEG runs once per SLURM array task, so
the manifests arrive in pieces. This concatenates them, makes every shard path
absolute, and checks the things that are only checkable once all the pieces are
in one place.

THE CHECK THAT MATTERS. No subject may appear in both splits. Each task decides
its own subjects' sides by hashing, which makes that true by construction -- but
"by construction" is an argument, and this is the measurement. A leak here means
the val loss is reporting memorisation, and it is worth one pass over a few
thousand JSON lines to know it did not happen.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physiowave.eeg_c1.routes import PRETRAIN_DATASETS      # noqa: E402


def load_rows(dataset_dir: str, split: str) -> List[Dict]:
    """Every manifest row for one dataset and split, sharded or not."""
    rows = []
    patterns = (os.path.join(dataset_dir, f"manifest_{split}.jsonl"),
                os.path.join(dataset_dir, f"manifest_{split}.*.jsonl"))
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if not os.path.isabs(rec["path"]):
                        rec["path"] = os.path.abspath(
                            os.path.join(dataset_dir, rec["path"]))
                    rows.append(rec)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--out-dir", default=None,
                   help="default: <corpus-root>/merged")
    p.add_argument("--datasets", default=None,
                   help="comma-separated subset; default: whatever is present")
    p.add_argument("--allow-missing", action="store_true",
                   help="merge what is there instead of naming what is not")
    p.add_argument("--check-shards", action="store_true",
                   help="also open every shard to confirm it is readable. Slow "
                        "on TUEG; worth it once before a long run.")
    args = p.parse_args(argv)

    root = args.corpus_root
    out_dir = args.out_dir or os.path.join(root, "merged")
    wanted = ([d.strip() for d in args.datasets.split(",") if d.strip()]
              if args.datasets else list(PRETRAIN_DATASETS))

    present, missing = [], []
    for d in wanted:
        if os.path.isdir(os.path.join(root, d)):
            present.append(d)
        else:
            missing.append(d)
    if missing and not args.allow_missing:
        print(f"ERROR: no directory under {root} for: {', '.join(missing)}\n"
              f"  Preprocess them first, or pass --allow-missing to merge the "
              f"{len(present)} that are there. A pretraining run on a subset is "
              f"a different run and should be a deliberate one.", file=sys.stderr)
        return 1
    if not present:
        print(f"ERROR: {root} holds no preprocessed dataset.", file=sys.stderr)
        return 1

    merged: Dict[str, List[Dict]] = {"train": [], "val": []}
    subjects: Dict[str, Dict[str, set]] = defaultdict(
        lambda: {"train": set(), "val": set()})
    per_dataset = {}

    for d in present:
        dataset_dir = os.path.join(root, d)
        counts = {}
        for split in ("train", "val"):
            rows = load_rows(dataset_dir, split)
            merged[split].extend(rows)
            counts[f"{split}_shards"] = len(rows)
            counts[f"{split}_windows"] = sum(r["n_windows"] for r in rows)
            for r in rows:
                subjects[d][split].update(r.get("subjects", ()))
        counts["route_id"] = PRETRAIN_DATASETS[d].route_id
        counts["subjects"] = len(subjects[d]["train"] | subjects[d]["val"])
        per_dataset[d] = counts

    # -- the check ---------------------------------------------------------- #
    leaks = {d: sorted(s["train"] & s["val"]) for d, s in subjects.items()
             if s["train"] & s["val"]}
    if leaks:
        print("ERROR: a subject appears in both splits, so the validation loss "
              "would be reporting memorisation:", file=sys.stderr)
        for d, names in leaks.items():
            print(f"  {d}: {len(names)} subject(s), e.g. {names[:8]}",
                  file=sys.stderr)
        print("\n  Preprocessing tasks disagreed about a subject's side. That "
              "happens when tasks ran with different --split-seed or "
              "--val-fraction, or when one ran with --split-mode exact.",
              file=sys.stderr)
        return 1

    seen_paths = set()
    duplicates = []
    for split in ("train", "val"):
        for r in merged[split]:
            if r["path"] in seen_paths:
                duplicates.append(r["path"])
            seen_paths.add(r["path"])
    if duplicates:
        print(f"ERROR: {len(duplicates)} shard path(s) listed more than once, "
              f"e.g. {duplicates[:4]}.\n  Overlapping --shard ranges, or a "
              f"stale manifest from an earlier run left in place.",
              file=sys.stderr)
        return 1

    if args.check_shards:
        import h5py
        bad = []
        for split in ("train", "val"):
            for r in merged[split]:
                try:
                    with h5py.File(r["path"], "r") as f:
                        if int(f["data"].shape[0]) != int(r["n_windows"]):
                            bad.append((r["path"], "window count disagrees"))
                except Exception as exc:                      # noqa: BLE001
                    bad.append((r["path"], str(exc)))
        if bad:
            print(f"ERROR: {len(bad)} shard(s) unreadable or inconsistent:",
                  file=sys.stderr)
            for path, why in bad[:8]:
                print(f"  {path}: {why}", file=sys.stderr)
            return 1

    # An empty merge is never a successful one. Writing a pair of empty
    # manifests and reporting a table of zeros is how a training run gets
    # pointed at nothing: the trainer's own check fires much later and much
    # further from the cause. This is also what "run it after squeue is empty"
    # is actually protecting against, and a rule that depends on remembering it
    # is not a check.
    n_shards = len(merged["train"]) + len(merged["val"])
    if n_shards == 0:
        print(f"ERROR: no manifest rows found under {root}.\n\n"
              f"  Looked for {'/'.join(present)}/manifest_{{train,val}}"
              f"[.NNNN].jsonl\n\n"
              f"  A preprocessing task writes its manifest when it FINISHES, so"
              f" this is\n"
              f"  what an empty or still-running corpus looks like. Check:\n"
              f"    squeue --me\n"
              f"    ls -la {os.path.join(root, present[0])}/manifest_*.jsonl\n"
              f"    sacct -j <jobid> --format=JobID,State,ExitCode -X\n\n"
              f"  Nothing was written; the previous merged/ is left as it was.",
              file=sys.stderr)
        return 1
    if not merged["val"]:
        print(f"ERROR: {len(merged['train'])} training shard(s) and no "
              f"validation shard.\n"
              f"  Every subject hashed to the training side, which at this "
              f"corpus size means\n"
              f"  the split seed or fraction differs between tasks, or too few "
              f"subjects finished.",
              file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val"):
        path = os.path.join(out_dir, f"manifest_{split}.jsonl")
        with open(path, "w") as f:
            for r in merged[split]:
                f.write(json.dumps(r) + "\n")

    summary = {"corpus_root": os.path.abspath(root),
               "datasets": per_dataset,
               "missing_datasets": missing,
               "total_train_windows": sum(r["n_windows"] for r in merged["train"]),
               "total_val_windows": sum(r["n_windows"] for r in merged["val"])}
    with open(os.path.join(out_dir, "corpus_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'dataset':<14} {'route':<10} {'subj':>6} {'train win':>12} "
          f"{'val win':>10} {'shards':>8}")
    print("-" * 66)
    for d in present:
        c = per_dataset[d]
        print(f"{d:<14} {c['route_id']:<10} {c['subjects']:>6} "
              f"{c['train_windows']:>12,} {c['val_windows']:>10,} "
              f"{c['train_shards'] + c['val_shards']:>8}")
    print("-" * 66)
    print(f"{'TOTAL':<14} {'':<10} {'':>6} {summary['total_train_windows']:>12,} "
          f"{summary['total_val_windows']:>10,}")
    if missing:
        print(f"\nmissing (merged anyway): {', '.join(missing)}")
    print(f"\nno subject in both splits: checked {sum(len(s['train'] | s['val']) for s in subjects.values())} subject(s)")
    print(f"wrote {out_dir}/manifest_train.jsonl")
    print(f"      {out_dir}/manifest_val.jsonl")
    print(f"      {out_dir}/corpus_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
