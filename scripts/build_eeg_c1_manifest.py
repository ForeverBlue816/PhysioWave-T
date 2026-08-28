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


#: How much of a shard to actually read. ``meta`` is the shape and attributes
#: alone; ``ends`` also inflates the first and last window; ``full`` inflates
#: every one.
CHECK_LEVELS = ("meta", "ends", "full")

#: Windows per read in the ``full`` pass. A shard is chunked one window per
#: chunk, so this only bounds memory -- every chunk is inflated either way.
_READ_BLOCK = 64


def _check_one(row):
    """Open one shard and confirm it is readable and the right length.

    THE SHAPE IS NOT THE DATA. ``f["data"].shape`` comes out of the object
    header, which HDF5 writes before any chunk. A task killed part-way through
    a write therefore leaves a file that answers this question correctly and
    raises ``OSError: Can't synchronously read data (inflate() failed)`` the
    first time training touches one of its windows. That is precisely what a
    ``meta`` check passed and a sixteen-rank job then died on, so ``full``
    -- which inflates every chunk, and is the only level that can prove a
    shard is readable -- is the default.

    A module-level function so it can cross a process boundary. h5py is not
    reliably thread-safe, so this is a process pool rather than a thread pool.
    """
    import h5py
    path, expected, level = row
    try:
        with h5py.File(path, "r") as f:
            data = f["data"]
            got = int(data.shape[0])
            if got != expected:
                return path, f"manifest says {expected} windows, file has {got}"
            if level == "ends" and got:
                # Truncation takes the tail, so the last chunk is the one that
                # tells you about it; the first is nearly free alongside.
                data[0], data[got - 1]
            elif level == "full":
                for k in range(0, got, _READ_BLOCK):
                    data[k:k + _READ_BLOCK]
    except Exception as exc:                                   # noqa: BLE001
        return path, str(exc)
    return None


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
                   help="read every shard to confirm it is readable and the "
                        "length its manifest row claims. On TUEG that is 67,000 "
                        "files, so give it --jobs; worth doing once before "
                        "committing to a long training run.")
    p.add_argument("--check-level", choices=CHECK_LEVELS, default="full",
                   help="how much of each shard --check-shards reads. "
                        "'full' (default) inflates every window and is the only "
                        "level that proves a shard is readable; 'ends' inflates "
                        "the first and last, which catches a truncated write; "
                        "'meta' reads the shape alone and cannot tell a good "
                        "shard from a half-written one.")
    p.add_argument("--drop-unreadable", action="store_true",
                   help="with --check-shards, leave the bad shards out of the "
                        "merged manifests and list them in "
                        "unreadable_shards.jsonl, instead of refusing to write. "
                        "The windows they held are lost either way; this makes "
                        "the loss explicit and lets training start.")
    p.add_argument("--jobs", type=int, default=1,
                   help="workers for --check-shards. The merge itself reads a "
                        "handful of text files and needs none of these.")
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
        import concurrent.futures as cf
        import time

        todo = [(r["path"], int(r["n_windows"]), args.check_level)
                for split in ("train", "val") for r in merged[split]]
        bad = []
        started = time.time()
        print(f"checking {len(todo):,} shard(s) on {args.jobs} worker(s) "
              f"at level '{args.check_level}'...", flush=True)
        if args.jobs > 1:
            with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for i, res in enumerate(
                        pool.map(_check_one, todo, chunksize=64), start=1):
                    if res is not None:
                        bad.append(res)
                    if i % 5000 == 0:
                        rate = i / max(1e-9, time.time() - started)
                        print(f"  {i:,}/{len(todo):,}  {rate:.0f}/s  "
                              f"{len(bad)} bad  "
                              f"ETA {(len(todo)-i)/max(1e-9, rate)/60:.0f} min",
                              flush=True)
        else:
            for row in todo:
                res = _check_one(row)
                if res is not None:
                    bad.append(res)
        print(f"  checked in {time.time()-started:.0f}s", flush=True)
        if bad and not args.drop_unreadable:
            print(f"ERROR: {len(bad)} of {len(todo)} shard(s) unreadable or "
                  f"inconsistent:", file=sys.stderr)
            for path, why in bad[:8]:
                print(f"  {path}: {why}", file=sys.stderr)
            if len(bad) > 8:
                print(f"  ... and {len(bad) - 8} more", file=sys.stderr)
            print(f"\n  'inflate() failed' means the header survived a write "
                  f"the data did not --\n"
                  f"  a preprocessing task killed part-way through. Re-run it "
                  f"for those\n"
                  f"  recordings, or re-merge with --drop-unreadable to leave "
                  f"them out and\n"
                  f"  start training without them.", file=sys.stderr)
            return 1
        if bad:
            os.makedirs(out_dir, exist_ok=True)
            report = os.path.join(out_dir, "unreadable_shards.jsonl")
            with open(report, "w") as f:
                for path, why in bad:
                    f.write(json.dumps({"path": path, "error": why}) + "\n")
            dropped = {path for path, _ in bad}
            lost = sum(int(r["n_windows"])
                       for split in ("train", "val") for r in merged[split]
                       if r["path"] in dropped)
            for split in ("train", "val"):
                merged[split] = [r for r in merged[split]
                                 if r["path"] not in dropped]
            # The table and the summary are built from the rows, so rebuild
            # them from what survived rather than reporting counts the written
            # manifests no longer back.
            for d in present:
                c, seen = per_dataset[d], {"train": set(), "val": set()}
                for split in ("train", "val"):
                    rows = [r for r in merged[split] if r["dataset_id"] == d]
                    c[f"{split}_shards"] = len(rows)
                    c[f"{split}_windows"] = sum(r["n_windows"] for r in rows)
                    for r in rows:
                        seen[split].update(r.get("subjects", ()))
                subjects[d] = seen
                c["subjects"] = len(seen["train"] | seen["val"])
            print(f"dropped {len(bad)} unreadable shard(s), {lost:,} window(s); "
                  f"listed in {report}", flush=True)

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
