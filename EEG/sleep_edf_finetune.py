#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sleep-EDF Expanded (Sleep Cassette) -> labelled HDF5 for 5-class sleep staging.
------------------------------------------------------------------------------
The preprocessing deliberately reproduces EEGPT's `datasets/downstream/
prepare_sleep.py` step for step, because the point of running this dataset is
to sit next to their published number rather than beside it:

    * source        braindecode SleepPhysionet, which fetches Sleep-EDFx SC
                    from PhysioNet through MNE the first time it is asked
    * channels      the two EEG derivations, Fpz-Cz and Pz-Oz, at 100 Hz
    * cropping      crop_wake_mins=30, so only 30 minutes of wake either side
                    of the sleep period is kept -- without this the class
                    balance is dominated by hours of pre-bed wake
    * scaling       volts to microvolts, then a 30 Hz lowpass
    * windows       30 s, non-overlapping, aligned to the hypnogram events,
                    i.e. 3000 samples
    * labels        W / N1 / N2 / N3 / REM, with the AASM merge of stages 3
                    and 4 into N3
    * normalising   per-channel z-score inside each window

What is NOT reproduced is their evaluation protocol. EEGPT trains 40 epochs
with no checkpoint callback and reports the validation fold, so the model is
selected on the set it is scored on. `--split holdout` keeps a test set that
nothing selects on; `--split eegpt-fold` reproduces their partition for a
like-for-like row. Report both and say which is which.

Two stages, because the download and the decoding are slow and the split is
not. `--stage cache` writes one .npz per subject; `--stage split` turns those
into HDF5. The default runs both.

Usage:
    pip install braindecode          # pulls in mne
    python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf

Output, matching what finetune.py reads:
    data          (N, 2, 3000) float32
    label         (N,)         int64      0=W 1=N1 2=N2 3=N3 4=REM
    channel_names (2,)         bytes
    subject       (N,)         int64      provenance; unused by the trainer
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

FS = 100.0
WINDOW_SAMPLES = 30 * int(FS)      # 3000
LOWPASS_HZ = 30.0
CROP_WAKE_MINS = 30
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
MAPPING = {                        # AASM: stages 3 and 4 are one stage
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

# braindecode's SleepPhysionet covers subjects 0..82 with these absent.
MISSING = {39, 68, 69, 78, 79}

# The subject list EEGPT's finetune_EEGPT_SleepEDF.py runs its 10 folds over.
# Copied verbatim so the partition is theirs, not a reconstruction of theirs.
EEGPT_SUBJECTS = [
    0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24,
    25, 26, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 42, 44, 45, 46, 47, 48, 49,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74,
    75, 76, 77, 81, 82,
]


# --------------------------------------------------------------------------- #
# Stage 1: fetch and preprocess one subject
# --------------------------------------------------------------------------- #
def cache_subject(subject: int, cache_dir: str, overwrite: bool = False) -> str | None:
    """Download, preprocess and window one subject into a .npz. Returns its path."""
    out = os.path.join(cache_dir, f"sub{subject:02d}.npz")
    if os.path.exists(out) and not overwrite:
        return out

    try:
        from braindecode.datasets import SleepPhysionet
        from braindecode.preprocessing import Preprocessor, preprocess
        from braindecode.preprocessing import create_windows_from_events
        from sklearn.preprocessing import scale as standard_scale
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n\nThis script needs braindecode (which pulls in mne):\n"
            f"    pip install braindecode"
        ) from exc

    ds = SleepPhysionet(subject_ids=[subject], crop_wake_mins=CROP_WAKE_MINS)
    preprocess(ds, [
        Preprocessor(lambda d: d * 1e6),                      # volts -> microvolts
        Preprocessor("filter", l_freq=None, h_freq=LOWPASS_HZ, n_jobs=1),
    ])
    windows = create_windows_from_events(
        ds, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=WINDOW_SAMPLES, window_stride_samples=WINDOW_SAMPLES,
        preload=True, mapping=MAPPING,
    )
    # Per-channel z-score inside each window, applied after windowing so a
    # window's scale does not depend on the rest of the night.
    preprocess(windows, [Preprocessor(standard_scale, channel_wise=True)])

    X = np.stack([w[0] for w in windows]).astype(np.float32)   # (N, C, 3000)
    y = np.array([w[1] for w in windows], dtype=np.int64)
    if X.shape[1] != len(CHANNELS) or X.shape[2] != WINDOW_SAMPLES:
        raise RuntimeError(f"subject {subject}: unexpected shape {X.shape}")

    np.savez_compressed(out, data=X, label=y)
    return out


# --------------------------------------------------------------------------- #
# Stage 2: subject-level splits
# --------------------------------------------------------------------------- #
def holdout_split(subjects: list[int], seed: int = 7) -> dict[str, list[int]]:
    """60/20/20 by subject.

    The split is by subject and not by window: consecutive 30 s epochs of one
    night are near-duplicates, so a window-level split would put a subject's
    own neighbouring epochs on both sides and report a number that says
    nothing about a new sleeper.
    """
    rng = np.random.default_rng(seed)
    order = np.array(subjects)[rng.permutation(len(subjects))]
    n_tr, n_va = int(0.6 * len(order)), int(0.2 * len(order))
    return {
        "train": sorted(order[:n_tr].tolist()),
        "val": sorted(order[n_tr:n_tr + n_va].tolist()),
        "test": sorted(order[n_tr + n_va:].tolist()),
    }


def eegpt_fold_split(fold: int, val_ratio: float = 0.15,
                     seed: int = 7) -> dict[str, list[int]]:
    """EEGPT's 10-fold partition of its 64-subject list, fold ``fold`` held out.

    EEGPT scores the held-out fold and selects on it. To keep the comparison
    honest a validation set is carved out of the *training* subjects here, so
    the held-out fold is only ever scored. That makes our number pessimistic
    relative to theirs rather than the other way round.
    """
    n = len(EEGPT_SUBJECTS) // 10
    held = EEGPT_SUBJECTS[fold * n:(fold + 1) * n]
    rest = [s for s in EEGPT_SUBJECTS if s not in set(held)]
    rng = np.random.default_rng(seed + fold)
    rest = np.array(rest)[rng.permutation(len(rest))]
    n_va = max(1, int(val_ratio * len(rest)))
    return {"train": sorted(rest[n_va:].tolist()),
            "val": sorted(rest[:n_va].tolist()),
            "test": sorted(held)}


def write_split(name: str, subjects: list[int], cache_dir: str, out_dir: str) -> dict:
    """Concatenate cached subjects into one HDF5, growing it incrementally."""
    path = os.path.join(out_dir, f"{name}.h5")
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    total = 0
    with h5py.File(path, "w") as f:
        data = f.create_dataset("data", shape=(0, len(CHANNELS), WINDOW_SAMPLES),
                                maxshape=(None, len(CHANNELS), WINDOW_SAMPLES),
                                dtype="float32", chunks=(64, len(CHANNELS), WINDOW_SAMPLES),
                                compression="lzf")
        label = f.create_dataset("label", shape=(0,), maxshape=(None,), dtype="int64")
        subj = f.create_dataset("subject", shape=(0,), maxshape=(None,), dtype="int64")
        for s in subjects:
            npz = os.path.join(cache_dir, f"sub{s:02d}.npz")
            if not os.path.exists(npz):
                print(f"  [{name}] subject {s}: no cache, skipped", file=sys.stderr)
                continue
            with np.load(npz) as z:
                X, y = z["data"], z["label"]
            n = len(y)
            data.resize(total + n, axis=0); data[total:total + n] = X
            label.resize(total + n, axis=0); label[total:total + n] = y
            subj.resize(total + n, axis=0); subj[total:total + n] = s
            counts += np.bincount(y, minlength=len(CLASS_NAMES))
            total += n
        f.create_dataset("channel_names",
                         data=np.array([c.encode() for c in CHANNELS], dtype="S32"))
    share = (counts / max(counts.sum(), 1) * 100)
    print(f"  {name:5s} {len(subjects):2d} subjects  {total:6d} windows  "
          + "  ".join(f"{c}={v}({p:.0f}%)" for c, v, p in zip(CLASS_NAMES, counts, share)))
    return {"subjects": subjects, "windows": total, "class_counts": counts.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default=None,
                   help="per-subject .npz cache (default: <out-dir>/cache)")
    p.add_argument("--stage", choices=["cache", "split", "all"], default="all")
    p.add_argument("--split", choices=["holdout", "eegpt-fold"], default="holdout")
    p.add_argument("--fold", type=int, default=0, help="0-9, for --split eegpt-fold")
    p.add_argument("--subjects", default=None,
                   help="comma-separated subject ids to cache; default is the "
                        "64 EEGPT runs its folds over")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]
    else:
        subjects = list(EEGPT_SUBJECTS)
    bad = sorted(set(subjects) & MISSING)
    if bad:
        raise SystemExit(f"subjects {bad} are not in Sleep-EDFx SC")

    if args.stage in ("cache", "all"):
        print(f"Caching {len(subjects)} subjects into {cache_dir}")
        print("The first subject downloads Sleep-EDFx SC through MNE; expect a wait.")
        from tqdm import tqdm
        failed = []
        for s in tqdm(subjects, ncols=100):
            try:
                cache_subject(s, cache_dir, args.overwrite)
            except SystemExit:
                raise
            except Exception as exc:                       # noqa: BLE001
                # One unreadable recording should not cost the whole corpus,
                # but it must be named -- a silently short dataset looks like
                # a modelling result.
                failed.append((s, repr(exc)))
        if failed:
            print(f"\n{len(failed)} subject(s) failed:", file=sys.stderr)
            for s, e in failed:
                print(f"  subject {s}: {e[:120]}", file=sys.stderr)

    if args.stage in ("split", "all"):
        if args.split == "holdout":
            split = holdout_split(subjects)
            tag = "holdout"
        else:
            split = eegpt_fold_split(args.fold)
            tag = f"eegpt-fold{args.fold}"
        print(f"\nSplit ({tag}), by subject:")
        meta = {"split": tag, "fs": FS, "window_samples": WINDOW_SAMPLES,
                "channels": CHANNELS, "classes": CLASS_NAMES, "splits": {}}
        for name in ("train", "val", "test"):
            meta["splits"][name] = write_split(name, split[name], cache_dir, args.out_dir)
        overlap = (set(split["train"]) & set(split["val"])) | \
                  (set(split["train"]) & set(split["test"])) | \
                  (set(split["val"]) & set(split["test"]))
        assert not overlap, f"subject appears in two splits: {sorted(overlap)}"
        with open(os.path.join(args.out_dir, "split.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nWrote {args.out_dir}/{{train,val,test}}.h5 and split.json")


if __name__ == "__main__":
    main()
