#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PhysioNet ERP-BCI (PhysioP300) -> labelled HDF5 for binary P300 detection.
--------------------------------------------------------------------------
Reproduces EEGPT's `datasets/downstream/prepare_PhysioNetP300.py` where it can,
so the result sits next to their published row rather than beside it:

    * source        PhysioNet erpbci 1.0.0, EDF, 2048 Hz, 64 EEG channels
                    (+ 6 ocular/ear channels, dropped)
    * channels      the 58 EEGPT's model actually consumes, in their order
    * epochs        tmin=-0.1 s, tmax=2.0 s around each flash, aligned to the
                    EDF's own annotations
    * filtering     IIR 0-120 Hz, applied after epoching, as they do
    * resampling    to 256 Hz, after filtering, as they do
    * labels        1 when the flashed row/column contains the run's target
                    character, else 0 -- a Donchin speller flashes 6 rows and
                    6 columns, 2 of which contain the target, so the positive
                    rate is exactly 1/6

Three deliberate departures, each because our model is not theirs:

1.  WINDOW LENGTH. Their epoch is 538 samples (2.1 s at 256 Hz). 538 = 2 x 269
    with 269 prime, so no sensible patch size divides it, and `patchify` in
    model.py asserts `T % patch_size == 0`. The array is cropped to the first
    512 samples, i.e. -0.1 s to 1.9 s. Nothing is lost that matters: the P300
    is a 250-500 ms deflection and the tail is there for context only.
    512 / 64 = 8 time patches, and 64 samples is 250 ms -- the same patch
    duration EEGPT uses (their d=64 at 256 Hz).

2.  AMPLITUDE. They scale volts by 1e3 and stop, because their classifier
    begins with a learnable per-channel scaling factor ("adaptive spatial
    filter", their Appendix C.2.5) that absorbs whatever the units are. Ours
    has no such layer, and 1e3 leaves the signal at std 0.0265 -- measured on
    s01/rc01 -- which is not a scale a wavelet filter bank should be handed.
    So a per-channel z-score is applied over the whole run.

    Over the *run*, not over each epoch: the P300 is defined by its amplitude
    relative to the non-target response, and per-epoch scaling would divide
    exactly that away. This is the same choice, for the same reason, as the
    recording-level z-score in sleep_edf_finetune.py.

3.  SUBJECT 1. The paper says subjects 8, 10 and 12 are dropped and "the data
    from the remaining 9 subjects were retained". Their preparation script
    loops over [2,3,4,5,6,7,9,11] -- eight subjects, subject 1 missing -- while
    their LOSO loop iterates [1,2,3,4,5,6,7,9,11]. The fold that holds out
    subject 1 therefore has nothing to hold out. This script follows the paper
    and includes subject 1; `--eegpt-subjects` reproduces their eight.

What is NOT reproduced is their evaluation protocol. EEGPT's LOSO loop builds
only `train_dataset` and `valid_dataset`, trains 100 epochs with
`callbacks = [lr_monitor]` -- no checkpoint callback, no early stopping -- and
reports the held-out subject it never held out from selection. Here the
held-out subject is the TEST set and a validation subject is carved out of the
training subjects, so nothing selects on what is scored. That makes our number
pessimistic relative to theirs rather than the other way round.

Two stages, because the decoding is slow and the split is not. `--stage cache`
writes one .npz per subject; `--stage split` turns those into HDF5 for one
LOSO fold. The default runs both.

Usage:
    source scripts/cineca_env.sh          # first: exports PW_DATA_EEG
    source $HOME/pwprep/bin/activate      # second: puts mne on PATH
    python EEG/download_p300.py --dest $PW_DATA_EEG/erpbci
    python EEG/physio_p300_finetune.py \
        --edf-dir $PW_DATA_EEG/erpbci --out-dir $PW_DATA_EEG/p300_f0 --fold 0

Output, matching what finetune.py reads:
    data          (N, 58, 512) float32
    label         (N,)         int64      0=non-target 1=target
    channel_names (58,)        bytes
    subject       (N,)         int64      provenance; unused by the trainer
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

FS_OUT = 256.0
TMIN, TMAX = -0.1, 2.0
LOWPASS_HZ = 120.0
# 538 samples come out of the resample; 512 is the largest patch-friendly
# prefix. See departure 1 in the module docstring.
WINDOW_SAMPLES = 512

CLASS_NAMES = ["nontarget", "target"]

# The 64 EEG channels in the EDF, in EDF order. The remaining six -- EARL,
# EARR, VEOGL, VEOGR, HEOGL, HEOGR -- are ocular and reference and are dropped.
EEG_64 = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]

# EEGPT's `use_channels_names1`, in their order: the 58 electrodes their
# encoder was pretrained on. Kept in this order rather than EDF order because
# it is topographic, and the 2d position embedding indexes rows by position.
CHANNELS_58 = [
    "Fp1", "Fpz", "Fp2",
    "AF3", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

# The nine the paper keeps (8, 10 and 12 dropped, following BENDR).
PAPER_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 11]
# What EEGPT's preparation script actually writes.
EEGPT_PREPARED_SUBJECTS = [2, 3, 4, 5, 6, 7, 9, 11]

VERBOSE = False    # set by --verbose


# --------------------------------------------------------------------------- #
# Stage 1: decode one subject
# --------------------------------------------------------------------------- #
def _epoch_one_run(path: str, channels: list[str]):
    """One EDF run -> (X, y) with X (n, C, 512) float32 and y (n,) int64.

    Returns (None, None) when the run carries no target annotation, which is
    how a calibration or aborted run presents itself.
    """
    import mne

    raw = mne.io.read_raw_edf(path)
    missing = [c for c in channels if c not in raw.ch_names]
    if missing:
        raise RuntimeError(f"{os.path.basename(path)}: missing channels {missing}")
    raw.pick(channels)

    events, event_id = mne.events_from_annotations(raw)
    # Annotations are '#TgtX_RCnn_SOAnn' (X is the target character), '#start',
    # '#end', and one code per flash naming the six characters in that row or
    # column, e.g. 'ABCDEF' or 'AGMSY5'.
    target_char = None
    for k in event_id:
        if k[:4] == "#Tgt":
            target_char = k[4]
            break
    if target_char is None:
        return None, None
    code_of = {v: k for k, v in event_id.items()}

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=TMIN, tmax=TMAX,
                        event_repeated="drop", preload=True, proj=False,
                        baseline=None)
    # Filter and resample after epoching, in that order, because that is what
    # EEGPT does; the 120 Hz lowpass doubles as the anti-alias filter for the
    # 2048 -> 256 Hz decimation (Nyquist 128).
    epochs.filter(0, LOWPASS_HZ, method="iir")
    epochs.resample(FS_OUT)

    data = epochs.get_data(copy=False)
    codes = [code_of[e[2]] for e in epochs.events]
    keep = [i for i, c in enumerate(codes) if not c.startswith("#")]
    if not keep:
        return None, None

    X = data[keep][:, :, :WINDOW_SAMPLES].astype(np.float32)
    if X.shape[2] != WINDOW_SAMPLES:
        raise RuntimeError(
            f"{os.path.basename(path)}: {X.shape[2]} samples after resampling, "
            f"expected at least {WINDOW_SAMPLES}"
        )
    y = np.array([1 if target_char in codes[i] else 0 for i in keep], dtype=np.int64)

    # Per-channel z-score over this run. See departure 2 in the docstring.
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True)
    X = (X - mu) / np.maximum(sd, 1e-8)
    return X, y


def cache_subject(subject: int, edf_dir: str, cache_dir: str,
                  channels: list[str], overwrite: bool = False,
                  verbose: bool = False) -> str | None:
    """Decode every run of one subject into a single .npz. Returns its path.

    ``verbose`` is a parameter rather than the module global because the worker
    processes are started with 'spawn', which re-imports this module in the
    child and resets the global to its default.
    """
    out = os.path.join(cache_dir, f"sub{subject:02d}.npz")
    if os.path.exists(out) and not overwrite:
        return out

    if not verbose:
        # 245 runs of MNE's filter-design banner scrolls the progress bar off
        # the screen and buries anything that actually went wrong.
        import mne
        mne.set_log_level("ERROR")

    sub_dir = os.path.join(edf_dir, f"s{subject:02d}")
    if not os.path.isdir(sub_dir):
        raise RuntimeError(
            f"no directory {sub_dir}\n"
            f"  fetch it with: python EEG/download_p300.py --dest {edf_dir}"
        )
    runs = sorted(f for f in os.listdir(sub_dir) if f.endswith(".edf"))
    if not runs:
        raise RuntimeError(f"{sub_dir} holds no .edf files")

    Xs, ys, skipped = [], [], []
    for r in runs:
        X, y = _epoch_one_run(os.path.join(sub_dir, r), channels)
        if X is None:
            # No target annotation, or no flash events. Named rather than
            # dropped in silence: a subject that is quietly short by a few runs
            # looks like data, not like a problem.
            skipped.append(r)
            continue
        Xs.append(X)
        ys.append(y)
    if not Xs:
        raise RuntimeError(f"subject {subject}: no run carried a target annotation")
    if skipped:
        print(f"  subject {subject}: {len(skipped)} of {len(runs)} runs had no "
              f"target annotation and were skipped ({', '.join(skipped[:5])}"
              f"{', ...' if len(skipped) > 5 else ''})", file=sys.stderr)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    np.savez_compressed(out, data=X, label=y, n_runs=len(Xs), n_skipped=len(skipped))
    return out


# --------------------------------------------------------------------------- #
# Stage 2: leave-one-subject-out splits
# --------------------------------------------------------------------------- #
def loso_split(subjects: list[int], fold: int, seed: int = 7) -> dict[str, list[int]]:
    """Subject ``subjects[fold]`` is the test set; one more becomes validation.

    EEGPT holds out one subject and calls it validation, then reports it. Here
    it is the test set and never selects anything; a second subject, drawn
    deterministically from the rest, is the validation set. Train/val/test is
    therefore 7/1/1 of the nine rather than their 8/1.
    """
    if not 0 <= fold < len(subjects):
        raise SystemExit(f"--fold must be in [0, {len(subjects)}), got {fold}")
    if len(subjects) < 3:
        raise SystemExit(
            f"a subject-level train/val/test split needs at least 3 subjects, "
            f"got {len(subjects)}. For a pipeline smoke test pass three, e.g. "
            f"--subjects 1,2,3."
        )
    test = [subjects[fold]]
    rest = [s for s in subjects if s != subjects[fold]]
    rng = np.random.default_rng(seed + fold)
    val = [rest[int(rng.integers(len(rest)))]]
    train = [s for s in rest if s not in set(val)]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_split(name: str, subjects: list[int], cache_dir: str, out_dir: str,
                channels: list[str]) -> dict:
    """Concatenate cached subjects into one HDF5, growing it incrementally."""
    path = os.path.join(out_dir, f"{name}.h5")
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    total = 0
    C = len(channels)
    with h5py.File(path, "w") as f:
        data = f.create_dataset("data", shape=(0, C, WINDOW_SAMPLES),
                                maxshape=(None, C, WINDOW_SAMPLES), dtype="float32",
                                chunks=(64, C, WINDOW_SAMPLES), compression="lzf")
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
                         data=np.array([c.encode() for c in channels], dtype="S32"))
    share = counts / max(counts.sum(), 1) * 100
    print(f"  {name:5s} {len(subjects):2d} subjects  {total:6d} epochs  "
          + "  ".join(f"{c}={v}({p:.1f}%)" for c, v, p in zip(CLASS_NAMES, counts, share)))
    return {"subjects": subjects, "epochs": total, "class_counts": counts.tolist()}


def preflight() -> None:
    """Import MNE once, before 245 runs fail the same way."""
    try:
        import mne  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"{exc}\n\nThis script needs mne:\n\n"
            f"    source scripts/cineca_env.sh\n"
            f"    source $HOME/pwprep/bin/activate\n\n"
            f"  In that order. cineca_env.sh activates its own virtualenv, so\n"
            f"  sourcing it second replaces the preparation environment -- the\n"
            f"  prompt flips from (pwprep) to (pw) and this import is what\n"
            f"  fails. Check with:  python -c \'import sys; print(sys.prefix)\'"
        ) from exc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edf-dir", required=True,
                   help="directory holding s01/ ... s12/, from download_p300.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default=None,
                   help="per-subject .npz cache (default: <edf-dir>/cache). Keep it "
                        "outside --out-dir so the nine folds share one decode.")
    p.add_argument("--stage", choices=["cache", "split", "all"], default="all")
    p.add_argument("--fold", type=int, default=0,
                   help="index into the subject list; that subject is the TEST set")
    p.add_argument("--subjects", default=None,
                   help="comma-separated ids; default is the paper's nine")
    p.add_argument("--eegpt-subjects", action="store_true",
                   help="use the eight EEGPT's prepare script actually writes "
                        "(subject 1 omitted) instead of the paper's nine")
    p.add_argument("--all-channels", action="store_true",
                   help="keep all 64 EEG channels instead of the 58 EEGPT uses")
    p.add_argument("--allow-missing", action="store_true",
                   help="write the split even though some subjects have no cache. "
                        "Only right for a pipeline smoke test -- otherwise every "
                        "fold is scored on a different corpus.")
    p.add_argument("--jobs", type=int, default=2,
                   help="subjects decoded in parallel; each is ~20 runs and holds "
                        "~0.5 GiB at peak. Default 2 rather than 4 because a login "
                        "node's cgroup kills the workers before it refuses them. "
                        "Workers are spawned, not forked (see the cache stage). "
                        "1 runs serially in this process -- the thing to try first "
                        "if anything looks stuck or gets killed.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    for label, path in (("--out-dir", args.out_dir), ("--edf-dir", args.edf_dir),
                        ("--cache-dir", args.cache_dir)):
        if path and os.path.dirname(os.path.abspath(path)) == os.sep:
            raise SystemExit(
                f"{label} is '{path}' -- directly under the filesystem root.\n\n"
                f"  That is almost always an unset variable: the shell expands\n"
                f"  $PW_DATA_EEG before this script runs, and it is empty unless\n"
                f"  you have sourced the environment:\n\n"
                f"      source scripts/cineca_env.sh\n"
            )

    channels = EEG_64 if args.all_channels else CHANNELS_58
    cache_dir = args.cache_dir or os.path.join(args.edf_dir, "cache")
    # The channel set changes what is in the cache, so it cannot be shared
    # between the two; keeping them apart avoids a silent shape mismatch.
    cache_dir = os.path.join(cache_dir, f"c{len(channels)}")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.subjects:
        subjects = [int(s) for s in args.subjects.split(",")]
    elif args.eegpt_subjects:
        subjects = list(EEGPT_PREPARED_SUBJECTS)
    else:
        subjects = list(PAPER_SUBJECTS)

    if args.stage in ("cache", "all"):
        preflight()
        print(f"Decoding {len(subjects)} subjects into {cache_dir}")
        print(f"  {len(channels)} channels, {TMIN}..{TMAX} s, IIR 0-{LOWPASS_HZ:.0f} Hz, "
              f"{FS_OUT:.0f} Hz, cropped to {WINDOW_SAMPLES} samples")
        import traceback

        failed = []

        def _note(subject, exc):
            # One unreadable subject should not cost the corpus, but it must be
            # named: a silently short dataset looks like a modelling result.
            # The first failure prints in full, because a bare repr from inside
            # MNE says nothing about which package raised it.
            if not failed:
                print("\n\nFirst failure, in full:\n", file=sys.stderr)
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            failed.append((subject, repr(exc)))

        todo = len(subjects)
        if args.jobs <= 1:
            # No pool at all. Not just max_workers=1: creating the executor is
            # what forks, and the fork is the problem below.
            for i, s in enumerate(subjects, 1):
                print(f"  [{i}/{todo}] subject {s} ...", flush=True)
                try:
                    cache_subject(s, args.edf_dir, cache_dir, channels,
                                  args.overwrite, VERBOSE)
                except Exception as exc:                       # noqa: BLE001
                    _note(s, exc)
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # 'spawn', not the Linux default 'fork'. preflight() has already
            # imported MNE in this process, which brings up numpy/scipy and
            # their BLAS thread pool. fork() copies those threads' locks in
            # whatever state they were in and does not copy the threads, so the
            # first BLAS call in the child -- here, the IIR filter -- blocks on
            # a lock nothing will ever release. The symptom is a progress bar
            # that sits at 0 forever with the workers idle, which is what this
            # replaced. macOS defaults to spawn, so the fork path only ever
            # appears on the cluster.
            ctx = mp.get_context("spawn")
            # One BLAS thread per worker. Each child would otherwise start a
            # pool sized to the whole node and the workers would fight for the
            # same cores -- slower than serial, on a login node shared with
            # everyone else.
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            done = 0
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
                futures = {pool.submit(cache_subject, s, args.edf_dir, cache_dir,
                                       channels, args.overwrite, VERBOSE): s
                           for s in subjects}
                for fut in as_completed(futures):
                    s = futures[fut]
                    done += 1
                    try:
                        fut.result()
                        print(f"  [{done}/{todo}] subject {s} done", flush=True)
                    except Exception as exc:                   # noqa: BLE001
                        print(f"  [{done}/{todo}] subject {s} FAILED", flush=True)
                        _note(s, exc)
        # A worker that is killed rather than raising takes down the pool, and
        # every subject still queued behind it fails with the same
        # BrokenProcessPool -- so the report would blame subjects that were
        # never even started. Retry them here, one at a time in this process,
        # where there is no contention for whatever limit did the killing.
        if failed and args.jobs > 1:
            retry = [s for s, _ in failed]
            print(f"\nRetrying {len(retry)} subject(s) serially: "
                  f"{', '.join(str(s) for s in retry)}", flush=True)
            still = []
            for i, s in enumerate(retry, 1):
                print(f"  [{i}/{len(retry)}] subject {s} ...", flush=True)
                try:
                    cache_subject(s, args.edf_dir, cache_dir, channels,
                                  args.overwrite, VERBOSE)
                except Exception as exc:                       # noqa: BLE001
                    still.append((s, repr(exc)))
            recovered = len(retry) - len(still)
            if recovered:
                print(f"  {recovered} recovered on the serial retry.", flush=True)
            failed = still

        if failed:
            print(f"\n{len(failed)} of {len(subjects)} subject(s) failed.", file=sys.stderr)
            for s, e in failed[:10]:
                print(f"  subject {s}: {e[:160]}", file=sys.stderr)
            if any("BrokenProcessPool" in e for _, e in failed):
                # BrokenProcessPool means the worker was killed, not that it
                # raised: there is no traceback because no Python exception
                # ever happened. On a login node it is a cgroup limit, and the
                # peak here is mne.Epochs(preload=True) holding a whole run at
                # 2048 Hz in float64 -- about 0.5 GiB per worker before the
                # filter and resample make their copies.
                print(
                    "\nBrokenProcessPool means a worker was KILLED, not that it "
                    "raised -- there is\nno traceback because no Python "
                    "exception occurred. On a login node that is\nalmost "
                    "always the per-user memory or CPU-time cgroup.\n\n"
                    "  Decode on a compute node instead. It needs no internet, "
                    "only the EDFs:\n\n"
                    "      srun -N1 -n1 -c8 -t 0:30:00 -A <account> -p <partition> \\\n"
                    "           $HOME/pwprep/bin/python EEG/physio_p300_finetune.py \\\n"
                    "           --edf-dir <dir> --out-dir <dir> --stage cache\n\n"
                    "  Or stay on the login node and drop to --jobs 1.\n"
                    "  Either way the subjects already cached are skipped, so "
                    "nothing is redone.",
                    file=sys.stderr)
            if len(failed) == len(subjects):
                raise SystemExit(
                    "\nEvery subject failed, so this is the environment or the "
                    "download rather than the data.\nThe traceback above names "
                    "the package. Nothing was written."
                )

    if args.stage in ("split", "all"):
        split = loso_split(subjects, args.fold)
        # Refuse before writing anything. A split assembled from whatever
        # happened to be cached is the failure this file exists to prevent: it
        # loads, it trains, and it reports a number for a corpus that is
        # quietly missing subjects. The cache stage prints its failures to
        # stderr, which is one scrollback away from being missed.
        missing = [s for s in subjects
                   if not os.path.exists(os.path.join(cache_dir, f"sub{s:02d}.npz"))]
        if missing and not args.allow_missing:
            raise SystemExit(
                f"\n{len(missing)} of {len(subjects)} subject(s) have no cache: "
                f"{', '.join(str(s) for s in missing)}\n\n"
                f"  Nothing was written. Rerun the cache stage -- subjects "
                f"already cached are skipped,\n  so only the missing ones are "
                f"decoded:\n\n"
                f"      python EEG/physio_p300_finetune.py --edf-dir {args.edf_dir} \\\n"
                f"          --out-dir {args.out_dir} --stage cache --jobs 1\n\n"
                f"  --allow-missing writes the split without them, which is "
                f"only ever right for a\n  pipeline smoke test: every fold "
                f"would then be scored on a different corpus."
            )
        if missing:
            print(f"WARNING: --allow-missing, so {len(missing)} subject(s) are "
                  f"absent from this split: {', '.join(str(s) for s in missing)}",
                  file=sys.stderr)
        print(f"\nLOSO fold {args.fold} (test subject {split['test'][0]}), by subject:")
        meta = {"split": f"loso-fold{args.fold}", "fs": FS_OUT,
                "window_samples": WINDOW_SAMPLES, "tmin": TMIN, "tmax": TMAX,
                "channels": channels, "classes": CLASS_NAMES, "splits": {}}
        for name in ("train", "val", "test"):
            meta["splits"][name] = write_split(name, split[name], cache_dir,
                                               args.out_dir, channels)
        empty = [n for n, m in meta["splits"].items() if m["epochs"] == 0]
        if empty:
            # Writing a 0-epoch HDF5 and printing "Wrote ..." is the failure
            # this script exists to avoid: the trainer would load it, report a
            # loss over nothing, and look like it ran.
            # Remove ALL three, not only the empty ones. Deleting just the
            # empty file leaves a directory holding a train.h5 and a test.h5
            # that look finished, and the next run of the trainer reports
            # "missing val.h5" -- pointing at the one file that is *not* the
            # problem. Worse, those survivors are short: the run that produced
            # this had two subjects fail to cache, so the train.h5 it left
            # behind was missing one of them, and had the validation subject
            # happened to be cached it would have trained on a six-subject
            # corpus without saying so.
            removed = []
            for n in ("train", "val", "test"):
                f_ = os.path.join(args.out_dir, f"{n}.h5")
                if os.path.exists(f_):
                    os.remove(f_)
                    removed.append(n)
            j_ = os.path.join(args.out_dir, "split.json")
            if os.path.exists(j_):
                os.remove(j_)
            raise SystemExit(
                f"\n{', '.join(empty)} came out empty -- no cached subject in "
                f"it had any epochs.\nThe cache stage has not run for those "
                f"subjects, or it failed.\n\n"
                f"  Removed {', '.join(removed)}.h5 and split.json so nothing "
                f"half-built is left behind.\n"
                f"  Fill the cache, then rerun this stage -- cached subjects "
                f"are skipped:\n\n"
                f"      python EEG/physio_p300_finetune.py --edf-dir {args.edf_dir} \\\n"
                f"          --out-dir {args.out_dir} --stage cache --jobs 1"
            )
        overlap = (set(split["train"]) & set(split["val"])) | \
                  (set(split["train"]) & set(split["test"])) | \
                  (set(split["val"]) & set(split["test"]))
        assert not overlap, f"subject appears in two splits: {sorted(overlap)}"
        with open(os.path.join(args.out_dir, "split.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nWrote {args.out_dir}/{{train,val,test}}.h5 and split.json")
        print(f"Train it with: DATA_DIR={args.out_dir} IN_CHANNELS={len(channels)} "
              f"bash EEG/finetune_p300.sh")


if __name__ == "__main__":
    main()
