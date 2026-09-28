#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BCI Competition IV 2a / 2b (motor imagery) -> labelled HDF5, on EEGPT's setup.
------------------------------------------------------------------------------
    python EEG/bcic_iv2_finetune.py --dataset 2a \\
        --raw-dir $PW_DATA_EEG/bcic_iv2a --out-dir $PW_DATA_EEG/bcic2a_f0 --fold 0

SOURCE. BNCI Horizon 2020's copies, data sets 001-2014 (= IV-2a) and 004-2014
(= IV-2b), fetched by EEG/download_eegpt_benchmarks.py. They are the same
recordings as the BBCI competition GDFs EEGPT reads, with two practical
differences: they download without a registration form, and the labels of the
evaluation sessions are inside the files rather than in a separate archive.

WHAT IS REPRODUCED from EEGPT's `downstream/Data_process` and
`linear_probe_EEGPT_BCIC2{A,B}.py`:

    epoch       0 to 4 s after the cue. The BNCI `trial` field marks the start
                of a trial, and the cue follows 2 s later on 2a and 3 s later
                on 2b (the paradigm description, and MOABB's intervals [2, 6]
                and [3, 7.5]).
    filter      0-38 Hz: a 4th-order zero-phase Butterworth low-pass, their
                `epochs.filter(0, 38, method='iir')`
    alignment   Euclidean alignment, once per session file (T, then E)
    reference   common average, AFTER alignment -- their
                `temporal_interpolation(use_avg=True)`, which runs after EA
    channels    2a: the 22 EEG electrodes (EOG dropped). 2b: C3, Cz, C4.
    classes     2a: left hand, right hand, feet, tongue.  2b: left, right.
    trials      all of them. The files flag artefact trials; EEGPT keeps them
                (their `reject` defaults to False) and so does this.
    folds       leave-one-subject-out over nine subjects, both sessions of the
                held-out subject as its test data

WHAT DIFFERS, and why, is in EEG/eegpt_bench_common.py: session-level rather
than per-trial amplitude normalisation, polyphase resampling rather than
nearest-neighbour stretching, and a held-out test subject that selects nothing
(so a 7/1/1 subject split rather than their 8/1). Two dataset-specific notes:

  * 2a's exponential moving standardisation (braindecode's) is not applied.
    It is a per-channel running z-score within each trial -- per-trial
    normalisation again -- and EA already sets the scale.
  * 2b's three channels are BIPOLAR derivations centred on C3, Cz and C4, not
    monopolar electrodes. They are named C3/Cz/C4 because that is where they
    sit, which is what the channel-name embedding keys on; the common average
    of three bipolar channels is what EEGPT computes, and is reproduced as such
    rather than because it is physiologically meaningful.

Output: see EEG/eegpt_bench_common.py. 1024 samples at 256 Hz per epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eegpt_bench_common as bc                                  # noqa: E402

DATASETS = {
    "2a": {
        "prefix": "A0{s}{part}.mat",
        "parts": ("T", "E"),
        "n_eeg": 22,
        "channels": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
                     "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                     "CP3", "CP1", "CPz", "CP2", "CP4",
                     "P1", "Pz", "P2", "POz"],
        "classes": ["left_hand", "right_hand", "feet", "tongue"],
        "cue_s": 2.0,
    },
    "2b": {
        "prefix": "B0{s}{part}.mat",
        "parts": ("T", "E"),
        "n_eeg": 3,
        "channels": ["C3", "Cz", "C4"],
        "classes": ["left_hand", "right_hand"],
        "cue_s": 3.0,
    },
}
SUBJECTS = list(range(1, 10))
FS_IN = 250.0
EPOCH_S = 4.0
LOWPASS_HZ = 38.0


def load_file(path: str, spec: dict):
    """One BNCI session file -> ``(epochs [N, C, 1024], labels [N], run [N])``.

    Filtered and resampled per RUN, continuous, before epoching. Runs with no
    trials -- 2a's three EOG calibration runs -- are skipped.
    """
    import scipy.io as sio

    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = m["data"]
    runs = list(runs) if isinstance(runs, np.ndarray) else [runs]
    length = int(round(EPOCH_S * bc.FS_OUT))
    X_all, y_all, r_all = [], [], []
    for ri, run in enumerate(runs):
        trial = np.atleast_1d(run.trial).astype(np.int64)
        y = np.atleast_1d(run.y).astype(np.int64)
        if trial.size == 0 or y.size == 0:
            continue
        if float(run.fs) != FS_IN:
            raise SystemExit(f"{path} run {ri}: fs {run.fs}, expected {FS_IN}")
        if len(trial) != len(y):
            raise SystemExit(f"{path} run {ri}: {len(trial)} trials, {len(y)} labels")
        x = np.asarray(run.X, dtype=np.float64)[:, :spec["n_eeg"]].T   # [C, T], µV
        x = bc.butter_filter(x, FS_IN, None, LOWPASS_HZ)
        x = bc.resample_to(x, FS_IN)
        # `trial` is 1-based MATLAB indexing into X.
        onsets = np.round((trial - 1 + spec["cue_s"] * FS_IN)
                          * bc.FS_OUT / FS_IN).astype(np.int64)
        X_all.append(bc.epoch_at(x, onsets, length))
        y_all.append(y - 1)
        r_all.append(np.full(len(y), ri, dtype=np.int64))
    if not X_all:
        raise SystemExit(f"{path}: no runs with trials")
    return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(r_all)


def prepare_subject(raw_dir: str, dataset: str, subject: int, ea: bool, car: bool):
    """Both session files of one subject, each normalised on its own."""
    spec = DATASETS[dataset]
    out = []
    for si, part in enumerate(spec["parts"]):
        path = os.path.join(raw_dir, spec["prefix"].format(s=subject, part=part))
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}\n  fetch it with: python "
                             f"EEG/download_eegpt_benchmarks.py --dataset {dataset} "
                             f"--dest {raw_dir}")
        X, y, _ = load_file(path, spec)
        # Per session FILE, as EEGPT's get_data applies EA(session_1) and
        # EA(session_2). EA fixes the scale; without it a per-channel z-score
        # over the session does. The common average comes AFTER, as theirs
        # does -- before, it leaves EA a rank-deficient covariance to invert.
        X = bc.euclidean_alignment(X) if ea else bc.session_zscore(X)
        if car:
            X = bc.common_average(X)
        X = bc.clip(X).astype(np.float32)
        n_cls = len(spec["classes"])
        if y.min() < 0 or y.max() >= n_cls:
            raise SystemExit(f"{path}: labels {np.unique(y)} outside 0..{n_cls - 1}")
        out.append((X, y, si))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--raw-dir", required=True,
                   help="directory holding A0?T.mat / A0?E.mat (or B0?...)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=0,
                   help="index into --subjects of the held-out TEST subject")
    p.add_argument("--subjects", default=",".join(map(str, SUBJECTS)),
                   help="comma-separated subject numbers (default: all nine)")
    p.add_argument("--no-ea", action="store_true",
                   help="skip Euclidean alignment; z-score per session instead")
    p.add_argument("--no-car", action="store_true",
                   help="skip the common-average reference")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    spec = DATASETS[args.dataset]
    bc.check_vocabulary(spec["channels"])
    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    split = bc.loso_split(subjects, args.fold, args.seed)
    ea, car = not args.no_ea, not args.no_car

    print(f"BCIC-IV-{args.dataset}: fold {args.fold}  "
          f"train {split['train']}  val {split['val']}  test {split['test']}")
    print(f"  {len(spec['channels'])} channels, cue+0..{EPOCH_S:g} s, "
          f"low-pass {LOWPASS_HZ:g} Hz, CAR {car}, EA {ea}, {bc.FS_OUT} Hz")

    provenance = {
        "dataset": f"BCIC-IV-{args.dataset}",
        "source": "BNCI Horizon 2020 " + ("001-2014" if args.dataset == "2a" else "004-2014"),
        "epoch": f"cue + [0, {EPOCH_S}) s, cue at trial + {spec['cue_s']} s",
        "lowpass_hz": LOWPASS_HZ, "car": car, "euclidean_alignment": ea,
        "normalisation": "EA per session file" if ea else "z-score per session",
        "clip_sigma": bc.CLIP_SIGMA, "sampling_rate": bc.FS_OUT,
        "resampler": "scipy.signal.resample_poly", "fold": args.fold,
        "split": split, "protocol": "leave-one-subject-out, test selects nothing",
    }

    rows = []
    for name, subs in split.items():
        X, Y, S, K = [], [], [], []
        for s in subs:
            for Xs, ys, si in prepare_subject(args.raw_dir, args.dataset, s, ea, car):
                X.append(Xs); Y.append(ys)
                S.append(np.full(len(ys), s)); K.append(np.full(len(ys), si))
        rows.append(bc.write_split(os.path.join(args.out_dir, f"{name}.h5"),
                                   X, Y, S, K, spec["channels"], spec["classes"],
                                   provenance))
    bc.print_summary(rows)
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"provenance": provenance, "files": rows}, f, indent=2)
    print(f"  wrote {args.out_dir}/{{train,val,test}}.h5  "
          f"(--num-classes {len(spec['classes'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
