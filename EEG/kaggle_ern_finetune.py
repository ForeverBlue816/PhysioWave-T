#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KaggleERN (BCI Challenge @ NER 2015) -> labelled HDF5, on EEGPT's setup.
-------------------------------------------------------------------------
    python EEG/kaggle_ern_finetune.py \\
        --raw-dir $PW_DATA_EEG/kaggle_ern --out-dir $PW_DATA_EEG/ern_f0 --fold 0

SOURCE. Kaggle competition `inria-bci-challenge`, fetched by
EEG/download_kaggle_ern.sh. Expected layout -- the one EEGPT's readme gives:

    <raw-dir>/TrainLabels.csv        IdFeedBack,Prediction   (16 subjects)
    <raw-dir>/true_labels.csv        the 10 test subjects' labels
    <raw-dir>/SampleSubmission.csv   the test IdFeedBack order (optional)
    <raw-dir>/train/Data_S02_Sess01.csv ...
    <raw-dir>/test/Data_S01_Sess01.csv  ...

Each CSV is one session at 200 Hz: a Time column, the 56 EEG channels, EOG,
and FeedBackEvent (1 at the sample the feedback appeared). The channel names
are read from the header rather than assumed.

The task: after each spelled letter the subject sees feedback, and the label
says whether the speller got it right (1) or wrong (0). An error elicits an
error-related potential. The classes are unbalanced -- roughly seven correct to
three wrong -- which is why EEGPT reports balanced accuracy and AUROC.

WHAT IS REPRODUCED from EEGPT's `read_kaggle_ern_{train,test}` and
`linear_probe_EEGPT_KaggleERN.py`:

    epoch       -0.7 s to +1.3 s around feedback onset (tmin=-0.7, tlen=2)
    reference   common average (their forward subtracts the channel mean)
    folds       their four: each trains on 12 of the 16 training subjects and
                scores the 10 Kaggle test subjects [1,3,4,5,8,9,10,15,19,25]
    labels      TrainLabels.csv for the training subjects, true_labels.csv
                for the test subjects
    S22 Sess05  EEGPT skips it as an "error file". Here any session whose
                feedback count disagrees with its label count is skipped and
                named -- S22 Sess05 included, if that is what is wrong with it
    channels    `--channels eegpt19` gives their 19 10-20 electrodes, in their
                order. The default is all 56: this model builds its own
                frontend and embeds channels by name, so nothing constrains it
                to their subset, exactly as with P300's 64

WHAT DIFFERS: EEGPT scales each trial by its own min-max into [-1, 1]. That
divides away the amplitude an error-related negativity is measured by, so here
the normalisation is a per-channel z-score over the whole session, after a
0.5 Hz high-pass (the pretraining pipeline's) on the continuous signal --
`--highpass 0` removes it. Their four folds VALIDATE on the test subjects; here
the four training subjects each fold leaves out become the validation set, so
the ten test subjects select nothing. See EEG/eegpt_bench_common.py.

Output: see EEG/eegpt_bench_common.py. 512 samples at 256 Hz per epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eegpt_bench_common as bc                                  # noqa: E402

FS_IN = 200.0
TMIN_S, TLEN_S = -0.7, 2.0
CLASSES = ["error", "correct"]
SESSIONS = [1, 2, 3, 4, 5]
TRAIN_SUBJECTS = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
TEST_SUBJECTS = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
# EEGPT's Folds dict, 1-based there, 0-based here. Each is the training set;
# the four training subjects it omits are this pipeline's validation set.
EEGPT_FOLDS = [
    [12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
    [2, 6, 7, 11, 17, 18, 20, 21, 22, 23, 24, 26],
    [2, 6, 7, 11, 12, 13, 14, 16, 22, 23, 24, 26],
    [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21],
]
EEGPT19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
           "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def session_path(raw_dir: str, split: str, s: int, k: int) -> str:
    return os.path.join(raw_dir, split, f"Data_S{s:02d}_Sess{k:02d}.csv")


def fb_id(s: int, k: int, fb: int) -> str:
    return f"S{s:02d}_Sess{k:02d}_FB{fb:03d}"


def read_session(path: str):
    """``(eeg [C, T] float64, channel names, feedback sample indices)``."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = list(df.columns)
    if cols[0].strip().lower() != "time" or "FeedBackEvent" not in cols:
        raise SystemExit(f"{path}: unexpected header {cols[:3]} ... {cols[-3:]}")
    eeg_cols = [c for c in cols[1:] if c not in ("EOG", "FeedBackEvent")]
    x = df[eeg_cols].to_numpy(dtype=np.float64).T
    onsets = np.flatnonzero(df["FeedBackEvent"].to_numpy() > 0)
    return x, [c.strip() for c in eeg_cols], onsets


def train_labels(raw_dir: str) -> dict:
    import pandas as pd
    path = os.path.join(raw_dir, "TrainLabels.csv")
    df = pd.read_csv(path)
    return dict(zip(df["IdFeedBack"].astype(str), df["Prediction"].astype(int)))


def test_labels(raw_dir: str, order: list) -> dict:
    """``{IdFeedBack: label}`` for the test subjects.

    By ID when either file carries one. Only if neither does is the label
    column read by POSITION against ``order`` -- the (subject, session,
    feedback) sequence EEGPT consumes it in -- and the counts must then agree
    exactly, because a positional read that is off by one is a label file
    shifted by one, silently.
    """
    import pandas as pd
    path = os.path.join(raw_dir, "true_labels.csv")
    if not os.path.isfile(path):
        raise SystemExit(
            f"no {path}.\n  The test subjects' labels are not in TrainLabels.csv"
            f" -- EEGPT scores against true_labels.csv. Check what the Kaggle "
            f"download contained (EEG/download_kaggle_ern.sh lists it).")
    df = pd.read_csv(path)
    lab_col = next((c for c in df.columns
                    if c.lower() in ("label", "prediction", "true_label")), None)
    if lab_col is None:
        raise SystemExit(f"{path}: no label column in {list(df.columns)}")
    labels = df[lab_col].astype(int).to_numpy()
    if "IdFeedBack" in df.columns:
        return dict(zip(df["IdFeedBack"].astype(str), labels))
    sub = os.path.join(raw_dir, "SampleSubmission.csv")
    if os.path.isfile(sub):
        ids = pd.read_csv(sub)["IdFeedBack"].astype(str).tolist()
        if len(ids) != len(labels):
            raise SystemExit(f"{path} has {len(labels)} labels and "
                             f"SampleSubmission.csv {len(ids)} ids")
        return dict(zip(ids, labels))
    if len(order) != len(labels):
        raise SystemExit(
            f"{path} has {len(labels)} labels and the test sessions have "
            f"{len(order)} feedback events; a positional match would be wrong")
    print("  NOTE: true_labels.csv has no ids and SampleSubmission.csv is "
          "absent; matching by position, as EEGPT does", file=sys.stderr)
    return dict(zip(order, labels))


def prepare_session(x: np.ndarray, names: list, onsets: np.ndarray,
                    keep: list, highpass: float):
    """One session -> ``(epochs [N, C, 512], kept feedback indices)``."""
    lower = {n.lower(): i for i, n in enumerate(names)}
    missing = [c for c in keep if c.lower() not in lower]
    if missing:
        raise SystemExit(f"channels {missing} not in the recording {names}")
    x = x[[lower[c.lower()] for c in keep]]
    if highpass:
        x = bc.butter_filter(x, FS_IN, highpass, None)
    x = bc.resample_to(x, FS_IN)
    x = bc.session_zscore(x)
    x = bc.common_average(x)
    length = int(round(TLEN_S * bc.FS_OUT))
    starts = np.round(onsets * bc.FS_OUT / FS_IN + TMIN_S * bc.FS_OUT).astype(np.int64)
    ok = (starts >= 0) & (starts + length <= x.shape[1])
    X = bc.epoch_at(x, starts[ok], length)
    return bc.clip(X).astype(np.float32), np.flatnonzero(ok)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fold", type=int, default=0, choices=range(len(EEGPT_FOLDS)),
                   help="EEGPT's fold, 0-based (their Folds[fold + 1])")
    p.add_argument("--channels", default="all", choices=["all", "eegpt19"],
                   help="all 56 (default) or EEGPT's 19")
    p.add_argument("--highpass", type=float, default=0.5,
                   help="Hz, on the continuous session; 0 disables")
    p.add_argument("--train-subjects", default=None,
                   help="override, comma-separated -- for smoke tests")
    p.add_argument("--test-subjects", default=None,
                   help="override, comma-separated -- for smoke tests")
    args = p.parse_args(argv)

    train = EEGPT_FOLDS[args.fold]
    val = [s for s in TRAIN_SUBJECTS if s not in train]
    test = TEST_SUBJECTS
    if args.train_subjects:
        pool = [int(v) for v in args.train_subjects.split(",")]
        train, val = pool[:-1], pool[-1:]
    if args.test_subjects:
        test = [int(v) for v in args.test_subjects.split(",")]
    split = {"train": train, "val": val, "test": test}

    # The montage: from the first file's header, so a column order different
    # from the one EEGPT hard-codes cannot silently relabel electrodes.
    first = session_path(args.raw_dir, "train", train[0], 1)
    if not os.path.isfile(first):
        raise SystemExit(f"missing {first}\n  fetch the data with: "
                         f"bash EEG/download_kaggle_ern.sh {args.raw_dir}")
    _, names, _ = read_session(first)
    keep = EEGPT19 if args.channels == "eegpt19" else names
    bc.check_vocabulary(keep)

    print(f"KaggleERN: fold {args.fold}  train {train}  val {val}  test {test}")
    print(f"  {len(keep)} channels, feedback {TMIN_S:+g}..{TMIN_S + TLEN_S:+g} s, "
          f"high-pass {args.highpass:g} Hz, session z-score, CAR, {bc.FS_OUT} Hz")

    tr_lab = train_labels(args.raw_dir)
    te_lab = None
    skipped, dropped = [], 0
    provenance = {
        "dataset": "KaggleERN (inria-bci-challenge)",
        "epoch": f"feedback {TMIN_S:+g} s, {TLEN_S:g} s long",
        "highpass_hz": args.highpass, "car": True,
        "normalisation": "z-score per channel per session",
        "clip_sigma": bc.CLIP_SIGMA, "sampling_rate": bc.FS_OUT,
        "resampler": "scipy.signal.resample_poly", "fold": args.fold,
        "eegpt_fold": args.fold + 1, "split": split, "channels": keep,
        "protocol": "EEGPT's fold; left-out training subjects validate, "
                    "test subjects select nothing",
    }

    rows = []
    for name, subs in split.items():
        folder = "test" if name == "test" else "train"
        X, Y, S, K = [], [], [], []
        for s in subs:
            for k in SESSIONS:
                path = session_path(args.raw_dir, folder, s, k)
                if not os.path.isfile(path):
                    skipped.append((os.path.basename(path), "missing"))
                    continue
                x, names_k, onsets = read_session(path)
                if names_k != names:
                    raise SystemExit(f"{path}: channel order differs from {first}")
                ids = [fb_id(s, k, i) for i in range(1, len(onsets) + 1)]
                if folder == "test":
                    if te_lab is None:
                        te_lab = test_labels(args.raw_dir, test_order(args.raw_dir, test))
                    lab = te_lab
                else:
                    lab = tr_lab
                have = [i for i in ids if i in lab]
                if len(have) != len(ids) or not ids:
                    skipped.append((os.path.basename(path),
                                    f"{len(onsets)} feedback events, "
                                    f"{len(have)} labels"))
                    continue
                Xs, kept = prepare_session(x, names, onsets, keep, args.highpass)
                dropped += len(ids) - len(kept)
                X.append(Xs)
                Y.append(np.array([lab[ids[i]] for i in kept], dtype=np.int64))
                S.append(np.full(len(kept), s)); K.append(np.full(len(kept), k))
        rows.append(bc.write_split(os.path.join(args.out_dir, f"{name}.h5"),
                                   X, Y, S, K, keep, CLASSES, provenance))

    for f, why in skipped:
        print(f"  SKIPPED {f}: {why}", file=sys.stderr)
    if dropped:
        print(f"  dropped {dropped} epoch(s) that ran off the end of a session",
              file=sys.stderr)
    bc.print_summary(rows)
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"provenance": provenance, "files": rows,
                   "skipped_sessions": skipped, "dropped_epochs": dropped}, f, indent=2)
    print(f"  wrote {args.out_dir}/{{train,val,test}}.h5  (--num-classes 2)")
    return 0


def test_order(raw_dir: str, subjects: list) -> list:
    """Every test feedback id, in the order EEGPT consumes true_labels.csv."""
    order = []
    for s in subjects:
        for k in SESSIONS:
            path = session_path(raw_dir, "test", s, k)
            if os.path.isfile(path):
                n = len(read_session(path)[2])
                order += [fb_id(s, k, i) for i in range(1, n + 1)]
    return order


if __name__ == "__main__":
    sys.exit(main())
