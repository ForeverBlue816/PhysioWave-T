#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What the three EEGPT benchmark converters share: BCIC-IV-2a, BCIC-IV-2b and
KaggleERN.

    EEG/bcic_iv2_finetune.py    --dataset 2a | 2b
    EEG/kaggle_ern_finetune.py

Each converter reproduces EEGPT's own preparation where it can -- the epoch,
the filter, the channel set, the cross-subject split -- and departs from it
only where their model and ours differ. The departures are the same three for
all of them, stated once here:

1.  AMPLITUDE. EEGPT normalises per TRIAL: exponential moving standardisation
    on 2a, x/10 on 2b, a per-trial min-max to [-1, 1] on KaggleERN. Their
    classifier then opens with a learnable per-channel scaling layer that
    absorbs whatever scale is left. Ours has no such layer, and per-trial
    scaling divides away exactly what these tasks are decided on: the
    left/right power asymmetry over C3/C4 that motor imagery IS, and the
    amplitude of an error-related negativity against a correct-feedback
    response. So the normalisation here is per SESSION -- Euclidean alignment
    on the motor-imagery sets (which EEGPT also applies, and which sets the
    scale as a side effect), a per-channel z-score over the session otherwise.
    This is the choice physio_p300_finetune.py and sleep_edf_finetune.py make,
    for the same reason.

2.  RESAMPLING. EEGPT stretches every epoch to a fixed length with
    `F.interpolate(mode='nearest')`, which is sample duplication: 1001 samples
    at 250 Hz become 1024 by repeating 23 of them. Here the continuous signal
    goes through scipy's polyphase resampler to 256 Hz before it is epoched, so
    a 4 s epoch is 1024 samples because 4 s at 256 Hz is 1024 samples.

3.  EVALUATION. EEGPT's scripts call `trainer.fit(model, train, test)` with no
    checkpoint callback: the held-out subjects are the validation set, scored
    every epoch, and the reported number is read off that curve. Here the
    held-out subjects are a TEST set that selects nothing, and validation is a
    further set of subjects carved out of the training ones. That makes these
    numbers pessimistic relative to theirs, not optimistic. Theirs is also a
    mean over folds (9 for BCIC, 4 for KaggleERN); each converter here builds
    one fold, whose shape matches one of theirs.

The output is what physiowave.train.finetune_main reads:

    data           (N, C, T) float32
    label          (N,)      int64
    subject        (N,)      int64     provenance; the trainer ignores it
    session        (N,)      int64     provenance; the trainer ignores it
    channel_names  (C,)      bytes     the montage, by name, in order
    attrs          sampling_rate, window_samples, provenance (JSON)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FS_OUT = 256
CLIP_SIGMA = 20.0


# --------------------------------------------------------------------------- #
# Signal steps. Every one of them works on a continuous [C, T] array or on a
# stack of epochs [N, C, T]; none of them looks at labels.
# --------------------------------------------------------------------------- #

def common_average(x: np.ndarray) -> np.ndarray:
    """Subtract the mean over channels at every sample.

    EEGPT does this implicitly -- `temporal_interpolation(..., use_avg=True)`
    subtracts `x.mean(dim=-2)` -- so it is part of their preparation even
    though no preparation script names it. It is the LAST spatial step there,
    after Euclidean alignment, and must be here too: see euclidean_alignment.
    """
    return x - x.mean(axis=-2, keepdims=True)


def butter_filter(x: np.ndarray, fs: float, low: Optional[float],
                  high: Optional[float], order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth, on a CONTINUOUS signal.

    MNE's `filter(l_freq, h_freq, method='iir')` is a 4th-order Butterworth run
    forward and backward, which is what this is. EEGPT applies it to 4 s epochs;
    here it runs on the whole run before epoching, so the filter's edge
    transient falls at the ends of a recording rather than at the ends of every
    trial.
    """
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    if low and high:
        sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    elif high:
        sos = butter(order, high / nyq, btype="lowpass", output="sos")
    elif low:
        sos = butter(order, low / nyq, btype="highpass", output="sos")
    else:
        return x
    return sosfiltfilt(sos, x, axis=-1)


def resample_to(x: np.ndarray, fs_in: float, fs_out: int = FS_OUT) -> np.ndarray:
    """Polyphase resampling along the last axis -- the pretraining resampler."""
    from physiowave.eeg_c1.preprocess import resample_to as _r
    return _r(x, fs_in, fs_out)


def euclidean_alignment(epochs: np.ndarray) -> np.ndarray:
    """``R^{-1/2} x`` with ``R`` the mean trial covariance of this set.

    EEGPT's `Data_process.utils.EA`, applied the way they apply it: once per
    session file, over every trial in it. Label-free, so it is applied to the
    test subject's sessions too -- as theirs is -- using that subject's own
    unlabelled trials. After it the mean spatial covariance of the session is
    the identity, which is the session-level normalisation this pipeline wants
    anyway.

    MUST RUN BEFORE the common-average reference, which is EEGPT's order too
    (EA in `get_data`, CAR later in `temporal_interpolation`). A common average
    makes the channels sum to zero at every sample, so R loses a rank: on
    2b's three channels it is singular outright, and on 2a's twenty-two it is
    singular to within rounding, where R^{-1/2} multiplies the all-ones
    direction by ~1e5 and the result is finite only by luck. Hence eigh and an
    explicit conditioning check, rather than a finiteness test that luck passes.
    """
    x = epochs.astype(np.float64)
    R = np.einsum("nct,ndt->cd", x, x) / len(x)
    w, V = np.linalg.eigh((R + R.T) / 2.0)
    if w.min() <= w.max() * 1e-8:
        raise ValueError(
            f"Euclidean alignment: the session covariance is rank-deficient "
            f"(eigenvalues {w.min():.3g} .. {w.max():.3g}). A common-average "
            f"reference applied before EA does this; so does a flat or "
            f"duplicated channel.")
    # errstate for the same reason as preprocess.detrend: numpy's matmul on
    # some BLAS builds reports divide-by-zero on well-conditioned input. The
    # conditioning is checked above and the outcome below, which is what
    # actually matters.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Rm = (V / np.sqrt(w)) @ V.T
        out = np.einsum("dc,nct->ndt", Rm, x)
    if not (np.all(np.isfinite(Rm)) and np.all(np.isfinite(out))):
        raise ValueError("Euclidean alignment produced non-finite values")
    return out


def session_zscore(x: np.ndarray) -> np.ndarray:
    """Per channel, over the whole session: ``[N, C, T]`` or ``[C, T]``.

    NOT per epoch. See point 1 at the top of this file.
    """
    axes = (0, 2) if x.ndim == 3 else (1,)
    mu = x.mean(axis=axes, keepdims=True)
    sd = x.std(axis=axes, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def clip(x: np.ndarray, sigma: float = CLIP_SIGMA) -> np.ndarray:
    """The pretraining pipeline's ±20 clip, so no single artefact dominates."""
    return np.clip(x, -sigma, sigma)


def epoch_at(x: np.ndarray, onsets: Sequence[int], length: int) -> np.ndarray:
    """``[C, T] -> [N, C, length]``. An epoch running off either end is an error.

    Not clipped and not padded. EEGPT's ERN reader clamps the start at 0 and
    the end at the file length, which yields a short epoch that
    `temporal_interpolation` then stretches -- a different duration passed off
    as the same one. None of the three datasets needs that, so it is refused.
    """
    C, T = x.shape
    out = np.empty((len(onsets), C, length), dtype=np.float32)
    for i, s in enumerate(onsets):
        s = int(s)
        if s < 0 or s + length > T:
            raise ValueError(f"epoch {i} at sample {s} (+{length}) runs off a "
                             f"{T}-sample recording")
        out[i] = x[:, s:s + length]
    return out


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def loso_split(subjects: List[int], fold: int, seed: int = 7) -> Dict[str, List[int]]:
    """Subject ``subjects[fold]`` is the test set; one more becomes validation.

    The same rule as physio_p300_finetune.py. EEGPT's BCIC fold trains on the
    other eight subjects and scores the ninth, validating on it as well; here
    the ninth is the TEST set, and a validation subject is drawn from the eight
    deterministically, so the split is 7/1/1.
    """
    if not 0 <= fold < len(subjects):
        raise SystemExit(f"--fold must be in [0, {len(subjects)}), got {fold}")
    if len(subjects) < 3:
        raise SystemExit(f"a subject-level train/val/test split needs at least "
                         f"3 subjects, got {len(subjects)}")
    test = [subjects[fold]]
    rest = [s for s in subjects if s != subjects[fold]]
    rng = np.random.default_rng(seed + fold)
    val = [rest[int(rng.integers(len(rest)))]]
    train = [s for s in rest if s not in val]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_split(path: str, epochs: List[np.ndarray], labels: List[np.ndarray],
                subjects: List[np.ndarray], sessions: List[np.ndarray],
                channels: Sequence[str], class_names: Sequence[str],
                provenance: Dict) -> Dict:
    """One HDF5 in the layout finetune_main reads. Returns a summary row."""
    import h5py

    X = np.concatenate(epochs).astype(np.float32) if epochs else \
        np.zeros((0, len(channels), 0), np.float32)
    y = np.concatenate(labels).astype(np.int64) if labels else np.zeros(0, np.int64)
    s = np.concatenate(subjects).astype(np.int64) if subjects else np.zeros(0, np.int64)
    k = np.concatenate(sessions).astype(np.int64) if sessions else np.zeros(0, np.int64)
    if len(X) == 0:
        raise SystemExit(f"{path}: no epochs -- refusing to write an empty split")
    if X.shape[1] != len(channels):
        raise SystemExit(f"{path}: {X.shape[1]} channels in the data, "
                         f"{len(channels)} names")
    if not np.all(np.isfinite(X)):
        raise SystemExit(f"{path}: non-finite values in the epochs")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=X, chunks=(min(64, len(X)),) + X.shape[1:],
                         compression="lzf")
        f.create_dataset("label", data=y)
        f.create_dataset("subject", data=s)
        f.create_dataset("session", data=k)
        f.create_dataset("channel_names",
                         data=np.array([c.encode() for c in channels], dtype="S32"))
        f.attrs["sampling_rate"] = float(FS_OUT)
        f.attrs["window_samples"] = int(X.shape[-1])
        f.attrs["class_names"] = json.dumps(list(class_names))
        f.attrs["provenance"] = json.dumps(provenance, default=str)

    counts = np.bincount(y, minlength=len(class_names))
    return {"file": os.path.basename(path), "epochs": int(len(y)),
            "subjects": sorted(set(s.tolist())),
            "class_counts": {c: int(n) for c, n in zip(class_names, counts)},
            "shape": list(X.shape)}


def check_vocabulary(channels: Sequence[str]) -> None:
    """Refuse a montage the channel-name embedding has no rows for.

    An unknown name would resolve to UNK and share one embedding row with every
    other unknown name -- a montage quietly losing its identity.
    """
    from channel_embedding import UNK_ID, channel_id
    unknown = [c for c in channels if channel_id(c) == UNK_ID]
    if unknown:
        raise SystemExit(f"channel(s) not in the vocabulary: {unknown}")


def print_summary(rows: List[Dict]) -> None:
    for r in rows:
        cc = "  ".join(f"{k}={v}" for k, v in r["class_counts"].items())
        print(f"  {r['file']:9s} {r['epochs']:6d} epochs  subjects "
              f"{r['subjects']}  {cc}  shape {tuple(r['shape'])}")
