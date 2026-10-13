#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPDB (Tao et al., Sci. Data 2024) -> aligned EEG + ECG + sEMG windows.

    python Fusion/mpdb_prep.py --raw-dir $R/mpdb --out-dir $D/mpdb

THE DATA. 35 participants in a driving simulator; 59-channel EEG and one
chest ECG electrode (Neuracle, 1 kHz), four limb sEMG channels (NeuSen WM,
1 kHz: right tibialis anterior, right gastrocnemius, both brachioradialis).
Ten simulator events fall into five behaviours, by the marker codes of the
paper's Table 2:

    smooth driving  133              lane change  129 131
    acceleration    135 143          turning      125 127
    deceleration    139 141 145

THE SOURCE is the authors' PREPROCESSED release (figshare 22192831): per
subject EEG_{n}.set, ECG_{n}.set, EMG_{n}.set, EEGLAB epochs of 2 s (0.5 s
before the marker to 1.5 s after), already band-passed. The raw release
(22193119, 23 GB of continuous .set files) is not needed. The three files of
a subject are ALIGNED by their marker sequence: a subject whose files do not
carry the same events in the same order is dropped and named in split.json,
never aligned by guesswork.

PREPROCESSING. The authors' filters stand (re-filtering a 2 s epoch at 0.5 Hz
would put the transient inside the window). Each epoch is resampled to the
encoder rates (EEG 256 Hz, ECG 250 Hz, sEMG 2000 Hz) and normalised the way
each pretraining corpus was (physiowave/fusion/prep_common.py).

SPLIT by subject, 25 / 5 / 5 (seed 42). sEMG from the pedal leg can carry
the label directly (stepping on the pedal IS the behaviour), which is why the
evaluation reports every modality subset and not only the complete one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.fusion.prep_common import (MODALITIES, MultimodalWriter,  # noqa: E402
                                           prepare_window, split_summary,
                                           subject_split)

CLASSES = ["smooth", "acceleration", "deceleration", "lane_change", "turning"]
MARKER_CLASS = {133: 0, 135: 1, 143: 1, 139: 2, 141: 2, 145: 2,
                129: 3, 131: 3, 125: 4, 127: 4}
EMG_CHANNELS = ["r_tibialis_anterior", "r_gastrocnemius",
                "brachioradialis_a", "brachioradialis_b"]
ECG_CHANNELS = ["patch1"]           # one chest patch: the L1_250 route's slot
WINDOW_SECONDS = 2.0                # the release's epoch: -0.5 s .. +1.5 s
SEED = 42
FILE_RE = {"eeg": r"EEG_(\d+)\.set$", "ecg": r"ECG_(\d+)\.set$", "emg": r"EMG_(\d+)\.set$"}


def load_eeglab_epochs(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """``(data [n, C, T], marker codes [n], channel names, sfreq)`` from a .set.

    mne reads both the in-file and the .fdt-backed layouts. The marker code
    is the integer in the event name ('133', 'S133', 'condition 133' all
    give 133).
    """
    import mne
    ep = mne.io.read_epochs_eeglab(path, verbose="error")
    by_id = {v: k for k, v in ep.event_id.items()}
    codes = []
    for ev in ep.events[:, 2]:
        name = by_id.get(int(ev), str(ev))
        m = re.search(r"(\d+)", str(name))
        codes.append(int(m.group(1)) if m else -1)
    return ep.get_data(), np.asarray(codes, np.int64), list(ep.ch_names), float(ep.info["sfreq"])


def find_subjects(raw: str) -> Dict[int, Dict[str, str]]:
    found: Dict[int, Dict[str, str]] = {}
    for p in glob.glob(os.path.join(raw, "**", "*.set"), recursive=True):
        for m, rx in FILE_RE.items():
            mm = re.search(rx, os.path.basename(p), re.I)
            if mm:
                found.setdefault(int(mm.group(1)), {})[m] = p
    return found


def align(codes: Dict[str, np.ndarray]) -> Optional[List[Tuple[int, ...]]]:
    """Epoch indices per modality that describe the same event, or None.

    Identical sequences align 1:1. Otherwise, if every file holds the same
    multiset of codes, the k-th occurrence of a code in one file is matched
    to the k-th in the others -- the files were cut from one synchronised
    session, so order within a code is preserved. Anything else is refused.
    """
    mods = list(codes)
    ref = codes[mods[0]]
    if all(len(codes[m]) == len(ref) and np.array_equal(codes[m], ref) for m in mods):
        return [tuple(i for _ in mods) for i in range(len(ref))]
    if not all(sorted(codes[m].tolist()) == sorted(ref.tolist()) for m in mods):
        return None
    pos = {m: {} for m in mods}
    for m in mods:
        for i, c in enumerate(codes[m].tolist()):
            pos[m].setdefault(c, []).append(i)
    out = []
    for i, c in enumerate(ref.tolist()):
        k = pos[mods[0]][c].index(i)
        out.append(tuple(pos[m][c][k] for m in mods))
    return out


def subject_windows(subj: int, files: Dict[str, str]) -> Tuple[List[dict], Dict]:
    loaded = {m: load_eeglab_epochs(files[m]) for m in MODALITIES if m in files}
    codes = {m: v[1] for m, v in loaded.items()}
    note = {"subject": subj, "files": {m: os.path.basename(p) for m, p in files.items()},
            "epochs": {m: int(len(c)) for m, c in codes.items()},
            "channels": {m: v[2] for m, v in loaded.items()},
            "sfreq": {m: v[3] for m, v in loaded.items()}}
    if set(loaded) != set(MODALITIES):
        note["dropped"] = f"missing {sorted(set(MODALITIES) - set(loaded))}"
        return [], note
    pairs = align(codes)
    if pairs is None:
        note["dropped"] = "marker sequences disagree between the three files"
        return [], note
    items = []
    unknown = 0
    for k, idx in enumerate(pairs):
        code = int(codes["eeg"][idx[0]])
        if code not in MARKER_CLASS:
            unknown += 1
            continue
        item = {"label": MARKER_CLASS[code], "subject": f"mpdb_s{subj:02d}",
                "segment": f"mpdb_s{subj:02d}/m{code}/{k}"}
        for m, i in zip(MODALITIES, idx):
            data, _, names, fs = loaded[m]
            x = data[i]
            if m == "ecg":
                x = x[:1]                       # the release's ECG file: one channel
            if m == "emg":
                x = x[:4]
            item[m] = prepare_window(x, fs, m, WINDOW_SECONDS)
        items.append(item)
    note["unknown_markers"] = unknown
    note["windows"] = len(items)
    return items, note


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-subjects", type=int, default=5)
    p.add_argument("--test-subjects", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--max-subjects", type=int, default=None, help="a quick look")
    args = p.parse_args(argv)
    t0 = time.time()

    found = find_subjects(args.raw_dir)
    if not found:
        raise SystemExit(f"no EEG_n.set / ECG_n.set / EMG_n.set under {args.raw_dir}")
    subjects = sorted(found)[: args.max_subjects] if args.max_subjects else sorted(found)
    print(f"mpdb: {len(subjects)} subject(s) under {args.raw_dir}", flush=True)

    # Load everything first: the split is drawn over the subjects that
    # actually yield windows, so a dropped subject cannot leave a split empty.
    loaded, notes, channels = {}, [], None
    for subj in subjects:
        items, note = subject_windows(subj, found[subj])
        notes.append(note)
        if not items:
            print(f"  subject {subj}: DROPPED -- {note.get('dropped')}", flush=True)
            continue
        eeg_names = [str(n) for n in note["channels"]["eeg"]]
        if channels is None:
            channels = {"eeg": eeg_names, "ecg": ECG_CHANNELS, "emg": EMG_CHANNELS}
        elif eeg_names != channels["eeg"]:
            note["dropped"] = "EEG channel names differ from the first subject's"
            print(f"  subject {subj}: DROPPED -- {note['dropped']}", flush=True)
            continue
        loaded[subj] = items
        print(f"  subject {subj}: {len(items)} windows", flush=True)
    if not loaded:
        raise SystemExit("every subject was dropped; see the notes above")
    side = subject_split([f"mpdb_s{s:02d}" for s in loaded],
                         args.val_subjects, args.test_subjects, args.split_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    writers = {s: MultimodalWriter(os.path.join(args.out_dir, f"{s}.h5"), "mpdb",
                                   WINDOW_SECONDS, CLASSES, channels,
                                   {"marker_class": MARKER_CLASS, "source": "figshare 22192831"})
               for s in ("train", "val", "test")}
    stats = {s: {"windows": 0, "subjects": set(), "per_class": np.zeros(len(CLASSES), np.int64),
                 "present": {m: 0 for m in MODALITIES}} for s in writers}
    for subj, items in loaded.items():
        s = side[f"mpdb_s{subj:02d}"]
        for it in items:
            writers[s].add(it)
            st = stats[s]
            st["windows"] += 1
            st["subjects"].add(it["subject"])
            st["per_class"][it["label"]] += 1
            for m in MODALITIES:
                st["present"][m] += int(it.get(m) is not None)
    for w in writers.values():
        w.close()
    summary = {"task": "mpdb", "classes": CLASSES, "channels": channels,
               "window_seconds": WINDOW_SECONDS, "marker_class": MARKER_CLASS,
               "splits": split_summary(stats),
               "subject_split": side, "subjects": notes,
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for s, v in summary["splits"].items():
        print(f"  {s:5s} {v['windows']:6,} windows  {v['subjects']} subjects  "
              f"per class {v['per_class']}")
    print(f"wrote {args.out_dir}/{{train,val,test}}.h5 and split.json in "
          f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
