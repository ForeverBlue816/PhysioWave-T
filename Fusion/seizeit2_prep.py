#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SeizeIT2 (Bhagubai et al., Sci. Data 2025; OpenNeuro ds005873) -> aligned
wearable EEG + ECG + EMG windows for seizure detection.

    python Fusion/seizeit2_prep.py --raw-dir $R/seizeit2 --out-dir $D/seizeit2

THE DATA. 125 patients with focal epilepsy, 5 European monitoring units,
~11,600 hours from the Sensor Dot wearable: two behind-the-ear EEG channels
(one same-side, one cross-head or second lateral), one ECG and one deltoid
EMG channel, all 256 Hz in the EDFs; 886 focal seizures (mean 58 s). BIDS
layout: sub-XXX/ses-01/{eeg,ecg,emg}/..._run-NN_{eeg,ecg,emg}.edf and an
events.tsv per run with onset / duration / eventType (bckg, impd, sz_*). In
3% of the data ECG/EMG are missing: that is REAL modality absence and is
kept as such (present flag False).

WINDOWS of 4 s. Seizure windows (label 1) are cut every --sz-stride seconds
across each seizure so that every window overlaps it by at least half.
Background windows (label 0) are cut on a 4 s grid away from every seizure
and impedance check by --guard seconds, and SUBSAMPLED to --bg-per-hour per
recorded hour (seed per run), because 11,600 hours on a 2 s grid would be
twenty million background windows against twenty-six thousand seizure ones.

PREPROCESSING per modality on the whole run (physiowave/fusion/prep_common.py),
then the window is resampled to the encoder rate (EEG stays 256, ECG 250,
EMG 2000) and normalised as that pretraining corpus was.

SPLIT by patient: --test-subjects 25, --val-subjects 15, the rest train
(seed 42). --subjects restricts to a regex over sub-XXX for a first pass.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from physiowave.fusion.prep_common import (MODALITIES, MultimodalWriter,  # noqa: E402
                                           cut, filter_continuous,
                                           natural_key, prepare_window,
                                           read_edf, split_summary,
                                           subject_split)

CLASSES = ["background", "seizure"]
EEG_CHANNELS = ["bte_sameside", "bte_other"]    # not in any vocabulary: <unk> codes
ECG_CHANNELS = ["patch1"]                        # one chest derivation: the L1_250 slot
EMG_CHANNELS = ["l_deltoid"]
MAINS_HZ = 50.0
SEED = 42


def find_runs(raw: str, subject_re: Optional[str]) -> List[dict]:
    runs = []
    for eeg in sorted(glob.glob(os.path.join(raw, "sub-*", "ses-*", "eeg", "*_eeg.edf")),
                      key=natural_key):
        sub = re.search(r"(sub-\d+)", eeg).group(1)
        if subject_re and not re.search(subject_re, sub):
            continue
        stem = eeg[: -len("_eeg.edf")]
        sesdir = os.path.dirname(os.path.dirname(eeg))
        base = os.path.basename(stem)
        run = {"subject": sub, "run": re.search(r"run-(\d+)", base).group(1),
               "eeg": eeg, "events": stem + "_events.tsv",
               "ecg": os.path.join(sesdir, "ecg", base + "_ecg.edf"),
               "emg": os.path.join(sesdir, "emg", base + "_emg.edf")}
        for m in ("ecg", "emg"):
            if not os.path.isfile(run[m]):
                run[m] = None
        runs.append(run)
    return runs


def read_events(path: str) -> List[Tuple[float, float, str]]:
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out.append((float(row["onset"]), float(row["duration"]), row["eventType"]))
            except (KeyError, ValueError):
                continue
    return out


def seizure_starts(onset: float, dur: float, W: float, stride: float, T: float) -> List[float]:
    """Window starts whose overlap with [onset, onset+dur] is at least W/2."""
    lo, hi = onset - W / 2, onset + dur - W / 2
    if hi < lo:
        lo = hi = onset + dur / 2 - W / 2
    starts, s = [], lo
    while s <= hi + 1e-9:
        if 0 <= s <= T - W:
            starts.append(round(s, 3))
        s += stride
    return starts


def background_starts(T: float, W: float, keep_out: List[Tuple[float, float]],
                      per_hour: float, rng: np.random.Generator) -> List[float]:
    grid = np.arange(0.0, T - W + 1e-9, W)
    ok = np.ones(len(grid), bool)
    for a, b in keep_out:
        ok &= ~((grid < b) & (grid + W > a))
    cand = grid[ok]
    k = int(round(per_hour * T / 3600.0))
    if k >= len(cand):
        return [float(s) for s in cand]
    return sorted(float(s) for s in cand[rng.choice(len(cand), k, replace=False)])


def run_windows(args) -> Tuple[List[dict], dict]:
    run, W, stride, guard, per_hour = args
    note = {"subject": run["subject"], "run": run["run"]}
    sig, rates, T = read_edf(run["eeg"])
    eeg_names = [n for n in sig]
    if len(eeg_names) < 1:
        note["dropped"] = "no EEG channel"
        return [], note
    fs = float(rates[eeg_names[0]])
    note["eeg_channels"] = eeg_names
    # Same-side channel first (BTEleft / BTEright), then whatever the other is.
    order = sorted(eeg_names, key=lambda n: (0 if n.upper().startswith("BTE") else 1, n))[:2]
    eeg = np.stack([sig[n] for n in order])
    if eeg.shape[0] == 1:
        eeg = np.concatenate([eeg, np.zeros_like(eeg)])
    streams = {"eeg": (filter_continuous(eeg, fs, "eeg", MAINS_HZ), fs)}
    for m, names in (("ecg", ECG_CHANNELS), ("emg", EMG_CHANNELS)):
        if run[m] is None:
            continue
        try:
            s2, r2, _ = read_edf(run[m])
        except (OSError, ValueError) as exc:
            note[f"{m}_error"] = str(exc)
            continue
        lab = next(iter(s2), None)
        if lab is None:
            continue
        x = s2[lab][None]
        streams[m] = (filter_continuous(x, float(r2[lab]), m, MAINS_HZ), float(r2[lab]))
    events = read_events(run["events"])
    sz = [(a, a + d) for a, d, t in events if t.startswith("sz")]
    impd = [(a, a + d) for a, d, t in events if t == "impd"]
    keep_out = [(a - guard, b + guard) for a, b in sz] + impd
    rng = np.random.default_rng([SEED, int(run["subject"][-3:]), int(run["run"])])
    starts = [(s, 1) for a, b in sz for s in seizure_starts(a, b - a, W, stride, T)]
    starts += [(s, 0) for s in background_starts(T, W, keep_out, per_hour, rng)]
    items = []
    for s, y in starts:
        item = {"label": y, "subject": run["subject"],
                "segment": f"{run['subject']}/run-{run['run']}/{s:.1f}"}
        for m in MODALITIES:
            if m not in streams:
                item[m] = None
                continue
            x, r = streams[m]
            seg = cut(x, r, s, W)
            item[m] = prepare_window(seg, r, m, W) if seg is not None else None
        if item["eeg"] is None:
            continue
        items.append(item)
    note.update({"seconds": T, "seizures": len(sz), "impedance_checks": len(impd),
                 "windows": len(items), "seizure_windows": sum(1 for _, y in starts if y)})
    return items, note


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window", type=float, default=4.0)
    p.add_argument("--sz-stride", type=float, default=1.0)
    p.add_argument("--guard", type=float, default=30.0,
                   help="seconds around a seizure or impedance check no background is cut from")
    p.add_argument("--bg-per-hour", type=float, default=30.0)
    p.add_argument("--subjects", default=None, help="regex over sub-XXX")
    p.add_argument("--val-subjects", type=int, default=15)
    p.add_argument("--test-subjects", type=int, default=25)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--max-runs", type=int, default=None, help="a quick look")
    args = p.parse_args(argv)
    t0 = time.time()

    runs = find_runs(args.raw_dir, args.subjects)
    if args.max_runs:
        runs = runs[: args.max_runs]
    if not runs:
        raise SystemExit(f"no sub-*/ses-*/eeg/*_eeg.edf under {args.raw_dir}")
    subjects = sorted({r["subject"] for r in runs})
    print(f"seizeit2: {len(runs)} run(s), {len(subjects)} subject(s) under {args.raw_dir}",
          flush=True)
    side = subject_split(subjects, args.val_subjects, args.test_subjects, args.split_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    channels = {"eeg": EEG_CHANNELS, "ecg": ECG_CHANNELS, "emg": EMG_CHANNELS}
    writers = {s: MultimodalWriter(os.path.join(args.out_dir, f"{s}.h5"), "seizeit2",
                                   args.window, CLASSES, channels,
                                   {"source": "openneuro ds005873", "guard_seconds": args.guard,
                                    "bg_per_hour": args.bg_per_hour, "sz_stride": args.sz_stride})
               for s in ("train", "val", "test")}
    stats = {s: {"windows": 0, "subjects": set(), "per_class": np.zeros(2, np.int64),
                 "present": {m: 0 for m in MODALITIES}} for s in writers}
    notes = []
    jobs = [(r, args.window, args.sz_stride, args.guard, args.bg_per_hour) for r in runs]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for run, (items, note) in zip(runs, ex.map(run_windows, jobs)):
            notes.append(note)
            s = side[run["subject"]]
            for it in items:
                writers[s].add(it)
                st = stats[s]
                st["windows"] += 1
                st["subjects"].add(it["subject"])
                st["per_class"][it["label"]] += 1
                for m in MODALITIES:
                    st["present"][m] += int(it.get(m) is not None)
            print(f"  {run['subject']} run-{run['run']}: {note.get('windows', 0)} windows "
                  f"({note.get('seizure_windows', 0)} seizure) -> {s}"
                  + (f"  [{note['dropped']}]" if "dropped" in note else ""), flush=True)
    for w in writers.values():
        w.close()
    summary = {"task": "seizeit2", "classes": CLASSES, "channels": channels,
               "window_seconds": args.window, "sz_stride": args.sz_stride,
               "guard_seconds": args.guard, "bg_per_hour": args.bg_per_hour,
               "splits": split_summary(stats), "subject_split": side, "runs": notes,
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for s, v in summary["splits"].items():
        print(f"  {s:5s} {v['windows']:7,} windows  {v['subjects']} subjects  "
              f"per class {v['per_class']}  present {v['present']}")
    print(f"wrote {args.out_dir}/{{train,val,test}}.h5 and split.json in "
          f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
