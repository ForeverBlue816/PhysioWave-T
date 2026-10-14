#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPDB from the RAW release -> aligned EEG + ECG + sEMG windows, on our own terms.

    python Fusion/mpdb_raw_prep.py --raw-dir $R/mpdb_raw --out-dir $D/mpdb
    python Fusion/mpdb_raw_prep.py --raw-dir $R/mpdb_raw --out-dir /tmp/x --inspect 2

THE RAW RELEASE (figshare 22193119, 23 GB): per subject four experiments --
brake, throttle, turn, change -- each a CONTINUOUS EEGLAB .set file per
device: EEG&ECG (59 electrodes + the chest ECG, one Neuracle amplifier, 1 kHz)
and EMG (four limb channels, NeuSen WM, 1 kHz), with the simulator's event
markers in each file's event list. Unlike the preprocessed release this has
timestamps, so an EMG window is paired with the EEG/ECG window of THE SAME
EVENT: within an experiment the two files' marker sequences are matched in
order (identical, or the longest common subsequence when one device missed
a trigger) and the inter-event intervals are compared -- a pairing whose
intervals disagree by more than --tolerance seconds is refused, not kept.

WINDOWS of --window seconds (4) starting --pre seconds (0.5) before the
marker, so the event's onset sits where the preprocessed release put it and
the window is the EEG encoder's pretraining length. Each record is filtered
whole with its modality's pretraining pipeline (physiowave/fusion/prep_common.py:
EEG detrend + 50 Hz notch + 0.5 Hz; ECG notch + 0.5 Hz; EMG notch + 20 Hz),
then cut, resampled to the encoder rate and normalised as that corpus was.
No ICA, no epoch rejection: what the authors pruned stays in, and quality is
the reliability estimator's problem, which is the point of the fusion.

LABELS from the marker codes (smooth 133; acceleration 135/137/143;
deceleration 139/141/145; lane change 129/131; turning 125/127). SPLIT by
subject, 25 / 5 / 5 (seed 42). ECG is the EEG file's ECG channel, named
``patch1`` so the L1_250 route's frontend transfers.
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
for p in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from mpdb_prep import (CLASSES, ECG_CHANNELS, EMG_CHANNELS, MARKER_CLASS,  # noqa: E402
                       SEED, _lcs_pairs)
from physiowave.fusion.prep_common import (MODALITIES, MultimodalWriter,  # noqa: E402
                                           cut, filter_continuous, prepare_window,
                                           split_summary, subject_split)

EXPERIMENTS = ("brake", "throttle", "turn", "change")
MAINS_HZ = 50.0


def load_eeglab_raw(path: str) -> Tuple[np.ndarray, List[str], float, List[Tuple[float, int]]]:
    """``(data [C, T], channel names, sfreq, [(onset s, marker code)])`` from a continuous .set."""
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    events = []
    for on, desc in zip(raw.annotations.onset, raw.annotations.description):
        m = re.search(r"(\d+)", str(desc))
        if m:
            events.append((float(on), int(m.group(1))))
    return raw.get_data(), list(raw.ch_names), float(raw.info["sfreq"]), events


def find_files(raw: str) -> Dict[int, Dict[str, Dict[str, str]]]:
    """``{subject: {experiment: {"eegecg": path, "emg": path}}}``."""
    found: Dict[int, Dict[str, Dict[str, str]]] = {}
    for p in glob.glob(os.path.join(raw, "**", "*.set"), recursive=True):
        m = re.match(r"(\d+)_(brake|throttle|turn|change)\.set$", os.path.basename(p), re.I)
        if not m:
            continue
        parent = os.path.basename(os.path.dirname(p)).upper()
        if "GSR" in parent:
            continue
        kind = "emg" if "EMG" in parent else "eegecg" if "EEG" in parent else None
        if kind is None:
            continue
        found.setdefault(int(m.group(1)), {}).setdefault(m.group(2).lower(), {})[kind] = p
    return found


def split_eeg_ecg(data: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """The EEG&ECG file's rows -> (eeg [59, T], eeg names, ecg [1, T])."""
    ecg_rows = [i for i, n in enumerate(names) if "ECG" in n.upper() or "EKG" in n.upper()]
    # The release's auxiliaries: ECG, and the four EOG channels named HEOR,
    # HEOL, VEOU, VEOL (horizontal right/left, vertical up/low).
    drop = set(ecg_rows) | {i for i, n in enumerate(names)
                            if re.search(r"trig|stim|marker|status|EOG|GSR|^[HV]EO[RLUD]$", n, re.I)}
    eeg_rows = [i for i in range(len(names)) if i not in drop]
    ecg = data[ecg_rows[:1]] if ecg_rows else None
    return data[eeg_rows], [names[i] for i in eeg_rows], ecg


def pair_events(a: List[Tuple[float, int]], b: List[Tuple[float, int]], tol: float
                ) -> Tuple[List[Tuple[int, int]], Dict]:
    """Indices into ``a`` (EEG&ECG file) and ``b`` (EMG file) of the same events.

    The two devices share the simulator's trigger stream but not a clock:
    the EMG file is offset by a constant (seconds) plus a drift of a few
    tens of ppm, and it may hold a few events the other file does not (the
    raw throttle files carry nine extra). Codes repeat hundreds of times, so
    order alone is ambiguous; TIME is not. The clock relation
    ``t_b = s * t_a + o`` is estimated from the codes that occur equally
    often in both files (the behaviour markers among them), refined by a
    least-squares fit over the pairs it explains, and every event of ``a``
    is then paired with the nearest same-code event of ``b`` within ``tol``
    of its predicted time, each event of ``b`` used once.
    """
    if not a or not b:
        return [], {"events_eeg": len(a), "events_emg": len(b), "paired": 0}
    ta, ca = np.array([t for t, _ in a]), np.array([c for _, c in a])
    tb, cb = np.array([t for t, _ in b]), np.array([c for _, c in b])
    # 1. the offset, as the mode of all same-code time differences. Right
    #    pairs pile into one 20 ms bin (drift spreads them over a few); wrong
    #    pairs scatter over the whole range. k-th-to-k-th pairing is NOT
    #    robust here: a code that fires every few seconds with one extra
    #    event in one file shifts every later pair.
    diffs = []
    for c in np.unique(ca):
        ia, ib = np.where(ca == c)[0], np.where(cb == c)[0]
        if not len(ib):
            continue
        d = (tb[ib][None, :] - ta[ia][:, None]).ravel()
        diffs.append(d[np.abs(d) < 300.0])
    diffs = np.concatenate(diffs) if diffs else np.zeros(0)
    if diffs.size < 3:
        return [], {"events_eeg": len(a), "events_emg": len(b), "paired": 0,
                    "reason": "no shared codes within 300 s of each other"}
    edges = np.arange(diffs.min() - 0.02, diffs.max() + 0.04, 0.02)
    hist, _ = np.histogram(diffs, bins=edges)
    k = int(np.argmax(hist))
    lo, hi = edges[max(0, k - 2)], edges[min(len(edges) - 1, k + 3)]
    off, scale = float(np.median(diffs[(diffs >= lo) & (diffs <= hi)])), 1.0

    def match(off, scale, tol_):
        used = np.zeros(len(b), bool)
        pairs = []
        for i in np.argsort(ta):
            pred = scale * ta[i] + off
            cand = np.where((cb == ca[i]) & ~used & (np.abs(tb - pred) <= tol_))[0]
            if cand.size:
                j = int(cand[np.argmin(np.abs(tb[cand] - pred))])
                used[j] = True
                pairs.append((int(i), j))
        return pairs
    # 2. refine with a linear fit (drift) over what the offset explains, at a
    #    tolerance that admits the drift but not a neighbouring event; then
    #    the final pairing at the caller's tolerance
    for tol_ in (0.1, tol):
        pairs = match(float(off), float(scale), tol_)
        if len(pairs) >= 3:
            xa = ta[[i for i, _ in pairs]]
            xb = tb[[j for _, j in pairs]]
            scale, off = np.polyfit(xa, xb, 1)
    pairs = match(float(off), float(scale), tol)
    resid = [abs(scale * ta[i] + off - tb[j]) for i, j in pairs]
    return pairs, {"events_eeg": len(a), "events_emg": len(b), "paired": len(pairs),
                   "unpaired_eeg": len(a) - len(pairs), "unused_emg": len(b) - len(pairs),
                   "clock_offset_s": round(float(off), 3),
                   "clock_drift_ppm": round((float(scale) - 1.0) * 1e6, 1),
                   "max_residual_ms": round(1000 * float(max(resid)), 1) if resid else None}


def subject_windows(subj: int, exps: Dict[str, Dict[str, str]], window: float, pre: float,
                    tol: float) -> Tuple[List[dict], Dict]:
    items: List[dict] = []
    note: Dict = {"subject": subj, "experiments": {}}
    eeg_names_ref: Optional[List[str]] = None
    for exp in EXPERIMENTS:
        files = exps.get(exp, {})
        en: Dict = {"files": {k: os.path.basename(v) for k, v in files.items()}}
        note["experiments"][exp] = en
        if "eegecg" not in files:
            en["skipped"] = "no EEG&ECG file"
            continue
        data, names, fs, ev_a = load_eeglab_raw(files["eegecg"])
        eeg, eeg_names, ecg = split_eeg_ecg(data, names)
        if eeg_names_ref is None:
            eeg_names_ref = eeg_names
        elif eeg_names != eeg_names_ref:
            en["skipped"] = f"EEG channels differ from {EXPERIMENTS[0]}'s: {eeg_names[:5]}..."
            continue
        en.update({"eeg_channels": len(eeg_names), "sfreq": fs, "seconds": eeg.shape[1] / fs,
                   "ecg": ecg is not None, "codes": {int(c): int(sum(1 for _, cc in ev_a if cc == c))
                                                     for c in sorted({cc for _, cc in ev_a})}})
        streams = {"eeg": (filter_continuous(eeg, fs, "eeg", MAINS_HZ), fs)}
        if ecg is not None:
            streams["ecg"] = (filter_continuous(ecg, fs, "ecg", MAINS_HZ), fs)
        emg_events: Dict[int, float] = {}
        if "emg" in files:
            d2, n2, fs2, ev_b = load_eeglab_raw(files["emg"])
            rows = [i for i, n in enumerate(n2) if not re.search(r"trig|stim|marker", n, re.I)][:4]
            emg = d2[rows]
            kept, how = pair_events(ev_a, ev_b, tol)
            en["emg_pairing"] = how
            en["emg_channels"] = [n2[i] for i in rows]
            if emg.shape[0] == 4:
                streams["emg"] = (filter_continuous(emg, fs2, "emg", MAINS_HZ), fs2)
                emg_events = {i: ev_b[j][0] for i, j in kept}
            else:
                en["emg_skipped"] = f"{emg.shape[0]} EMG channels, not 4"
        else:
            en["emg_skipped"] = "no EMG file"
        n_win = 0
        for k, (t, code) in enumerate(ev_a):
            if code not in MARKER_CLASS:
                continue
            item = {"label": MARKER_CLASS[code], "subject": f"mpdb_s{subj:02d}",
                    "segment": f"mpdb_s{subj:02d}/{exp}/m{code}/{k}"}
            x, r = streams["eeg"]
            seg = cut(x, r, t - pre, window)
            if seg is None:
                continue
            item["eeg"] = prepare_window(seg, r, "eeg", window)
            if "ecg" in streams:
                x, r = streams["ecg"]
                seg = cut(x, r, t - pre, window)
                item["ecg"] = prepare_window(seg, r, "ecg", window) if seg is not None else None
            else:
                item["ecg"] = None
            item["emg"] = None
            if "emg" in streams and k in emg_events:
                x, r = streams["emg"]
                seg = cut(x, r, emg_events[k] - pre, window)
                if seg is not None:
                    item["emg"] = prepare_window(seg, r, "emg", window)
            items.append(item)
            n_win += 1
        en["windows"] = n_win
    note["eeg_channels"] = eeg_names_ref
    note["windows"] = len(items)
    return items, note


def inspect_files(raw: str, n: int, tol: float) -> int:
    found = find_files(raw)
    print(f"{len(found)} subject(s) with raw files under {raw}: {sorted(found)}")
    for subj in sorted(found)[:n]:
        print(f"subject {subj}")
        for exp in EXPERIMENTS:
            files = found[subj].get(exp, {})
            print(f"  {exp}: {', '.join(f'{k}={os.path.basename(v)}' for k, v in files.items()) or 'no files'}")
            evs = {}
            for kind, path in files.items():
                try:
                    data, names, fs, ev = load_eeglab_raw(path)
                except Exception as exc:  # noqa: BLE001
                    print(f"    {kind}: mne could not read it: {exc!r}")
                    continue
                evs[kind] = ev
                hist = {}
                for _, c in ev:
                    hist[c] = hist.get(c, 0) + 1
                print(f"    {kind}: {data.shape[0]} ch x {data.shape[1] / fs:.0f} s @ {fs:g} Hz; "
                      f"events {len(ev)} {dict(sorted(hist.items()))}")
                print(f"      channels ({len(names)}): {names}")
                beh = {c: n for c, n in sorted(hist.items()) if c in MARKER_CLASS}
                print(f"      behaviour markers: {beh}  (others are simulator state codes)")
                print(f"      first events (s, code): {[(round(t, 2), c) for t, c in ev[:8]]}")
            if "eegecg" in evs and "emg" in evs:
                kept, how = pair_events(evs["eegecg"], evs["emg"], tol)
                print(f"    pairing EEG<->EMG: {how}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window", type=float, default=4.0)
    p.add_argument("--pre", type=float, default=0.5, help="seconds before the marker")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="max disagreement (s) between the devices' inter-event intervals")
    p.add_argument("--val-subjects", type=int, default=5)
    p.add_argument("--test-subjects", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=SEED)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--inspect", type=int, default=0, metavar="N")
    args = p.parse_args(argv)
    if args.inspect:
        return inspect_files(args.raw_dir, args.inspect, args.tolerance)
    t0 = time.time()
    found = find_files(args.raw_dir)
    if not found:
        raise SystemExit(f"no <n>_{{brake,throttle,turn,change}}.set under {args.raw_dir}")
    subjects = sorted(found)[: args.max_subjects] if args.max_subjects else sorted(found)
    print(f"mpdb raw: {len(subjects)} subject(s) under {args.raw_dir}", flush=True)

    loaded, notes, channels = {}, [], None
    for subj in subjects:
        items, note = subject_windows(subj, found[subj], args.window, args.pre, args.tolerance)
        notes.append(note)
        if not items:
            print(f"  subject {subj}: DROPPED -- no windows", flush=True)
            continue
        if channels is None:
            channels = {"eeg": note["eeg_channels"], "ecg": ECG_CHANNELS, "emg": EMG_CHANNELS}
        elif note["eeg_channels"] != channels["eeg"]:
            note["dropped"] = "EEG channel names differ from the first subject's"
            print(f"  subject {subj}: DROPPED -- {note['dropped']}", flush=True)
            continue
        loaded[subj] = items
        with_emg = sum(1 for it in items if it["emg"] is not None)
        with_ecg = sum(1 for it in items if it["ecg"] is not None)
        print(f"  subject {subj}: {len(items)} windows ({with_ecg} with ECG, {with_emg} with EMG)  "
              + " ".join(f"{e}:{v.get('windows', 0)}" for e, v in note["experiments"].items()),
              flush=True)
    if not loaded:
        raise SystemExit("every subject was dropped; see the notes above")
    side = subject_split([f"mpdb_s{s:02d}" for s in loaded], args.val_subjects,
                         args.test_subjects, args.split_seed)
    os.makedirs(args.out_dir, exist_ok=True)
    writers = {s: MultimodalWriter(os.path.join(args.out_dir, f"{s}.h5"), "mpdb", args.window,
                                   CLASSES, channels,
                                   {"marker_class": MARKER_CLASS, "source": "figshare 22193119 (raw)",
                                    "pre_seconds": args.pre})
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
    summary = {"task": "mpdb", "source": "raw", "classes": CLASSES, "channels": channels,
               "window_seconds": args.window, "pre_seconds": args.pre,
               "tolerance_seconds": args.tolerance, "marker_class": MARKER_CLASS,
               "splits": split_summary(stats), "subject_split": side, "subjects": notes,
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for s, v in summary["splits"].items():
        print(f"  {s:5s} {v['windows']:6,} windows  {v['subjects']} subjects  "
              f"per class {v['per_class']}  present {v['present']}")
    print(f"wrote {args.out_dir}/{{train,val,test}}.h5 and split.json in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
