#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The sEMG downstream benchmarks -> train/val/test HDF5 for the sEMG C1 encoder.

    python EMG/emg_downstream_prep.py --task epn612      --raw-dir $R/epn612      --out-dir $D/epn612
    python EMG/emg_downstream_prep.py --task grabmyo     --raw-dir $R/grabmyo     --out-dir $D/grabmyo
    python EMG/emg_downstream_prep.py --task ninapro_db2 --raw-dir $R/ninapro_db2 --out-dir $D/ninapro_db2

TASKS

  epn612         EMG-EPN-612 (Zenodo 4421500): 612 users, Myo armband, 8 ch at
                 200 Hz, 6 classes (noGesture, waveIn, waveOut, pinch, open,
                 fist), one 5 s recording per sample. THE PAPER'S PROTOCOL
                 (EMG/epn_finetune.py): the 306 training users' trainingSamples
                 train and their testingSamples validate; each of the 306
                 testing users gives its first 10 trainingSamples per gesture
                 to training and the other 15 to test -- so test users are
                 seen in training (user-dependent).
  epn612_xuser   The same data, USER-INDEPENDENT: training users train (both
                 sample sets; 10% of them, seed 42, validate), testing users'
                 labelled trainingSamples are the test set, never trained on.
  grabmyo        GRABMyo 1.1.0 (PhysioNet): 43 participants, 3 sessions (days
                 1, 8, 29), 16 gestures + rest, 7 trials of 5 s, 2048 Hz; the
                 16 forearm (F1-F16) and 12 wrist (W1-W12) electrodes, U1-U4
                 (unused) dropped. INTER-SESSION: sessions 1-2 train (trial 7
                 validates), session 3 -- three weeks later -- tests.
  grabmyo_xsubj  The same, SUBJECT-INDEPENDENT: participants 70/15/15 (seed
                 42), every session.
  ninapro_db2    NinaPro DB2: 40 subjects, 12 Delsys electrodes at 2 kHz,
                 exercises B (17 movements, labels 1-17), C (23 grasps, 18-40)
                 and D (9 force patterns, 41-49) + rest (0): 50 classes, from
                 the refined labels (restimulus/rerepetition). The standard
                 repetition split (Atzori et al. 2014): repetitions 1, 3, 4, 6
                 train and 2, 5 test, every subject in both; 10% of the
                 training (subject, movement, repetition) segments -- whole
                 segments, so no window overlaps one in training -- validate.
                 Rest is subsampled per subject and exercise to the mean
                 window count of one movement, so it does not outnumber them
                 50 to 1. --db2-exercises picks a subset (B, C, D).

PREPROCESSING IS THE PRETRAINING CORPUS'S (EMG/preprocess_emg_corpus.py):
to mV -> DC removal -> mains notch with harmonics below Nyquist -> 20 Hz
high-pass -> resample to the task rate -> windows -> one scale per window
across the channels (window_shared) -> clip. No window is dropped for
quality. EPN stays at its 200 Hz (resampling it to 2000 Hz would add samples,
not information): one 5 s window per sample, 40 patches of 0.125 s.
GRABMyo and DB2 are at 2000 Hz in 1 s windows with a 0.5 s stride, the
pretraining window; a window must lie inside one label.

None of these montages is a pretraining route, so the downstream model builds
its own wavelet frontend and patch embedding and takes the pretrained channel
encoder and Transformer (physiowave/emg_c1/downstream.py).

Output per split: data (N, C, T) float32, label (N,) int64, subject and
segment (N,) bytes, channel_names; attrs sampling_rate, window_samples,
class_names, task, prep_version, provenance. split.json says what went where.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(ROOT, "ECG")):
    if p not in sys.path:
        sys.path.insert(0, p)

from physiowave.ecg_c1.preprocess import ECGPreprocessConfig, process_ecg_record  # noqa: E402
from physiowave.eeg_c1.routes import Route                                         # noqa: E402

PREP_VERSION = "emg-c1-downstream-v1"
SEED = 42

EPN_CLASSES = ["noGesture", "waveIn", "waveOut", "pinch", "open", "fist"]
EPN_CHANNELS = [f"myo{k}" for k in range(1, 9)]
GRAB_CLASSES = ["lateral_prehension", "thumb_adduction", "thumb_little_opposition",
                "thumb_index_opposition", "thumb_index_extension",
                "thumb_little_extension", "index_middle_extension",
                "little_extension", "index_extension", "thumb_extension",
                "wrist_extension", "wrist_flexion", "forearm_supination",
                "forearm_pronation", "hand_open", "hand_close", "rest"]
GRAB_CHANNELS = [f"F{k}" for k in range(1, 17)] + [f"W{k}" for k in range(1, 13)]
DB2_CHANNELS = [f"db2_e{k}" for k in range(1, 13)]
DB2_EXERCISES = {"B": (1, range(1, 18)), "C": (2, range(18, 41)), "D": (3, range(41, 50))}

TASKS = {
    "epn612":        {"fs": 200,  "window": 5.0, "stride": None, "mains": 60.0},
    "epn612_xuser":  {"fs": 200,  "window": 5.0, "stride": None, "mains": 60.0},
    "grabmyo":       {"fs": 2000, "window": 1.0, "stride": 0.5,  "mains": 60.0},
    "grabmyo_xsubj": {"fs": 2000, "window": 1.0, "stride": 0.5,  "mains": 60.0},
    "ninapro_db2":   {"fs": 2000, "window": 1.0, "stride": 0.5,  "mains": 50.0},
}


def classes_for(task: str, exercises="BCD") -> List[str]:
    if task.startswith("epn"):
        return EPN_CLASSES
    if task.startswith("grabmyo"):
        return GRAB_CLASSES
    return ["rest"] + [f"m{k}" for k in range(1, 50)]


def channels_for(task: str) -> List[str]:
    return EPN_CHANNELS if task.startswith("epn") else \
        GRAB_CHANNELS if task.startswith("grabmyo") else DB2_CHANNELS


def config_for(task: str) -> ECGPreprocessConfig:
    t = TASKS[task]
    return ECGPreprocessConfig(
        highpass_hz=20.0, notch_harmonics=3, normalization="window_shared",
        window_seconds=t["window"], stride_seconds=t["stride"],
        flat_lead_std_mv=0.0, max_abs_mv=float("inf"), derive_limb_leads=False)


def route_for(task: str) -> Route:
    t = TASKS[task]
    ch = channels_for(task)
    return Route("downstream", len(ch), t["fs"], window_seconds=t["window"],
                 patch_seconds=0.125, slots=tuple(ch))


def pipeline(x: np.ndarray, fs: float, task: str, key: str):
    """``[C, T]`` mV -> (windows [N, C, W] float32, starts in output samples)."""
    from preprocess_emg_corpus import process_emg_record
    route = route_for(task)
    out = process_emg_record(x, channels_for(task), fs, "mV", route,
                             TASKS[task]["mains"], config_for(task),
                             record_key=key, slots=route.slots)
    starts = np.round(out.starts_seconds * route.sampling_rate).astype(np.int64)
    return out.windows.astype(np.float32), starts


# --------------------------------------------------------------------------- #
# EPN-612
# --------------------------------------------------------------------------- #

def _epn_members(raw: str) -> List[Tuple[str, str]]:
    """(container, member) for every user json, from zips or a tree."""
    out = []
    for z in sorted(glob.glob(os.path.join(raw, "**", "*.zip"), recursive=True)):
        with zipfile.ZipFile(z) as zf:
            out += [(z, n) for n in zf.namelist() if re.search(r"(training|testing)JSON/user\d+/user\d+\.json$", n)]
    out += [("", p) for p in glob.glob(os.path.join(raw, "**", "user*.json"), recursive=True)
            if re.search(r"(training|testing)JSON", p)]
    return sorted(set(out))


def _epn_load(container: str, member: str) -> dict:
    if container:
        with zipfile.ZipFile(container) as zf:
            return json.loads(zf.read(member))
    with open(member) as f:
        return json.load(f)


def epn_unit(args):
    """One user's labelled samples: [(split, subject, segment, label, window)]."""
    container, member, task = args
    part = "tr" if "trainingJSON" in member else "te"
    user = re.search(r"(user\d+)\.json$", member).group(1)
    subject = f"epn_{part}_{user}"
    d = _epn_load(container, member)
    win = int(TASKS[task]["window"] * 200)
    rows = []
    per_gesture: Dict[str, int] = {}
    for setname in ("trainingSamples", "testingSamples"):
        for idx, s in d.get(setname, {}).items():
            g = s.get("gestureName")
            if g not in EPN_CLASSES:
                continue                         # unlabelled (testing users' testingSamples)
            if task == "epn612":
                if part == "tr":
                    split = "train" if setname == "trainingSamples" else "val"
                else:
                    if setname != "trainingSamples":
                        continue
                    n = per_gesture.get(g, 0)
                    per_gesture[g] = n + 1
                    split = "train" if n < 10 else "test"
            else:                                # epn612_xuser; val users picked later
                if part == "te" and setname != "trainingSamples":
                    continue
                split = "train" if part == "tr" else "test"
            x = np.stack([np.asarray(s["emg"][f"ch{k}"], np.float64) for k in range(1, 9)]) / 128.0
            if x.shape[1] < win:
                x = np.concatenate([x, np.zeros((8, win - x.shape[1]))], axis=1)
            w, _ = pipeline(x[:, :win], 200.0, task, f"{subject}/{idx}")
            if len(w):
                rows.append((split, subject, f"{subject}/{setname}/{idx}",
                             EPN_CLASSES.index(g), w[0]))
    return rows


# --------------------------------------------------------------------------- #
# GRABMyo
# --------------------------------------------------------------------------- #

_GRAB = re.compile(r"session(\d)_participant(\d+)_gesture(\d+)_trial(\d+)$")


def grab_units(raw: str) -> List[str]:
    return sorted(h[:-4] for h in glob.glob(os.path.join(raw, "**", "*.hea"), recursive=True)
                  if _GRAB.search(os.path.basename(h)[:-4]))


def grab_unit(args):
    stem, task = args
    import wfdb
    sess, part, gest, trial = map(int, _GRAB.search(os.path.basename(stem)).groups())
    rec = wfdb.rdrecord(stem)
    names = list(rec.sig_name)
    rows_idx = [names.index(c) for c in GRAB_CHANNELS]
    x = np.asarray(rec.p_signal, np.float64).T[rows_idx]       # mV
    subject = f"grabmyo_p{part:02d}"
    w, _ = pipeline(x, float(rec.fs), task, os.path.basename(stem))
    if task == "grabmyo":
        split = "test" if sess == 3 else ("val" if trial == 7 else "train")
    else:
        split = None                                           # by subject, later
    seg = f"{subject}/s{sess}/g{gest}/t{trial}"
    return [(split, subject, seg, gest - 1, wi) for wi in w]


# --------------------------------------------------------------------------- #
# NinaPro DB2
# --------------------------------------------------------------------------- #

def db2_units(raw: str, exercises: str) -> List[Tuple[str, str]]:
    want = {DB2_EXERCISES[e][0] for e in exercises}
    out = []
    for z in sorted(glob.glob(os.path.join(raw, "**", "DB2_s*.zip"), recursive=True)):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                m = re.search(r"S(\d+)_E(\d)_A1\.mat$", n)
                if m and int(m.group(2)) in want:
                    out.append((z, n))
    for p in glob.glob(os.path.join(raw, "**", "S*_E*_A1.mat"), recursive=True):
        m = re.search(r"S(\d+)_E(\d)_A1\.mat$", p)
        if m and int(m.group(2)) in want:
            out.append(("", p))
    return sorted(set(out))


def db2_unit(args):
    container, member, task = args
    import scipy.io
    src = io.BytesIO(zipfile.ZipFile(container).read(member)) if container else member
    m = scipy.io.loadmat(src, variable_names=["emg", "restimulus", "rerepetition"])
    s, e = map(int, re.search(r"S(\d+)_E(\d)_A1\.mat$", member).groups())
    n = min(len(m["emg"]), len(m["restimulus"]), len(m["rerepetition"]))
    x = np.asarray(m["emg"][:n], np.float64).T * 1000.0          # V -> mV
    lab = np.asarray(m["restimulus"][:n]).ravel().astype(np.int64)
    rep = np.asarray(m["rerepetition"][:n]).ravel().astype(np.int64)
    # Rest carries repetition 0; it belongs to the repetition it follows (the
    # first rest to repetition 1), so a repetition split splits rest too.
    filled = rep.copy()
    last = 1
    for i in range(n):
        if filled[i] > 0:
            last = filled[i]
        else:
            filled[i] = last
    w, starts = pipeline(x, 2000.0, task, f"db2_s{s}_e{e}")
    W = int(TASKS[task]["window"] * 2000)
    subject = f"db2_s{s:02d}"
    rows, rest = [], []
    for wi, st in zip(w, starts):
        seg_lab = lab[st:st + W]
        if seg_lab.size < W or seg_lab.min() != seg_lab.max():
            continue                                 # straddles a label change
        k = int(seg_lab[0])
        r = int(filled[st])
        split = "test" if r in (2, 5) else "train"
        item = (split, subject, f"{subject}/e{e}/m{k}/r{r}", k, wi)
        (rest if k == 0 else rows).append(item)
    if rows and rest:
        per_class = len(rows) / max(1, len({it[3] for it in rows}))
        keep = np.random.default_rng([SEED, s, e]).permutation(len(rest))[:int(round(per_class))]
        rows += [rest[i] for i in sorted(keep)]
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

class Writer:
    def __init__(self, path, task, C, W, classes, cfg):
        import h5py
        self.path = path
        self.f = h5py.File(path + ".part", "w")
        self.data = self.f.create_dataset("data", (0, C, W), maxshape=(None, C, W),
                                          dtype="float32", chunks=(32, C, W))
        self.label = self.f.create_dataset("label", (0,), maxshape=(None,), dtype="int64")
        dt = h5py.string_dtype()
        self.subject = self.f.create_dataset("subject", (0,), maxshape=(None,), dtype=dt)
        self.segment = self.f.create_dataset("segment", (0,), maxshape=(None,), dtype=dt)
        self.f.create_dataset("channel_names", data=np.array(channels_for(task), dtype="S16"))
        self.f.attrs.update({
            "sampling_rate": float(TASKS[task]["fs"]), "window_samples": int(W),
            "class_names": json.dumps(classes), "task": task, "multilabel": False,
            "prep_version": PREP_VERSION,
            "provenance": json.dumps(cfg.provenance({"task": task}), default=str)})
        self.n = 0
        self.buf = []

    def add(self, item):
        self.buf.append(item)
        if len(self.buf) >= 512:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        k = len(self.buf)
        for ds in (self.data, self.label, self.subject, self.segment):
            ds.resize(self.n + k, axis=0)
        self.data[self.n:self.n + k] = np.stack([b[4] for b in self.buf])
        self.label[self.n:self.n + k] = [b[3] for b in self.buf]
        self.subject[self.n:self.n + k] = [b[1] for b in self.buf]
        self.segment[self.n:self.n + k] = [b[2] for b in self.buf]
        self.n += k
        self.buf = []

    def close(self):
        self.flush()
        self.f.close()
        os.replace(self.path + ".part", self.path)


def _subject_sides(subjects, frac_val, frac_test):
    subs = sorted(subjects)
    order = np.random.default_rng(SEED).permutation(len(subs))
    n_te, n_va = int(round(frac_test * len(subs))), int(round(frac_val * len(subs)))
    side = {}
    for rank, i in enumerate(order):
        side[subs[i]] = "test" if rank < n_te else "val" if rank < n_te + n_va else "train"
    return side


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--db2-exercises", default="BCD",
                   help="NinaPro DB2 exercises to use, a subset of BCD")
    p.add_argument("--max-units", type=int, default=None, help="a quick look")
    args = p.parse_args(argv)
    task, t0 = args.task, time.time()

    if task.startswith("epn"):
        units = [(c, m, task) for c, m in _epn_members(args.raw_dir)]
        fn = epn_unit
    elif task.startswith("grabmyo"):
        units = [(s, task) for s in grab_units(args.raw_dir)]
        fn = grab_unit
    else:
        units = [(c, m, task) for c, m in db2_units(args.raw_dir, args.db2_exercises)]
        fn = db2_unit
    if args.max_units:
        units = units[:args.max_units]
    if not units:
        raise SystemExit(f"nothing for {task} under {args.raw_dir}")
    print(f"{task}: {len(units):,} unit(s) under {args.raw_dir}", flush=True)

    # Subject-level splits need every subject first; the others are decided
    # per sample. Collect, then write.
    items, failed = [], []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for u, res in zip(units, ex.map(_safe, [(fn, u) for u in units], chunksize=4)):
            if isinstance(res, str):
                failed.append({"unit": str(u[:-1]), "reason": res})
            else:
                items += res
            if len(items) and len(items) % 20000 < 200:
                print(f"  {len(items):,} windows so far", flush=True)

    if task == "epn612_xuser":
        tr_users = sorted({it[1] for it in items if it[0] == "train"})
        val_users = set(np.array(tr_users)[np.random.default_rng(SEED).permutation(
            len(tr_users))[:int(round(0.1 * len(tr_users)))]].tolist())
        items = [("val" if it[1] in val_users else it[0],) + it[1:] for it in items]
    elif task == "grabmyo_xsubj":
        side = _subject_sides({it[1] for it in items}, 0.15, 0.15)
        items = [(side[it[1]],) + it[1:] for it in items]
    elif task == "ninapro_db2":
        segs = sorted({it[2] for it in items if it[0] == "train"})
        pick = np.random.default_rng(SEED).permutation(len(segs))[:int(round(0.1 * len(segs)))]
        val_segs = {segs[i] for i in pick}
        items = [("val" if it[2] in val_segs else it[0],) + it[1:] for it in items]

    classes = classes_for(task)
    C, W = items[0][4].shape
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = config_for(task)
    writers = {s: Writer(os.path.join(args.out_dir, f"{s}.h5"), task, C, W, classes, cfg)
               for s in ("train", "val", "test")}
    stats = {s: {"windows": 0, "subjects": set(), "segments": set(),
                 "per_class": np.zeros(len(classes), np.int64)} for s in writers}
    rng = np.random.default_rng(SEED)
    for i in rng.permutation(len(items)) if False else range(len(items)):
        it = items[i]
        writers[it[0]].add(it)
        st = stats[it[0]]
        st["windows"] += 1
        st["subjects"].add(it[1])
        st["segments"].add(it[2])
        st["per_class"][it[3]] += 1
    for w in writers.values():
        w.close()

    sides = list(stats)
    overlap = {f"{a}/{b}": len(stats[a]["subjects"] & stats[b]["subjects"])
               for i, a in enumerate(sides) for b in sides[i + 1:]}
    seg_overlap = {f"{a}/{b}": len(stats[a]["segments"] & stats[b]["segments"])
                   for i, a in enumerate(sides) for b in sides[i + 1:]}
    summary = {"task": task, "classes": classes, "channels": channels_for(task),
               "sampling_rate": TASKS[task]["fs"], "window_seconds": TASKS[task]["window"],
               "stride_seconds": TASKS[task]["stride"], "prep_version": PREP_VERSION,
               "db2_exercises": args.db2_exercises if task == "ninapro_db2" else None,
               "splits": {s: {"windows": v["windows"], "subjects": len(v["subjects"]),
                              "segments": len(v["segments"]),
                              "per_class": v["per_class"].tolist()} for s, v in stats.items()},
               "subject_overlap": overlap, "segment_overlap": seg_overlap,
               "failed_units": len(failed), "failures": failed[:100],
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for s, v in summary["splits"].items():
        print(f"  {s:5s} {v['windows']:>8,} windows  {v['subjects']:>4} subjects  "
              f"{v['segments']:>7,} segments")
    print(f"  subject overlap {overlap}; segment overlap {seg_overlap}; "
          f"{len(failed)} unit(s) failed")
    if any(seg_overlap.values()):
        print("ERROR: a segment is on two sides of the split", file=sys.stderr)
        return 1
    print(f"wrote {args.out_dir}/{{train,val,test}}.h5 and split.json "
          f"in {time.time() - t0:.0f}s")
    return 0


def _safe(arg):
    fn, unit = arg
    try:
        return fn(unit)
    except Exception as exc:                                    # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    sys.exit(main())
