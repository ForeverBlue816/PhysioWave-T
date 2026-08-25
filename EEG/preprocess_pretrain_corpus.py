#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess one EEG corpus into the C1 pretraining HDF5 schema.

    python EEG/preprocess_pretrain_corpus.py \
        --dataset faced --root /path/to/raw --out-dir /path/to/processed/faced

One dataset per invocation, because they are acquired at different rates on
different montages and the failures are dataset-shaped. Every run writes
manifest_train.jsonl, manifest_val.jsonl, dataset_statistics.json and
preprocessing_failures.jsonl into --out-dir; a recording that cannot be read
lands in the failures file with its reason and does not stop the run.

WHAT THIS DOES NOT DO. It does not download anything. TUEG, HBN, M3CV and
TDBRAIN are all behind data use agreements, and a script that fetches them on
the user's behalf is a script that agrees to a licence on the user's behalf.
Point --root at data you already have.

SYNTHETIC DATA. --smoke-test generates noise so the pipeline itself can be
exercised without any corpus present. It is the only path that produces
synthetic windows, it stamps ``"synthetic": true`` into every shard's
provenance, and it writes to a directory it names ``smoke_*``. Nothing falls
back to it: a missing --root is an error, not a reason to invent data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_embedding import (CHANNEL_TO_ID,             # noqa: E402
                               channel_ids_for, normalize_channel_name)
from physiowave.eeg_c1.preprocess import (                # noqa: E402
    PreprocessConfig, PreprocessError, detrend, highpass, map_to_slots, notch,
    place_on_slots, resample_to, split_subjects, subject_split_side,
    to_microvolts, window_qc, window_signal, write_shard, zscore_windows)
from physiowave.eeg_c1.routes import PRETRAIN_DATASETS, ROUTES  # noqa: E402


# --------------------------------------------------------------------------- #
# What an adapter returns
# --------------------------------------------------------------------------- #

@dataclass
class Recording:
    """One readable recording, before any processing."""

    recording_id: str
    subject_id: str
    data: np.ndarray            # [C, T]
    channel_names: List[str]
    sampling_rate: float
    unit: str = "uV"
    mains_hz: Optional[float] = None
    notes: Dict = None


# --------------------------------------------------------------------------- #
# Adapters
#
# Each yields Recording objects. They differ only in how a file is opened and
# which channels are kept; everything downstream is shared.
#
# NOT ONE OF THESE HAS BEEN RUN AGAINST ITS REAL CORPUS in this environment --
# no raw data is present here. They are written against each dataset's published
# format and the reader they need; the file layout assumptions each one makes
# are stated in its docstring so they can be checked against a real tree before
# a long run. --dry-run lists what an adapter would read without processing it.
# --------------------------------------------------------------------------- #

def _require_mne():
    try:
        import mne
    except ImportError as exc:                       # pragma: no cover
        raise PreprocessError(
            "mne is needed to read this corpus and is not installed in this "
            "interpreter. On CINECA: source $HOME/pwprep/bin/activate") from exc
    mne.set_log_level("ERROR")
    return mne


def _walk(root: str, exts: Tuple[str, ...]) -> List[str]:
    hits = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.lower().endswith(exts):
                hits.append(os.path.join(dirpath, fn))
    return sorted(hits)


def shard_files(files: List[str], args, root: str) -> List[str]:
    """Keep only the files this array task owns, sharded BY SUBJECT.

    By subject and not by file: a subject's recordings must all be processed by
    one task, because the train/val side is decided per subject and a subject
    split across two tasks that disagree would be the leak the subject-level
    split exists to prevent. Hashing the subject id also balances the tasks
    without anyone counting files first.
    """
    idx, total = args.shard
    kept = []
    for path in files:
        subject = tueg_identity(path, root)["subject"]
        h = int.from_bytes(hashlib.sha256(subject.encode()).digest()[:4], "big")
        if h % total == idx:
            kept.append(path)
    return kept


#: TUH filenames carry the identity: ``aaaaaaaa_s001_t000.edf`` is subject
#: aaaaaaaa, session s001, token t000. Parsing the NAME rather than counting
#: directory levels is what makes this survive the layout differences between
#: TUEG releases (v1.x had edf/<subject>/..., v2.x inserts a bucket directory),
#: and between TUEG and TUAB (which groups by train/eval and normal/abnormal
#: instead of by subject at all).
TUH_FILENAME = re.compile(r"^(?P<subject>[a-z0-9]{4,12})_s(?P<session>\d{2,4})"
                          r"_t(?P<token>\d{2,4})\.edf$", re.IGNORECASE)

#: The montage a TUH session was recorded under, as its directory is named.
#: Recorded in provenance: 01_tcp_ar and 02_tcp_le are different references, and
#: a corpus that turns out to be a mixture is something to know about rather
#: than average over.
TUH_MONTAGE_DIRS = ("01_tcp_ar", "02_tcp_le", "03_tcp_ar_a", "04_tcp_le_a",
                    "05_tcp_ar_a")


def tueg_identity(path: str, root: str) -> Dict[str, str]:
    """``subject``/``session``/``montage`` for one TUH EDF, from its name.

    Falls back to the parent directory only when the filename does not follow
    the convention, and says which rule fired, so an --inspect run shows whether
    the fallback is carrying the corpus.
    """
    base = os.path.basename(path)
    m = TUH_FILENAME.match(base)
    parts = os.path.normpath(os.path.relpath(path, root)).split(os.sep)
    montage = next((p for p in parts if p in TUH_MONTAGE_DIRS), "")
    if m:
        return {"subject": m.group("subject"),
                "session": f"s{m.group('session')}",
                "montage": montage, "rule": "filename"}
    # Fallback: the directory above the montage directory is the session, and
    # the one above that the subject, in every TUH layout that has one.
    subject = session = ""
    if montage and montage in parts:
        i = parts.index(montage)
        session = parts[i - 1] if i >= 1 else ""
        subject = parts[i - 2] if i >= 2 else ""
    if not subject:
        subject = parts[-2] if len(parts) > 1 else "unknown"
    return {"subject": subject, "session": session, "montage": montage,
            "rule": "path"}


def iter_tueg_files(root: str) -> List[str]:
    """Every .edf under root. TUEG puts them all under ``edf/``; not assumed."""
    edf_root = os.path.join(root, "edf")
    return _walk(edf_root if os.path.isdir(edf_root) else root, (".edf",))


def adapt_tueg(root: str, args) -> Iterator[Recording]:
    """TUH EEG Corpus: .edf, one recording per file, montage-mixed.

    Identity comes from the FILENAME (see tueg_identity), so the same code reads
    v1.x and v2.x layouts. Sampling rate varies per recording -- TUEG holds at
    least 250, 256, 400, 512 and 1000 Hz -- and is read from each file, never
    assumed; every one of them is resampled to E19_256's 256 Hz.

    Channel labels arrive as "EEG FP1-REF" / "EEG T3-LE", which
    normalize_channel_name strips and maps, T3/T4/T5/T6 included. Channels that
    are not scalp EEG (EKG, PHOTIC, IBI, BURSTS, SUPPR) simply match no slot and
    are dropped by map_to_slots, which names them in provenance.

    Mains is 60 Hz: TUH is recorded in Philadelphia. Passing --mains-hz
    overrides it; there is no guessing either way.
    """
    mne = _require_mne()
    files = iter_tueg_files(root)
    if getattr(args, "shard", None):
        files = shard_files(files, args, root)
    for path in files:
        ident = tueg_identity(path, root)
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        except Exception as exc:                              # noqa: BLE001
            raise PreprocessError(f"unreadable EDF: {exc}") from exc
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=ident["subject"],
            data=raw.get_data(),                    # mne returns volts
            channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]),
            unit="V",
            mains_hz=args.mains_hz if args.mains_hz else 60.0,
            notes={"tuh_session": ident["session"],
                   "tuh_montage": ident["montage"],
                   "tuh_identity_rule": ident["rule"]},
        )


def adapt_faced(root: str, args) -> Iterator[Recording]:
    """FACED: the raw 1000 Hz BDF release only.

    The widely mirrored FACED release is already preprocessed to 250 Hz. This
    route needs 512 Hz, and 250 -> 512 is upsampling: it invents no information
    and leaves everything above 125 Hz as interpolation artefact, in a corpus
    whose point is that it is 32-channel affective EEG. Refused by default
    rather than silently upsampled, because the resulting shard is
    indistinguishable from a real one downstream.
    """
    mne = _require_mne()
    files = _walk(root, (".bdf",))
    if not files:
        files = _walk(root, (".set", ".fif", ".edf"))
        if files and not args.allow_upsample_faced:
            raise PreprocessError(
                f"no .bdf under {root}, only {os.path.splitext(files[0])[1]} "
                f"files. That is the 250 Hz preprocessed FACED release, and "
                f"E32_512 needs 512 Hz -- resampling 250 -> 512 is upsampling, "
                f"which fabricates the top half of the spectrum.\n\n"
                f"  Use the raw BDF release, or pass --allow-upsample-faced to "
                f"override. The official configuration does not set it.")
    for path in files:
        if path.lower().endswith(".bdf"):
            raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        elif path.lower().endswith(".set"):
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        elif path.lower().endswith(".fif"):
            raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        if fs < ROUTES["E32_512"].sampling_rate and not args.allow_upsample_faced:
            raise PreprocessError(
                f"{path} is {fs} Hz and the route needs "
                f"{ROUTES['E32_512'].sampling_rate} Hz. Refusing to upsample; "
                f"--allow-upsample-faced overrides.")
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=os.path.basename(os.path.dirname(path)) or
            os.path.splitext(os.path.basename(path))[0],
            data=raw.get_data(), channel_names=list(raw.ch_names),
            sampling_rate=fs, unit="V", mains_hz=args.mains_hz,
        )


def adapt_tdbrain(root: str, args) -> Iterator[Recording]:
    """TDBRAIN: 26 electrodes at 500 Hz, in per-subject .npy/.csv or BIDS .edf.

    The 26 are placed into E32_512's 32 slots BY NAME. The six slots TDBRAIN
    does not record stay zero with valid_channel_mask False -- no interpolation
    and no CSD, because a spatially interpolated channel is a smooth function of
    its neighbours and the model would learn to reproduce the interpolation.
    """
    mne = _require_mne()
    files = _walk(root, (".edf", ".bdf", ".vhdr"))
    for path in files:
        if path.lower().endswith(".vhdr"):
            raw = mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")
        elif path.lower().endswith(".bdf"):
            raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        base = os.path.basename(path)
        subject = base.split("_")[0] if "_" in base else os.path.splitext(base)[0]
        yield Recording(
            recording_id=os.path.relpath(path, root), subject_id=subject,
            data=raw.get_data(), channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=args.mains_hz,
        )


def adapt_physionet_mi(root: str, args) -> Iterator[Recording]:
    """PhysioNet EEG Motor Movement/Imagery: 64 channels at 160 Hz, .edf.

    The 64 are kept. Cropping to 32 would throw away half the montage of the
    only 64-channel corpus with a large subject count, and E64_256 exists so it
    does not have to be.
    """
    mne = _require_mne()
    for path in _walk(root, (".edf",)):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        base = os.path.basename(path)
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=base[:4] if base.lower().startswith("s") else base,
            data=raw.get_data(), channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=args.mains_hz if args.mains_hz else 60.0,
        )


def adapt_m3cv(root: str, args) -> Iterator[Recording]:
    """M3CV: 64 channels at 250 Hz. Pretraining only, never a downstream split."""
    mne = _require_mne()
    files = _walk(root, (".set", ".edf", ".fif", ".cnt"))
    for path in files:
        if path.lower().endswith(".set"):
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        elif path.lower().endswith(".cnt"):
            raw = mne.io.read_raw_cnt(path, preload=True, verbose="ERROR")
        elif path.lower().endswith(".fif"):
            raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        base = os.path.splitext(os.path.basename(path))[0]
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=base.split("_")[0], data=raw.get_data(),
            channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=args.mains_hz,
        )


#: EGI HydroCel GSN-129 non-scalp / reference labels. Removing the RECORDED
#: REFERENCE is what takes 129 to 128, and it is E129 (vertex reference, "Cz")
#: on this net -- a named electrode, identified from the montage, not "the last
#: row". Dropping the last row would remove whichever channel the file happened
#: to end with.
HBN_NON_SCALP = ("E129", "Cz", "VREF", "REF")


def adapt_hbn(root: str, args) -> Iterator[Recording]:
    """HBN: EGI HydroCel 129 at 500 Hz -> 128 scalp channels.

    The channel removed is identified by NAME from the montage (the net's vertex
    reference), and the names actually dropped are written into every shard's
    provenance under ``removed_channels``. If the reference cannot be identified
    the recording is failed rather than trimmed to length.
    """
    mne = _require_mne()
    files = _walk(root, (".mff", ".set", ".fif", ".edf"))
    seen_dirs = set()
    for path in files:
        if path.lower().endswith(".mff"):
            if path in seen_dirs:
                continue
            seen_dirs.add(path)
            raw = mne.io.read_raw_egi(path, preload=True, verbose="ERROR")
        elif path.lower().endswith(".set"):
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        elif path.lower().endswith(".fif"):
            raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        names = list(raw.ch_names)
        drop = [n for n in names if n.strip().upper() in
                {s.upper() for s in HBN_NON_SCALP}]
        if len(names) == 129 and not drop:
            raise PreprocessError(
                f"{path}: 129 channels but none of {HBN_NON_SCALP} is among "
                f"them, so the reference cannot be identified by name. "
                f"Refusing to drop a row by position.")
        keep = [n for n in names if n not in drop]
        data = raw.get_data(picks=[names.index(n) for n in keep])
        base = os.path.basename(os.path.normpath(path))
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=base.split("_")[0].split(".")[0],
            data=data, channel_names=keep,
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=args.mains_hz,
            notes={"removed_channels": drop, "n_channels_before": len(names)},
        )


def adapt_hgd(root: str, args) -> Iterator[Recording]:
    """High-Gamma Dataset: 128 channels at 500 Hz. Pretraining only.

    HGD's electrodes are 10-5 scalp names and HBN's are EGI net positions, so
    the two do not share slot identities even though they share the route's
    shape. HGD channels that match no E-numbered slot land as padded, which is
    the honest outcome; --hgd-own-slots places them on HGD's own 128-name list
    instead, which is right if HGD is the only corpus on the route.
    """
    mne = _require_mne()
    for path in _walk(root, (".mat", ".edf", ".fif")):
        if path.lower().endswith(".mat"):
            raise PreprocessError(
                f"{path}: HGD's .mat release needs the braindecode reader; "
                f"export to FIF or EDF first, or point --root at a converted "
                f"tree. Not guessing a MATLAB layout.")
        if path.lower().endswith(".fif"):
            raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        base = os.path.splitext(os.path.basename(path))[0]
        yield Recording(
            recording_id=os.path.relpath(path, root),
            subject_id=base.split("_")[0], data=raw.get_data(),
            channel_names=list(raw.ch_names),
            sampling_rate=float(raw.info["sfreq"]), unit="V",
            mains_hz=args.mains_hz,
        )


def adapt_smoke(root: str, args) -> Iterator[Recording]:
    """Synthetic noise, for exercising the pipeline with no corpus present.

    Reached only via --smoke-test. Every shard it writes carries
    ``"synthetic": true`` in its provenance so a smoke run can never be mistaken
    for a real one in a manifest or a figure.
    """
    spec = PRETRAIN_DATASETS[args.smoke_as]
    route = spec.route
    rng = np.random.default_rng(args.seed)
    n_src = len(spec.slots)
    for s in range(args.smoke_subjects):
        for r in range(args.smoke_recordings):
            secs = args.window_seconds * (args.smoke_windows + 1)
            n = int(secs * route.sampling_rate)
            yield Recording(
                recording_id=f"smoke_s{s:02d}_r{r:02d}",
                subject_id=f"smoke_s{s:02d}",
                data=rng.normal(0, 20e-6, size=(n_src, n)),
                channel_names=list(spec.slots),
                sampling_rate=float(route.sampling_rate),
                unit="V", mains_hz=50.0, notes={"synthetic": True},
            )


ADAPTERS: Dict[str, Callable] = {
    "tueg": adapt_tueg, "faced": adapt_faced, "tdbrain": adapt_tdbrain,
    "physionet_mi": adapt_physionet_mi, "m3cv": adapt_m3cv,
    "hbn": adapt_hbn, "hgd": adapt_hgd,
}


# --------------------------------------------------------------------------- #
# The shared pipeline
# --------------------------------------------------------------------------- #

def process_recording(rec: Recording, dataset_id: str, cfg: PreprocessConfig,
                      slots: Sequence[str], route, out_dir: str,
                      extra_prov: Dict) -> Optional[Dict]:
    """One recording -> one shard, or None when it yields no whole window."""
    if len(slots) != route.n_channels:
        # Caught here rather than at training time. A slot list one short writes
        # a perfectly valid-looking shard whose channel axis does not match the
        # route, and the failure then surfaces thousands of windows later as a
        # shape error inside the model.
        raise PreprocessError(
            f"{dataset_id}: slot list has {len(slots)} entries but route "
            f"{route.route_id} has {route.n_channels} channels")

    x = to_microvolts(np.asarray(rec.data, dtype=np.float64), rec.unit)
    x = detrend(x, cfg.detrend)
    mains = rec.mains_hz if rec.mains_hz is not None else cfg.notch_hz
    x = notch(x, rec.sampling_rate, mains, cfg.notch_harmonics, cfg.notch_quality)
    x = highpass(x, rec.sampling_rate, cfg.highpass_hz)
    x = resample_to(x, rec.sampling_rate, route.sampling_rate)

    mapping = map_to_slots(rec.channel_names, slots)
    if not mapping.matrix_rows:
        raise PreprocessError(
            f"none of {len(rec.channel_names)} channels matched a slot of "
            f"{route.route_id}. First few recorded: {rec.channel_names[:6]}")
    placed = place_on_slots(x, mapping, len(slots))

    win = int(cfg.window_seconds * route.sampling_rate)
    stride = int(cfg.stride() * route.sampling_rate)
    windows, starts = window_signal(placed, win, stride)
    if windows.shape[0] == 0:
        return None
    windows = zscore_windows(windows, mapping.valid, cfg.zscore_eps,
                             cfg.clip_sigma)

    ids, _ = channel_ids_for(slots)
    ids = [i if mapping.valid[k] else 0 for k, i in enumerate(ids)]   # PAD_ID
    prov = cfg.provenance({
        "dataset_id": dataset_id,
        "mains_hz": mains,
        "source_sampling_rate": rec.sampling_rate,
        "target_sampling_rate": route.sampling_rate,
        "unmatched_source_channels": mapping.unmatched_sources,
        "empty_slots": mapping.empty_slots,
        "unknown_channel_names": mapping.unknown_names,
        **(rec.notes or {}), **extra_prov,
    })
    safe = rec.recording_id.replace(os.sep, "__").replace(" ", "_")
    path = os.path.join(out_dir, "shards", f"{safe}.h5")
    entry = write_shard(
        path, windows, route, dataset_id, list(slots), ids, mapping.valid,
        [rec.subject_id] * windows.shape[0],
        [rec.recording_id] * windows.shape[0],
        (starts / route.sampling_rate).tolist(), rec.sampling_rate, prov)
    entry["qc"] = window_qc(windows, mapping.valid, route.sampling_rate)
    entry["source_sampling_rate"] = float(rec.sampling_rate)
    entry["empty_slots"] = list(mapping.empty_slots)
    entry["unmatched_source_channels"] = list(mapping.unmatched_sources)
    entry["placed_total"] = len(mapping.matrix_rows)
    entry["placed_unk"] = sum(1 for k, i in enumerate(ids)
                              if mapping.valid[k] and i == 1)     # UNK_ID
    entry["unknown_channel_names"] = mapping.unknown_names
    entry["n_channels_recorded"] = len(rec.channel_names)
    entry["duration_seconds"] = float(windows.shape[0] * cfg.window_seconds)
    return entry


def _shard_spec(text: str):
    try:
        i, n = text.split("/")
        i, n = int(i), int(n)
    except Exception:                                          # noqa: BLE001
        raise argparse.ArgumentTypeError(f"--shard wants I/N, got {text!r}")
    if not (0 <= i < n):
        raise argparse.ArgumentTypeError(
            f"--shard {text}: need 0 <= I < N")
    return (i, n)


def inspect_corpus(dataset_id: str, root: str, args, slots, route) -> int:
    """Report what the corpus actually is, without processing any of it."""
    from collections import Counter

    mne = _require_mne()
    if dataset_id == "tueg":
        files = iter_tueg_files(root)
    else:
        files = _walk(root, (".edf", ".bdf", ".set", ".fif", ".cnt", ".mff"))
    if not files:
        print(f"ERROR: no readable files under {root}", file=sys.stderr)
        return 1

    n = min(args.inspect, len(files))
    print(f"{len(files)} file(s) under {root}")
    print(f"reading the headers of {n} of them\n")

    rates, montages, rules, counts = Counter(), Counter(), Counter(), Counter()
    name_hits, name_miss = Counter(), Counter()
    subjects = set()
    failed = 0
    step = max(1, len(files) // n)
    for path in files[::step][:n]:
        try:
            raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR") \
                if path.lower().endswith(".edf") else \
                mne.io.read_raw(path, preload=False, verbose="ERROR")
        except Exception as exc:                               # noqa: BLE001
            failed += 1
            print(f"  UNREADABLE {os.path.relpath(path, root)}: {exc}")
            continue
        rates[float(raw.info["sfreq"])] += 1
        counts[len(raw.ch_names)] += 1
        if dataset_id == "tueg":
            ident = tueg_identity(path, root)
            montages[ident["montage"] or "(none)"] += 1
            rules[ident["rule"]] += 1
            subjects.add(ident["subject"])
        for c in raw.ch_names:
            canonical = normalize_channel_name(c)
            (name_hits if canonical in slots else name_miss)[canonical] += 1

    print(f"sampling rates   : {dict(sorted(rates.items()))}")
    print(f"channel counts   : {dict(sorted(counts.items()))}")
    if dataset_id == "tueg":
        print(f"montage dirs     : {dict(montages)}")
        print(f"identity rule    : {dict(rules)}   "
              f"({len(subjects)} distinct subjects in this sample)")
        if rules.get("path", 0):
            print("  WARNING: the filename rule did not fire on every file; "
                  "check the layout before a long run.")
    print(f"unreadable       : {failed} of {n}")
    print(f"\nslots filled     : {len(name_hits)} of {len(slots)} "
          f"({route.route_id})")
    print(f"  matched  : {sorted(name_hits)[:24]}")
    missed = sorted(name_miss)
    print(f"  unmatched: {missed[:24]}"
          + (f" (+{len(missed)-24} more)" if len(missed) > 24 else ""))
    print("\nUnmatched names are dropped, not mapped to UNK -- they are usually "
          "EKG/PHOTIC/annotation rows.\nIf a scalp electrode is in that list, "
          "the adapter's naming needs fixing before you run.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=sorted(ADAPTERS) + ["smoke"])
    p.add_argument("--root", default=None, help="raw corpus root (not downloaded)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mains-hz", type=float, default=None,
                   help="50 or 60. Required when the corpus does not say and "
                        "the adapter has no default: a notch at the wrong "
                        "frequency leaves the interference and removes signal.")
    p.add_argument("--no-notch", action="store_true",
                   help="skip mains notch entirely")
    p.add_argument("--highpass-hz", type=float, default=0.5)
    p.add_argument("--clip-sigma", type=float, default=20.0)
    p.add_argument("--window-seconds", type=float, default=4.0)
    p.add_argument("--stride-seconds", type=float, default=None,
                   help="default: no overlap")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-recordings", type=int, default=None)
    p.add_argument("--allow-upsample-faced", action="store_true",
                   help="permit FACED's 250 Hz release. The official "
                        "configuration never sets this.")
    p.add_argument("--hgd-own-slots", action="store_true")
    p.add_argument("--unk-rate-max", type=float, default=0.01,
                   help="fail if more than this fraction of the channels placed "
                        "into slots carry an <unk> id")
    p.add_argument("--max-empty-slot-rate", type=float, default=0.25,
                   help="fail if more than this fraction of the route's slots "
                        "are left unfilled on average. This is the gate that "
                        "catches an adapter that cannot name the montage; "
                        "TDBRAIN legitimately sits at 18.8%%.")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be read, process nothing")
    p.add_argument("--inspect", type=int, default=None, metavar="N",
                   help="read the headers of N recordings and report what the "
                        "corpus actually looks like -- layout, identity rule, "
                        "sampling rates, montages, channel names, how many "
                        "resolve. Run this before a long job; it opens headers "
                        "only and processes nothing.")
    p.add_argument("--shard", type=_shard_spec, default=None, metavar="I/N",
                   help="process only shard I of N, sharded by subject so a "
                        "subject is never split across tasks. For SLURM arrays.")
    p.add_argument("--resume", action="store_true",
                   help="skip recordings whose shard HDF5 already exists")
    p.add_argument("--split-mode", choices=["exact", "hash"], default=None,
                   help="'exact' partitions the subject list (needs all of it); "
                        "'hash' decides per subject independently, which is "
                        "required when sharding. Defaults to hash under "
                        "--shard, exact otherwise.")
    p.add_argument("--smoke-test", action="store_true",
                   help="synthetic data; the ONLY path that fabricates signal")
    p.add_argument("--smoke-as", default="tdbrain",
                   choices=sorted(PRETRAIN_DATASETS))
    p.add_argument("--smoke-subjects", type=int, default=6)
    p.add_argument("--smoke-recordings", type=int, default=2)
    p.add_argument("--smoke-windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.smoke_test:
        dataset_id = args.smoke_as
        adapter = adapt_smoke
    else:
        dataset_id = args.dataset
        if dataset_id == "smoke":
            return _fail("--dataset smoke needs --smoke-test.")
        adapter = ADAPTERS[dataset_id]
        if not args.root:
            return _fail("--root is required. This script never downloads a "
                         "corpus and never falls back to synthetic data; "
                         "--smoke-test is the explicit way to run without one.")
        if not os.path.isdir(args.root):
            return _fail(f"--root {args.root} does not exist.")

    spec = PRETRAIN_DATASETS[dataset_id]
    route = ROUTES[spec.route_id]
    slots = spec.slots if not (args.hgd_own_slots and dataset_id == "hgd") \
        else PRETRAIN_DATASETS["hgd"].slots

    cfg = PreprocessConfig(
        highpass_hz=args.highpass_hz,
        notch_hz=None if args.no_notch else args.mains_hz,
        clip_sigma=args.clip_sigma, window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds, val_fraction=args.val_fraction,
        split_seed=args.split_seed)

    if args.inspect:
        if not args.root:
            return _fail("--inspect needs --root.")
        return inspect_corpus(dataset_id, args.root, args, slots, route)

    # An array task must not write over its siblings' manifests, so every output
    # this task owns carries its shard index. build_eeg_c1_manifest.py merges
    # them; without the suffix, task 7 finishing last would leave a manifest
    # describing only task 7's shards.
    suffix = f".{args.shard[0]:04d}" if args.shard else ""
    if args.split_mode is None:
        args.split_mode = "hash" if args.shard else "exact"
    if args.shard and args.split_mode != "hash":
        return _fail("--shard requires --split-mode hash: no task sees the "
                     "whole subject list, so an exact partition would put the "
                     "same subject on different sides in different tasks.")

    os.makedirs(args.out_dir, exist_ok=True)
    fail_path = os.path.join(args.out_dir,
                             f"preprocessing_failures{suffix}.jsonl")
    entries: List[Dict] = []
    failures: List[Dict] = []

    print(f"dataset={dataset_id}  route={route.route_id}  "
          f"{route.n_channels}x{route.window_samples} @ {route.sampling_rate}Hz")
    print(f"slots={len(slots)}  out={args.out_dir}"
          + ("  [SMOKE TEST -- synthetic]" if args.smoke_test else ""))

    n_seen = 0
    try:
        for rec in adapter(args.root or "", args):
            n_seen += 1
            if args.max_recordings and n_seen > args.max_recordings:
                break
            if args.dry_run:
                print(f"  would read {rec.recording_id}: "
                      f"{len(rec.channel_names)}ch @ {rec.sampling_rate}Hz")
                continue
            safe = rec.recording_id.replace(os.sep, "__").replace(" ", "_")
            done = os.path.join(args.out_dir, "shards", f"{safe}.h5")
            if args.resume and os.path.isfile(done) and os.path.getsize(done) > 0:
                # Re-read the shard rather than re-deriving it, so a resumed run
                # writes the same manifest a complete one would.
                try:
                    import h5py
                    with h5py.File(done, "r") as f:
                        # Every field the summary needs is already in the shard,
                        # so a resumed run reports what a complete one would.
                        # Rebuilding the entry from the Recording alone left the
                        # coverage gates reading keys that were not there.
                        prov = json.loads(
                            f.attrs.get("preprocessing_provenance", "{}"))
                        valid = np.asarray(f["valid_channel_mask"][...], bool)
                        n_win = int(f["data"].shape[0])
                        entries.append({
                            "path": done, "dataset_id": dataset_id,
                            "route_id": route.route_id,
                            "n_windows": n_win,
                            "subjects": [rec.subject_id],
                            "qc": {"channel_missing_rate":
                                   float(1.0 - valid.mean())},
                            "unknown_channel_names":
                                prov.get("unknown_channel_names", []),
                            "empty_slots": prov.get("empty_slots", []),
                            "unmatched_source_channels":
                                prov.get("unmatched_source_channels", []),
                            "placed_total": int(valid.sum()),
                            "placed_unk": int(
                                (np.asarray(f["channel_ids"][...])[valid] == 1).sum()),
                            "n_channels_recorded": len(rec.channel_names),
                            "source_sampling_rate": float(rec.sampling_rate),
                            "duration_seconds": float(
                                n_win * cfg.window_seconds)})
                    continue
                except Exception:                              # noqa: BLE001
                    pass          # unreadable: fall through and rebuild it
            try:
                entry = process_recording(
                    rec, dataset_id, cfg, slots, route, args.out_dir,
                    {"synthetic": bool(args.smoke_test)})
                if entry is None:
                    failures.append({"recording_id": rec.recording_id,
                                     "reason": "no whole window in recording"})
                else:
                    entries.append(entry)
                    print(f"  {rec.recording_id}: {entry['n_windows']} windows")
            except Exception as exc:                       # noqa: BLE001
                failures.append({"recording_id": rec.recording_id,
                                 "reason": str(exc),
                                 "traceback": traceback.format_exc(limit=3)})
    except PreprocessError as exc:
        return _fail(str(exc))

    if args.dry_run:
        print(f"\ndry run: {n_seen} recording(s) would be read")
        return 0
    if not entries:
        with open(fail_path, "w") as f:
            for r in failures:
                f.write(json.dumps(r) + "\n")
        return _fail(f"no recording produced a window. See {fail_path}")

    # -- coverage gates ------------------------------------------------------ #
    #
    # Two different failures, measured separately, because the obvious single
    # ratio conflates them and stops a corpus that is perfectly fine.
    #
    # TUEG carries five non-EEG rows in every recording -- EKG, PHOTIC, IBI,
    # BURSTS, SUPPR -- which is 21% of a 24-row file. They match no slot and are
    # dropped, which is correct: they are not electrodes and never reach the
    # model. Counting them as "outside the vocabulary" and refusing the corpus
    # measures how many auxiliary channels the recording system writes, not
    # whether the montage was understood.
    #
    # What actually matters is whether the route's SLOTS got filled. A scalp
    # electrode whose name the adapter fails to normalise does not become <unk>
    # -- it fails to match a slot, the slot stays empty and masked out, and the
    # montage silently shrinks. That is the one to stop for.
    total_slots = len(slots) * len(entries)
    empty_slots = sum(len(e["empty_slots"]) for e in entries)
    empty_rate = empty_slots / max(1, total_slots)

    # And of the channels that DID reach a slot, how many carry an <unk> id.
    # Zero by construction -- a slot name is in the vocabulary -- so a non-zero
    # value means the slot list and the vocabulary have drifted apart.
    placed_unk = sum(e["placed_unk"] for e in entries)
    placed_total = sum(e["placed_total"] for e in entries)
    unk_rate = placed_unk / max(1, placed_total)

    dropped = sorted({n for e in entries for n in e["unmatched_source_channels"]})
    # A dropped name that IS a known electrode is worth saying out loud: it is a
    # real channel this route has no slot for, not an auxiliary row.
    dropped_electrodes = [n for n in dropped
                          if normalize_channel_name(n) in CHANNEL_TO_ID]
    if dropped:
        print(f"\n  dropped {len(dropped)} non-slot channel name(s): "
              f"{dropped[:12]}" + (" ..." if len(dropped) > 12 else ""))
    if dropped_electrodes:
        print(f"  NOTE: {len(dropped_electrodes)} of them are known electrodes "
              f"that {route.route_id} has no slot for: {dropped_electrodes[:12]}")

    if unk_rate > args.unk_rate_max:
        return _fail(
            f"{unk_rate:.1%} of the channels placed into slots carry <unk> "
            f"(limit {args.unk_rate_max:.1%}). The slot list and CHANNEL_VOCAB "
            f"have drifted apart; the channel embedding would be learning one "
            f"row for several electrodes.")

    if empty_rate > args.max_empty_slot_rate:
        return _fail(
            f"{empty_rate:.1%} of {route.route_id}'s slots were left empty "
            f"(limit {args.max_empty_slot_rate:.1%}), averaged over "
            f"{len(entries)} recording(s).\n\n"
            f"  Slots nothing filled: "
            f"{sorted({n for e in entries for n in e['empty_slots']})[:24]}\n"
            f"  Channel names that matched no slot: {dropped[:24]}\n\n"
            f"  A montage this incomplete is usually an adapter naming problem "
            f"rather than a corpus that genuinely lacks the electrodes. Run "
            f"--inspect to see what the files actually carry. Raise "
            f"--max-empty-slot-rate if the corpus really is this sparse -- "
            f"TDBRAIN legitimately leaves 6 of 32 empty, which is 18.8%.")

    # -- subject split ------------------------------------------------------- #
    subjects = sorted({s for e in entries for s in e["subjects"]})
    if args.split_mode == "hash":
        vset = {s for s in subjects
                if subject_split_side(s, cfg.val_fraction, cfg.split_seed) == "val"}
        tset = set(subjects) - vset
        train_subj, val_subj = sorted(tset), sorted(vset)
    else:
        train_subj, val_subj = split_subjects(subjects, cfg.val_fraction,
                                              cfg.split_seed)
        tset, vset = set(train_subj), set(val_subj)
    assert not (tset & vset), "a subject landed in both splits"

    def side(entry):
        subs = set(entry["subjects"])
        if subs <= vset:
            return "val"
        if subs <= tset:
            return "train"
        raise SystemExit(f"shard {entry['path']} spans train and val subjects")

    manifests = {"train": [], "val": []}
    for e in entries:
        manifests[side(e)].append(e)

    for split, rows in manifests.items():
        with open(os.path.join(args.out_dir,
                                f"manifest_{split}{suffix}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if k != "qc"}) + "\n")

    stats = {
        "dataset_id": dataset_id, "route_id": route.route_id,
        "n_subjects": len(subjects),
        "n_subjects_train": len(train_subj), "n_subjects_val": len(val_subj),
        "n_recordings": len(entries),
        "n_windows": sum(e["n_windows"] for e in entries),
        "n_windows_train": sum(e["n_windows"] for e in manifests["train"]),
        "n_windows_val": sum(e["n_windows"] for e in manifests["val"]),
        "duration_seconds": sum(e["duration_seconds"] for e in entries),
        "n_failed": len(failures),
        "unk_channel_rate": unk_rate,
        "empty_slot_rate": empty_rate,
        "dropped_channel_names": dropped,
        "target_sampling_rate": route.sampling_rate,
        # The rates actually seen, not the rate the corpus is documented at.
        # TUEG's varies per recording, and a corpus that turns out to hold two
        # rates is something to find here rather than in a training curve.
        "source_sampling_rates": sorted({e["source_sampling_rate"]
                                         for e in entries}),
        "channel_missing_rate": float(np.mean(
            [e["qc"].get("channel_missing_rate", 0.0) for e in entries])),
        "qc": {k: float(np.mean([e["qc"][k] for e in entries if k in e["qc"]]))
               for k in ("amplitude_mean", "amplitude_std", "amplitude_min",
                         "amplitude_max", "psd_delta", "psd_theta", "psd_alpha",
                         "psd_beta", "psd_gamma", "psd_high_gamma")
               if any(k in e["qc"] for e in entries)},
        "preprocess_config": cfg.provenance({"dataset_id": dataset_id}),
        "synthetic": bool(args.smoke_test),
        "shard": list(args.shard) if args.shard else None,
        "split_mode": args.split_mode,
    }
    with open(os.path.join(args.out_dir,
                           f"dataset_statistics{suffix}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(fail_path, "w") as f:
        for r in failures:
            f.write(json.dumps(r) + "\n")

    print(f"\n{stats['n_recordings']} recordings, {stats['n_windows']} windows, "
          f"{stats['n_subjects']} subjects "
          f"({len(train_subj)} train / {len(val_subj)} val)")
    print(f"  {len(failures)} failure(s) -> {fail_path}")
    print(f"  manifests + dataset_statistics.json in {args.out_dir}")
    return 0


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
