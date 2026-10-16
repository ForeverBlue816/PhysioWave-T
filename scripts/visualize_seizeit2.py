#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper figures for the SeizeIT2 multimodal runs, from their own artefacts.

    python scripts/visualize_seizeit2.py --raw-dir $FUSION_ROOT/raw/seizeit2 \\
        --run-dir $PW_CKPT_ROOT/fusion/ralf_frozen_szo \\
        --compare $PW_CKPT_ROOT/fusion/{concat_frozen,weighted_logit_frozen,ralf_finetune,ralf_scratch}_szo \\
        --eeg-runs $PW_CKPT_ROOT/fusion/eeg_{c1,eegpt,labram}_finetune_sze \\
        --out-dir docs/runs/seizeit2_fusion --format svg,pdf --paper

Nothing is illustrative. The traces are the test patients' recordings, the
probabilities are the ones the event-level evaluation cached
(<run>/predictions/*.npz, Fusion/seizeit2_events.py), the curves are
events_cross.json, the reliabilities and AUPRCs are results.json, the
learning curves are history.json.

  fig_seizure_example      one test recording around one seizure: the two
                           behind-the-ear EEG channels, ECG and EMG as recorded;
                           the model's per-second probability under every
                           modality subset; the RMS gate; the detection after
                           post-processing. The recording is CHOSEN: the
                           candidates are ranked and the ranking is kept in
                           figure_metadata, so the caption can say so.
  fig_sensitivity_fa       sensitivity against false alarms per hour (the
                           dataset paper's scoring) over the threshold curve:
                           every run, every modality subset of the main run,
                           the published SVM and ChronoNet as points.
  fig_modality_subsets     what each modality subset reaches at <= 11 FA/h
                           (the SVM's operating point) and its window AUPRC.
  fig_reliability          RALF's per-modality reliability on the clean test
                           set and under each corruption; the AUPRC every run
                           keeps under the same corruptions.
  fig_per_patient          per test patient: seizures, sensitivity at
                           threshold 0.5, FA/h.
  fig_learning_curves      validation AUPRC per epoch, the selected epoch marked.

Style and plumbing (Times New Roman bold, Okabe-Ito, svg+pdf with editable
text, figure_data / figure_metadata beside every figure) are those of
scripts/visualize_eeg_pretraining.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "Fusion"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402

from visualize_eeg_pretraining import (OKABE_ITO, FigureWriter,  # noqa: E402
                                       apply_style, git_commit)
from physiowave.fusion.prep_common import (MODALITIES, edf_units, filter_continuous,  # noqa: E402
                                           read_edf, to_microvolts)
from physiowave.fusion.seizure_events import post_process, score_recording  # noqa: E402
from seizeit2_events import SUBSETS                            # noqa: E402
from seizeit2_prep import MAINS_HZ, find_runs, read_events     # noqa: E402

MOD_COLOR = {"eeg": OKABE_ITO[0], "ecg": OKABE_ITO[1], "emg": OKABE_ITO[2]}
SUBSET_COLOR = {"complete": "#000000", "eeg_only": MOD_COLOR["eeg"], "ecg_only": MOD_COLOR["ecg"],
                "emg_only": MOD_COLOR["emg"], "eeg_ecg": OKABE_ITO[3], "eeg_emg": OKABE_ITO[4],
                "ecg_emg": OKABE_ITO[5]}
SUBSET_LABEL = {"complete": "EEG+ECG+EMG", "eeg_only": "EEG", "ecg_only": "ECG",
                "emg_only": "EMG", "eeg_ecg": "EEG+ECG", "eeg_emg": "EEG+EMG",
                "ecg_emg": "ECG+EMG"}
MOD_LABEL = {"eeg": "bte-EEG", "ecg": "ECG", "emg": "EMG"}
#: (condition id in results.json, modality, label)
CORRUPTIONS = [("eeg_noise", "eeg", "noise"), ("eeg_bad_channel", "eeg", "bad channel"),
               ("eeg_drift", "eeg", "drift"), ("ecg_saturation", "ecg", "saturation"),
               ("ecg_noise", "ecg", "noise"), ("emg_dropout", "emg", "dropout"),
               ("emg_noise", "emg", "noise")]
PUBLISHED = [("SVM (Bhagubai et al.)", 71.1, 11.0, "s"), ("ChronoNet (Bhagubai et al.)", 84.2, 100.5, "D")]
FA_BUDGET = 11.0


# --------------------------------------------------------------------------- #
# Run artefacts
# --------------------------------------------------------------------------- #

class Run:
    def __init__(self, path: str):
        self.path = path.rstrip("/")
        self.name = os.path.basename(self.path)
        self.results = self._json("results.json")
        self.history = self._json("history.json") or []
        self.events = self._json("events_cross.json")
        self.pred_dir = os.path.join(self.path, "predictions")
        self.fusion = (self.results or {}).get("fusion", "?")
        self.mode = (self.results or {}).get("encoder_mode", "?")

    def _json(self, name):
        f = os.path.join(self.path, name)
        return json.load(open(f)) if os.path.isfile(f) else None

    @property
    def label(self) -> str:
        n = self.name
        for suf in ("_szo", "_sze"):
            n = n.replace(suf, "")
        if n.startswith("eeg_"):
            enc = {"c1": "C1", "c1native": "C1 (native)", "eegpt": "EEGPT", "labram": "LaBraM"}
            parts = n.split("_")
            return f"{enc.get(parts[1], parts[1])} {parts[2]}, EEG"
        return f"{self.fusion.replace('weighted_logit', 'weighted logit')} / {self.mode}"

    def test(self) -> Dict[str, Dict]:
        return ((self.results or {}).get("tasks", {}).get("seizeit2", {}) or {}).get("test", {})

    def curve(self, cond: str = "complete") -> Optional[List[Dict]]:
        c = (self.events or {}).get("conditions", {}).get(cond)
        return c["curve"] if c else None

    def sens_at(self, cond: str, budget: float) -> Optional[float]:
        cur = self.curve(cond)
        if not cur:
            return None
        ok = [r["official"]["sens_ovlp"] for r in cur
              if r["official"]["fa_ovlp_h"] == r["official"]["fa_ovlp_h"]
              and r["official"]["fa_ovlp_h"] <= budget]
        return max(ok) if ok else 0.0

    def predictions(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.pred_dir, "*.npz")))


def gated(z, cond: str) -> np.ndarray:
    p = z[f"prob_{cond}"].astype(np.float64)
    g = z["gate_cross"][: len(p)]
    return np.where(g, np.nan_to_num(p), 0.0)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def choose_example(run: Run, preferred: Optional[str]) -> Tuple[str, List[Dict]]:
    """Rank the recordings with a seizure and all three modalities by how far the
    complete model's probability inside the seizure sits above its surroundings;
    the best detected one is drawn. The ranking is returned for the metadata."""
    rows = []
    for f in run.predictions():
        z = np.load(f)
        y = z["label"].astype(bool)
        if not y.any() or not bool(z["present"].all()):
            continue
        p = gated(z, "complete")
        det = post_process(p, 0.5)
        from physiowave.fusion.seizure_events import mask_to_events
        for a, b in mask_to_events(y):
            a, b = int(a), int(b)
            lo, hi = max(0, a - 120), min(len(p), b + 120)
            around = np.concatenate([p[lo:a], p[b:hi]])
            contrast = float(p[a:b].mean() - (around.mean() if around.size else 0.0))
            rows.append({"recording": os.path.basename(f)[:-4], "onset_s": a, "offset_s": b,
                         "duration_s": b - a, "contrast": contrast,
                         "detected": bool(det[a:b].any()), "n_seizures": len(mask_to_events(y))})
    rows.sort(key=lambda r: (not r["detected"], -r["contrast"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    if preferred:
        pick = next((r for r in rows if r["recording"] == preferred), None)
        if pick is None:
            raise SystemExit(f"{preferred}: not among the recordings with a seizure and all "
                             f"three modalities ({len(rows)} candidates)")
    else:
        if not rows:
            raise SystemExit("no recording with a seizure and all three modalities")
        pick = rows[0]
    pick["chosen"] = True
    return pick["recording"], rows


def fig_seizure_example(w: FigureWriter, run: Run, raw_dir: str, preferred: Optional[str],
                        margin: float):
    rec, ranking = choose_example(run, preferred)
    subject, runid = rec.split("_run-")
    cand = [r for r in find_runs(raw_dir, subject + "$") if r["run"] == runid]
    if not cand:
        raise SystemExit(f"{rec}: not under {raw_dir}")
    r = cand[0]
    z = np.load(os.path.join(run.pred_dir, rec + ".npz"))
    pick = next(x for x in ranking if x.get("chosen"))
    t0, t1 = pick["onset_s"] - margin, pick["offset_s"] + margin
    t0 = max(0.0, t0)
    sig, rates, T = read_edf(r["eeg"])
    units = edf_units(r["eeg"])
    t1 = min(float(T), t1)
    names = sorted(sig, key=lambda n: (0 if n.upper().startswith("BTE") else 1, n))[:2]
    traces = []                                     # (label, t, y, unit, colour)
    fs = float(rates[names[0]])
    eeg = np.stack([to_microvolts(np.asarray(sig[n], np.float64), units.get(n, "uV"))
                    for n in names])
    eeg = filter_continuous(eeg, fs, "eeg", MAINS_HZ)
    a, b = int(t0 * fs), int(t1 * fs)
    for n, row in zip(names, eeg):
        traces.append((n.replace(" SD", ""), np.arange(a, b) / fs, row[a:b], "µV", MOD_COLOR["eeg"]))
    for m in ("ecg", "emg"):
        if not r.get(m) or not os.path.isfile(r[m]):
            continue
        s2, r2, _ = read_edf(r[m])
        u2 = edf_units(r[m])
        lab = next(iter(s2))
        f2 = float(r2[lab])
        x = filter_continuous(np.asarray(s2[lab], np.float64)[None], f2, m, MAINS_HZ)[0]
        a2, b2 = int(t0 * f2), int(t1 * f2)
        traces.append((MOD_LABEL[m], np.arange(a2, b2) / f2, x[a2:b2], u2.get(lab, ""), MOD_COLOR[m]))
    conds = [c for c, _ in SUBSETS if c in ("complete", "eeg_only", "ecg_only", "emg_only")]
    n_rows = len(traces) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, 1.0 * len(traces) + 1.9), sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(traces) + [1.9]})
    onset, offset = pick["onset_s"], pick["offset_s"]
    fig.subplots_adjust(right=0.84, hspace=0.25)
    for ax, (lab, t, y, unit, col) in zip(axes, traces):
        ax.plot(t - onset, y, color=col, lw=0.5)
        ax.axvspan(0, offset - onset, color="0.85", alpha=0.6, lw=0, zorder=0)
        amp = float(np.percentile(np.abs(y), 99.5)) if y.size else 1.0
        ax.set_ylabel(f"{lab}\n±{amp:.0f} {unit}".rstrip(), rotation=0, ha="right", va="center",
                      labelpad=6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
        ax.set_ylim(-amp * 1.1, amp * 1.1)
    ax = axes[-1]
    sec = np.arange(int(t0), int(t1))
    for c in conds:
        p = gated(z, c)
        if np.isnan(z[f"prob_{c}"]).all():
            continue
        ax.step(sec - onset + 0.5, p[int(t0):int(t1)], where="mid", color=SUBSET_COLOR[c],
                lw=1.4 if c == "complete" else 0.9, label=SUBSET_LABEL[c],
                alpha=1.0 if c == "complete" else 0.9)
    gate = z["gate_cross"][int(t0):int(t1)]
    for s, g in zip(sec, gate):
        if not g:
            ax.axvspan(s - onset, s - onset + 1, color="0.6", alpha=0.35, lw=0, hatch="///")
    det = post_process(gated(z, "complete"), 0.5)[int(t0):int(t1)]
    ax.fill_between(sec - onset + 0.5, 1.02, 1.10, where=det, step="mid", color="black", lw=0)
    ax.text(1.005, 1.06 / 1.12, "detected", fontsize=6, ha="left", va="center",
            transform=ax.transAxes)
    ax.axhline(0.5, color="0.5", lw=0.6, ls="--")
    ax.axvspan(0, offset - onset, color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("P(seizure)")
    ax.set_xlabel("time from annotated onset (s)")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(1.0, 0.92), handlelength=1.5,
              borderaxespad=0.2)
    ax.set_xlim(t0 - onset, t1 - onset)
    fig.suptitle(f"{run.label}: {subject} run-{runid}, seizure {pick['rank']} of {len(ranking)} "
                 f"ranked candidates ({pick['duration_s']} s)", y=0.995)
    fig.text(0.5, 0.03, "grey band: annotated seizure; hatched: seconds the 13-150 µV RMS gate "
                        "forces to 0; black bar: detection after post-processing at threshold 0.5",
             ha="center", fontsize=6.5, color="0.35")
    w.save(fig, "fig_seizure_example",
           data={"t0": t0, "t1": t1, "onset": onset, "offset": offset,
                 **{f"prob_{c}": gated(z, c)[int(t0):int(t1)] for c in conds},
                 "gate": gate, "label": z["label"][int(t0):int(t1)],
                 **{f"trace_{i}": y for i, (_, _, y, _, _) in enumerate(traces)}},
           meta={"recording": rec, "eeg_file": r["eeg"], "chosen": pick, "ranking": ranking[:20],
                 "n_candidates": len(ranking), "margin_s": margin})


def _plot_curve(ax, cur, color, label, lw=1.2, ls="-"):
    fa = np.array([c["official"]["fa_ovlp_h"] for c in cur])
    se = np.array([c["official"]["sens_ovlp"] for c in cur]) * 100
    ok = np.isfinite(fa) & np.isfinite(se)
    o = np.argsort(fa[ok])
    ax.plot(np.maximum(fa[ok][o], 0.05), se[ok][o], color=color, lw=lw, ls=ls, label=label)


def fig_sensitivity_fa(w: FigureWriter, main: Run, others: List[Run], eeg_runs: List[Run]):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    runs = [main] + others
    for i, r in enumerate(runs):
        cur = r.curve("complete")
        if cur:
            _plot_curve(a, cur, OKABE_ITO[i % 8], r.label, lw=1.6 if r is main else 1.0)
    for j, r in enumerate(eeg_runs):
        cur = r.curve("complete")
        if cur:
            _plot_curve(a, cur, OKABE_ITO[(len(runs) + j) % 8], r.label, lw=0.9, ls="--")
    for lab, s, fa, mk in PUBLISHED:
        a.plot(fa, s, marker=mk, color="black", ms=5, ls="none", label=lab, zorder=5)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlim(0.05, 300)
        ax.set_ylim(0, 100)
        ax.axvline(FA_BUDGET, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel("false alarms per hour (any-overlap, mean over recordings)")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    a.set_ylabel("sensitivity (%)")
    a.set_title("(a) runs, all modalities present")
    a.legend(frameon=False, fontsize=5.5, loc="lower right")
    for c, _ in SUBSETS:
        cur = main.curve(c)
        if cur:
            _plot_curve(b, cur, SUBSET_COLOR[c], SUBSET_LABEL[c], lw=1.6 if c == "complete" else 0.9)
    b.set_title(f"(b) {main.label}: modality subsets")
    b.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.suptitle("SeizeIT2 test patients (sub-097..125), the dataset paper's scoring; dotted: "
                 f"{FA_BUDGET:g} FA/h, the SVM's operating point", y=1.02)
    w.save(fig, "fig_sensitivity_fa",
           data={f"{r.name}_{c}": np.array([[x["th"], x["official"]["fa_ovlp_h"],
                                             x["official"]["sens_ovlp"]] for x in r.curve(c)])
                 for r in runs + eeg_runs for c, _ in SUBSETS if r.curve(c)},
           meta={"runs": [r.name for r in runs], "eeg_runs": [r.name for r in eeg_runs],
                 "published": PUBLISHED})


def fig_modality_subsets(w: FigureWriter, main: Run, others: List[Run]):
    runs = [main] + others
    subsets = [c for c, _ in SUBSETS if any(r.curve(c) for r in runs)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    width = 0.8 / len(runs)
    x = np.arange(len(subsets))
    sens, auprc = {}, {}
    for i, r in enumerate(runs):
        s = [100 * (r.sens_at(c, FA_BUDGET) or 0) for c in subsets]
        p = [r.test().get(c, {}).get("auprc", np.nan) for c in subsets]
        sens[r.name], auprc[r.name] = s, p
        a.bar(x + (i - (len(runs) - 1) / 2) * width, s, width, color=OKABE_ITO[i % 8])
        b.bar(x + (i - (len(runs) - 1) / 2) * width, p, width, color=OKABE_ITO[i % 8], label=r.label)
    a.axhline(71.1, color="black", lw=0.7, ls="--")
    a.text(len(subsets) - 0.5, 72, "SVM, bte-EEG: 71.1", fontsize=6, ha="right", va="bottom")
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels([SUBSET_LABEL[c] for c in subsets], rotation=25, ha="right")
    a.set_ylabel(f"sensitivity at ≤ {FA_BUDGET:g} FA/h (%)")
    a.set_title("(a) event-level, continuous recordings")
    a.set_ylim(0, 100)
    b.set_ylabel("window-level AUPRC")
    b.set_title("(b) window-level test (background sub-sampled)")
    top = np.nanmax([v for vv in auprc.values() for v in vv] + [0.1])
    b.set_ylim(0, top * 1.35)
    b.legend(frameon=False, fontsize=5.5, ncol=1, loc="upper right")
    fig.suptitle("Modality subsets: which signals carry the detection", y=1.02)
    w.save(fig, "fig_modality_subsets", data={"subsets": np.array(subsets),
                                              **{f"sens_{k}": v for k, v in sens.items()},
                                              **{f"auprc_{k}": v for k, v in auprc.items()}},
           meta={"fa_budget": FA_BUDGET, "runs": [r.name for r in runs]})


def fig_reliability(w: FigureWriter, main: Run, others: List[Run]):
    t = main.test()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    labels, ticks, vals, cols, edges, pos, groups, k = [], [], [], [], [], [], [], 0
    for m in MODALITIES:
        items = [("complete", "clean")] + [(cid, lab) for cid, mm, lab in CORRUPTIONS if mm == m]
        start = k
        for cid, lab in items:
            v = t.get(cid, {}).get(f"reliability_{m}")
            if v is None:
                continue
            labels.append(f"{MOD_LABEL[m]} {lab}")
            ticks.append(lab.replace("bad channel", "bad ch."))
            vals.append(v)
            cols.append(MOD_COLOR[m] if lab == "clean" else matplotlib.colors.to_rgba(MOD_COLOR[m], 0.45))
            edges.append(MOD_COLOR[m])
            pos.append(k)
            k += 1
        if k > start:
            groups.append((MOD_LABEL[m], (start + k - 1) / 2, MOD_COLOR[m]))
        k += 0.7
    a.bar(pos, vals, 0.8, color=cols, edgecolor=edges, lw=0.6)
    a.set_xticks(pos)
    a.set_xticklabels(ticks, fontsize=6, rotation=45, ha="right")
    for name, centre, col in groups:
        a.text(centre, 1.02, name, ha="center", va="bottom", fontsize=7, color=col)
    a.set_ylim(0, 1)
    a.set_ylabel("predicted reliability (test mean)")
    a.set_title(f"(a) {main.label}: the reliability head", pad=16)
    runs = [main] + others
    conds = ["complete"] + [cid for cid, _, _ in CORRUPTIONS if any(cid in r.test() for r in runs)]
    width = 0.8 / len(runs)
    x = np.arange(len(conds))
    au = {}
    for i, r in enumerate(runs):
        v = [r.test().get(c, {}).get("auprc", np.nan) for c in conds]
        au[r.name] = v
        b.bar(x + (i - (len(runs) - 1) / 2) * width, v, width, color=OKABE_ITO[i % 8], label=r.label)
    b.set_xticks(x)
    b.set_xticklabels(["clean"] + [f"{MOD_LABEL[m]} {lab}" for cid, m, lab in CORRUPTIONS
                                   if cid in conds], rotation=30, ha="right", fontsize=5.5)
    b.set_ylabel("window-level AUPRC")
    b.set_title("(b) every run under the same corruptions (severity 0.7)")
    b.legend(frameon=False, fontsize=5.5)
    fig.suptitle("Corrupted inputs: the reliability head notices, and what each fusion keeps", y=1.1)
    w.save(fig, "fig_reliability", data={"reliability_labels": np.array(labels), "reliability": vals,
                                         "conditions": np.array(conds),
                                         **{f"auprc_{k}": v for k, v in au.items()}},
           meta={"severity": (main.results or {}).get("corruption_severity")})


def fig_per_patient(w: FigureWriter, main: Run):
    per: Dict[str, Dict[str, float]] = {}
    for f in main.predictions():
        z = np.load(f)
        sub = os.path.basename(f).split("_run-")[0]
        r = score_recording(gated(z, "complete"), z["label"], 0.5)
        d = per.setdefault(sub, {"n_seizures": 0, "hit": 0, "false_events": 0, "hours": 0.0})
        for key in d:
            d[key] += r[key]
    subs = sorted(per, key=lambda s: (per[s]["n_seizures"] == 0,
                                      -(per[s]["hit"] / per[s]["n_seizures"]) if per[s]["n_seizures"] else 0))
    sens = [100 * per[s]["hit"] / per[s]["n_seizures"] if per[s]["n_seizures"] else np.nan for s in subs]
    fah = [per[s]["false_events"] / per[s]["hours"] if per[s]["hours"] else np.nan for s in subs]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 3.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.4, 1]})
    x = np.arange(len(subs))
    a.bar(x, sens, color=OKABE_ITO[0])
    for i, s in enumerate(subs):
        a.text(i, (sens[i] if sens[i] == sens[i] else 0) + 2, str(per[s]["n_seizures"]), ha="center",
               va="bottom", fontsize=5.5)
    a.set_ylabel("sensitivity at 0.5 (%)")
    a.set_ylim(0, 112)
    a.set_title(f"{main.label}: per test patient (number above each bar: annotated seizures)")
    b.bar(x, fah, color=OKABE_ITO[1])
    b.set_ylim(bottom=0)
    b.set_ylabel("FA / h")
    b.set_xticks(x)
    b.set_xticklabels([s.replace("sub-", "") for s in subs], rotation=90, fontsize=6)
    b.set_xlabel("patient")
    w.save(fig, "fig_per_patient", data={"patients": np.array(subs), "sens": sens, "fa_h": fah,
                                         "hours": [per[s]["hours"] for s in subs],
                                         "n_seizures": [per[s]["n_seizures"] for s in subs]},
           meta={"threshold": 0.5, "gate": "cross"})


def fig_learning_curves(w: FigureWriter, main: Run, others: List[Run], eeg_runs: List[Run]):
    groups = [("(a) three modalities", [main] + others), ("(b) EEG alone", eeg_runs)]
    groups = [(t, rs) for t, rs in groups if rs]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 2.6), squeeze=False)
    data = {}
    for ax, (title, rs) in zip(axes[0], groups):
        for i, r in enumerate(rs):
            v = [e.get("val", {}).get("seizeit2", {}).get("auprc", np.nan) for e in r.history]
            if not v:
                continue
            data[r.name] = v
            ax.plot(range(len(v)), v, color=OKABE_ITO[i % 8], lw=1.1, label=r.label)
            be = (r.results or {}).get("best_epoch")
            if be is not None and be < len(v):
                ax.plot(be, v[be], "o", color=OKABE_ITO[i % 8], ms=4)
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=5.5)
    axes[0][0].set_ylabel("validation AUPRC (windows)")
    fig.suptitle("Learning curves; the dot is the epoch the test figures come from", y=1.03)
    w.save(fig, "fig_learning_curves", data=data,
           meta={"runs": [r.name for r in [main] + others + eeg_runs]})


# --------------------------------------------------------------------------- #

def write_readme(out_dir: str, w: FigureWriter, main: Run, args):
    lines = [f"# SeizeIT2 multimodal figures: {main.name}", "",
             f"Drawn by `scripts/visualize_seizeit2.py` at commit `{git_commit()}` from "
             f"`{main.path}` (and the runs it is compared with). Vector figures with editable "
             "text; the arrays behind each are in `figure_data/` (not committed), what produced "
             "each in `figure_metadata/`.", ""]
    for name in w.written:
        meta = json.load(open(os.path.join(w.meta_dir, f"{name}.json")))
        lines += [f"## {name}", ""]
        for t in meta.get("figure_text", []):
            lines.append(t)
        if name == "fig_seizure_example":
            c = meta["chosen"]
            lines.append(f"\nRecording `{meta['recording']}`, chosen as rank {c['rank']} of "
                         f"{meta['n_candidates']} candidates by probability contrast "
                         f"({c['contrast']:.2f}); say so in the caption.")
        lines += ["", f"![{name}](figures/{name}.{w.formats[-1]})", ""]
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="SeizeIT2 BIDS root")
    ap.add_argument("--run-dir", required=True, help="the main run (RALF)")
    ap.add_argument("--compare", nargs="*", default=[], help="other trimodal runs")
    ap.add_argument("--eeg-runs", nargs="*", default=[], help="EEG-only runs (C1, EEGPT, LaBraM)")
    ap.add_argument("--out-dir", default=None, help="default <run-dir>/figures_seizeit2")
    ap.add_argument("--format", default="svg,pdf")
    ap.add_argument("--paper", action="store_true", help="no titles/notes/stamp on the figures")
    ap.add_argument("--example", default=None, help="recording for fig_seizure_example, "
                                                     "e.g. sub-101_run-03 (default: ranked)")
    ap.add_argument("--margin", type=float, default=60.0, help="seconds around the seizure")
    ap.add_argument("--only", nargs="*", default=None, help="figure names to draw")
    args = ap.parse_args(argv)
    apply_style()
    main_run = Run(args.run_dir)
    others = [Run(p) for p in args.compare]
    eeg_runs = [Run(p) for p in args.eeg_runs]
    for r in [main_run] + others + eeg_runs:
        print(f"  {r.name}: results {'yes' if r.results else 'NO'}, events "
              f"{'yes' if r.events else 'NO'}, predictions {len(r.predictions())}, "
              f"epochs {len(r.history)}")
    out_dir = args.out_dir or os.path.join(main_run.path, "figures_seizeit2")
    os.makedirs(out_dir, exist_ok=True)
    fmts = [f.strip() for f in args.format.split(",") if f.strip()]
    w = FigureWriter(out_dir, fmts, {"checkpoint": os.path.join(main_run.path, "best.pth"),
                                     "epoch": (main_run.results or {}).get("best_epoch"),
                                     "run": main_run.name, "git_commit": git_commit()},
                     paper=args.paper)
    want = set(args.only) if args.only else None

    def do(name):
        return want is None or name in want
    if do("fig_seizure_example") and main_run.predictions():
        fig_seizure_example(w, main_run, args.raw_dir, args.example, args.margin)
    if do("fig_sensitivity_fa") and main_run.events:
        fig_sensitivity_fa(w, main_run, others, eeg_runs)
    if do("fig_modality_subsets") and main_run.events:
        fig_modality_subsets(w, main_run, others)
    if do("fig_reliability") and main_run.results:
        fig_reliability(w, main_run, others)
    if do("fig_per_patient") and main_run.predictions():
        fig_per_patient(w, main_run)
    if do("fig_learning_curves") and main_run.history:
        fig_learning_curves(w, main_run, others, eeg_runs)
    write_readme(out_dir, w, main_run, args)
    print(f"wrote {len(w.written)} figure(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
