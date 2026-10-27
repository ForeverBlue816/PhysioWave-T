#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What the learned wavelet front ends do: one figure for EEG, ECG and sEMG.

    python scripts/visualize_wavelet_frontend.py \
        --eeg $PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth \
        --ecg $PW_CKPT_ROOT/pretrain_ecg_c1_moe/best.pth \
        --emg $PW_CKPT_ROOT/pretrain_emg_c1_moe/epoch38.pth \
        --out-dir figures --report docs/runs/fig_wavelet.md

Each route owns a bank of M=5 learnable analysis filter pairs (16 taps, one
pair per electrode, initialised from named PyWavelets filters) and a selector
that predicts, from the window, softmax weights over the five. The filter a
window is actually analysed with is the bank mixed by those weights. This
script runs --windows pre-training windows of every corpus through the
selector, averages the weights, mixes the bank with the averages and draws:

  a  per route, the magnitude response of the learned low-pass and high-pass
     (mean over electrodes, 10-90th percentile band) against the
     initialisation mixed with uniform weights, in Hz, with the modality's
     physiological bands shaded and the -3 dB cut-off of the learned low-pass
     marked;
  b  per corpus, the mean selector weights as a stacked bar, with the uniform
     initialisation as a reference bar carrying the wavelet names;
  c  the learned low-pass cut-off of every route against its sampling rate,
     with the initialisation's fs/4 as the dashed reference.

The selector is applied at every decomposition level to the current
approximation (one selector per route, three calls per window). The figure
shows LEVEL 1 -- the analysis of the raw window, where Hz on the axis means
what it says; levels 2 and 3 are measured on a subset and reported in the
markdown. Nothing is decimated between levels in this architecture, so a
level's filter always acts at the route's sampling rate.

Loading is checked tensor by tensor: for every route the count of front-end
tensors present in the checkpoint and equal to the loaded module's is
printed, and a route whose front end or selector is not fully loaded aborts.

Writes:
    <out-dir>/fig_wavelet.{pdf,png,svg}   the figure (FigureWriter layout:
                                           figures/, figure_data/, figure_metadata/)
    <report>                               docs/runs/fig_wavelet.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
import scripts.visualize_eeg_pretraining as viz                # noqa: E402
from scripts.visualize_eeg_pretraining import (               # noqa: E402
    PAPER, PAPER_FONT, PAPER_LETTER, PAPER_TITLE, PAPER_WIDTH, FigureWriter,
    _cap_threads, _paper_rc, apply_style, git_commit, sha256_file)
from physiowave.eeg_c1.data import CorpusIndex, EEGWindowDataset  # noqa: E402
from wavelet_modules import load_wavelet_kernel               # noqa: E402

MODALITIES = ("eeg", "ecg", "emg")
LABEL = {"eeg": "EEG", "ecg": "ECG", "emg": "sEMG"}
#: Modality colours from the journal palette: the learned low-pass in the
#: colour, the learned high-pass in its darker shade.
MOD_COLOR = {"eeg": PAPER["indigo"], "ecg": PAPER["mauve"], "emg": PAPER["gold"]}
MOD_DARK = {"eeg": "#23264F", "ecg": "#7A3466", "emg": "#9A6214"}
#: Five sequential steps for the five basis slots, light to dark.
BASIS_COLORS = ["#E3E8F2", "#BBD0E8", "#8FA6D0", "#6A64AC", "#3F4280"]
#: Physiological bands, (label, lo, hi, emphasis) in Hz.
BANDS = {
    "eeg": [("δ", 0.5, 4, 0), ("θ", 4, 8, 0), ("α", 8, 13, 0), ("β", 13, 30, 0),
            ("γ", 30, 80, 0)],
    "ecg": [("P/T", 0.5, 10, 0), ("QRS", 10, 40, 0)],
    "emg": [("20–450 Hz", 20, 450, 0), ("", 50, 150, 1)],
}
#: Corpus display names (the EEG ones are visualize_eeg_pretraining's).
CORPUS_LABEL = {**viz.DATASET_LABEL,
                "mimic_iv_ecg": "MIMIC-IV-ECG", "code15": "CODE-15%", "sph": "SPH",
                "georgia": "Georgia", "medalcare_xl": "MedalCare-XL",
                "icentia11k": "Icentia11k", "pulsedb": "PulseDB",
                "emg2pose": "emg2pose", "emg2qwerty": "emg2qwerty", "hyser": "Hyser",
                "cemhsey_8x8": "CEMHSEY 8×8", "cemhsey_5x13": "CEMHSEY 5×13",
                "putemg": "putEMG"}
NFFT = 2048


# --------------------------------------------------------------------------- #
# Loading, with the tensor check the task asks for
# --------------------------------------------------------------------------- #

def load_checked(path: str, modality: str):
    """``(ck, cfg, mcfg, model, routes, datasets_spec, report)``.

    ``load_trained_model`` loads strictly and switches the figure module's
    registry to the modality. On top of that, every front-end tensor of every
    route is compared with the checkpoint's value after loading, so "loaded"
    means equal, not merely present.
    """
    ck, cfg, mcfg, objective, mod, model_cls, model = viz.load_trained_model(path, modality)
    if mod != modality:
        raise SystemExit(f"{path} is a {mod} checkpoint, given as --{modality}")
    sd_ck = ck["model"]
    sd_model = model.state_dict()
    report = {"checkpoint": path, "epoch": ck.get("epoch"),
              "tensors_in_checkpoint": len(sd_ck), "tensors_in_model": len(sd_model),
              "missing_in_checkpoint": sorted(set(sd_model) - set(sd_ck)),
              "unexpected_in_checkpoint": sorted(set(sd_ck) - set(sd_model)),
              "routes": {}}
    for rid in viz.ROUTES:
        prefix = f"wavelet_frontends.{rid}."
        keys = [k for k in sd_model if k.startswith(prefix)]
        matched = sum(1 for k in keys if k in sd_ck and torch.equal(sd_model[k], sd_ck[k]))
        sel_keys = [k for k in keys if ".decomp.selector." in k]
        sel_matched = sum(1 for k in sel_keys
                          if k in sd_ck and torch.equal(sd_model[k], sd_ck[k]))
        report["routes"][rid] = {"frontend_tensors": len(keys), "frontend_matched": matched,
                                 "selector_tensors": len(sel_keys),
                                 "selector_matched": sel_matched}
        print(f"  {LABEL[modality]} {rid}: front end {matched}/{len(keys)} tensors "
              f"matched, selector {sel_matched}/{len(sel_keys)}")
        if matched != len(keys) or sel_matched != len(sel_keys) or not keys:
            raise SystemExit(f"ABORT: {rid} front end did not load from {path}")
    return ck, cfg, mcfg, model, dict(viz.ROUTES), dict(viz.PRETRAIN_DATASETS), report


def open_datasets(cfg: Dict, split: str, routes) -> Dict[str, EEGWindowDataset]:
    manifest = cfg.get("data", {}).get(f"manifest_{split}")
    if not manifest or not os.path.isfile(manifest):
        raise SystemExit(f"manifest for split {split!r} not found: {manifest!r}")
    index = CorpusIndex.from_manifest(manifest)
    out = {}
    for d in sorted(index.by_dataset()):
        try:
            out[d] = EEGWindowDataset(index, d, routes=routes)
        except Exception as exc:                              # noqa: BLE001
            print(f"  skipping {d}: {exc}")
    return out


def spread_indices(n_total: int, n: int) -> List[int]:
    if n_total <= 0:
        return []
    return sorted({int(round(v)) for v in np.linspace(0, n_total - 1, min(n, n_total))})


# --------------------------------------------------------------------------- #
# Selector weights and effective filters
# --------------------------------------------------------------------------- #

@torch.no_grad()
def selector_weights(model, route, ds, n_windows: int, n_deep: int, device,
                     chunk: int = 128):
    """Mean softmax weights per level: ``{1: [M], 2: [M], 3: [M]}`` and counts.

    Level 1 reads the selector's own MLP on the raw window (cheap, all
    ``n_windows``); levels 2 and 3 need the full decomposition and are
    measured on the first ``n_deep`` of the same windows.
    """
    front = model.wavelet_frontends[route.route_id].to(device)
    sel = front.decomp.selector
    idx = spread_indices(len(ds), n_windows)
    sums = {1: None, 2: None, 3: None}
    counts = {1: 0, 2: 0, 3: 0}
    captured: List[torch.Tensor] = []
    hook = sel.selector.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    try:
        for start in range(0, len(idx), chunk):
            x = ds.read_batch(idx[start:start + chunk]).to(device)
            w1 = sel.selector(x).float()                         # [B, M]
            sums[1] = w1.sum(0) if sums[1] is None else sums[1] + w1.sum(0)
            counts[1] += w1.shape[0]
            if counts[2] < n_deep:
                captured.clear()
                xb = x[: max(1, min(x.shape[0], n_deep - counts[2]))]
                front.decomp(xb)
                # One call per level, in order.
                for lvl, w in zip((1, 2, 3), captured[:3]):
                    if lvl == 1:
                        continue
                    w = w.float()
                    sums[lvl] = w.sum(0) if sums[lvl] is None else sums[lvl] + w.sum(0)
                    counts[lvl] += w.shape[0]
    finally:
        hook.remove()
    means = {lvl: (sums[lvl] / counts[lvl]).cpu().numpy() if counts[lvl] else None
             for lvl in sums}
    return means, counts


def bank_kernels(model, route):
    """``(low [M, C, K], high [M, C, K])`` learned taps of the route's bank."""
    sel = model.wavelet_frontends[route.route_id].decomp.selector
    lo = torch.stack([f.low_filter.weight.detach().float().cpu()[:, 0, :]
                      for f in sel.wavelet_filters])
    hi = torch.stack([f.high_filter.weight.detach().float().cpu()[:, 0, :]
                      for f in sel.wavelet_filters])
    return lo.numpy(), hi.numpy()


def init_kernels(names: List[str], kernel_size: int, mode: str):
    lo, hi = zip(*[load_wavelet_kernel(n, kernel_size, mode) for n in names])
    return torch.stack(lo).numpy(), torch.stack(hi).numpy()         # [M, K]


def magnitude(kernels: np.ndarray, fs: float):
    """``kernels [..., K] -> (f [NFFT/2+1], |H| normalised to peak 1 per kernel)``."""
    H = np.abs(np.fft.rfft(kernels, n=NFFT, axis=-1))
    H = H / np.maximum(H.max(axis=-1, keepdims=True), 1e-12)
    f = np.fft.rfftfreq(NFFT, d=1.0 / fs)
    return f, H


def cutoff_lowpass(f: np.ndarray, H: np.ndarray) -> float:
    """-3 dB point of a low-pass response: the first frequency above the peak
    where |H| crosses 1/sqrt(2), linearly interpolated."""
    k0 = int(np.argmax(H))
    thr = 1.0 / math.sqrt(2.0)
    for k in range(k0, len(H) - 1):
        if H[k] >= thr > H[k + 1]:
            t = (H[k] - thr) / max(H[k] - H[k + 1], 1e-12)
            return float(f[k] + t * (f[k + 1] - f[k]))
    return float("nan")


def cutoff_highpass(f: np.ndarray, H: np.ndarray) -> float:
    """-3 dB point of a high-pass response: the last frequency below the peak
    where |H| crosses 1/sqrt(2) rising."""
    k0 = int(np.argmax(H))
    thr = 1.0 / math.sqrt(2.0)
    for k in range(k0, 0, -1):
        if H[k] >= thr > H[k - 1]:
            t = (H[k] - thr) / max(H[k] - H[k - 1], 1e-12)
            return float(f[k] - t * (f[k] - f[k - 1]))
    return float("nan")


def analyse_route(model, route, mcfg, w_mean: Dict[int, Optional[np.ndarray]]):
    """Effective filters and responses for one route at every level measured."""
    names = list(mcfg.get("wavelet_names"))
    K = int(mcfg.get("wave_kernel_size", 16))
    mode = mcfg.get("wave_init_mode", "pad")
    lo, hi = bank_kernels(model, route)                       # [M, C, K]
    lo0, hi0 = init_kernels(names, K, mode)                   # [M, K]
    M = lo.shape[0]
    fs = float(route.sampling_rate)
    out = {"wavelet_names": names, "levels": {}}
    uni = np.full(M, 1.0 / M)
    eff0_lo = np.tensordot(uni, lo0, axes=1)                  # [K]
    eff0_hi = np.tensordot(uni, hi0, axes=1)
    f, H0_lo = magnitude(eff0_lo, fs)
    _, H0_hi = magnitude(eff0_hi, fs)
    out["f_hz"] = f
    out["init"] = {"H_lo": H0_lo, "H_hi": H0_hi,
                   "cutoff_lo_hz": cutoff_lowpass(f, H0_lo),
                   "cutoff_hi_hz": cutoff_highpass(f, H0_hi)}
    # How far the taps moved from their initialisation, per basis, relative.
    lo_dev = np.sqrt(((lo - lo0[:, None, :]) ** 2).mean(axis=(1, 2)))
    out["tap_rms_change_rel"] = (lo_dev / np.sqrt((lo0 ** 2).mean(axis=1))).tolist()
    for lvl, w in w_mean.items():
        if w is None:
            continue
        eff_lo = np.tensordot(w, lo, axes=1)                  # [C, K]
        eff_hi = np.tensordot(w, hi, axes=1)
        _, H_lo = magnitude(eff_lo, fs)                       # [C, F]
        _, H_hi = magnitude(eff_hi, fs)
        cut_lo = np.array([cutoff_lowpass(f, h) for h in H_lo])
        cut_hi = np.array([cutoff_highpass(f, h) for h in H_hi])
        out["levels"][lvl] = {
            "weights": w.tolist(),
            "H_lo_mean": H_lo.mean(0), "H_lo_p10": np.percentile(H_lo, 10, axis=0),
            "H_lo_p90": np.percentile(H_lo, 90, axis=0),
            "H_hi_mean": H_hi.mean(0), "H_hi_p10": np.percentile(H_hi, 10, axis=0),
            "H_hi_p90": np.percentile(H_hi, 90, axis=0),
            "cutoff_lo_hz_mean": float(np.nanmean(cut_lo)),
            "cutoff_lo_hz_p10": float(np.nanpercentile(cut_lo, 10)),
            "cutoff_lo_hz_p90": float(np.nanpercentile(cut_lo, 90)),
            "cutoff_lo_hz_of_mean_response": cutoff_lowpass(f, H_lo.mean(0)),
            "cutoff_hi_hz_mean": float(np.nanmean(cut_hi)),
            "cutoff_hi_hz_of_mean_response": cutoff_highpass(f, H_hi.mean(0)),
        }
    return out


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

def _route_title(route) -> str:
    return f"C={route.n_channels}, {route.sampling_rate:g} Hz"


def draw(w: FigureWriter, R: Dict, out_name: str = "fig_wavelet"):
    """``R[modality] = {"routes": {rid: analysis}, "corpora": [...], "route_order": [...]}``."""
    mods = [m for m in MODALITIES if m in R]
    n_cols = max(len(R[m]["route_order"]) for m in mods)
    with _paper_rc():
        fig = plt.figure(figsize=(PAPER_WIDTH, 5.5))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.28,
                                 left=0.06, right=0.985, top=0.9, bottom=0.085)
        ga = outer[0].subgridspec(len(mods), n_cols, wspace=0.12, hspace=0.55)
        gr = outer[1].subgridspec(2, 1, height_ratios=[1.9, 1.0], hspace=0.45)

        # -- (a) ------------------------------------------------------------- #
        first_ax = None
        for r_, m in enumerate(mods):
            routes = R[m]["routes"]
            for c_, rid in enumerate(R[m]["route_order"]):
                A = routes[rid]
                route = A["route"]
                ax = fig.add_subplot(ga[r_, c_])
                first_ax = first_ax or ax
                fs = float(route.sampling_rate)
                f = A["f_hz"]
                L = A["levels"][1]
                sel = (f >= 0.5) & (f <= fs / 2)
                for lab, lo, hi, emph in BANDS[m]:
                    ax.axvspan(lo, min(hi, fs / 2), color=PAPER["pale_blue"],
                               alpha=0.35 if emph else 0.18, lw=0, zorder=0)
                    if c_ == 0 and lab:
                        ax.text(math.sqrt(lo * min(hi, fs / 2)), 1.14, lab, ha="center",
                                va="bottom", fontsize=PAPER_FONT - 1.5, color=PAPER["rule"],
                                clip_on=False)
                if m == "ecg":
                    for mains in (50, 60):
                        ax.axvline(mains, color=PAPER["rule"], lw=0.6, ls=":", zorder=1)
                ax.fill_between(f[sel], L["H_lo_p10"][sel], L["H_lo_p90"][sel],
                                color=MOD_COLOR[m], alpha=0.2, lw=0, zorder=2)
                ax.fill_between(f[sel], L["H_hi_p10"][sel], L["H_hi_p90"][sel],
                                color=MOD_DARK[m], alpha=0.2, lw=0, zorder=2)
                ax.plot(f[sel], A["init"]["H_lo"][sel], color=PAPER["grey"], lw=0.9,
                        ls="--", zorder=3)
                ax.plot(f[sel], A["init"]["H_hi"][sel], color=PAPER["grey"], lw=0.9,
                        ls="--", zorder=3)
                ax.plot(f[sel], L["H_lo_mean"][sel], color=MOD_COLOR[m], lw=1.3, zorder=4)
                ax.plot(f[sel], L["H_hi_mean"][sel], color=MOD_DARK[m], lw=1.3, zorder=4)
                fc = L["cutoff_lo_hz_of_mean_response"]
                if np.isfinite(fc):
                    ax.plot([fc], [1 / math.sqrt(2)], marker="o", ms=4, color=MOD_COLOR[m],
                            mec="white", mew=0.6, zorder=6)
                ax.set_xscale("log")
                ax.set_xlim(0.5, fs / 2)
                ax.set_ylim(0, 1.12)
                ax.set_title(_route_title(route), pad=10, fontsize=PAPER_FONT)
                if c_ == 0:
                    ax.set_ylabel(f"{LABEL[m]}\n|H(f)|", linespacing=1.3)
                    ax.set_yticks([0, 0.5, 1])
                else:
                    ax.set_yticks([0, 0.5, 1])
                    ax.tick_params(axis="y", labelleft=False)
                ticks = [t for t in (1, 10, 100, 1000) if 0.5 < t < fs / 2]
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=PAPER["ink"], lw=1.3),
                   Line2D([], [], color="0.55", lw=1.3),
                   Line2D([], [], color=PAPER["grey"], lw=0.9, ls="--"),
                   Line2D([], [], color=PAPER["ink"], marker="o", ls="", ms=4, mec="white")]
        box_a = outer[0].get_position(fig)
        fig.legend(handles, ["Learned low-pass", "Learned high-pass (darker shade)",
                             "Initialisation (uniform mix)", "−3 dB cut-off"],
                   loc="lower left", bbox_to_anchor=(box_a.x0 + 0.03, 0.935),
                   ncol=2, frameon=False, fontsize=PAPER_FONT - 1, handlelength=1.5,
                   columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0,
                   labelspacing=0.3)
        fig.text(0.5 * (box_a.x0 + box_a.x1), 0.012, "Frequency (Hz)", ha="center",
                 va="bottom", gid="paper-keep")
        fig.text(box_a.x0 - 0.04, 0.99, "a", fontsize=PAPER_LETTER, fontweight="bold",
                 ha="left", va="top", gid="paper-keep")

        # -- (b) stacked selector weights per corpus ----------------------- #
        axb = fig.add_subplot(gr[0])
        y = 0
        yt, yl, ycol = [], [], []
        group_pos = {}
        for m in mods:
            names = R[m]["wavelet_names"]
            y0 = y
            # reference bar: uniform 1/M with the wavelet names in the segments
            M = len(names)
            left = 0.0
            # The reference bar is the bank's names on the uniform mix, drawn
            # lighter than the corpora; at this row height an outline would
            # run through the names, so none is drawn.
            for i, nm in enumerate(names):
                axb.barh(y, 1.0 / M, left=left, height=0.72, color=BASIS_COLORS[i], lw=0,
                         alpha=0.45, zorder=2)
                axb.text(left + 0.5 / M, y, nm, ha="center", va="center",
                         fontsize=PAPER_FONT - 1.5, zorder=5, color=PAPER["ink"])
                left += 1.0 / M
            yt.append(y); yl.append("uniform init"); ycol.append(PAPER["rule"])
            y += 1
            for cdict in R[m]["corpora"]:
                wv = np.asarray(cdict["weights_level1"])
                left = 0.0
                for i in range(M):
                    axb.barh(y, wv[i], left=left, height=0.72, color=BASIS_COLORS[i], lw=0,
                             zorder=2)
                    left += wv[i]
                yt.append(y); yl.append(CORPUS_LABEL.get(cdict["id"], cdict["id"]))
                ycol.append(PAPER["ink"])
                y += 1
            group_pos[m] = (y0, y - 1)
            y += 0.6
        axb.set_yticks(yt)
        axb.set_yticklabels(yl)
        for lab, col in zip(axb.get_yticklabels(), ycol):
            lab.set_color(col)
            lab.set_fontsize(PAPER_FONT - 1.5)       # 23 rows: 6.5 pt clears the pitch
        axb.invert_yaxis()
        axb.set_xlim(0, 1)
        axb.set_xticks([0, 0.5, 1])
        axb.set_xlabel("Mean selector weight")
        axb.tick_params(axis="y", length=0)
        for m, (a, b) in group_pos.items():
            axb.text(1.02, 0.5 * (a + b), LABEL[m], color=MOD_COLOR[m], fontweight="bold",
                     ha="left", va="center", fontsize=PAPER_FONT, rotation=90)
        for sp in ("top", "right"):
            axb.spines[sp].set_visible(False)
        box_b = gr[0].get_position(fig)
        fig.text(box_b.x0 - 0.085, 0.99, "b", fontsize=PAPER_LETTER, fontweight="bold",
                 ha="left", va="top", gid="paper-keep")

        # -- (c) cut-off against sampling rate ----------------------------- #
        axc = fig.add_subplot(gr[1])
        fs_all = []
        seen: Dict[float, int] = {}
        for m in mods:
            for rid in R[m]["route_order"]:
                A = R[m]["routes"][rid]
                fs = float(A["route"].sampling_rate)
                fs_all.append(fs)
                fc = A["levels"][1]["cutoff_lo_hz_of_mean_response"]
                axc.plot([fs], [fc], marker="o", ms=5, color=MOD_COLOR[m], ls="", zorder=4)
                # Routes that share a rate share an x: their labels alternate
                # right and left so they do not print over each other.
                k = seen.get(fs, 0)
                seen[fs] = k + 1
                side = 1 if k % 2 == 0 else -1
                axc.annotate(f"C={A['route'].n_channels}", (fs, fc),
                             xytext=(side * 5, 3 - 9 * (k // 2)), textcoords="offset points",
                             ha="left" if side > 0 else "right", va="bottom",
                             fontsize=PAPER_FONT - 1.5, color=MOD_COLOR[m])
        fs_line = np.array(sorted(set(fs_all)))
        axc.plot(fs_line, fs_line / 4, color=PAPER["grey"], ls="--", lw=0.9, zorder=2)
        axc.plot(fs_line, fs_line / 4, marker="o", ms=5, mfc="white", mec=PAPER["grey"],
                 ls="", zorder=3)
        # At the line's upper end, where no route sits.
        axc.annotate("init: fs / 4", (fs_line[-1], fs_line[-1] / 4), xytext=(-7, 7),
                     textcoords="offset points", ha="right", va="bottom",
                     color=PAPER["rule"], fontsize=PAPER_FONT - 1)
        axc.set_xscale("log")
        axc.set_yscale("log")
        axc.set_xlabel("Sampling rate (Hz)")
        axc.set_ylabel("Low-pass −3 dB cut-off (Hz)")
        xt = [250, 500, 1000, 2000]
        axc.set_xticks(xt)
        axc.set_xticklabels([str(t) for t in xt])
        axc.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ys = [A["levels"][1]["cutoff_lo_hz_of_mean_response"]
              for m in mods for A in R[m]["routes"].values()] + list(fs_line / 4)
        ys = [v for v in ys if np.isfinite(v) and v > 0]
        lo_y, hi_y = min(ys) / 1.6, max(ys) * 1.8
        axc.set_ylim(lo_y, hi_y)
        yt_c = [t for t in (10, 20, 50, 100, 200, 500) if lo_y <= t <= hi_y]
        axc.set_yticks(yt_c)
        axc.set_yticklabels([str(t) for t in yt_c])
        axc.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axc.set_xlim(fs_line[0] / 1.4, fs_line[-1] * 1.6)
        for m in mods:
            axc.plot([], [], marker="o", ms=5, color=MOD_COLOR[m], ls="", label=LABEL[m])
        axc.legend(frameon=False, fontsize=PAPER_FONT - 1, loc="upper left",
                   handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
        for sp in ("top", "right"):
            axc.spines[sp].set_visible(False)
        box_c = gr[1].get_position(fig)
        fig.text(box_c.x0 - 0.085, box_c.y1 + 0.05, "c", fontsize=PAPER_LETTER,
                 fontweight="bold", ha="left", va="top", gid="paper-keep")

        data, meta = {}, {"modalities": {}}
        for m in mods:
            for rid, A in R[m]["routes"].items():
                data[f"{rid}_f_hz"] = A["f_hz"]
                data[f"{rid}_init_H_lo"] = A["init"]["H_lo"]
                data[f"{rid}_init_H_hi"] = A["init"]["H_hi"]
                for lvl, L in A["levels"].items():
                    for k in ("H_lo_mean", "H_lo_p10", "H_lo_p90", "H_hi_mean", "H_hi_p10",
                              "H_hi_p90"):
                        data[f"{rid}_L{lvl}_{k}"] = L[k]
            meta["modalities"][m] = {
                "wavelet_names": R[m]["wavelet_names"], "load": R[m]["load"],
                "corpora": R[m]["corpora"],
                "routes": {rid: {"init_cutoff_lo_hz": A["init"]["cutoff_lo_hz"],
                                 "init_cutoff_hi_hz": A["init"]["cutoff_hi_hz"],
                                 "tap_rms_change_rel": A["tap_rms_change_rel"],
                                 "levels": {str(l): {k: v for k, v in L.items()
                                                     if not isinstance(v, np.ndarray)}
                                            for l, L in A["levels"].items()}}
                           for rid, A in R[m]["routes"].items()}}
        w.save(fig, out_name, data, meta)
        return meta


# --------------------------------------------------------------------------- #
# The markdown report
# --------------------------------------------------------------------------- #

def write_report(path: str, R: Dict, n_windows: int, n_deep: int, split: str):
    mods = [m for m in MODALITIES if m in R]
    L = []
    L.append("# The learned wavelet front ends\n")
    L.append(f"Selector weights and effective analysis filters of the three C1 encoders, "
             f"from up to {n_windows} {split} windows per corpus (levels 2-3 on the first "
             f"{n_deep}). Figure: `figures/fig_wavelet.pdf` "
             f"(`scripts/visualize_wavelet_frontend.py`).\n")
    L.append("## Checkpoints and tensor matching\n")
    L.append("| modality | checkpoint | epoch | route | front-end tensors matched | selector |")
    L.append("|---|---|---|---|---|---|")
    for m in mods:
        ld = R[m]["load"]
        for rid, v in ld["routes"].items():
            L.append(f"| {LABEL[m]} | `{os.path.basename(os.path.dirname(ld['checkpoint']))}/"
                     f"{os.path.basename(ld['checkpoint'])}` | {ld['epoch']} | {rid} | "
                     f"{v['frontend_matched']}/{v['frontend_tensors']} | "
                     f"{v['selector_matched']}/{v['selector_tensors']} |")
    L.append("")
    L.append("## Windows used per corpus\n")
    L.append("| modality | corpus | route | windows (level 1) | windows (levels 2-3) |")
    L.append("|---|---|---|---|---|")
    for m in mods:
        for c in R[m]["corpora"]:
            L.append(f"| {LABEL[m]} | {CORPUS_LABEL.get(c['id'], c['id'])} | {c['route']} | "
                     f"{c['n_level1']} | {c['n_deep']} |")
    L.append("")
    L.append("## Mean selector weights, level 1 (corpus × basis)\n")
    flags = []
    for m in mods:
        names = R[m]["wavelet_names"]
        L.append(f"**{LABEL[m]}** (bank: {', '.join(names)})\n")
        L.append("| corpus | " + " | ".join(names) + " | max |w − 1/M| |")
        L.append("|---|" + "---|" * len(names) + "---|")
        for c in R[m]["corpora"]:
            wv = np.asarray(c["weights_level1"])
            dev = float(np.abs(wv - 1.0 / len(names)).max())
            L.append(f"| {CORPUS_LABEL.get(c['id'], c['id'])} | "
                     + " | ".join(f"{v:.3f}" for v in wv) + f" | {dev:.3f} |")
            if dev < 0.03:
                flags.append(f"{LABEL[m]} / {CORPUS_LABEL.get(c['id'], c['id'])}: level-1 "
                             f"selector weights within 0.03 of uniform (max deviation {dev:.3f}).")
        L.append("")
        L.append("Levels 2 and 3, per route (mean over the route's corpora):\n")
        L.append("| route | level | " + " | ".join(names) + " |")
        L.append("|---|---|" + "---|" * len(names))
        for rid, A in R[m]["routes"].items():
            for lvl in (2, 3):
                if lvl in A["levels"]:
                    L.append(f"| {rid} | {lvl} | "
                             + " | ".join(f"{v:.3f}" for v in A["levels"][lvl]["weights"]) + " |")
        L.append("")
    L.append("## −3 dB cut-offs (Hz)\n")
    L.append("Low-pass: the learned effective filter (bank mixed with the level's mean "
             "weights), cut-off of the channel-mean response, with the 10-90th percentile "
             "across channels; initialisation is the bank mixed uniformly (fs/4 for an "
             "orthogonal half-band pair). High-pass: the rising −3 dB point.\n")
    L.append("| route | fs | init LP | learned LP L1 (p10–p90) | L2 | L3 | init HP | learned HP L1 | "
             "tap RMS change / init RMS (per basis) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in mods:
        for rid, A in R[m]["routes"].items():
            fs = float(A["route"].sampling_rate)
            l1 = A["levels"][1]
            l2 = A["levels"].get(2, {}).get("cutoff_lo_hz_of_mean_response", float("nan"))
            l3 = A["levels"].get(3, {}).get("cutoff_lo_hz_of_mean_response", float("nan"))
            L.append(f"| {rid} | {fs:g} | {A['init']['cutoff_lo_hz']:.1f} | "
                     f"{l1['cutoff_lo_hz_of_mean_response']:.1f} "
                     f"({l1['cutoff_lo_hz_p10']:.1f}–{l1['cutoff_lo_hz_p90']:.1f}) | "
                     f"{l2:.1f} | {l3:.1f} | {A['init']['cutoff_hi_hz']:.1f} | "
                     f"{l1['cutoff_hi_hz_of_mean_response']:.1f} | "
                     + ", ".join(f"{v:.2f}" for v in A["tap_rms_change_rel"]) + " |")
            rel = abs(l1["cutoff_lo_hz_of_mean_response"] - A["init"]["cutoff_lo_hz"]) \
                / max(A["init"]["cutoff_lo_hz"], 1e-9)
            if rel < 0.05 and max(A["tap_rms_change_rel"]) < 0.05:
                flags.append(f"{rid}: learned low-pass cut-off within 5% of initialisation "
                             f"and taps moved < 5% RMS -- the filters barely moved.")
    L.append("")
    # Do the cut-offs still scale with the sampling rate within a modality?
    for m in mods:
        pts = [(float(A["route"].sampling_rate), A["levels"][1]["cutoff_lo_hz_of_mean_response"])
               for A in R[m]["routes"].values()]
        rates = sorted({p[0] for p in pts})
        if len(rates) >= 2:
            by_rate = {r: np.mean([c for f, c in pts if f == r]) for r in rates}
            ratio_fs = rates[-1] / rates[0]
            ratio_fc = by_rate[rates[-1]] / by_rate[rates[0]]
            verdict = ("scale with the sampling rate (not a shared physical frequency)"
                       if ratio_fc > 0.5 * ratio_fs + 0.5 else
                       "are closer to a shared physical frequency than to fs/4")
            flags.append(f"{LABEL[m]}: fs ratio {ratio_fs:.2f}, learned cut-off ratio "
                         f"{ratio_fc:.2f} -- the cut-offs {verdict}.")
    L.append("## What does not show the expected pattern\n")
    L += [f"- {s}" for s in flags] or ["- (nothing flagged by the automatic checks)"]
    L.append("")
    L.append("The automatic checks above use fixed thresholds (uniform within 0.03; "
             "cut-off within 5% and taps within 5% RMS of initialisation; cut-off ratio "
             "against sampling-rate ratio). They flag, they do not interpret.\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eeg"); p.add_argument("--ecg"); p.add_argument("--emg")
    p.add_argument("--windows", type=int, default=2000)
    p.add_argument("--deep-windows", type=int, default=256,
                   help="windows per corpus run through the full decomposition for "
                        "the level-2/3 selector weights")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--report", default="docs/runs/fig_wavelet.md")
    p.add_argument("--format", default="pdf,png,svg")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-paper", action="store_true")
    args = p.parse_args(argv)
    if not (args.eeg or args.ecg or args.emg):
        p.error("give at least one of --eeg, --ecg, --emg")
    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    R: Dict[str, Dict] = {}
    base_meta = {"plotting_script_git_commit": git_commit(), "checkpoints": {}}
    # EEG first: load_trained_model switches the registry to ECG/EMG and never
    # back, so the EEG tables must be read before either of the others.
    for m in MODALITIES:
        path = getattr(args, m)
        if not path:
            continue
        print(f"== {LABEL[m]}: {path}")
        ck, cfg, mcfg, model, routes, specs, load_rep = load_checked(path, m)
        model.eval()
        base_meta["checkpoints"][m] = {"path": path, "sha256": sha256_file(path),
                                       "epoch": ck.get("epoch")}
        datasets = open_datasets(cfg, args.split, routes)
        corpora, per_route_w = [], {rid: {1: [], 2: [], 3: []} for rid in routes}
        for dsid in sorted(datasets):
            ds = datasets[dsid]
            if len(ds) == 0:
                continue
            route = routes[ds.route_id]
            means, counts = selector_weights(model, route, ds, args.windows,
                                             args.deep_windows, device)
            print(f"  {dsid:14s} {route.route_id:9s} n={counts[1]:5d} level-1 weights "
                  + " ".join(f"{v:.3f}" for v in means[1]))
            corpora.append({"id": dsid, "route": route.route_id, "n_level1": counts[1],
                            "n_deep": counts[2],
                            "weights_level1": means[1].tolist(),
                            "weights_level2": None if means[2] is None else means[2].tolist(),
                            "weights_level3": None if means[3] is None else means[3].tolist()})
            for lvl in (1, 2, 3):
                if means[lvl] is not None:
                    per_route_w[route.route_id][lvl].append((means[lvl], counts[lvl]))
            ds.close()
        analyses = {}
        for rid, route in routes.items():
            w_mean = {}
            for lvl in (1, 2, 3):
                items = per_route_w[rid][lvl]
                if items:
                    tot = sum(n for _, n in items)
                    w_mean[lvl] = sum(wv * n for wv, n in items) / tot
                else:
                    w_mean[lvl] = None
            if w_mean[1] is None:
                print(f"  {rid}: no windows in the {args.split} split; skipped")
                continue
            A = analyse_route(model, route, mcfg, w_mean)
            A["route"] = route
            analyses[rid] = A
            print(f"  {rid}: low-pass -3 dB {A['levels'][1]['cutoff_lo_hz_of_mean_response']:.1f} Hz "
                  f"(init {A['init']['cutoff_lo_hz']:.1f}), high-pass "
                  f"{A['levels'][1]['cutoff_hi_hz_of_mean_response']:.1f} Hz "
                  f"(init {A['init']['cutoff_hi_hz']:.1f})")
        R[m] = {"routes": analyses, "route_order": [r for r in routes if r in analyses],
                "corpora": corpora, "wavelet_names": list(mcfg.get("wavelet_names")),
                "load": load_rep}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    writer = FigureWriter(args.out_dir, [s.strip() for s in args.format.split(",") if s.strip()],
                          base_meta, stamp=args.no_paper, paper=not args.no_paper)
    draw(writer, R)
    write_report(args.report, R, args.windows, args.deep_windows, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
