#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scale-fold weights alpha by sleep stage, from a fine-tuned Sleep-EDF model.

    python scripts/visualize_sleep_fold.py \
        --run-dir $PW_CKPT_ROOT/eeg_downstream/sleep_edf_ft_lr1e-4 \
        --data-dir $PW_DATA_EEG/sleep_edf --format svg,pdf

The dynamic ScaleFold decides, per channel and per 0.5 s block, how much of
each wavelet scale goes into the one row per electrode the Transformer sees
(wavelet_modules.ScaleFold). The methods section calls this a differentiable
best-basis selection. This figure is the check that the selection is
physiological rather than arbitrary: if the fold leans on the approximation
band (0-6.25 Hz at 100 Hz) in N3, where the EEG is slow-wave, and on the
detail bands in wake and REM, the weights follow the signal's spectral
content, which is what a best basis is supposed to do.

Only the frontend runs: alpha is decided before the Transformer, so the whole
test split goes through in minutes on a CPU. The stage labels are never shown
to the fold -- it sees 30 s of two-channel EEG and nothing else.

The alpha drawn is the fold's UNDROPPED softmax over scales (``alpha_blocks``,
the field scripts/alpha_probe.py reads), in eval mode. Nominal band edges are
the dyadic ones for the sampling rate; the learned filters are not brick-wall,
so the labels say what the band is centred on, not where it ends.

Writes, under --out-dir (default: the run directory):
    figures/fig_paper_fold_by_stage.{pdf,svg}
    figure_data/fig_paper_fold_by_stage.npz      per-window alpha, labels, subjects,
                                                 band-power fractions, the night's blocks
    figure_metadata/fig_paper_fold_by_stage.json the numbers in the figure
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
from physiowave.config import load_config                     # noqa: E402
from physiowave.models.build import build_model               # noqa: E402
from scripts.visualize_eeg_pretraining import (               # noqa: E402
    OKABE_ITO, PAPER_FONT, PAPER_WIDTH, FigureWriter, _cap_threads, _letter,
    _paper_rc, apply_style, git_commit, sha256_file)

STAGES = ["W", "N1", "N2", "N3", "REM"]
#: Okabe-Ito, fixed order, never cycled: one hue per stage.
STAGE_COLOR = {"W": OKABE_ITO[4], "N1": OKABE_ITO[5], "N2": OKABE_ITO[0],
               "N3": OKABE_ITO[7], "REM": OKABE_ITO[3]}
#: The scales are ordinal (high to low frequency), so they take a sequential
#: ramp rather than a second categorical set beside the stages.
SCALE_CMAP = "viridis"


# --------------------------------------------------------------------------- #
# The model, rebuilt the way finetune_main built it
# --------------------------------------------------------------------------- #

def read_split(path: str):
    """``(data, label, subject, channel_names, sampling_rate)`` of one HDF5."""
    import h5py
    with h5py.File(path, "r") as f:
        X = np.asarray(f["data"][:], dtype=np.float32)
        y = np.asarray(f["label"][:]).astype(np.int64)
        subj = (np.asarray(f["subject"][:]).astype(np.int64) if "subject" in f
                else np.zeros(len(y), np.int64))
        names = ([c.decode() if isinstance(c, bytes) else str(c)
                  for c in f["channel_names"][:]] if "channel_names" in f else None)
        fs = float(f.attrs.get("sampling_rate", 0.0)) or None
    return X, y, subj, names, fs


def load_downstream(run_dir: str, checkpoint: str, X, names, fs):
    """The fine-tuned model from its checkpoint and the config it recorded.

    finetune_main stores ``{"args": vars(args), "config": <config name>}`` and
    fills the montage facts (channels, window, names, rate) from the data file
    at build time; the same is done here, so the model has the shapes the
    weights were saved with.
    """
    ckpt = checkpoint if os.path.isabs(checkpoint) else os.path.join(run_dir, checkpoint)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("config", {})
    a = saved.get("args", {})
    cfg_name = a.get("config") or saved.get("config")
    if not cfg_name:
        raise SystemExit(f"{ckpt} does not record the config it was trained with")
    cfg = load_config(cfg_name, a.get("set") or [])
    mcfg = dict(cfg.get("model", {}) or {})
    name = mcfg.get("name")
    if name != "eeg_c1":
        raise SystemExit(f"{ckpt} is a {name!r} model; this reads the EEG C1 "
                         f"downstream's dynamic fold")
    mcfg["num_classes"] = int(a.get("num_classes", 5))
    c1 = dict(mcfg.get("eeg_c1", {}) or {})
    c1.pop("pretrained", None)
    if a.get("freeze_encoder"):
        c1["freeze_encoder"] = True
    c1.setdefault("in_channels", int(X.shape[1]))
    c1.setdefault("window_samples", int(X.shape[2]))
    if names and "channel_names" not in c1:
        c1["channel_names"] = list(names)
    if fs and "sampling_rate" not in c1:
        c1["sampling_rate"] = float(fs)
    mcfg["eeg_c1"] = c1
    cfg["model"] = mcfg
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if not k.startswith("spatial_filter")]
    if missing or unexpected:
        raise SystemExit(f"{ckpt} does not fit the rebuilt model: missing "
                         f"{missing[:5]}, unexpected {unexpected[:5]}")
    model.eval()
    fold = model.wavelet_frontend.fold
    if fold.mode != "dynamic":
        raise SystemExit(f"scale_fold is {fold.mode!r}; only 'dynamic' decides "
                         f"alpha per block")
    return ck, ckpt, cfg, model


def band_labels(fs: float, n_scales: int):
    """Nominal dyadic band of each scale, scale-major as the frontend emits
    them: detail 1 (highest) ... detail L, then the approximation."""
    L = n_scales - 1
    out = []
    for s in range(L):
        out.append((f"d{s + 1}", fs / 2 ** (s + 2), fs / 2 ** (s + 1)))
    out.append((f"a{L}", 0.0, fs / 2 ** (L + 1)))
    return out


def band_power_fractions(X: np.ndarray, fs: float, edges=(0.5, 4.0, 8.0, 13.0, 30.0)):
    """Fraction of 0.5-30 Hz power in delta / theta / alpha / beta: ``[N, C, 4]``."""
    F = np.fft.rfft(X - X.mean(-1, keepdims=True), axis=-1)
    P = (F.real ** 2 + F.imag ** 2)
    f = np.fft.rfftfreq(X.shape[-1], 1.0 / fs)
    tot = P[..., (f >= edges[0]) & (f < edges[-1])].sum(-1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(P[..., (f >= lo) & (f < hi)].sum(-1) / np.maximum(tot, 1e-12))
    return np.stack(out, -1)


@torch.no_grad()
def fold_alpha(model, X: np.ndarray, batch_size: int, keep_blocks_for=None):
    """Per-window block-mean alpha ``[N, C, S]`` and, for the windows in
    ``keep_blocks_for`` (a boolean mask), the full ``[n, C, N_blocks, S]``."""
    front = model.wavelet_frontend
    fold = front.fold
    fold.keep_alpha = True
    means, blocks = [], []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size])
        spec = front.decomp(x)
        fold(spec)
        a = fold.alpha_blocks                                   # [B, C, N, S]
        means.append(a.mean(dim=2).cpu().numpy())
        if keep_blocks_for is not None:
            sel = keep_blocks_for[i:i + batch_size]
            if sel.any():
                blocks.append(a[torch.from_numpy(sel)].cpu().numpy())
    fold.keep_alpha = False
    A = np.concatenate(means)
    B = np.concatenate(blocks) if blocks else None
    return A, B


def subject_mean_ci(values: np.ndarray, subj: np.ndarray):
    """Mean over subjects of each subject's mean, and a 95% t-interval."""
    from scipy.stats import t as student_t
    per = np.stack([values[subj == s].mean(0) for s in np.unique(subj)])
    n = per.shape[0]
    m = per.mean(0)
    if n < 2:
        return m, np.zeros_like(m), n
    h = student_t.ppf(0.975, n - 1) * per.std(0, ddof=1) / math.sqrt(n)
    return m, h, n


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

def fig_fold_by_stage(w: FigureWriter, A, y, subj, names, fs, bands,
                      night, delta_frac, model_note: str):
    C, S = A.shape[1], A.shape[2]
    scale_colors = plt.get_cmap(SCALE_CMAP)(np.linspace(0.9, 0.1, S))
    approx = S - 1
    stats = {"n_windows": int(len(y)), "n_subjects": int(len(np.unique(subj))),
             "per_stage": {}, "bands": [{"scale": b[0], "lo_hz": b[1], "hi_hz": b[2]}
                                        for b in bands]}
    with _paper_rc():
        fig = plt.figure(figsize=(PAPER_WIDTH, 2.75))
        outer = fig.add_gridspec(1, 3, width_ratios=[1.9, 1.55, 1.0], wspace=0.36,
                                 left=0.06, right=0.985, top=0.85, bottom=0.3)
        # (a) alpha per scale and stage, one small panel per channel
        ga = outer[0].subgridspec(1, C, wspace=0.12)
        present = [k for k in range(len(STAGES)) if (y == k).any()]
        xs = np.arange(len(present))
        for c in range(C):
            ax = fig.add_subplot(ga[c])
            for s in range(S):
                m, h = [], []
                for k in present:
                    sel = y == k
                    mm, hh, n = subject_mean_ci(A[sel, c, s], subj[sel])
                    m.append(mm)
                    h.append(hh)
                    stats["per_stage"].setdefault(STAGES[k], {}).setdefault(
                        names[c] if names else f"ch{c}", {})[bands[s][0]] = {
                        "mean_over_subjects": float(mm), "ci95_half": float(hh),
                        "n_subjects": int(n)}
                lab = (f"{bands[s][0]}  {bands[s][1]:g}–{bands[s][2]:g} Hz" if s < approx
                       else f"{bands[s][0]}  <{bands[s][2]:g} Hz")
                ax.errorbar(xs, m, yerr=h, color=scale_colors[s], marker="o", ms=3.2,
                            lw=1.3, capsize=1.5, elinewidth=0.7, label=lab)
            ax.axhline(1.0 / S, color="0.6", lw=0.5, ls=":")
            ax.set_xticks(xs)
            ax.set_xticklabels([STAGES[k] for k in present])
            ax.set_title(names[c] if names else f"channel {c}", pad=2)
            if c == 0:
                ax.set_ylabel("Fold weight α")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlim(-0.4, len(present) - 0.6)
            if c == C - 1:
                handles, labels = ax.get_legend_handles_labels()
        # Share y across the channel panels
        axs_a = [a_ for a_ in fig.axes if a_.get_subplotspec() is not None
                 and a_.get_subplotspec().get_gridspec() is ga]
        lo = min(a_.get_ylim()[0] for a_ in axs_a)
        hi = max(a_.get_ylim()[1] for a_ in axs_a)
        for a_ in axs_a:
            a_.set_ylim(lo, hi + 0.05 * (hi - lo))
        box0 = outer[0].get_position(fig)
        xc0 = 0.5 * (box0.x0 + box0.x1)
        fig.text(xc0, 0.175, "Sleep stage", ha="center", va="bottom", gid="paper-keep")
        # The scale legend under the panels, off the data, once for both channels.
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(xc0, 0.0),
                   ncol=2, frameon=False, fontsize=PAPER_FONT - 1, handlelength=1.4,
                   columnspacing=1.2, labelspacing=0.25, borderaxespad=0.0)
        _letter(fig, 0.004, 0.99, "a")

        # (b) one night: hypnogram over alpha of the approximation band
        gb = outer[1].subgridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.38)
        ax_h = fig.add_subplot(gb[0])
        ax_b = fig.add_subplot(gb[1], sharex=ax_h)
        ny, nA = night["y"], night["alpha"]                       # [n], [n, C, N, S]
        ep = np.arange(len(ny)) * 0.5 / 60.0                       # hours (30 s epochs)
        depth = np.array([{0: 4, 1: 2, 2: 1, 3: 0, 4: 3}[k] for k in ny])   # W top, N3 bottom
        ax_h.step(ep, depth, where="post", color="0.25", lw=0.7)
        for k in present:
            sel = ny == k
            ax_h.scatter(ep[sel], depth[sel], s=2, color=STAGE_COLOR[STAGES[k]], lw=0,
                         zorder=3)
        ax_h.set_yticks([4, 3, 2, 1, 0])
        ax_h.set_yticklabels(["W", "REM", "N1", "N2", "N3"], fontsize=PAPER_FONT - 1)
        ax_h.tick_params(axis="x", labelbottom=False)
        ax_h.spines["bottom"].set_visible(False)
        ax_h.set_title(f"Subject {night['subject']}, {names[0] if names else 'channel 0'}",
                       pad=2)
        a_night = nA[:, 0].mean(1)                                 # [n, S], channel 0
        for s in (approx, 0):
            ax_b.plot(ep, a_night[:, s], color=scale_colors[s], lw=0.9,
                      label=bands[s][0])
        # The legend in the gap between the hypnogram and the traces, where
        # there is no data for it to sit on.
        ax_b.legend(frameon=False, fontsize=PAPER_FONT - 1, ncol=2, loc="lower center",
                    bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.2,
                    borderaxespad=0.05)
        ax_b.axhline(1.0 / S, color="0.6", lw=0.5, ls=":")
        ax_b.set_xlim(float(ep[0]), float(ep[-1]))
        ax_b.set_xlabel("Time in night (h)")
        ax_b.set_ylabel("α")
        _letter(fig, outer[1].get_position(fig).x0 - 0.045, 0.99, "b")

        # (c) alpha of the approximation band against the window's delta share
        ax_c = fig.add_subplot(outer[2])
        from scipy.stats import spearmanr
        xd, ya = delta_frac[:, 0, 0], A[:, 0, approx]        # channel 0, delta band
        order = np.random.default_rng(0).permutation(len(xd))  # draw order, not a subset
        for k in present:
            sel = y[order] == k
            ax_c.scatter(xd[order][sel], ya[order][sel], s=1.6, color=STAGE_COLOR[STAGES[k]],
                         alpha=0.35, lw=0, rasterized=True, label=STAGES[k])
        rho = float(spearmanr(xd, ya).correlation)
        stats["delta_vs_alpha_approx"] = {"channel": names[0] if names else "ch0",
                                          "spearman_rho": rho, "n": int(len(xd))}
        ax_c.set_xlabel(f"Delta share of 0.5–30 Hz power\nSpearman ρ = {rho:.2f}",
                        linespacing=1.3)
        ax_c.set_ylabel(f"α of {bands[approx][0]}")
        ax_c.set_xlim(0, 1)
        # Stage legend above the panel, off the point cloud.
        leg = ax_c.legend(frameon=False, fontsize=PAPER_FONT - 1, loc="lower center",
                          bbox_to_anchor=(0.5, 1.0), ncol=len(present), markerscale=4,
                          handlelength=0.8, handletextpad=0.3, columnspacing=0.8,
                          borderaxespad=0.1)
        for h_ in leg.legend_handles:
            h_.set_alpha(1.0)
        _letter(fig, outer[2].get_position(fig).x0 - 0.045, 0.99, "c")

        w.save(fig, "fig_paper_fold_by_stage",
               {"alpha_window_mean": A, "label": y, "subject": subj,
                "band_fractions": delta_frac, "night_alpha_blocks": nA,
                "night_label": ny, "night_subject": np.asarray(night["subject"]),
                "channel_names": np.asarray(names or []),
                "band_edges_hz": np.asarray([[b[1], b[2]] for b in bands])},
               {"stats": stats, "model": model_note, "stages": STAGES,
                "uncertainty": "mean over test subjects of each subject's mean "
                               "window alpha; 95% t-interval across subjects",
                "alpha": "fold.alpha_blocks: the undropped softmax over scales, "
                         "per channel and 0.5 s block, averaged over the 60 blocks "
                         "of a 30 s window; eval mode",
                "bands": "nominal dyadic bands at the data's sampling rate; the "
                         "learned filters are not brick-wall",
                "night": "the test subject with the most windows; panel (b) is "
                         "channel 0 and the approximation / finest detail scales",
                "delta_share": "power in 0.5-4 Hz over power in 0.5-30 Hz, per "
                               "window, from a plain periodogram of the z-scored EEG"})
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="a Sleep-EDF downstream run")
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--data-dir", required=True, help="holds <split>.h5")
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", default=None, help="default: the run directory")
    p.add_argument("--format", default="svg,pdf")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--night-subject", type=int, default=None,
                   help="subject id for panel (b); default: the one with most windows")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-paper", action="store_true", help="keep the corner stamp")
    args = p.parse_args(argv)

    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()
    apply_style()

    path = os.path.join(args.data_dir, f"{args.split}.h5")
    X, y, subj, names, fs = read_split(path)
    if fs is None:
        raise SystemExit(f"{path} records no sampling_rate; rebuild it with "
                         f"EEG/sleep_edf_finetune.py --stage split")
    print(f"  {path}: {len(y)} windows, {X.shape[1]} ch x {X.shape[2]} samples at "
          f"{fs:g} Hz, {len(np.unique(subj))} subjects, channels {names}")
    ck, ckpt, cfg, model = load_downstream(args.run_dir, args.checkpoint, X, names, fs)
    S = model.wavelet_frontend.fold.num_scales
    bands = band_labels(fs, S)
    print(f"  fold: dynamic, {S} scales, blocks of "
          f"{model.wavelet_frontend.fold.patch_len} samples; bands "
          + ", ".join(f"{b[0]} {b[1]:g}-{b[2]:g} Hz" for b in bands))

    night_subject = (args.night_subject if args.night_subject is not None
                     else int(np.bincount(subj).argmax()))
    keep = subj == night_subject
    A, blocks = fold_alpha(model, X, args.batch_size, keep_blocks_for=keep)
    frac = band_power_fractions(X, fs)
    night = {"subject": night_subject, "y": y[keep], "alpha": blocks}
    for k in range(len(STAGES)):
        if (y == k).any():
            print(f"  {STAGES[k]:>3}: alpha mean over windows "
                  + " ".join(f"{bands[s][0]}={A[y == k, :, s].mean():.3f}" for s in range(S)))

    out_dir = args.out_dir or args.run_dir
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    writer = FigureWriter(out_dir, formats, {
        "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
        "epoch": ck.get("epoch"), "data": path, "split": args.split,
        "plotting_script_git_commit": git_commit()},
        stamp=args.no_paper, paper=not args.no_paper)
    note = (f"{cfg.get('model', {}).get('name')} downstream, "
            f"{os.path.basename(args.run_dir)}, epoch {ck.get('epoch')}")
    stats = fig_fold_by_stage(writer, A, y, subj, names, fs, bands, night, frac, note)
    print(f"  delta share vs alpha_{bands[-1][0]}: Spearman rho "
          f"{stats['delta_vs_alpha_approx']['spearman_rho']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
