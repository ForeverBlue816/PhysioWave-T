#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-panel SVGs of one window's path through a trained C1 model, as
building blocks for an architecture diagram.

    python scripts/draw_architecture_panels.py --run-dir $PW_CKPT_ROOT/pretrain_eeg_c1_moe
    python scripts/draw_architecture_panels.py --run-dir ... --route E19_256 --channels 0,4,8
    python scripts/draw_architecture_panels.py --run-dir ... --axes      # with axes and labels

Every panel is drawn from the checkpoint run on one real validation window
under its real mask -- the same computation as the training step -- so the
pictures in the diagram are what the model does, not illustrations. By
default the window is the best-ranked one of the run's reconstruction survey
(figure_metadata/reconstruction_survey.json, written by
visualize_eeg_pretraining.py --recon-windows N) for that route.

Written to <run-dir>/architecture_panels/<route>/ (or --out-dir), one SVG
each, transparent background, no axes or text unless --axes:

    01_raw_input              the preprocessed window, a few channels stacked
    02_mask_grid              which patches are masked (channel x patch)
    03_masked_input           the window with the masked patches zeroed -- what
                              the wavelet frontend of the online view receives
    04_wavelet_band_<b>       each scale of the learned decomposition
                              (d1..dJ detail bands, then the approximation)
    04_wavelet_bands_stacked  all scales of the first drawn channel, stacked
    05_folded_spec            the ScaleFold output: the scales folded back to
                              one row per channel (the spec target, before the
                              per-patch normalisation)
    06_spec_target_heatmap    the normalised spec target, channel x time
    07_spec_reconstruction    target on visible patches, prediction on masked
    08_raw_reconstruction     the same for the raw-waveform head, as traces
    09_raw_overlay            target vs prediction on the masked spans only,
                              masked spans shaded
    10_spec_overlay           the spec head's target vs prediction, likewise

Text, when there is any, is Times New Roman bold, as in the paper figures.
The window, mask seed and checkpoint are written to panels.json beside them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE, os.path.join(ROOT, "ECG"), os.path.join(ROOT, "EMG")):
    if p not in sys.path:
        sys.path.insert(0, p)

import visualize_eeg_pretraining as viz                      # noqa: E402
from visualize_eeg_pretraining import plt                    # noqa: E402

# Okabe-Ito, as in the paper figures.
C_SIGNAL = "#0072B2"
C_PRED = "#D55E00"
C_MASK = "#E69F00"
C_GREY = "0.55"


# --------------------------------------------------------------------------- #
# The window through the model
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_window(model, ds, route, mask_seed: int, window_index: int):
    """Everything a panel needs, on the [C, T] sample grid."""
    item = ds[window_index]
    batch = viz.collate_windows([item])
    meta = ds.montage()
    x = batch["x"]
    gen = viz._mask_generator(mask_seed, ds.dataset_id, window_index)
    out = model(x, route.route_id, channel_meta=meta,
                mask_ratio=model.mask_ratio, mask_generator=gen)
    C, P, pt = route.n_channels, route.patches_per_channel, route.patch_t
    clean_x = model._zero_padded_channels(x, meta)
    front = model.wavelet_frontends[route.route_id]
    bands = front.decomp(clean_x)[0].cpu().numpy()          # [(J+1)*C, T]
    J1 = bands.shape[0] // C
    mask_cp = out["mask"][0].cpu().numpy().reshape(C, P)

    def flat(key):
        return model.unpatchify(out[key], C, pt)[0].cpu().numpy()

    m = np.repeat(mask_cp, pt, axis=1)                       # [C, T]
    raw = clean_x[0].cpu().numpy()
    return {
        "raw": raw,
        "masked_raw": np.where(m, 0.0, raw),
        "mask_cp": mask_cp, "mask": m,
        "bands": bands.reshape(J1, C, -1),                   # [J+1, C, T]
        "folded": out["clean_spec"][0].cpu().numpy(),
        "target_spec": flat("target_spec"), "pred_spec": flat("pred_spec"),
        "target_raw": flat("target_raw"), "pred_raw": flat("pred_raw"),
        "subject_id": item["subject_id"], "recording_id": item["recording_id"],
        "fs": route.sampling_rate, "patch_t": pt,
    }


def pick_window(run_dir, route_id, datasets, dataset=None, window=None):
    """The survey's best window for the route, unless one is named."""
    if dataset and window is not None:
        return dataset, int(window)
    survey = os.path.join(run_dir, "figure_metadata", "reconstruction_survey.json")
    if os.path.isfile(survey):
        rows = [r for r in json.load(open(survey)).get("windows", [])
                if r.get("route_id") == route_id
                and (dataset is None or r.get("dataset_id") == dataset)
                and r.get("dataset_id") in datasets]
        if rows:
            best = min(rows, key=lambda r: (r.get("raw_nmse_rank", 99),
                                            r.get("spec_nmse_rank", 99)))
            return best["dataset_id"], int(best["window_index"])
    members = sorted(d for d in datasets
                     if datasets[d].route_id == route_id and len(datasets[d]))
    if dataset is None:
        if not members:
            raise SystemExit(f"no validation data for route {route_id}")
        dataset = members[0]
    return dataset, int(window or 0)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

class Panels:
    def __init__(self, out_dir, size, axes: bool, lw: float):
        self.out_dir, self.size, self.axes, self.lw = out_dir, size, axes, lw
        os.makedirs(out_dir, exist_ok=True)
        self.written = []

    def new(self, h_scale=1.0):
        fig, ax = plt.subplots(figsize=(self.size[0], self.size[1] * h_scale))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        return fig, ax

    def finish(self, fig, ax, name, xlabel=None, ylabel=None, title=None):
        if self.axes:
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if title:
                ax.set_title(title)
        else:
            ax.set_axis_off()
            ax.margins(x=0)
        path = os.path.join(self.out_dir, f"{name}.svg")
        fig.savefig(path, format="svg", transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        self.written.append(path)
        print(f"  {path}")


def _scale(r):
    r = np.asarray(r, float)
    v = np.nanpercentile(np.abs(r), 99.5) if np.isfinite(r).any() else 1.0
    return v if v > 0 else 1.0


def _stack(rows, ref=None, gap=1.2):
    """Rows scaled one by one (by ``ref``'s rows when given) and stacked.

    Per row, because the panels are diagram parts, not measurements: the
    wavelet scales differ by an order of magnitude and on one shared scale
    the high-frequency bands draw as flat lines. A pair that is compared --
    target and prediction -- shares its row's scale through ``ref``.
    """
    ref = rows if ref is None else ref
    out = [np.asarray(r, float) / _scale(q) for r, q in zip(rows, ref)]
    return out, [-i * 2 * gap for i in range(len(out))]


def traces(pn, name, rows, t, color=C_SIGNAL, shade=None, xlabel="time (s)",
           ylabel=None, title=None, ref=None):
    fig, ax = pn.new()
    rows, offs = _stack(rows, ref)
    for r, o in zip(rows, offs):
        ax.plot(t, r + o, color=color, lw=pn.lw)
    if shade is not None:
        _shade(ax, shade, t, offs)
    if pn.axes:
        ax.set_yticks([])
    pn.finish(fig, ax, name, xlabel, ylabel, title)


def _spans(mrow):
    """[start, end) sample spans where a mask row is True."""
    d = np.diff(np.concatenate([[0], mrow.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _shade(ax, mrows, t, offs, gap=1.2):
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    for mrow, o in zip(mrows, offs):
        for a, b in _spans(mrow):
            ax.fill_between([t[a], t[b - 1] + dt], o - gap, o + gap,
                            color=C_MASK, alpha=0.18, lw=0)


def overlay(pn, name, target, pred, mrows, t, title=None):
    """Target throughout, prediction only where it was masked."""
    fig, ax = pn.new()
    tg, offs = _stack(target)
    pr, _ = _stack(pred, ref=target)
    _shade(ax, mrows, t, offs)
    for r, p, o, mrow in zip(tg, pr, offs, mrows):
        ax.plot(t, r + o, color=C_SIGNAL, lw=pn.lw)
        ax.plot(t, np.where(mrow, p, np.nan) + o, color=C_PRED, lw=pn.lw)
    if pn.axes:
        ax.set_yticks([])
    pn.finish(fig, ax, name, "time (s)", None, title)


def heatmap(pn, name, arr, lim=None, cmap="RdBu_r", title=None, h_scale=1.0):
    fig, ax = pn.new(h_scale)
    lim = lim or max(float(np.nanpercentile(np.abs(arr), 99.5)), 1e-9)
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
              interpolation="nearest")
    if pn.axes:
        ax.set_yticks([])
    pn.finish(fig, ax, name, "time (samples)", "channel", title)


def mask_grid(pn, name, mask_cp, title=None):
    fig, ax = pn.new(h_scale=max(0.25, min(1.0, mask_cp.shape[0] / 12)))
    from matplotlib.colors import ListedColormap
    ax.imshow(mask_cp.astype(float), aspect="auto",
              cmap=ListedColormap(["#F2F2F2", C_MASK]), vmin=0, vmax=1,
              interpolation="nearest")
    # patch borders, so the grid reads as tokens
    C, P = mask_cp.shape
    for i in range(1, P):
        ax.axvline(i - 0.5, color="white", lw=0.6)
    for j in range(1, C):
        ax.axhline(j - 0.5, color="white", lw=0.6)
    pn.finish(fig, ax, name, "patch", "channel", title)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--modality", default="auto", choices=["auto", "eeg", "ecg", "emg"])
    p.add_argument("--route", default=None, help="default: every route")
    p.add_argument("--dataset", default=None)
    p.add_argument("--window-index", type=int, default=None)
    p.add_argument("--channels", default=None,
                   help="comma-separated channel rows to draw (default: the "
                        "three closest to half masked, or the only one)")
    p.add_argument("--mask-seed", type=int, default=None)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--size", default="3.2,1.6", help="panel size in inches, W,H")
    p.add_argument("--linewidth", type=float, default=1.0)
    p.add_argument("--axes", action="store_true",
                   help="draw axes, labels and titles (default: bare panels)")
    args = p.parse_args(argv)

    viz.apply_style()
    run_dir = args.run_dir
    ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
            else os.path.join(run_dir, args.checkpoint))
    ck, cfg, mcfg, objective, modality, _cls, model = viz.load_trained_model(
        ckpt, args.modality)
    mask_seed = (args.mask_seed if args.mask_seed is not None
                 else int(cfg.get("train", {}).get("val_mask_seed", 1234)))
    manifest = cfg.get("data", {}).get(f"manifest_{args.split}")
    if not manifest or not os.path.isfile(manifest):
        raise SystemExit(f"the run's {args.split} manifest is not here: {manifest!r}")
    index = viz.CorpusIndex.from_manifest(manifest)
    datasets = {}
    for d in sorted(index.by_dataset()):
        try:
            datasets[d] = viz.EEGWindowDataset(index, d, routes=viz.ROUTES)
        except Exception as exc:                              # noqa: BLE001
            print(f"  skipping {d}: {exc}")

    size = tuple(float(v) for v in args.size.split(","))
    routes = [args.route] if args.route else list(viz.ROUTES)
    record = {"checkpoint": ckpt, "epoch": ck.get("epoch"),
              "mask_seed": mask_seed, "modality": modality, "routes": {}}
    for rid in routes:
        route = viz.ROUTES[rid]
        if not any(ds.route_id == rid for ds in datasets.values()):
            print(f"  {rid}: no validation data, skipped")
            continue
        did, widx = pick_window(run_dir, rid, datasets, args.dataset,
                                args.window_index)
        ex = run_window(model, datasets[did], route, mask_seed, widx)
        C = route.n_channels
        if args.channels:
            chans = [int(c) for c in args.channels.split(",")]
        else:
            # The three rows closest to half masked: each then shows visible
            # and masked patches side by side, which is the point of the
            # masked-input and reconstruction panels. Ties go to lower rows.
            frac = ex["mask_cp"].mean(1)
            chans = sorted(sorted(range(C), key=lambda c: (abs(frac[c] - 0.5), c))
                           [:min(3, C)])
        out_dir = os.path.join(args.out_dir or os.path.join(run_dir,
                                                            "architecture_panels"), rid)
        pn = Panels(out_dir, size, args.axes, args.linewidth)
        T = ex["raw"].shape[-1]
        t = np.arange(T) / float(ex["fs"])
        m = ex["mask"][chans]
        print(f"{rid}: {did} window {widx} ({ex['recording_id']}), channels {chans}")

        traces(pn, "01_raw_input", ex["raw"][chans], t, title="input")
        mask_grid(pn, "02_mask_grid", ex["mask_cp"][chans], title="patch mask")
        traces(pn, "03_masked_input", ex["masked_raw"][chans], t, shade=m,
               title="masked input", ref=ex["raw"][chans])
        J1 = ex["bands"].shape[0]
        names = [f"d{j + 1}" for j in range(J1 - 1)] + ["approx"]
        for j, bname in enumerate(names):
            traces(pn, f"04_wavelet_band_{j + 1}_{bname}",
                   ex["bands"][j][chans], t, title=f"wavelet {bname}")
        traces(pn, "04_wavelet_bands_stacked", ex["bands"][:, chans[0]], t,
               title=f"wavelet scales, channel {chans[0]}")
        traces(pn, "05_folded_spec", ex["folded"][chans], t,
               title="ScaleFold output")
        lim = max(float(np.nanpercentile(np.abs(ex["target_spec"][chans]), 99.5)), 1e-9)
        heatmap(pn, "06_spec_target_heatmap", ex["target_spec"][chans], lim,
                title="spec target", h_scale=max(0.35, min(1, len(chans) / 6)))
        comp = np.where(m, ex["pred_spec"][chans], ex["target_spec"][chans])
        heatmap(pn, "07_spec_reconstruction", comp, lim,
                title="spec reconstruction", h_scale=max(0.35, min(1, len(chans) / 6)))
        traces(pn, "08_raw_reconstruction",
               np.where(m, ex["pred_raw"][chans], ex["target_raw"][chans]), t,
               shade=m, title="raw reconstruction", ref=ex["target_raw"][chans])
        overlay(pn, "09_raw_overlay", ex["target_raw"][chans],
                ex["pred_raw"][chans], m, t, title="raw: target vs prediction")
        overlay(pn, "10_spec_overlay", ex["target_spec"][chans],
                ex["pred_spec"][chans], m, t, title="spec: target vs prediction")
        record["routes"][rid] = {"dataset": did, "window_index": widx,
                                 "recording_id": ex["recording_id"],
                                 "subject_id": ex["subject_id"],
                                 "channels": chans, "out_dir": out_dir,
                                 "panels": [os.path.basename(x) for x in pn.written]}
        with open(os.path.join(out_dir, "panels.json"), "w") as f:
            json.dump(record["routes"][rid] | {k: v for k, v in record.items()
                                                if k != "routes"}, f, indent=2,
                      default=str)
    for ds in datasets.values():
        ds.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
