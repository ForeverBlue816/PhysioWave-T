#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper figures for an EEG C1 pretraining run.

    python scripts/visualize_eeg_pretraining.py \
        --run-dir /path/to/run --checkpoint best.pth --split val --format svg

Every figure is drawn from the run's own artefacts -- the checkpoint, the
metrics files and the corpus the run trained on. Nothing is illustrative: the
masks drawn are the masks the model was given, the reconstructions are that
checkpoint's outputs, and the wavelet responses are its learned filters.

WHAT THE MODEL RECONSTRUCTS. The head predicts FOLDED WAVELET PATCHES, not raw
EEG. Spec(x) is decomposed into J+1 scales and the dynamic fold reduces the
scale axis back to one row per electrode; that folded representation is what is
patched, masked and predicted. Raw EEG appears in the figures only as the
context strip above, and is labelled as such. Calling the bottom rows "raw EEG
reconstruction" would claim an inverse transform this model does not compute.

Alongside every figure:
    figures/<name>.svg          the figure
    figure_data/<name>.npz      the arrays it was drawn from
    figure_metadata/<name>.json checkpoint hash, step, sample identity, seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
from channel_embedding import CHANNEL_VOCAB                    # noqa: E402
from physiowave.eeg_c1.data import (CorpusIndex, EEGWindowDataset,  # noqa: E402
                                    collate_windows)
from physiowave.eeg_c1.model import (MultiRouteEEGPretrainer,  # noqa: E402
                                     masked_reconstruction_loss)
from physiowave.eeg_c1.routes import (PRETRAIN_DATASETS, ROUTES,  # noqa: E402
                                      Route)
from physiowave.eeg_c1.train import _mask_generator            # noqa: E402


# --------------------------------------------------------------------------- #
# House style
#
# Okabe-Ito: eight hues distinguishable under the common forms of colour vision
# deficiency, and still distinguishable printed in greyscale by lightness.
# svg.fonttype="none" leaves text as text in the SVG, so the figure can be
# restyled by the journal's template instead of arriving as outlines.
# --------------------------------------------------------------------------- #
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
ROUTE_COLOR = {"E19_256": OKABE_ITO[0], "E32_512": OKABE_ITO[1],
               "E64_256": OKABE_ITO[2], "E128_512": OKABE_ITO[3]}
DATASET_COLOR = {d: OKABE_ITO[i % len(OKABE_ITO)]
                 for i, d in enumerate(PRETRAIN_DATASETS)}


def apply_style():
    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


# --------------------------------------------------------------------------- #
# Output plumbing
# --------------------------------------------------------------------------- #

class FigureWriter:
    """Writes the figure, the arrays behind it, and what produced both."""

    def __init__(self, out_dir: str, fmt: str, base_meta: Dict):
        self.fig_dir = os.path.join(out_dir, "figures")
        self.data_dir = os.path.join(out_dir, "figure_data")
        self.meta_dir = os.path.join(out_dir, "figure_metadata")
        for d in (self.fig_dir, self.data_dir, self.meta_dir):
            os.makedirs(d, exist_ok=True)
        self.fmt = fmt
        self.base_meta = base_meta
        self.written: List[str] = []

    def save(self, fig, name: str, data: Optional[Dict] = None,
             meta: Optional[Dict] = None):
        path = os.path.join(self.fig_dir, f"{name}.{self.fmt}")
        fig.savefig(path, format=self.fmt)
        plt.close(fig)
        if data:
            np.savez_compressed(os.path.join(self.data_dir, f"{name}.npz"),
                                **{k: np.asarray(v) for k, v in data.items()})
        payload = dict(self.base_meta)
        payload.update({"figure": name,
                        "generated_utc": datetime.now(timezone.utc).isoformat()})
        if meta:
            payload.update(meta)
        with open(os.path.join(self.meta_dir, f"{name}.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
        self.written.append(path)
        print(f"  wrote {path}")


def sha256_file(path: str, limit: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while read < limit:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def read_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# 1. datasets and routes
# --------------------------------------------------------------------------- #

def fig_dataset_routes(w: FigureWriter, manifest: Dict):
    datasets = list(PRETRAIN_DATASETS)
    counts = [manifest.get("datasets", {}).get(d, {}).get("n_windows", 0)
              for d in datasets]
    mix = manifest.get("realised_mixture", {})
    by_step = [mix.get("by_step", {}).get(d, 0.0) * 100 for d in datasets]
    by_window = [mix.get("by_window", {}).get(d, 0.0) * 100 for d in datasets]
    target = [manifest.get("target_weights", {}).get(d, 0.0) * 100 for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4),
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})

    ax = axes[0]
    y = np.arange(len(datasets))
    ax.barh(y, counts, color=[ROUTE_COLOR[PRETRAIN_DATASETS[d].route_id]
                              for d in datasets])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}\n{PRETRAIN_DATASETS[d].route_id}" for d in datasets])
    ax.invert_yaxis()
    ax.set_xlabel("windows in the training manifest")
    ax.set_title("corpus size by dataset")
    if max(counts or [0]) > 0:
        ax.set_xscale("log")

    ax = axes[1]
    width = 0.38
    ax.barh(y - width / 2, target, height=width, color=OKABE_ITO[7],
            label="target")
    ax.barh(y + width / 2, by_step, height=width, color=OKABE_ITO[0],
            label="realised")
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.invert_yaxis()
    ax.set_xlabel("share of optimizer steps (%)")
    ax.set_title("mixture, by step")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.barh(y, by_window, color=[ROUTE_COLOR[PRETRAIN_DATASETS[d].route_id]
                                 for d in datasets])
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.invert_yaxis()
    ax.set_xlabel("share of windows (%)")
    ax.set_title("mixture, by window\n(differs: micro-batch is route-specific)")

    handles = [plt.Line2D([0], [0], color=c, lw=4,
                          label=f"{r} · {ROUTES[r].n_channels}ch @ "
                                f"{ROUTES[r].sampling_rate}Hz · "
                                f"{ROUTES[r].n_tokens} tokens")
               for r, c in ROUTE_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Seven corpora, four hard routes, four wavelet frontends", y=1.02)
    w.save(fig, "fig_dataset_routes",
           {"datasets": datasets, "n_windows": counts, "target_pct": target,
            "by_step_pct": by_step, "by_window_pct": by_window})


# --------------------------------------------------------------------------- #
# 2 & 3. convergence
# --------------------------------------------------------------------------- #

def _series(rows: List[Dict], key: str):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            xs.append(r.get("global_step", r.get("step", len(xs))))
            ys.append(v)
    return np.asarray(xs, float), np.asarray(ys, float)


def fig_pretraining_convergence(w: FigureWriter, epoch_rows, step_rows):
    metrics = [("loss_total", "total loss"), ("loss_masked_mse", "masked MSE"),
               ("masked_rmse", "masked RMSE"), ("masked_mae", "masked MAE")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    data = {}
    for ax, (key, label) in zip(axes, metrics):
        sx, sy = _series(step_rows, key)
        if sx.size:
            # Raw step values, not a smoothed curve: a window long enough to
            # hide the step-to-step spread would also hide whether the run is
            # actually converging or bouncing.
            ax.plot(sx, sy, color=OKABE_ITO[5], lw=0.5, alpha=0.35,
                    label="step")
            data[f"step_x_{key}"], data[f"step_y_{key}"] = sx, sy
        ex, ey = _series(epoch_rows, f"train/{key}")
        if ex.size:
            ax.plot(ex, ey, color=OKABE_ITO[0], marker="o", ms=2.5,
                    label="train (epoch)")
            data[f"train_x_{key}"], data[f"train_y_{key}"] = ex, ey
        vx, vy = _series(epoch_rows, f"val/{key}")
        if vx.size:
            ax.plot(vx, vy, color=OKABE_ITO[1], marker="s", ms=2.5,
                    label="val (epoch)")
            data[f"val_x_{key}"], data[f"val_y_{key}"] = vx, vy
        ax.set_title(label)
        ax.set_xlabel("global step")
        if key != "loss_total":
            ax.set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)
    fig.suptitle("Masked wavelet-patch reconstruction: convergence", y=1.03)
    w.save(fig, "fig_pretraining_convergence", data)


def fig_route_convergence(w: FigureWriter, epoch_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    data = {}
    for rid, color in ROUTE_COLOR.items():
        x, y = _series(epoch_rows, f"val/route/{rid}/loss_masked_mse")
        if x.size:
            axes[0].plot(x, y, color=color, marker="o", ms=2.5, label=rid)
            data[f"route_{rid}_x"], data[f"route_{rid}_y"] = x, y
    axes[0].set_title("validation masked MSE, by route")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel("masked MSE")
    axes[0].legend(frameon=False)

    for d in PRETRAIN_DATASETS:
        x, y = _series(epoch_rows, f"val/dataset/{d}/loss_masked_mse")
        if x.size:
            axes[1].plot(x, y, color=DATASET_COLOR[d], marker=".", ms=3,
                         label=d)
            data[f"dataset_{d}_x"], data[f"dataset_{d}_y"] = x, y
    axes[1].set_title("validation masked MSE, by dataset (supplementary)")
    axes[1].set_xlabel("global step")
    axes[1].legend(frameon=False, ncol=2)
    w.save(fig, "fig_route_convergence", data)


# --------------------------------------------------------------------------- #
# 4 & 5. real masks and real reconstructions
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _fixed_example(model, ds: EEGWindowDataset, route: Route, mask_seed: int,
                   window_index: int = 0, device="cpu"):
    """One fixed validation window under its fixed mask, through the model."""
    item = ds[window_index]
    batch = collate_windows([item])
    meta = {k: v.to(device) for k, v in ds.montage().items()}
    gen = _mask_generator(mask_seed, ds.dataset_id, window_index)
    out = model(batch["x"].to(device), route.route_id, channel_meta=meta,
                mask_ratio=model.mask_ratio, mask_generator=gen)
    _, metrics = masked_reconstruction_loss(out)
    C, P = route.n_channels, route.patches_per_channel
    return {
        "raw": batch["x"][0].cpu().numpy(),
        "spec": out["spec"][0].detach().cpu().numpy(),
        "target": out["target"][0].detach().cpu().numpy().reshape(C, P, -1),
        "pred": out["pred"][0].detach().cpu().numpy().reshape(C, P, -1),
        "mask": out["mask"][0].detach().cpu().numpy().reshape(C, P),
        "valid": (out["valid_tokens"][0].detach().cpu().numpy().reshape(C, P)
                  if out["valid_tokens"] is not None else np.ones((C, P), bool)),
        "metrics": metrics,
        "subject_id": item["subject_id"],
        "recording_id": item["recording_id"],
        "window_index": window_index,
    }


def fig_mask_reconstruction(w: FigureWriter, model, datasets, mask_seed, device):
    rows = []
    for rid in ROUTES:
        pick = next((d for d in datasets
                     if datasets[d].route_id == rid), None)
        if pick:
            rows.append((rid, pick))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 6, figsize=(17, 2.5 * len(rows)),
                             squeeze=False)
    data, meta_rows = {}, []
    for r, (rid, dsid) in enumerate(rows):
        route = ROUTES[rid]
        ex = _fixed_example(model, datasets[dsid], route, mask_seed,
                            device=device)
        C, P = route.n_channels, route.patches_per_channel
        recon = ex["pred"].reshape(C, -1)
        tgt = ex["target"].reshape(C, -1)
        masked_repr = tgt.copy()
        masked_repr[np.repeat(ex["mask"], route.patch_t, axis=1)] = np.nan
        err = np.abs(recon - tgt)

        panels = [
            (ex["raw"], "raw EEG (context only)", "RdBu_r"),
            (ex["spec"], "folded-wavelet target", "viridis"),
            (ex["mask"].astype(float), "binary patch mask", "gray_r"),
            (masked_repr, "masked representation", "viridis"),
            (recon, "predicted folded-wavelet patches", "viridis"),
            (err, "|reconstruction error|", "magma"),
        ]
        for c, (arr, title, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest")
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"{rid}\n{dsid}\n{C} ch")
            ax.set_xticks([])
            ax.set_yticks([])
            # Grey overlay marking the masked TIME patches, on the panels whose
            # x axis is samples rather than patches.
            if c in (1, 3, 4, 5):
                for p in range(P):
                    if ex["mask"][:, p].any():
                        ax.axvspan(p * route.patch_t, (p + 1) * route.patch_t,
                                   color="0.5", alpha=0.16, lw=0)
        m = ex["metrics"]
        axes[r][5].set_xlabel(
            f"MSE {m['loss_masked_mse']:.4f}  RMSE {m['masked_rmse']:.4f}  "
            f"MAE {m['masked_mae']:.4f}  r {m['masked_corr']:.3f}")
        data[f"{rid}_target"] = tgt
        data[f"{rid}_pred"] = recon
        data[f"{rid}_mask"] = ex["mask"]
        meta_rows.append({"route_id": rid, "dataset_id": dsid,
                          "subject_id": ex["subject_id"],
                          "recording_id": ex["recording_id"],
                          "window_index": ex["window_index"],
                          "mask_seed": mask_seed, **m})
    fig.suptitle("Fixed validation windows under their fixed masks. The head "
                 "predicts folded wavelet patches; raw EEG is context only.",
                 y=1.005)
    w.save(fig, "fig_mask_reconstruction", data, {"examples": meta_rows})


def fig_mask_examples_by_dataset(w: FigureWriter, model, datasets, mask_seed,
                                 device):
    present = [d for d in PRETRAIN_DATASETS if d in datasets]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 3.0),
                             squeeze=False)
    data, meta_rows = {}, []
    for i, dsid in enumerate(present):
        route = ROUTES[datasets[dsid].route_id]
        ex = _fixed_example(model, datasets[dsid], route, mask_seed,
                            device=device)
        ax = axes[0][i]
        ax.imshow(ex["mask"].astype(float), aspect="auto", cmap="gray_r",
                  interpolation="nearest", vmin=0, vmax=1)
        ratio = ex["mask"].sum() / max(1, ex["valid"].sum())
        ax.set_title(f"{dsid}\n{route.route_id}  masked {ratio*100:.0f}%")
        ax.set_xlabel("time patch")
        if i == 0:
            ax.set_ylabel("channel slot")
        ax.set_xticks(range(route.patches_per_channel))
        data[f"{dsid}_mask"] = ex["mask"]
        meta_rows.append({"dataset_id": dsid, "route_id": route.route_id,
                          "subject_id": ex["subject_id"],
                          "recording_id": ex["recording_id"],
                          "window_index": ex["window_index"],
                          "mask_seed": mask_seed})
    fig.suptitle("The actual masks the model was given, one fixed window per "
                 "dataset", y=1.04)
    w.save(fig, "fig_mask_examples_by_dataset", data, {"examples": meta_rows})


def fig_mask_statistics(w: FigureWriter, model, datasets, mask_seed, device):
    present = [d for d in PRETRAIN_DATASETS if d in datasets]
    if not present:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    data = {}
    ratios, labels, colors = [], [], []
    for dsid in present:
        route = ROUTES[datasets[dsid].route_id]
        ex = _fixed_example(model, datasets[dsid], route, mask_seed,
                            device=device)
        ratios.append(float(ex["mask"].sum() / max(1, ex["valid"].sum())))
        labels.append(dsid)
        colors.append(ROUTE_COLOR[route.route_id])
        if dsid == present[0]:
            # Importance is the summed magnitude spectrum along the token
            # sequence. It is a rank over TOKENS and does not map to a physical
            # frequency, so the axis says token importance and not Hz.
            with torch.no_grad():
                item = datasets[dsid][0]
                batch = collate_windows([item])
                meta = {k: v.to(device) for k, v in datasets[dsid].montage().items()}
                spec = model.wavelet_frontends[route.route_id](
                    batch["x"].to(device))
                tok = model.patch_embed_by_rate[route.rate_key](spec.unsqueeze(1))
                imp = torch.abs(torch.fft.rfft(tok, dim=1)).sum(dim=2)
                imp = torch.nn.functional.interpolate(
                    imp.unsqueeze(1), size=tok.shape[1], mode="linear",
                    align_corners=True).squeeze(1)[0].cpu().numpy()
            sel = ex["mask"].reshape(-1)
            axes[1].scatter(np.arange(imp.size)[~sel], imp[~sel], s=3,
                            color="0.7", label="kept")
            axes[1].scatter(np.arange(imp.size)[sel], imp[sel], s=3,
                            color=OKABE_ITO[1], label="masked")
            axes[1].set_title(f"token importance vs selection ({dsid})")
            axes[1].set_xlabel("token index")
            axes[1].set_ylabel("token/patch importance (arb. units)")
            axes[1].legend(frameon=False, markerscale=3)
            data["importance"] = imp
            data["selected"] = sel

    axes[0].bar(range(len(ratios)), [r * 100 for r in ratios], color=colors)
    axes[0].axhline(model.mask_ratio * 100, color="0.3", ls="--", lw=1,
                    label=f"requested {model.mask_ratio*100:.0f}%")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("actual mask ratio over valid tokens (%)")
    axes[0].set_title("realised mask ratio")
    axes[0].legend(frameon=False)

    for rid, color in ROUTE_COLOR.items():
        members = [i for i, d in enumerate(labels)
                   if PRETRAIN_DATASETS[d].route_id == rid]
        if members:
            axes[2].bar([labels[i] for i in members],
                        [ratios[i] * 100 for i in members], color=color,
                        label=rid)
    axes[2].set_title("mask ratio by route")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(frameon=False)
    data["ratios"] = np.asarray(ratios)
    data["labels"] = np.asarray(labels)
    w.save(fig, "fig_mask_statistics", data, {"mask_seed": mask_seed})


# --------------------------------------------------------------------------- #
# 7, 8, 9. what the model learned
# --------------------------------------------------------------------------- #

def _filter_bank(frontend) -> Optional[np.ndarray]:
    """The learned analysis kernels of one frontend, as ``[n_filters, K]``."""
    kernels = []
    for name, p in frontend.named_parameters():
        if p.dim() >= 2 and p.shape[-1] >= 4 and ("filter" in name or "wave" in name
                                                  or "kernel" in name):
            kernels.append(p.detach().float().reshape(-1, p.shape[-1]).cpu().numpy())
    if not kernels:
        return None
    return np.concatenate(kernels, axis=0)


def fig_wavelet_frequency_response(w: FigureWriter, model,
                                   init_model: Optional[object]):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.1))
    data = {}
    for ax, (rid, route) in zip(axes, ROUTES.items()):
        for label, m, color, alpha in (
                ("at init", init_model, "0.6", 0.9),
                ("trained", model, ROUTE_COLOR[rid], 1.0)):
            if m is None:
                continue
            bank = _filter_bank(m.wavelet_frontends[rid])
            if bank is None:
                continue
            K = bank.shape[-1]
            H = np.abs(np.fft.rfft(bank, n=max(256, K * 8), axis=-1))
            H = H.mean(axis=0)
            H = H / max(H.max(), 1e-12)
            freqs = np.fft.rfftfreq(max(256, K * 8), d=1.0 / route.sampling_rate)
            ax.plot(freqs, 20 * np.log10(np.maximum(H, 1e-6)), color=color,
                    alpha=alpha, label=label)
            data[f"{rid}_{label.replace(' ', '_')}_freq"] = freqs
            data[f"{rid}_{label.replace(' ', '_')}_db"] = 20 * np.log10(
                np.maximum(H, 1e-6))
        ax.set_title(f"{rid}  ({route.sampling_rate} Hz)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_xlim(0, route.sampling_rate / 2)
        ax.set_ylim(-60, 2)
    axes[0].set_ylabel("mean magnitude response (dB)")
    axes[0].legend(frameon=False)
    fig.suptitle("Learned wavelet analysis filters, per route, on that route's "
                 "own frequency axis", y=1.03)
    w.save(fig, "fig_wavelet_frequency_response", data)


def fig_scale_fold_weights(w: FigureWriter, model, datasets, device):
    present = [d for d in PRETRAIN_DATASETS if d in datasets]
    if not present:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    data = {}
    alphas, spreads, labels, colors = [], [], [], []
    with torch.no_grad():
        for dsid in present:
            route = ROUTES[datasets[dsid].route_id]
            batch = collate_windows([datasets[dsid][0]])
            model.wavelet_frontends[route.route_id](batch["x"].to(device))
            fold = model.wavelet_frontends[route.route_id].fold
            a = fold.alpha_mean
            if a is None:
                continue
            alphas.append(a.detach().float().cpu().numpy())
            st, sc = fold.alpha_std_time, fold.alpha_std_chan
            spreads.append((float(st.mean()) if st is not None else 0.0,
                            float(sc.mean()) if sc is not None else 0.0))
            labels.append(dsid)
            colors.append(ROUTE_COLOR[route.route_id])
    if not alphas:
        return
    A = np.stack(alphas)
    S = A.shape[1]
    xs = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for s in range(S):
        axes[0].bar(xs, A[:, s], bottom=bottom, label=f"scale {s}",
                    color=OKABE_ITO[s % len(OKABE_ITO)])
        bottom += A[:, s]
    axes[0].axhline(1.0, color="0.3", lw=0.6)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("mean mixing weight")
    axes[0].set_title("ScaleFold alpha by dataset\n(a single full bar = collapse "
                      "onto one scale)")
    axes[0].legend(frameon=False, ncol=2)

    sp = np.asarray(spreads)
    axes[1].bar(xs - 0.2, sp[:, 0], width=0.4, color=OKABE_ITO[0],
                label="std over time blocks")
    axes[1].bar(xs + 0.2, sp[:, 1], width=0.4, color=OKABE_ITO[1],
                label="std over channels")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_title("fold spread (zero over time = a static fold)")
    axes[1].legend(frameon=False)
    data["alpha"] = A
    data["spread"] = sp
    data["labels"] = np.asarray(labels)
    w.save(fig, "fig_scale_fold_weights", data)


#: Lobe by 10-20 name prefix. A NAMING rule, applied to the channel's label --
#: C1 is a learned embedding of a name and contains no geometry, so the colours
#: are a reader's aid and not a property the model was given. Recorded in the
#: figure metadata so the grouping can be checked.
LOBE_RULES = (("frontal", ("FP", "AF", "F")), ("central", ("FC", "C", "CP")),
              ("parietal", ("P", "PO")), ("occipital", ("O", "I")),
              ("temporal", ("T",)))


def _lobe(name: str) -> str:
    u = name.upper()
    for lobe, prefixes in LOBE_RULES:
        for p in sorted(prefixes, key=len, reverse=True):
            if u.startswith(p):
                return lobe
    return "other"


def fig_channel_embedding(w: FigureWriter, model, datasets):
    if model.channel_encoder is None:
        return
    used = sorted({int(i) for ds in datasets.values()
                   for i in ds.montage()["channel_ids"].tolist() if i > 1})
    if len(used) < 3:
        used = list(range(2, min(len(CHANNEL_VOCAB), 40)))
    E = model.channel_encoder.id_embed.weight.detach().float().cpu().numpy()[used]
    names = [CHANNEL_VOCAB[i] for i in used]
    lobes = [_lobe(n) for n in names]
    palette = {l: OKABE_ITO[i] for i, l in enumerate(sorted(set(lobes)))}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    norms = np.linalg.norm(E, axis=1)
    axes[0].bar(range(len(names)), norms,
                color=[palette[l] for l in lobes])
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=90, fontsize=4)
    axes[0].set_ylabel("embedding L2 norm")
    axes[0].set_title("C1 channel-name embedding norm")

    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    # This machine's BLAS raises spurious FP warnings from matmul on ordinary
    # input; the result is checked below rather than trusted silently.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Cm = En @ En.T
    if not np.all(np.isfinite(Cm)):
        raise SystemExit("channel embedding cosine matrix is not finite; the "
                         "checkpoint's embedding table contains NaN or Inf.")
    Cm = np.clip(Cm, -1.0, 1.0)
    im = axes[1].imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("cosine similarity")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    Ec = E - E.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Ec @ Vt[:2].T
    for l in sorted(set(lobes)):
        sel = [i for i, x in enumerate(lobes) if x == l]
        axes[2].scatter(P[sel, 0], P[sel, 1], s=12, color=palette[l], label=l)
    axes[2].set_title("PCA (2 components)")
    axes[2].set_xlabel(f"PC1 ({S[0]**2/np.sum(S**2)*100:.0f}% var)")
    axes[2].set_ylabel(f"PC2 ({S[1]**2/np.sum(S**2)*100:.0f}% var)")
    axes[2].legend(frameon=False)
    fig.suptitle("C1 is a learned embedding of the channel NAME. Lobe colours "
                 "come from a name-prefix rule, not from electrode geometry.",
                 y=1.04)
    w.save(fig, "fig_channel_embedding",
           {"embedding": E, "names": np.asarray(names),
            "lobes": np.asarray(lobes), "cosine": Cm, "pca": P},
           {"lobe_rules": {k: list(v) for k, v in LOBE_RULES},
            "channel_ids": used})


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--format", default="svg", choices=["svg", "pdf", "png"])
    p.add_argument("--manifest", default=None,
                   help="override the manifest the run recorded")
    p.add_argument("--mask-seed", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None,
                   help="figure names to regenerate")
    args = p.parse_args(argv)

    # Rank 0 only. Under torchrun every rank would otherwise write the same
    # files at the same time and the SVGs would interleave.
    if int(os.environ.get("RANK", "0")) != 0:
        return 0

    apply_style()
    run_dir = args.run_dir
    ckpt_path = (args.checkpoint if os.path.isabs(args.checkpoint)
                 else os.path.join(run_dir, args.checkpoint))
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}", file=sys.stderr)
        return 1

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mcfg = cfg.get("model", {})
    device = torch.device("cpu")

    model = MultiRouteEEGPretrainer(
        embed_dim=int(mcfg.get("embed_dim", 384)),
        depth=int(mcfg.get("depth", 6)), num_heads=int(mcfg.get("num_heads", 6)),
        dropout=float(mcfg.get("dropout", 0.1)),
        norm=mcfg.get("norm", "rmsnorm"), ffn=mcfg.get("ffn", "swiglu"),
        qk_norm=bool(mcfg.get("qk_norm", True)),
        max_level=int(mcfg.get("max_level", 3)),
        wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
        wavelet_names=mcfg.get("wavelet_names"),
        wave_init_mode=mcfg.get("wave_init_mode", "pad"),
        fold_synthesis=int(mcfg.get("fold_synthesis", 3)),
        fold_gamma=float(mcfg.get("fold_gamma", 0.1)),
        mask_ratio=float(mcfg.get("mask_ratio", 0.5)),
        channel_encoding=mcfg.get("channel_encoding", "id"),
        channel_injection=mcfg.get("channel_injection", "token"),
        channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
        channel_vocab_size=ck.get("channel_vocab_size"),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # A second model at initialisation, for the "before training" trace. Built
    # rather than stored: it is a deterministic function of the seed and the
    # config, and carrying a copy in every checkpoint would double their size.
    torch.manual_seed(int(cfg.get("seed", 42)))
    init_model = MultiRouteEEGPretrainer(
        embed_dim=int(mcfg.get("embed_dim", 384)),
        depth=int(mcfg.get("depth", 6)), num_heads=int(mcfg.get("num_heads", 6)),
        max_level=int(mcfg.get("max_level", 3)),
        wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
        wavelet_names=mcfg.get("wavelet_names"),
        wave_init_mode=mcfg.get("wave_init_mode", "pad"),
        channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
        channel_vocab_size=ck.get("channel_vocab_size"))
    init_model.eval()

    manifest_path = args.manifest or cfg.get("data", {}).get(
        f"manifest_{args.split}")
    datasets: Dict[str, EEGWindowDataset] = {}
    if manifest_path and os.path.isfile(manifest_path):
        index = CorpusIndex.from_manifest(manifest_path)
        for d in sorted(index.by_dataset()):
            try:
                datasets[d] = EEGWindowDataset(index, d)
            except Exception as exc:                          # noqa: BLE001
                print(f"  skipping {d}: {exc}")
    else:
        print(f"  note: manifest for split {args.split!r} is not available "
              f"({manifest_path!r}); figures that need real windows are skipped.")

    run_manifest = {}
    mf = os.path.join(run_dir, "dataset_manifest.json")
    if os.path.isfile(mf):
        with open(mf) as f:
            run_manifest = json.load(f)

    mask_seed = (args.mask_seed if args.mask_seed is not None
                 else int(cfg.get("train", {}).get("val_mask_seed", 1234)))
    writer = FigureWriter(run_dir, args.format, {
        "checkpoint": ckpt_path,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
        "channel_vocab_sha256": ck.get("channel_vocab_sha256"),
        "plotting_script_git_commit": git_commit(),
        "split": args.split, "mask_seed": mask_seed,
        "manifest": manifest_path,
    })

    epoch_rows = read_jsonl(os.path.join(run_dir, "metrics_epoch.jsonl"))
    step_rows = read_jsonl(os.path.join(run_dir, "metrics_step.jsonl"))

    todo = set(args.only) if args.only else None

    def want(name):
        return todo is None or name in todo

    print(f"figures for {run_dir} (checkpoint {os.path.basename(ckpt_path)})")
    if want("fig_dataset_routes"):
        fig_dataset_routes(writer, run_manifest)
    if want("fig_pretraining_convergence"):
        fig_pretraining_convergence(writer, epoch_rows, step_rows)
    if want("fig_route_convergence"):
        fig_route_convergence(writer, epoch_rows)
    if datasets:
        if want("fig_mask_reconstruction"):
            fig_mask_reconstruction(writer, model, datasets, mask_seed, device)
        if want("fig_mask_examples_by_dataset"):
            fig_mask_examples_by_dataset(writer, model, datasets, mask_seed,
                                         device)
        if want("fig_mask_statistics"):
            fig_mask_statistics(writer, model, datasets, mask_seed, device)
        if want("fig_scale_fold_weights"):
            fig_scale_fold_weights(writer, model, datasets, device)
    if want("fig_wavelet_frequency_response"):
        fig_wavelet_frequency_response(writer, model, init_model)
    if want("fig_channel_embedding"):
        fig_channel_embedding(writer, model, datasets)

    for ds in datasets.values():
        ds.close()
    print(f"\n{len(writer.written)} figure(s) in {writer.fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
