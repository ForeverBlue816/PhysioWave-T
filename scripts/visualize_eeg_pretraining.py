#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper figures for an EEG C1 pretraining run.

    python scripts/visualize_eeg_pretraining.py \
        --run-dir /path/to/run --checkpoint best.pth --split val --format svg

``--recon-windows N`` draws the two reconstruction figures on N different
validation windows per route instead of one, ranks them by masked NMSE and
writes the ranking to figure_metadata/reconstruction_survey.json. Choosing the
best of several is a normal way to pick a figure; the rank exists so the chosen
one can be captioned as chosen rather than as the window that came up.

Every figure is drawn from the run's own artefacts -- the checkpoint, the
metrics files and the corpus the run trained on. Nothing is illustrative: the
masks drawn are the masks the model was given, the reconstructions are that
checkpoint's outputs, and the wavelet responses are its learned filters.

WHAT THE MODEL RECONSTRUCTS. TWO things, at equal weight. One head predicts
FOLDED WAVELET PATCHES: Spec(x) is decomposed into J+1 scales and the dynamic
fold reduces the scale axis back to one row per electrode, and that folded
representation is what is patched, masked and predicted. The other head
predicts the PREPROCESSED EEG of the same masked patches -- a waveform, in the
z-scored units the preprocessing produced, and not an inverse wavelet
transform. Both appear in fig_mask_reconstruction; the raw head has
14_raw_waveform_reconstruction to itself.

THREE RULES THE RECONSTRUCTION FIGURES FOLLOW, because breaking any of them
flatters the model:

  * the spec target drawn is ``target_spec`` -- the per-patch NORMALISED
    tensor the loss is computed against -- not the unnormalised ``clean_spec``.
    Comparing a normalised prediction to an unnormalised target is a
    comparison of two different quantities;
  * a target and its composite share colour limits, taken from the target.
    Independent autoscaling makes a prediction with a third of the amplitude
    draw at the same contrast;
  * predictions are shown on MASKED patches only. No gradient ever reached a
    visible token's decoder output, so displaying one as reconstruction
    reports a number nothing trained.

Alongside every figure:
    figures/<name>.svg          the figure
    figure_data/<name>.npz      the arrays it was drawn from
    figure_metadata/<name>.json checkpoint hash, step, sample identity, seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from physiowave.eeg_c1.objective import (objective_equation,  # noqa: E402
                                         resolve_eeg_c1_objective)
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


#: ``(logging key, fallback keys, label)``. The fallbacks are the pre-dual
#: names, so a run from before the raw head still renders. New code names the
#: head it means: `loss_masked_mse` does not say whether it is spec or raw.
CONVERGENCE_PANELS = (
    (("loss_total",), "total loss"),
    (("loss_masked_spec_mse", "loss_masked_mse"), "masked spec MSE"),
    (("loss_masked_raw_smoothl1",), "masked raw SmoothL1"),
    (("masked_spec_corr", "masked_corr"), "masked spec corr"),
)


def _first_key(rows: List[Dict], keys) -> str:
    """The first of ``keys`` this run actually logged, under any prefix.

    metrics_step.jsonl holds bare names and metrics_epoch.jsonl holds
    ``train/`` and ``val/`` ones, so a probe for the bare name alone would miss
    a run whose step file is absent and fall back to a key that does not exist.
    """
    for k in keys:
        if any(k in r or f"train/{k}" in r or f"val/{k}" in r for r in rows):
            return k
    return keys[0]


def fig_pretraining_convergence(w: FigureWriter, epoch_rows, step_rows):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    data = {}
    for ax, (keys, label) in zip(axes, CONVERGENCE_PANELS):
        key = _first_key(step_rows + epoch_rows, keys)
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
        if key.endswith("mse") or key.endswith("smoothl1"):
            ax.set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)
    fig.suptitle("Masked dual reconstruction: convergence. The spec and raw "
                 "panels are different units and are not comparable to each "
                 "other.", y=1.03)
    w.save(fig, "fig_pretraining_convergence", data)


def fig_route_convergence(w: FigureWriter, epoch_rows):
    """Per route and per dataset, on the TOTAL loss -- what is being selected on.

    These curves are globally aggregated across ranks: the validation sweep is
    partitioned, so before that reduction each of these lines was one rank's
    slice of the dataset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    data = {}
    metric = _first_key(epoch_rows, ("loss_total", "loss_masked_mse"))
    for rid, color in ROUTE_COLOR.items():
        x, y = _series(epoch_rows, f"val/route/{rid}/{metric}")
        if x.size:
            axes[0].plot(x, y, color=color, marker="o", ms=2.5, label=rid)
            data[f"route_{rid}_x"], data[f"route_{rid}_y"] = x, y
    mx, my = _series(epoch_rows, "val/macro_route_loss_total")
    if mx.size:
        # The unweighted mean of the four lines above, which is what
        # best_macro_total.pth is selected on. It is NOT the global validation
        # loss: that one is 97% TUEG and HBN by batch count.
        axes[0].plot(mx, my, color="0.2", ls="--", lw=1.4, label="macro (mean)")
        data["macro_x"], data["macro_y"] = mx, my
    axes[0].set_title(f"validation {metric}, by route")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel(metric)
    axes[0].legend(frameon=False)

    for d in PRETRAIN_DATASETS:
        x, y = _series(epoch_rows, f"val/dataset/{d}/{metric}")
        if x.size:
            axes[1].plot(x, y, color=DATASET_COLOR[d], marker=".", ms=3,
                         label=d)
            data[f"dataset_{d}_x"], data[f"dataset_{d}_y"] = x, y
    axes[1].set_title(f"validation {metric}, by dataset (supplementary)")
    axes[1].set_xlabel("global step")
    axes[1].legend(frameon=False, ncol=2)
    w.save(fig, "fig_route_convergence", data)


# --------------------------------------------------------------------------- #
# 4 & 5. real masks and real reconstructions
# --------------------------------------------------------------------------- #

def _expand(mask_cp: np.ndarray, patch_t: int) -> np.ndarray:
    """``[C, P]`` patch mask to ``[C, P*patch_t]``, the sample grid."""
    return np.repeat(mask_cp, patch_t, axis=1)


def _composite(target: np.ndarray, pred: np.ndarray,
               m: np.ndarray) -> np.ndarray:
    """Target where the model could SEE, prediction where it could not.

    Showing the decoder's output on visible patches as though it were
    reconstruction is the thing this function exists to prevent: nothing in the
    loss ever looks at a visible token, so those values are unsupervised, and a
    figure that displays them is reporting a number no gradient produced.
    """
    return np.where(m, pred, target)


def _hide(target: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The target with the masked patches removed -- NaN, drawn as blank."""
    out = target.astype(float).copy()
    out[m] = np.nan
    return out


def _corrupt(target: np.ndarray, m: np.ndarray) -> np.ndarray:
    """What the frontend was actually handed: masked patches zeroed.

    Zero rather than NaN: this is the SIGNAL the model saw, and the fill is
    zero because the window is z-scored, so zero is its mean.
    """
    out = target.astype(float).copy()
    out[m] = 0.0
    return out


def _masked_error(pred: np.ndarray, target: np.ndarray, m: np.ndarray,
                  valid_rows: np.ndarray) -> np.ndarray:
    """``|pred - target|`` on masked valid samples, NaN everywhere else."""
    e = np.abs(pred.astype(float) - target.astype(float))
    e[~m] = np.nan
    e[~valid_rows] = np.nan
    return e


def _sym_limits(target: np.ndarray, valid_rows: np.ndarray,
                q: float = 99.5) -> float:
    """A robust symmetric limit from the TARGET, shared by every panel of a pair.

    Autoscaling the target and the prediction independently is what makes a
    flat prediction look like a good one: matplotlib stretches whatever range
    it is given, so two panels with different limits are not comparable however
    similar they look. One limit, taken from the target, and a prediction that
    is too small stays visibly too small.
    """
    vals = target[valid_rows]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(vals), q))
    return v if v > 0 else 1.0


@torch.no_grad()
def _fixed_example(model, ds: EEGWindowDataset, route: Route, mask_seed: int,
                   window_index: int = 0, device="cpu",
                   objective: Optional[Dict] = None):
    """One fixed validation window under its fixed mask, through the model.

    Everything is returned on the ``[C, T]`` sample grid, unpatchified with the
    model's own inverse, so a panel can be indexed by channel and time without
    the caller reconstructing the channel-major token order.
    """
    item = ds[window_index]
    batch = collate_windows([item])
    meta = {k: v.to(device) for k, v in ds.montage().items()}
    gen = _mask_generator(mask_seed, ds.dataset_id, window_index)
    out = model(batch["x"].to(device), route.route_id, channel_meta=meta,
                mask_ratio=model.mask_ratio, mask_generator=gen)
    o = objective or {}
    _, metrics = masked_reconstruction_loss(
        out,
        spec_weight=o.get("spec_weight"), raw_weight=o.get("raw_weight"),
        fold_kl=o.get("fold_kl"), raw_beta=o.get("raw_beta"))
    C, P, pt = route.n_channels, route.patches_per_channel, route.patch_t

    def flat(key):
        return model.unpatchify(out[key], C, pt)[0].detach().cpu().numpy()

    mask = out["mask"][0].detach().cpu().numpy().reshape(C, P)
    valid = (out["valid_tokens"][0].detach().cpu().numpy().reshape(C, P)
             if out["valid_tokens"] is not None else np.ones((C, P), bool))
    return {
        "raw": batch["x"][0].cpu().numpy(),
        # The UNNORMALISED frontend output. Kept for the wavelet-analysis
        # figures; it is NOT the reconstruction target when
        # normalize_spec_target is on, and nothing below labels it as one.
        "clean_spec": out["clean_spec"][0].detach().cpu().numpy(),
        "spec": out["clean_spec"][0].detach().cpu().numpy(),
        # What the loss actually compares, on the sample grid.
        "target_spec": flat("target_spec"),
        "pred_spec": flat("pred_spec"),
        "target_raw": flat("target_raw"),
        "pred_raw": flat("pred_raw"),
        "normalize_spec_target": bool(out.get("normalize_spec_target", True)),
        # -- compatibility aliases: the older figures asked for these -------- #
        "target": flat("target_spec"),
        "pred": flat("pred_spec"),
        "mask": mask,
        "valid": valid,
        "metrics": metrics,
        "subject_id": item["subject_id"],
        "recording_id": item["recording_id"],
        "window_index": window_index,
    }


def _route_rows(datasets) -> List:
    """One deterministic ``(route, dataset)`` pair per route, in route order.

    ``sorted`` rather than "the first one the iteration happens to yield": a
    figure whose sample moves because a manifest was rebuilt in a different
    order is not the fixed sample it claims to be.
    """
    rows = []
    for rid in ROUTES:
        members = sorted(d for d in datasets
                         if datasets[d].route_id == rid and len(datasets[d]))
        if members:
            rows.append((rid, members[0]))
    return rows


def _survey_indices(ds, n: int) -> List[int]:
    """``n`` window indices spread across the dataset, deterministically.

    NOT ``range(n)``. The corpus is one shard per recording and windows sit
    consecutively inside it, so the first n indices are n consecutive windows
    of ONE recording -- often one minute of one subject. A survey drawn from
    those varies the mask and almost nothing else, and picking "the best of
    eight" out of it selects over noise. linspace spans the dataset, so the
    candidates come from different recordings and the spread between them is a
    spread over the corpus.
    """
    n_total = len(ds)
    if n_total <= 0:
        return []
    if n <= 1 or n_total == 1:
        return [0]
    return sorted({int(round(float(i)))
                   for i in np.linspace(0, n_total - 1, min(n, n_total))})


def survey_reconstruction_windows(model, datasets, mask_seed, device,
                                  objective, n_windows: int = 1):
    """Every window the two reconstruction figures draw, modelled once.

    Returns ``(rows, grid, ranks)``:

    * ``rows``      -- ``(route_id, dataset_id)``, one per route;
    * ``grid[k][i]``-- the example for ``rows[i]``'s k-th surveyed window;
    * ``ranks[(route_id, k)]`` -- ``{"raw": r, "spec": s, "n": K}``, where 1 is
      the LOWEST masked NMSE among the windows surveyed for that route.

    The rank is the point of surveying at all. Drawing eight windows and
    publishing the one that came out best is a legitimate way to choose a
    figure; publishing it as though it were simply the window that came up is
    not. The rank travels in the figure's caption and its metadata so the two
    stay distinguishable after the figure leaves this directory.

    One pass, shared by both figures. They show the same windows from two
    sides; computing them separately would run the model twice and let the two
    drift onto different samples the moment either index rule is edited.
    """
    rows = _route_rows(datasets)
    if not rows:
        return [], [], {}
    n_windows = max(1, int(n_windows))
    picks = {rid: _survey_indices(datasets[dsid], n_windows)
             for rid, dsid in rows}
    n_drawn = max((len(v) for v in picks.values()), default=0)

    grid: List[List[Optional[Dict]]] = []
    for k in range(n_drawn):
        col: List[Optional[Dict]] = []
        for rid, dsid in rows:
            if not picks[rid]:
                col.append(None)
                continue
            # A dataset with fewer windows than the survey repeats its last one
            # rather than dropping out of the later figures. A four-row figure
            # that silently becomes three rows reads as "this route failed".
            wi = picks[rid][min(k, len(picks[rid]) - 1)]
            col.append(_fixed_example(model, datasets[dsid], ROUTES[rid],
                                      mask_seed, window_index=wi,
                                      device=device, objective=objective))
        grid.append(col)

    return rows, grid, _rank_windows(rows, grid)


def _rank_windows(rows, grid) -> Dict:
    """``{(route_id, k): {"raw": r, "spec": s, "n": K}}``, 1 = lowest NMSE.

    Per route, because the routes are not comparable to each other: E19_256's
    NMSE and E128_512's are computed on different numbers of channels at
    different sampling rates, and a single pooled ranking would report "the
    best window" as "the window from the easiest route".

    The two heads rank independently. They are separately weighted halves of
    the objective and a window can be the best on one and the worst on the
    other; collapsing them into one score would hide exactly that.
    """
    n_drawn = len(grid)
    ranks: Dict = {}
    for i, (rid, _ds) in enumerate(rows):
        for tag, key in (("raw", "masked_raw_nmse"),
                         ("spec", "masked_spec_nmse")):
            scored = []
            for k in range(n_drawn):
                ex = grid[k][i]
                v = float(ex["metrics"][key]) if ex else float("nan")
                # NaN compares false against everything, so an unguarded sort
                # can leave a diverged window in first place -- and first place
                # is what someone reading the survey for a figure reaches for.
                scored.append((v if math.isfinite(v) else math.inf, k))
            scored.sort(key=lambda t: t[0])
            for pos, (_v, k) in enumerate(scored, start=1):
                ranks.setdefault((rid, k), {})[tag] = pos
    for key in ranks:
        ranks[key]["n"] = n_drawn
    return ranks


def write_reconstruction_survey(w: FigureWriter, rows, grid, ranks,
                                mask_seed: int) -> List[Dict]:
    """The table a figure gets picked from, printed and written next to them.

    Written, not only printed: "we showed window 431 of TUEG" is not a
    reproducible statement unless the windows it was chosen over are recorded
    too. The file is what a caption's "best of 8" can be checked against.
    """
    out: List[Dict] = []
    for k, col in enumerate(grid):
        for i, (rid, dsid) in enumerate(rows):
            ex = col[i]
            if ex is None:
                continue
            mt, rk = ex["metrics"], ranks.get((rid, k), {})
            out.append({
                "figure_index": k,
                "figure": _recon_name("fig_mask_reconstruction", k),
                "waveform_figure": _recon_name(
                    "14_raw_waveform_reconstruction", k),
                "route_id": rid, "dataset_id": dsid,
                "window_index": ex["window_index"],
                "subject_id": ex["subject_id"],
                "recording_id": ex["recording_id"],
                "mask_seed": mask_seed,
                "raw_corr": float(mt["masked_raw_corr"]),
                "raw_nmse": float(mt["masked_raw_nmse"]),
                "spec_corr": float(mt["masked_spec_corr"]),
                "spec_nmse": float(mt["masked_spec_nmse"]),
                "raw_nmse_rank": rk.get("raw"),
                "spec_nmse_rank": rk.get("spec"),
                "n_windows_surveyed": rk.get("n"),
            })
    if not out:
        return out

    n_drawn = len(grid)
    print(f"\n  reconstruction survey: {n_drawn} window(s) per route "
          f"(rank 1 = lowest masked NMSE for that route)")
    print(f"  {'fig':>3}  {'route':<10} {'dataset':<10} {'win':>7}  "
          f"{'raw r':>7} {'raw NMSE':>9} {'spec r':>7} {'spec NMSE':>10}  "
          f"{'rank raw/spec':>13}")
    for r in out:
        print(f"  {r['figure_index']:>3}  {r['route_id']:<10} "
              f"{r['dataset_id']:<10} {r['window_index']:>7}  "
              f"{r['raw_corr']:>7.3f} {r['raw_nmse']:>9.3f} "
              f"{r['spec_corr']:>7.3f} {r['spec_nmse']:>10.3f}  "
              f"{str(r['raw_nmse_rank']) + '/' + str(r['spec_nmse_rank']):>13}")
    path = os.path.join(w.meta_dir, "reconstruction_survey.json")
    with open(path, "w") as f:
        json.dump({"mask_seed": mask_seed, "n_windows": n_drawn,
                   "note": "one row per drawn window. rank 1 is the lowest "
                           "masked NMSE among the windows surveyed for that "
                           "route; a figure chosen by this table is a SELECTED "
                           "example and its caption should say so",
                   "windows": out}, f, indent=2)
    print(f"  wrote {path}")
    return out


def _recon_name(base: str, k: int) -> str:
    """``base`` for the first window, ``base_wNN`` after it.

    The first keeps the historical name so a paper script, a test or a figure
    reference written before the survey existed still resolves.
    """
    return base if k == 0 else f"{base}_w{k:02d}"


#: Panel titles, in order. The figure and the tests read the same list.
RECONSTRUCTION_PANEL_TITLES = (
    "raw EEG target", "raw EEG, mask applied", "raw composite reconstruction",
    "raw |error|, masked only", "spec target", "masked spec target",
    "spec composite reconstruction", "spec |error|, masked only",
)


def reconstruction_panels(ex: Dict, route: Route) -> List:
    """``[(array, title, cmap, vmin, vmax)] x 8`` for one route row.

    Pure, so the three properties that make the figure honest can be checked
    without rendering anything:

    * panels 0-2 share colour limits, and 4-6 share theirs. Taken from the
      TARGET, so a prediction with a third of the amplitude draws at a third of
      the contrast instead of being stretched to match.
    * the spec panels are ``target_spec``, the per-patch normalised tensor the
      loss is computed against -- never ``clean_spec``, which is on a different
      scale and is not what any gradient compared against.
    * the two error panels are NaN off the masked valid patches. Nothing
      supervises a visible token's decoder output.
    """
    pt = route.patch_t
    m = _expand(ex["mask"], pt)
    valid_rows = _expand(ex["valid"], pt)
    raw_t, raw_p = ex["target_raw"], ex["pred_raw"]
    spec_t, spec_p = ex["target_spec"], ex["pred_spec"]
    raw_lim = _sym_limits(raw_t, valid_rows)
    spec_lim = _sym_limits(spec_t, valid_rows)
    spec_label = ("normalised spec target"
                  if ex.get("normalize_spec_target", True)
                  else "spec target (raw scale)")
    titles = list(RECONSTRUCTION_PANEL_TITLES)
    titles[4] = spec_label
    titles[5] = f"masked {spec_label}"
    return [
        (raw_t, titles[0], "RdBu_r", -raw_lim, raw_lim),
        (_corrupt(raw_t, m), titles[1], "RdBu_r", -raw_lim, raw_lim),
        (_composite(raw_t, raw_p, m), titles[2], "RdBu_r", -raw_lim, raw_lim),
        (_masked_error(raw_p, raw_t, m, valid_rows), titles[3], "magma",
         0.0, None),
        (spec_t, titles[4], "RdBu_r", -spec_lim, spec_lim),
        (_hide(spec_t, m), titles[5], "RdBu_r", -spec_lim, spec_lim),
        (_composite(spec_t, spec_p, m), titles[6], "RdBu_r",
         -spec_lim, spec_lim),
        (_masked_error(spec_p, spec_t, m, valid_rows), titles[7], "magma",
         0.0, None),
    ]


def fig_mask_reconstruction(w: FigureWriter, rows, grid, ranks, mask_seed,
                            objective: Optional[Dict] = None):
    """Both heads, one row per route, on fixed windows under fixed masks.

    Eight panels, in two groups of four that read the same way: target, what
    the model was handed, the composite reconstruction, and the error on the
    masked patches alone.

    One figure per surveyed window. With ``--recon-windows 1`` that is the
    single window this figure always showed; above 1 the later figures are
    ``fig_mask_reconstruction_w01`` and so on, each carrying its rank among the
    survey so a chosen one can be captioned as chosen.

    Four things the previous version of this figure got wrong, each of which
    flattered the model:

    * it showed raw EEG only as context, so the second head -- half the
      objective -- never appeared;
    * it compared the UNNORMALISED clean_spec against a prediction trained
      against the normalised target, which is a comparison of two different
      quantities;
    * it autoscaled target and prediction independently, so a prediction with a
      third of the target's amplitude drew at the same contrast;
    * its error map covered visible patches too, and the visible patches are
      not supervised: their error is small because the encoder passes the token
      through, not because anything reconstructed them.
    """
    if not rows or not grid:
        return
    n_drawn = len(grid)
    for k, col in enumerate(grid):
        drawn = [(i, rid, dsid) for i, (rid, dsid) in enumerate(rows)
                 if col[i] is not None]
        if not drawn:
            continue
        fig, axes = plt.subplots(len(drawn), 8,
                                 figsize=(21, 2.6 * len(drawn)), squeeze=False)
        data, meta_rows = {}, []
        for r, (i, rid, dsid) in enumerate(drawn):
            ex = col[i]
            route = ROUTES[rid]
            pt = route.patch_t
            n_patches = route.patches_per_channel
            m = _expand(ex["mask"], pt)
            valid_rows = _expand(ex["valid"], pt)
            raw_t, raw_p = ex["target_raw"], ex["pred_raw"]
            spec_t, spec_p = ex["target_spec"], ex["pred_spec"]
            raw_err = _masked_error(raw_p, raw_t, m, valid_rows)
            spec_err = _masked_error(spec_p, spec_t, m, valid_rows)
            raw_lim = _sym_limits(raw_t, valid_rows)
            spec_lim = _sym_limits(spec_t, valid_rows)

            panels = reconstruction_panels(ex, route)
            for c, (arr, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[r][c]
                ax.imshow(arr, aspect="auto", cmap=cmap,
                          interpolation="nearest", vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title(title)
                if c == 0:
                    ax.set_ylabel(f"{rid}\n{dsid}\n{route.n_channels} ch")
                ax.set_xticks([])
                ax.set_yticks([])
                # The columns any channel has a masked patch in, marked on the
                # panels whose x axis is samples.
                for pi in range(n_patches):
                    if ex["mask"][:, pi].any():
                        ax.axvspan(pi * pt, (pi + 1) * pt, color="0.5",
                                   alpha=0.10, lw=0)
            mt = ex["metrics"]
            rk = ranks.get((rid, k), {})
            ratio = ex["mask"].sum() / max(1, ex["valid"].sum())
            axes[r][3].set_xlabel(
                f"SmoothL1 {mt['loss_masked_raw_smoothl1']:.4f}  "
                f"r {mt['masked_raw_corr']:.3f}  "
                f"NMSE {mt['masked_raw_nmse']:.3f}")
            axes[r][7].set_xlabel(
                f"MSE {mt['loss_masked_spec_mse']:.4f}  "
                f"r {mt['masked_spec_corr']:.3f}  "
                f"NMSE {mt['masked_spec_nmse']:.3f}")
            # The realised ratio, not the requested one: the budget is per
            # sample over the VALID tokens, so a route with padded slots masks
            # fewer. The rank rides along whenever there was a choice to make.
            label = f"window {ex['window_index']}   " \
                    f"masked {ratio * 100:.1f}% of valid patches"
            if n_drawn > 1 and rk:
                label += (f"\nNMSE rank raw {rk.get('raw')}/{rk.get('n')}, "
                          f"spec {rk.get('spec')}/{rk.get('n')}")
            axes[r][0].set_xlabel(label)
            data[f"{rid}_target_raw"] = raw_t
            data[f"{rid}_pred_raw"] = raw_p
            data[f"{rid}_target_spec"] = spec_t
            data[f"{rid}_pred_spec"] = spec_p
            data[f"{rid}_masked_error_raw"] = raw_err
            data[f"{rid}_masked_error_spec"] = spec_err
            data[f"{rid}_mask"] = ex["mask"]
            data[f"{rid}_valid"] = ex["valid"]
            # Compatibility with the pre-dual arrays, which meant the spec pair.
            data[f"{rid}_target"] = spec_t
            data[f"{rid}_pred"] = spec_p
            meta_rows.append({"route_id": rid, "dataset_id": dsid,
                              "subject_id": ex["subject_id"],
                              "recording_id": ex["recording_id"],
                              "window_index": ex["window_index"],
                              "mask_seed": mask_seed,
                              "actual_mask_ratio": float(ratio),
                              "raw_color_limit": raw_lim,
                              "spec_color_limit": spec_lim,
                              "spec_target_is_normalised":
                                  ex["normalize_spec_target"],
                              "raw_nmse_rank": rk.get("raw"),
                              "spec_nmse_rank": rk.get("spec"),
                              "n_windows_surveyed": rk.get("n"),
                              **mt})
        title = ("Fixed validation windows under their fixed masks. Composite "
                 "= target on visible patches, prediction on masked ones; the "
                 "error maps are masked patches only, because visible patches "
                 "are not supervised. Target and composite share colour "
                 "limits.")
        if n_drawn > 1:
            title += (f"  Window {k + 1} of {n_drawn} surveyed per route -- a "
                      f"figure picked from this set is a SELECTED example.")
        fig.suptitle(title, y=1.005)
        w.save(fig, _recon_name("fig_mask_reconstruction", k), data,
               {"examples": meta_rows,
                "panels": list(RECONSTRUCTION_PANEL_TITLES),
                "survey_index": k, "n_windows_surveyed": n_drawn,
                "note": "spec panels use target_spec -- the per-patch "
                        "normalised target the loss is computed against -- not "
                        "clean_spec"})


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


# --------------------------------------------------------------------------- #
# 10. the two objectives, side by side
# --------------------------------------------------------------------------- #

def fig_dual_objective(w: FigureWriter, epoch_rows):
    """Both reconstruction terms over training, on their own scales.

    They are not comparable in magnitude -- one is an MSE on folded wavelet
    coefficients, the other a SmoothL1 on z-scored EEG -- so plotting them
    against a shared axis would say more about their units than about their
    progress. Two panels, and a third for the weighted contributions, which IS
    the comparison that matters: it shows which term the optimizer is actually
    following.
    """
    if not epoch_rows:
        return
    ep = [r["epoch"] for r in epoch_rows]

    def series(prefix, key):
        return [r.get(f"{prefix}/{key}", float("nan")) for r in epoch_rows]

    spec_tr, spec_va = series("train", "loss_masked_spec_mse"), \
        series("val", "loss_masked_spec_mse")
    raw_tr, raw_va = series("train", "loss_masked_raw_smoothl1"), \
        series("val", "loss_masked_raw_smoothl1")
    kl_tr = series("train", "loss_fold_kl")
    if all(math.isnan(v) for v in spec_tr):
        return

    # From the run's own resolved configuration, through the same helper the
    # trainer used. NOT a literal: a plotting default is exactly how a figure
    # comes to caption weights the run never trained under.
    meta = resolve_eeg_c1_objective({"objective": w.base_meta.get("objective")})
    ws = float(meta["spec_weight"])
    wr = float(meta["raw_weight"])
    wk = float(meta["fold_kl"])

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    for ax, (tr, va, title, ylab) in zip(axes[:2], (
            (spec_tr, spec_va, "Spec reconstruction",
             "masked MSE, folded wavelet"),
            (raw_tr, raw_va, "Raw EEG reconstruction",
             "masked SmoothL1, z-scored"))):
        ax.plot(ep, tr, color=OKABE_ITO[0], label="train", marker="o", ms=2.5)
        ax.plot(ep, va, color=OKABE_ITO[1], label="val", marker="s", ms=2.5,
                ls="--")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False)

    ax = axes[2]
    contrib = {
        f"spec x{ws:g}": ([ws * v for v in spec_tr], OKABE_ITO[0]),
        f"raw x{wr:g}": ([wr * v for v in raw_tr], OKABE_ITO[1]),
        f"foldKL x{wk:g}": ([wk * v for v in kl_tr], OKABE_ITO[2]),
    }
    bottom = np.zeros(len(ep))
    for label, (vals, colour) in contrib.items():
        v = np.nan_to_num(np.asarray(vals, dtype=float))
        ax.fill_between(ep, bottom, bottom + v, label=label, color=colour,
                        alpha=0.85, lw=0)
        bottom = bottom + v
    ax.set_title("Weighted contribution to the total")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weight x loss")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle(f"The two reconstruction targets:  spec x{ws:g}   "
                 f"raw x{wr:g}   fold KL x{wk:g}", y=1.04, fontsize=10)
    fig.tight_layout()
    w.save(fig, "10_dual_objective",
           data={"epoch": ep, "spec_train": spec_tr, "spec_val": spec_va,
                 "raw_train": raw_tr, "raw_val": raw_va, "fold_kl": kl_tr},
           meta={"weights": {"spec": ws, "raw": wr, "fold_kl": wk},
                 "objective_equation": objective_equation(meta),
                 "note": "panels 1-2 are unweighted losses on their own "
                         "scales and are NOT comparable to each other; panel 3 "
                         "is what enters the total"})


# --------------------------------------------------------------------------- #
# 11. masked against visible -- the control the loss cannot provide
# --------------------------------------------------------------------------- #

def fig_masked_vs_visible(w: FigureWriter, run_dir: str):
    """Where the reconstruction error sits, on hidden tokens and on seen ones.

    The loss only ever looks at masked tokens, so it cannot distinguish a model
    that has learned structure from one that has learned to copy its input: the
    second has a near-zero VISIBLE error and an unimproved masked one. The gap
    between the two distributions is the diagnostic, and it is why the visible
    half is collected during validation despite contributing nothing to
    training.
    """
    path = os.path.join(run_dir, "error_histogram.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        h = json.load(f)
    edges = np.asarray(h["edges"], dtype=float)
    centres = np.sqrt(np.maximum(edges[:-1], 1e-4) * edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for ax, tag, title in ((axes[0], "spec", "Folded-wavelet patches"),
                           (axes[1], "raw", "Preprocessed EEG patches")):
        for half, colour, ls in (("masked", OKABE_ITO[1], "-"),
                                 ("visible", OKABE_ITO[0], "--")):
            key = f"{tag}_{half}"
            counts = np.asarray(h["counts"].get(key, []), dtype=float)
            if counts.sum() == 0:
                continue
            dens = counts / counts.sum()
            ax.step(centres, dens, where="mid", color=colour, ls=ls,
                    label=f"{half}  (mean {h['mean_abs_error'][key]:.3g})")
            ax.axvline(h["mean_abs_error"][key], color=colour, lw=0.6,
                       alpha=0.5)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("|prediction - target|")
        ax.set_ylabel("fraction of tokens")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False)

    ratio = {}
    for tag in ("spec", "raw"):
        m = h["mean_abs_error"].get(f"{tag}_masked", 0.0)
        v = h["mean_abs_error"].get(f"{tag}_visible", 0.0)
        ratio[tag] = (m / v) if v else float("nan")
    fig.suptitle(
        f"Masked vs visible reconstruction error   "
        f"(masked/visible: spec {ratio['spec']:.2f}x, raw {ratio['raw']:.2f}x)",
        y=1.05, fontsize=10)
    fig.tight_layout()
    w.save(fig, "11_masked_vs_visible_error",
           data={"edges": edges,
                 **{k: np.asarray(v) for k, v in h["counts"].items()}},
           meta={"epoch": h.get("epoch"), "n_tokens": h.get("n"),
                 "mean_abs_error": h.get("mean_abs_error"),
                 "masked_over_visible": ratio,
                 "note": "a ratio near 1 means masking costs the model "
                         "nothing, which is what copying looks like"})


# --------------------------------------------------------------------------- #
# 12. where the gradient goes
# --------------------------------------------------------------------------- #

def fig_gradient_flow(w: FigureWriter, step_rows):
    """Gradient norm per branch over training.

    A single global norm cannot answer the question this model raises: with two
    decoders, a frontend run twice and a channel path gated to zero at
    initialisation, "the gradient is healthy" has to be a statement about
    branches. A branch pinned at exactly zero for the whole run is either
    disconnected or deliberately gated, and those look identical in the total.
    """
    keys = sorted({k for r in step_rows for k in r if k.startswith("gradnorm/")})
    if not keys:
        return
    steps = [r["step"] for r in step_rows if any(k in r for k in keys)]
    if len(steps) < 2:
        return

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, key in enumerate(keys):
        xs = [r["step"] for r in step_rows if key in r]
        ys = [r[key] for r in step_rows if key in r]
        if not ys or all(v == 0 for v in ys):
            ax.plot([], [], color=OKABE_ITO[i % 8], ls=":",
                    label=f"{key.split('/', 1)[1]}  (zero throughout)")
            continue
        ax.plot(xs, ys, color=OKABE_ITO[i % 8], lw=1.0,
                label=key.split("/", 1)[1])
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Gradient norm by branch")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=6, loc="upper right")
    fig.tight_layout()
    w.save(fig, "12_gradient_flow",
           data={"step": steps,
                 **{k.replace("/", "_"): [r.get(k, float("nan"))
                                          for r in step_rows] for k in keys}},
           meta={"branches": keys,
                 "note": "the channel encoder is zero while tanh(gate) is "
                         "zero -- that is the initialisation, not a fault"})


# --------------------------------------------------------------------------- #
# 13. what each route cost
# --------------------------------------------------------------------------- #

def fig_route_cost(w: FigureWriter, epoch_rows):
    """Share of steps, share of windows and share of WALL CLOCK, per route.

    The three differ and the third is the one nobody configures. E128_512 draws
    12 windows to E19_256's 64 and each carries 6.7x the tokens, so a mixture
    set by share of steps buys a very different share of the compute -- and a
    run can be spending most of its hours on the route with the smallest share
    of the data.
    """
    if not epoch_rows:
        return
    last = epoch_rows[-1]
    routes = sorted({k.split("/")[-1] for k in last
                     if k.startswith("train/route_share_of_time/")})
    if not routes:
        return

    time_share = [last.get(f"train/route_share_of_time/{r}", 0.0) for r in routes]
    windows = [last.get(f"train/route_windows/{r}", 0.0) for r in routes]
    tot_w = sum(windows) or 1.0
    window_share = [v / tot_w for v in windows]

    x = np.arange(len(routes))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - 0.2, window_share, 0.4, label="share of windows",
           color=[ROUTE_COLOR.get(r, OKABE_ITO[0]) for r in routes], alpha=0.55)
    ax.bar(x + 0.2, time_share, 0.4, label="share of wall clock",
           color=[ROUTE_COLOR.get(r, OKABE_ITO[0]) for r in routes])
    for xi, (ws_, ts) in enumerate(zip(window_share, time_share)):
        if ws_ > 0:
            ax.text(xi + 0.2, ts + 0.01, f"{ts / ws_:.1f}x", ha="center",
                    fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(routes)
    ax.set_ylabel("share of the epoch")
    ax.set_title("What each route cost, against what it contributed")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    w.save(fig, "13_route_cost",
           data={"route": routes, "window_share": window_share,
                 "time_share": time_share, "windows": windows},
           meta={"epoch": last.get("epoch"),
                 "note": "the multiplier above each bar is wall clock per "
                         "window relative to that route's share of windows"})


# --------------------------------------------------------------------------- #
# 14. the raw decoder's output, as a waveform
# --------------------------------------------------------------------------- #

def fig_raw_waveform_reconstruction(w: FigureWriter, rows, grid, ranks,
                                    mask_seed, mask_ratio: float):
    """The second head's prediction against the EEG it was asked to predict.

    One row per route, so E19_256's reconstruction can be read against
    E128_512's rather than against nothing. The previous version showed the
    first dataset the iteration happened to yield and one channel of it, which
    is a sample, not a comparison.

    One figure per surveyed window, on the SAME windows as
    fig_mask_reconstruction, so the two figures at index k describe one sample
    from two sides.

    This is the only figure in the set showing a WAVEFORM in the units the
    recording was preprocessed into. The spec figures show folded wavelet
    patches, which are not EEG and must not be captioned as though they were.

    The prediction is drawn ONLY inside masked intervals. Outside them the
    decoder is unsupervised -- no gradient ever reached those outputs -- and a
    continuous orange line across the whole window would be read as a
    reconstruction of the whole window.
    """
    if not rows or not grid:
        return
    n_drawn = len(grid)
    for k, col in enumerate(grid):
        present = [(i, rid, dsid) for i, (rid, dsid) in enumerate(rows)
                   if col[i] is not None]
        if not present:
            continue
        fig, axes = plt.subplots(
            2 * len(present), 1, figsize=(9.4, 2.3 * len(present)),
            gridspec_kw={"height_ratios": [3, 1] * len(present)})
        axes = np.atleast_1d(axes).reshape(-1)
        data, meta_rows = {}, []

        for i_row, (i, rid, dsid) in enumerate(present):
            route = ROUTES[rid]
            ex = col[i]
            pt, n_patches = route.patch_t, route.patches_per_channel
            mask, valid = ex["mask"], ex["valid"]
            n_masked = mask.sum(axis=1)
            # A VALID channel with SOME masked patches and some visible ones.
            # Not simply the most-masked channel: at mask_ratio 0.70 a good
            # many channels have all their patches hidden, and a row with no
            # visible stretch shows neither the context the model had nor the
            # boundary between the two -- the shading covers the whole window
            # and the composite is the prediction everywhere.
            ok = valid.all(axis=1)
            mixed = np.where(ok & (n_masked > 0)
                             & (n_masked < n_patches))[0]
            if mixed.size:
                want = max(1, min(n_patches - 1,
                                  int(round(mask_ratio * n_patches))))
                ch = int(mixed[np.argmin(np.abs(n_masked[mixed] - want))])
            else:
                fallback = np.where(n_masked > 0)[0]
                if fallback.size == 0:
                    continue
                ch = int(fallback[np.argmax(n_masked[fallback])])

            target = ex["target_raw"][ch]
            pred = ex["pred_raw"][ch]
            m = _expand(mask[ch:ch + 1], pt)[0]
            composite = np.where(m, pred, target)
            t = np.arange(target.size) / route.sampling_rate

            # Dimensionless, and computed on the masked samples of THIS
            # channel, so the numbers under the row are the numbers the row
            # shows. float64 throughout. In float32 the sums of squares behind
            # a correlation over a few thousand samples overflow to inf on a
            # route whose decoder has not converged, and numpy reports it as
            # "overflow encountered in matmul" rather than as a number.
            pm = pred[m].astype(np.float64)
            tm = target[m].astype(np.float64)
            if tm.size:
                resid = float(np.mean((pm - tm) ** 2))
                base = float(np.mean(tm ** 2))
                nmse = resid / max(base, 1e-12)
                pc, tc = pm - pm.mean(), tm - tm.mean()
                den = float(np.linalg.norm(pc) * np.linalg.norm(tc))
                corr = (float(np.dot(pc, tc) / den)
                        if den > 0 and np.isfinite(den) else float("nan"))
                mae = float(np.mean(np.abs(pm - tm)))
                rmse = math.sqrt(resid)
            else:
                nmse = corr = mae = rmse = float("nan")

            ax, ax_err = axes[2 * i_row], axes[2 * i_row + 1]
            ax.plot(t, target, color="0.25", lw=0.9,
                    label="preprocessed EEG (target)")
            ax.plot(t, np.where(m, composite, np.nan), color=OKABE_ITO[1],
                    lw=1.1, label="raw head, masked patches only")
            ax.plot(t, composite, color=OKABE_ITO[0], lw=0.6, alpha=0.55,
                    label="composite (target visible + prediction masked)")
            for pi in range(n_patches):
                if mask[ch, pi]:
                    ax.axvspan(pi * pt / route.sampling_rate,
                               (pi + 1) * pt / route.sampling_rate,
                               color=OKABE_ITO[4], alpha=0.16, lw=0)
            ax.set_ylabel("amplitude\n(z-scored)")
            ax.set_title(
                f"{rid}  {dsid}  channel {route.slots[ch]}  "
                f"window {ex['window_index']}   "
                f"r {corr:.3f}   NMSE {nmse:.3f}   MAE {mae:.3f}   "
                f"RMSE {rmse:.3f}   masked {mask[ch].sum()}/{n_patches} "
                f"patches", loc="left")
            ax.grid(alpha=0.2, lw=0.5)
            ax.set_xticklabels([])
            if i_row == 0:
                ax.legend(frameon=False, ncol=3, fontsize=6.5,
                          loc="upper right")

            err = np.abs(pred - target)
            err[~m] = np.nan
            ax_err.fill_between(t, 0, np.nan_to_num(err), where=m,
                                color=OKABE_ITO[3], lw=0)
            ax_err.set_ylabel("|error|\nmasked")
            ax_err.grid(alpha=0.2, lw=0.5)
            if i_row == len(present) - 1:
                ax_err.set_xlabel("time (s)")
            else:
                ax_err.set_xticklabels([])

            rk = ranks.get((rid, k), {})
            data[f"{rid}_time"] = t
            data[f"{rid}_target"] = target
            data[f"{rid}_pred"] = pred
            data[f"{rid}_composite"] = composite
            data[f"{rid}_mask"] = mask[ch]
            data[f"{rid}_masked_error"] = err
            meta_rows.append({"route_id": rid, "dataset_id": dsid,
                              "channel": route.slots[ch], "channel_slot": ch,
                              "subject_id": ex["subject_id"],
                              "recording_id": ex["recording_id"],
                              "window_index": ex["window_index"],
                              "mask_seed": mask_seed,
                              "raw_corr": corr, "raw_nmse": nmse,
                              "raw_mae": mae, "raw_rmse": rmse,
                              "window_raw_nmse_rank": rk.get("raw"),
                              "n_windows_surveyed": rk.get("n")})

        if not meta_rows:
            plt.close(fig)
            continue
        # Compatibility with the single-route version of this figure, whose npz
        # held bare `time`/`target`/`pred`/`mask` and whose metadata was flat.
        # Paper scripts written against those names keep working and get the
        # first route's row, which is what they used to get.
        first = meta_rows[0]["route_id"]
        for bare in ("time", "target", "pred", "composite", "mask",
                     "masked_error"):
            data[bare] = data[f"{first}_{bare}"]
        title = ("Raw-head reconstruction, one row per route. The trace is "
                 "PREPROCESSED, z-scored EEG -- not raw EDF values, and not "
                 "the folded wavelet representation. Shaded = masked before "
                 "the frontend; the prediction is drawn only there, because "
                 "visible patches are unsupervised.")
        if n_drawn > 1:
            title += (f"  Window {k + 1} of {n_drawn} surveyed per route.")
        fig.suptitle(title, y=1.005, fontsize=8.5)
        fig.tight_layout()
        w.save(fig, _recon_name("14_raw_waveform_reconstruction", k), data,
               {"rows": meta_rows,
                "survey_index": k, "n_windows_surveyed": n_drawn,
                # The flat keys the single-route version wrote, from row one.
                "dataset_id": meta_rows[0]["dataset_id"],
                "route_id": meta_rows[0]["route_id"],
                "channel": meta_rows[0]["channel"],
                "note": "z-scored preprocessed EEG, not raw EDF values, and "
                        "not the folded wavelet representation; the prediction "
                        "is shown on masked patches only. One row per route; "
                        "the unprefixed arrays are the first route's, for "
                        "scripts written against the single-route version."})


#: Threads for the figures that run the model. A login node has many cores and
#: a small per-user thread quota, and torch defaults to one thread per core: the
#: figure that first touches the model then dies with "libgomp: Thread creation
#: failed: Resource temporarily unavailable", followed by "free(): corrupted
#: unsorted chunks" and a segfault, because libgomp failing part-way through a
#: parallel region leaves the allocator inconsistent. None of that names the
#: cause. PW_VIZ_THREADS overrides; the work here is small enough that four is
#: not the constraint.
DEFAULT_THREADS = 4


def _cap_threads() -> int:
    n = int(os.environ.get("PW_VIZ_THREADS") or DEFAULT_THREADS)
    n = max(1, min(n, os.cpu_count() or n))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)
    return n


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
    p.add_argument("--threads", type=int, default=None,
                   help=f"threads for the figures that run the model "
                        f"(default {DEFAULT_THREADS}; a login node's per-user "
                        f"quota is what one-thread-per-core runs into)")
    p.add_argument("--recon-windows", type=int, default=1, metavar="N",
                   help="how many DIFFERENT validation windows per route the "
                        "two reconstruction figures draw (default 1). Above 1 "
                        "they are numbered _w01, _w02 ... and each carries its "
                        "rank among the survey, so a figure chosen for looking "
                        "good can be captioned as chosen. The ranking table is "
                        "printed and written to "
                        "figure_metadata/reconstruction_survey.json")
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
    objective = resolve_eeg_c1_objective(cfg)

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
        # Not cosmetic. normalize_spec_target decides what target_spec IS, and
        # mask_before_frontend decides whether the online view was corrupted at
        # all; defaulting both would render an ablation run as though it had
        # been trained the standard way.
        mask_before_frontend=bool(objective["mask_before_frontend"]),
        normalize_spec_target=bool(objective["normalize_spec_target"]),
        mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
        use_separate_channel=bool(mcfg.get("use_separate_channel", True)),
        masking_strategy=mcfg.get("masking_strategy", "frequency_guided"),
        importance_ratio=float(mcfg.get("importance_ratio", 0.6)),
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
        # The weights the run was trained under, resolved by the SAME helper
        # the trainer uses. This used to read train.spec_recon_weight and
        # train.raw_recon_weight -- keys the config stopped writing -- and fell
        # back to hard-coded 1.0/0.25, which happened to be right. At 0.5/0.5
        # it would have captioned every figure with weights the run never
        # trained under, and nothing in the pipeline would have disagreed.
        "objective": objective,
        "objective_equation": objective_equation(objective),
    })

    epoch_rows = read_jsonl(os.path.join(run_dir, "metrics_epoch.jsonl"))
    step_rows = read_jsonl(os.path.join(run_dir, "metrics_step.jsonl"))

    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()

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
    if want("fig_dual_objective"):
        fig_dual_objective(writer, epoch_rows)
    if want("fig_masked_vs_visible"):
        fig_masked_vs_visible(writer, run_dir)
    if want("fig_gradient_flow"):
        fig_gradient_flow(writer, step_rows)
    if want("fig_route_cost"):
        fig_route_cost(writer, epoch_rows)
    if datasets:
        # One survey, both reconstruction figures. Built only if one of them is
        # wanted: it is the only part of this script that runs the model over
        # more than a single window, and `--only fig_channel_embedding` should
        # not pay for it.
        if want("fig_mask_reconstruction") or \
                want("fig_raw_waveform_reconstruction"):
            rows, grid, ranks = survey_reconstruction_windows(
                model, datasets, mask_seed, device, objective,
                n_windows=args.recon_windows)
            if want("fig_mask_reconstruction"):
                fig_mask_reconstruction(writer, rows, grid, ranks, mask_seed,
                                        objective=objective)
            if want("fig_raw_waveform_reconstruction"):
                fig_raw_waveform_reconstruction(writer, rows, grid, ranks,
                                                mask_seed, model.mask_ratio)
            if len(grid) > 1:
                write_reconstruction_survey(writer, rows, grid, ranks,
                                            mask_seed)
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
