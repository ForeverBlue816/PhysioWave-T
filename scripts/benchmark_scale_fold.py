#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What ScaleFold buys: the encoder with the fold against the same encoder fed
the unfolded bands, on every route, measured on one GPU.

    python scripts/benchmark_scale_fold.py --run-dir $PW_CKPT_ROOT/pretrain_eeg_c1_moe

The wavelet stage emits (J+1) bands per electrode at the full sampling rate.
Without the fold those (J+1)·C rows are what gets patched, so the Transformer
sees (J+1) times the tokens and attention pays (J+1)^2. The fold collapses the
scale axis to one row per electrode before patching, with an MLP shared across
channels deciding the mixture per 0.5 s block. This measures, per route and at
the training batch size:

    tokens per window        C·P with the fold, (J+1)·C·P without
    forward FLOPs            of the encoder (decomposition, patcher, Transformer)
    latency per window       forward, bf16 autocast as in training, CUDA-timed
    peak memory per batch    forward only (training adds activations and
                             optimiser state in the same proportion)
    time breakdown           decomposition / fold / patcher + Transformer, with
                             the fold's own cost beside a plain mean fold

The unfolded path is the model's own modules with the fold bypassed: the
decomposition's [B, (J+1)·C, T] output goes straight to the rate's patcher,
the 2-D position code is built for (J+1)·C rows, and no channel code is added
(a name code per band would be a design decision this benchmark does not
make). The Transformer, patcher and decomposition weights are the checkpoint's.

This is a COMPUTE comparison. Whether the fold loses reconstruction quality
against the unfolded bands is a training ablation, not a benchmark, and is
not measured here.

Writes, under --out-dir (default: the run directory):
    fold_efficiency.json                        every number, per route
    figures/fig_paper_fold_efficiency.{pdf,svg}  the journal figure
    figure_data/, figure_metadata/               as for the other figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                  # noqa: E402
from channel_embedding import channel_ids_for                 # noqa: E402
from physiowave.eeg_c1.routes import ROUTES                   # noqa: E402
from scripts.visualize_eeg_pretraining import (               # noqa: E402
    PAPER, PAPER_FONT, PAPER_WIDTH, FigureWriter, _cap_threads, _letter,
    _paper_rc, apply_style, git_commit, load_trained_model, sha256_file)
from wavelet_modules import ScaleFold                         # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device, warmup: int, iters: int) -> float:
    """Median wall time of ``fn()`` in seconds, after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def _peak_mb(fn, device) -> float:
    if device.type != "cuda":
        fn()
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    _sync(device)
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def _flops(fn) -> float:
    """Forward FLOPs of ``fn()`` via torch's flop counter (NaN if unavailable)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return float("nan")
    try:
        counter = FlopCounterMode(display=False)
        with torch.enable_grad(), counter:
            fn()
        return float(counter.get_total_flops())
    except Exception as exc:                                  # noqa: BLE001
        print(f"  flop counter failed: {exc}")
        return float("nan")


def _meta_for(route, device):
    ids, _ = channel_ids_for(route.slots)
    return {"channel_ids": torch.as_tensor(ids, dtype=torch.long, device=device),
            "valid_channel_mask": torch.ones(route.n_channels, dtype=torch.bool,
                                             device=device)}


class _Paths:
    """The two encoder paths, built from one model's modules."""

    def __init__(self, model, route, device):
        self.model, self.route, self.device = model, route, device
        self.front = model.wavelet_frontends[route.route_id]
        self.patch = model.patch_embed_by_rate[route.rate_key]
        self.meta = _meta_for(route, device)
        self.n_scales = model.max_level + 1
        # A plain mean fold, for the fold's own cost: same bands in, one row
        # per channel out, no statistics and no MLP.
        self.mean_fold = ScaleFold(mode="mean", num_scales=self.n_scales,
                                   in_channels=route.n_channels,
                                   patch_len=route.patch_t).to(device).eval()

    # -- pieces ------------------------------------------------------------ #
    def decomp(self, x):
        return self.front.decomp(x)

    def fold(self, bands):
        return self.front.fold(bands)

    def tokens_folded(self, spec):
        return self.model._build_token_view(
            spec, self.route.rate_key, self.model._channel_code(self.meta),
            self.route.n_channels, self.route.patches_per_channel)

    def tokens_unfolded(self, bands):
        tok = self.patch(bands.unsqueeze(1))                   # [B, (J+1)C*P, D]
        return self.model.pos_embed(tok, freq_size=self.n_scales * self.route.n_channels,
                                    time_size=self.route.patches_per_channel)

    def transformer(self, tokens):
        return self.model.shared_transformer(tokens)

    # -- whole paths ------------------------------------------------------- #
    def encode_folded(self, x):
        return self.transformer(self.tokens_folded(self.fold(self.decomp(x))))

    def encode_unfolded(self, x):
        return self.transformer(self.tokens_unfolded(self.decomp(x)))


@torch.no_grad()
def benchmark_route(model, route, batch_size: int, device, warmup: int, iters: int,
                    amp: bool) -> Dict:
    P = _Paths(model, route, device)
    B, C, T = batch_size, route.n_channels, route.window_samples
    x = torch.randn(B, C, T, device=device)
    n_tok = C * route.patches_per_channel
    row = {"route": route.route_id, "channels": C, "sampling_rate": route.sampling_rate,
           "window_samples": T, "patch_samples": route.patch_t, "batch_size": B,
           "tokens_folded": n_tok, "tokens_unfolded": P.n_scales * n_tok,
           "scales": P.n_scales}
    ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
           if amp and device.type == "cuda" else torch.autocast("cpu", enabled=False))

    def run(fn):
        def f():
            with ctx:
                return fn(x)
        return f

    for tag, fn in (("folded", P.encode_folded), ("unfolded", P.encode_unfolded)):
        try:
            lat = _timed(run(fn), device, warmup, iters)
            mem = _peak_mb(run(fn), device)
            row[f"latency_s_per_batch_{tag}"] = lat
            row[f"latency_ms_per_window_{tag}"] = 1e3 * lat / B
            row[f"peak_mem_mb_{tag}"] = mem
            row[f"windows_per_s_{tag}"] = B / lat
            row[f"status_{tag}"] = "ok"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row[f"status_{tag}"] = "oom"
            print(f"  {route.route_id} {tag}: out of memory at batch {B}")
    # FLOPs on ONE window in fp32 (the counter sees the same ops whatever the
    # dtype; a single window keeps the trace short).
    x1 = x[:1]
    with torch.no_grad():
        bands1 = P.decomp(x1)
        spec1 = P.fold(bands1)
        tok_f = P.tokens_folded(spec1)
        tok_u = P.tokens_unfolded(bands1)
    f_dec = _flops(lambda: P.decomp(x1))
    f_fold = _flops(lambda: P.fold(bands1))
    f_tf_f = _flops(lambda: P.transformer(tok_f))
    f_tf_u = _flops(lambda: P.transformer(tok_u))
    f_pat_f = _flops(lambda: P.tokens_folded(spec1))
    f_pat_u = _flops(lambda: P.tokens_unfolded(bands1))
    row.update({"gflops_per_window_decomp": f_dec / 1e9, "gflops_per_window_fold": f_fold / 1e9,
                "gflops_per_window_patch_folded": f_pat_f / 1e9,
                "gflops_per_window_patch_unfolded": f_pat_u / 1e9,
                "gflops_per_window_transformer_folded": f_tf_f / 1e9,
                "gflops_per_window_transformer_unfolded": f_tf_u / 1e9,
                "gflops_per_window_folded": (f_dec + f_fold + f_pat_f + f_tf_f) / 1e9,
                "gflops_per_window_unfolded": (f_dec + f_pat_u + f_tf_u) / 1e9})
    # Time breakdown of the folded encoder, at the training batch.
    with torch.no_grad(), ctx:
        bands = P.decomp(x)
        spec = P.fold(bands)
        tok = P.tokens_folded(spec)
    t_dec = _timed(run(P.decomp), device, warmup, iters)
    t_fold = _timed(lambda: P.fold(bands), device, warmup, iters)
    t_mean = _timed(lambda: P.mean_fold(bands), device, warmup, iters)
    t_pat = _timed(lambda: P.tokens_folded(spec), device, warmup, iters)
    t_tf = _timed(lambda: P.transformer(tok), device, warmup, iters)
    row.update({"t_ms_decomp": 1e3 * t_dec, "t_ms_fold_dynamic": 1e3 * t_fold,
                "t_ms_fold_mean": 1e3 * t_mean, "t_ms_patch": 1e3 * t_pat,
                "t_ms_transformer_folded": 1e3 * t_tf})
    del x, bands, spec, tok
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #

def _short(rid: str) -> str:
    r = ROUTES[rid]
    return f"{rid.split('_')[0]}\n{r.n_channels} ch, {r.sampling_rate:g} Hz"


def fig_fold_efficiency(w: FigureWriter, rows: List[Dict], device_name: str):
    rids = [r["route"] for r in rows]
    x = np.arange(len(rids))
    wd = 0.36
    col_f, col_u = PAPER["indigo"], PAPER["mauve"]
    with _paper_rc():
        fig = plt.figure(figsize=(PAPER_WIDTH, 2.55))
        gs = fig.add_gridspec(1, 4, wspace=0.55, left=0.065, right=0.995, top=0.86,
                              bottom=0.24)
        axes = [fig.add_subplot(gs[i]) for i in range(4)]

        def pair(ax, key, ylabel, log=True, fmt="{:.0f}×"):
            vf = np.array([r.get(f"{key}_folded", np.nan) for r in rows], float)
            vu = np.array([r.get(f"{key}_unfolded", np.nan) for r in rows], float)
            ax.bar(x - wd / 2, vf, wd, color=col_f, label="With ScaleFold")
            ax.bar(x + wd / 2, vu, wd, color=col_u, label="Unfolded bands")
            if log:
                ax.set_yscale("log")
            top = np.nanmax(np.r_[vf, vu])
            for i, (a, b) in enumerate(zip(vf, vu)):
                if np.isfinite(a) and np.isfinite(b) and a > 0:
                    ax.text(x[i], b * (1.12 if log else 1.0) + (0 if log else 0.02 * top),
                            fmt.format(b / a), ha="center", va="bottom",
                            fontsize=PAPER_FONT - 1)
                elif np.isfinite(a) and not np.isfinite(b):
                    ax.text(x[i] + wd / 2, a * 1.12, "OOM", ha="center", va="bottom",
                            fontsize=PAPER_FONT - 1, color=col_u, rotation=90)
            ax.set_ylabel(ylabel)
            if log:
                ax.set_ylim(top=top * (4.0 if log else 1.25))
            ax.set_xticks(x)
            ax.set_xticklabels([_short(r) for r in rids], fontsize=PAPER_FONT - 1.5,
                               linespacing=1.15)
            ax.tick_params(axis="x", length=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

        pair(axes[0], "tokens", "Tokens per window")
        pair(axes[1], "gflops_per_window", "Encoder GFLOPs per window", fmt="{:.1f}×")
        pair(axes[2], "latency_ms_per_window", "Forward latency (ms / window)",
             fmt="{:.1f}×")
        # (d) breakdown of the folded encoder's forward time, as shares
        ax = axes[3]
        parts = [("t_ms_decomp", "Wavelet decomposition", PAPER["pale_blue"]),
                 ("t_ms_fold_dynamic", "ScaleFold", PAPER["gold"]),
                 ("t_ms_patch", "Patcher", PAPER["lavender"]),
                 ("t_ms_transformer_folded", "Transformer", col_f)]
        tot = np.array([sum(r[k] for k, _, _ in parts) for r in rows])
        bottom = np.zeros(len(rows))
        for k, lab, col in parts:
            v = np.array([r[k] for r in rows]) / tot * 100
            ax.bar(x, v, 0.6, bottom=bottom, color=col, label=lab, lw=0)
            bottom += v
        fold_share = [100 * r["t_ms_fold_dynamic"] / t for r, t in zip(rows, tot)]
        for i, s in enumerate(fold_share):
            ax.text(x[i], 101, f"fold {s:.1f}%", ha="center", va="bottom",
                    fontsize=PAPER_FONT - 1.5, color=PAPER["ink"])
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Share of forward time (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(r) for r in rids], fontsize=PAPER_FONT - 1.5,
                           linespacing=1.15)
        ax.tick_params(axis="x", length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.065, 1.0), ncol=2,
                   frameon=False, fontsize=PAPER_FONT - 1, handlelength=1.2,
                   columnspacing=1.2, borderaxespad=0.0)
        h, l = ax.get_legend_handles_labels()
        fig.legend(h[::-1], l[::-1], loc="upper right", bbox_to_anchor=(0.995, 1.0),
                   ncol=4, frameon=False, fontsize=PAPER_FONT - 1, handlelength=1.0,
                   columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0)
        for i, ch in enumerate("abcd"):
            _letter(fig, gs[i].get_position(fig).x0 - 0.045, 0.99, ch)
        data = {k: np.array([r.get(k, np.nan) for r in rows], float)
                for k in rows[0] if isinstance(rows[0][k], (int, float))}
        data["route"] = np.asarray(rids)
        w.save(fig, "fig_paper_fold_efficiency", data,
               {"rows": rows, "device": device_name,
                "unfolded_path": "decomposition output [(J+1)·C rows] patched "
                                 "directly, 2-D position code for (J+1)·C rows, no "
                                 "channel code; same weights",
                "latency": "median of forward passes at the training batch size, "
                           "bf16 autocast, CUDA-synchronised",
                "flops": "torch.utils.flop_counter on one window, fp32",
                "breakdown": "panel (d): each stage timed alone at the training "
                             "batch; the label above each bar is the dynamic "
                             "fold's share, t_ms_fold_mean in the rows is a "
                             "plain mean fold for comparison",
                "not_measured": "reconstruction quality of the unfolded path"})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pth")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--format", default="svg,pdf")
    p.add_argument("--routes", nargs="*", default=None)
    p.add_argument("--batch-sizes", default=None,
                   help="route=batch,... ; default the run's train.batch_size_by_route")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-amp", action="store_true", help="fp32 instead of bf16 autocast")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-paper", action="store_true")
    args = p.parse_args(argv)

    if args.threads:
        os.environ["PW_VIZ_THREADS"] = str(args.threads)
    _cap_threads()
    apply_style()
    ckpt = (args.checkpoint if os.path.isabs(args.checkpoint)
            else os.path.join(args.run_dir, args.checkpoint))
    ck, cfg, mcfg, objective, modality, model_cls, model = load_trained_model(ckpt)
    if modality != "eeg":
        raise SystemExit("this benchmark is written for the EEG routes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dev_name = (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    print(f"  device: {dev_name}")
    if device.type != "cuda":
        print("  WARNING: no GPU; latencies and memory will not be the training node's")

    bs = dict(cfg.get("train", {}).get("batch_size_by_route", {}) or {})
    if args.batch_sizes:
        for kv in args.batch_sizes.split(","):
            k, v = kv.split("=")
            bs[k.strip()] = int(v)
    rids = args.routes or list(ROUTES)
    rows = []
    for rid in rids:
        route = ROUTES[rid]
        B = int(bs.get(rid, 16))
        print(f"  {rid}: batch {B}, {route.n_channels} ch x {route.window_samples} samples")
        row = benchmark_route(model, route, B, device, args.warmup, args.iters,
                              amp=not args.no_amp)
        rows.append(row)
        print(f"    tokens {row['tokens_folded']} vs {row['tokens_unfolded']};  "
              f"GFLOPs/window {row['gflops_per_window_folded']:.2f} vs "
              f"{row['gflops_per_window_unfolded']:.2f};  "
              f"ms/window {row.get('latency_ms_per_window_folded', float('nan')):.2f} vs "
              f"{row.get('latency_ms_per_window_unfolded', float('nan')):.2f};  "
              f"peak MB {row.get('peak_mem_mb_folded', float('nan')):.0f} vs "
              f"{row.get('peak_mem_mb_unfolded', float('nan')):.0f};  "
              f"fold {row['t_ms_fold_dynamic']:.2f} ms (mean fold "
              f"{row['t_ms_fold_mean']:.2f} ms) of "
              f"{row['t_ms_decomp'] + row['t_ms_fold_dynamic'] + row['t_ms_patch'] + row['t_ms_transformer_folded']:.1f} ms")

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fold_efficiency.json"), "w") as f:
        json.dump({"device": dev_name, "checkpoint": ckpt, "amp": not args.no_amp,
                   "warmup": args.warmup, "iters": args.iters, "rows": rows}, f, indent=2)
    writer = FigureWriter(out_dir, [s.strip() for s in args.format.split(",") if s.strip()],
                          {"checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
                           "epoch": ck.get("epoch"), "device": dev_name,
                           "plotting_script_git_commit": git_commit()},
                          stamp=args.no_paper, paper=not args.no_paper)
    fig_fold_efficiency(writer, rows, dev_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
