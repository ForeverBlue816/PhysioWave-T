"""
Training loop for the EEG C1 multi-route pretrainer.

The objective is unchanged from the legacy pretrainer:

    loss = masked patch MSE  +  fold_kl * ScaleFold KL

and nothing else. No reference consistency, no contrastive term, no query
specialisation -- those belong to the WAST/TARE path, which this one does not
touch.

Two things here are less obvious than the loop:

**Validation masks are fixed.** A validation loss computed under a fresh random
mask each epoch moves because the mask moved, and the curve then measures the
sampler. Each validation batch draws its mask from a generator seeded by
(mask seed, dataset, first window index), so epoch 12 masks exactly what epoch 1
masked and a change in the curve is a change in the model.

**Resume restores the sampling sequence, not just the weights.** The schedule is
a pure function of (seed, epoch), and the step within the epoch is checkpointed,
so a resumed run draws the batches the interrupted one would have drawn next.
Restoring the model and letting the sampler start over would quietly re-train on
the first half of the epoch and skip the second.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from channel_embedding import save_vocab, vocab_payload
from .data import (DEFAULT_BATCH_BY_ROUTE, CorpusIndex, EEGWindowDataset,
                   RouteBatchLoader, RouteSchedule, collate_windows)
from .model import MultiRouteEEGPretrainer, masked_reconstruction_loss
from .routes import PRETRAIN_DATASETS, ROUTES


# --------------------------------------------------------------------------- #
# Deterministic validation
# --------------------------------------------------------------------------- #

def _mask_generator(seed: int, dataset_id: str, first_index: int) -> torch.Generator:
    """A CPU generator fixed by the sample identity, not by the epoch."""
    g = torch.Generator()
    h = (seed * 1_000_003
         + (abs(hash(dataset_id)) % 1_000_003) * 1009
         + int(first_index))
    g.manual_seed(h % (2 ** 63 - 1))
    return g


class ValIterator:
    """Every validation window once, in file order, in route-pure batches.

    Not the training schedule: validation is not a mixture to be sampled, it is
    a fixed set to be swept, and sweeping it in a fixed order is what lets the
    mask seed depend on the window's identity.
    """

    def __init__(self, index: CorpusIndex, batch_by_route: Dict[str, int],
                 num_replicas: int = 1, rank: int = 0,
                 max_batches_per_dataset: Optional[int] = None):
        self.datasets = {d: EEGWindowDataset(index, d)
                         for d in sorted(index.by_dataset())}
        self.batch_by_route = dict(batch_by_route)
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.max_batches = max_batches_per_dataset

    def __iter__(self):
        for dataset_id, ds in self.datasets.items():
            bs = self.batch_by_route[ds.route_id]
            stride = bs * self.num_replicas
            n = len(ds)
            emitted = 0
            for start in range(0, n - stride + 1, stride):
                lo = start + self.rank * bs
                idx = list(range(lo, min(lo + bs, n)))
                if len(idx) < bs:
                    break
                batch = collate_windows([ds[i] for i in idx])
                batch["channel_meta"] = ds.montage()
                batch["mask_seed_index"] = start
                yield batch
                emitted += 1
                if self.max_batches and emitted >= self.max_batches:
                    break

    def close(self):
        for ds in self.datasets.values():
            ds.close()


# --------------------------------------------------------------------------- #
# Metric accumulation
# --------------------------------------------------------------------------- #

class Accumulator:
    """Running means, globally and per route and per dataset."""

    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, metrics: Dict[str, float], route_id: str, dataset_id: str):
        for k, v in metrics.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            for key in (k, f"route/{route_id}/{k}", f"dataset/{dataset_id}/{k}"):
                self.sums[key] = self.sums.get(key, 0.0) + float(v)
                self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {k: self.sums[k] / self.counts[k] for k in self.sums
                if self.counts[k]}


def _reduce_mean(value: float, distributed: bool, device) -> float:
    if not distributed:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / dist.get_world_size())


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class EEGC1Trainer:
    def __init__(self, cfg: Dict, out_dir: str, info, max_steps: Optional[int] = None):
        self.cfg = cfg
        self.out_dir = out_dir
        self.info = info
        self.max_steps = max_steps
        self.is_main = getattr(info, "is_main", True)
        self.device = getattr(info, "device", torch.device("cpu"))
        self.distributed = getattr(info, "distributed", False)

        tcfg = cfg.get("train", {})
        mcfg = cfg.get("model", {})
        dcfg = cfg.get("data", {})

        self.epochs = int(tcfg.get("epochs", 10))
        self.grad_accum = int(tcfg.get("grad_accumulation_steps", 1))
        self.fold_kl = float(tcfg.get("fold_kl", 1e-3))
        self.mask_ratio = float(mcfg.get("mask_ratio", 0.5))
        self.val_mask_seed = int(tcfg.get("val_mask_seed", 1234))
        self.clip_grad = float(tcfg.get("clip_grad_norm", 1.0))
        self.vis_every = int(tcfg.get("vis_every_epochs", 5))
        self.batch_by_route = {**DEFAULT_BATCH_BY_ROUTE,
                               **(tcfg.get("batch_size_by_route") or {})}

        # -- data ---------------------------------------------------------- #
        self.train_index = CorpusIndex.from_manifest(dcfg["manifest_train"])
        self.val_index = (CorpusIndex.from_manifest(dcfg["manifest_val"])
                          if dcfg.get("manifest_val") else None)

        self.schedule = RouteSchedule(
            self.train_index, weights=dcfg.get("weights"),
            steps_per_epoch=tcfg.get("steps_per_epoch"),
            seed=int(cfg.get("seed", 42)),
            batch_by_route=self.batch_by_route,
            num_replicas=getattr(info, "world_size", 1),
            rank=getattr(info, "rank", 0))
        self.loader = RouteBatchLoader(self.train_index, self.schedule)

        # -- model --------------------------------------------------------- #
        self.model = MultiRouteEEGPretrainer(
            embed_dim=int(mcfg.get("embed_dim", 384)),
            depth=int(mcfg.get("depth", 6)),
            num_heads=int(mcfg.get("num_heads", 6)),
            mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
            dropout=float(mcfg.get("dropout", 0.1)),
            norm=mcfg.get("norm", "rmsnorm"), ffn=mcfg.get("ffn", "swiglu"),
            qk_norm=bool(mcfg.get("qk_norm", True)),
            max_level=int(mcfg.get("max_level", 3)),
            wave_kernel_size=int(mcfg.get("wave_kernel_size", 16)),
            wavelet_names=mcfg.get("wavelet_names"),
            wave_init_mode=mcfg.get("wave_init_mode", "pad"),
            use_separate_channel=bool(mcfg.get("use_separate_channel", True)),
            fold_synthesis=int(mcfg.get("fold_synthesis", 3)),
            fold_gamma=float(mcfg.get("fold_gamma", 0.1)),
            masking_strategy=mcfg.get("masking_strategy", "frequency_guided"),
            importance_ratio=float(mcfg.get("importance_ratio", 0.6)),
            mask_ratio=self.mask_ratio,
            channel_encoding=mcfg.get("channel_encoding", "id"),
            channel_injection=mcfg.get("channel_injection", "token"),
            channel_embed_dim=int(mcfg.get("channel_embed_dim", 64)),
            channel_token_gate_init=float(mcfg.get("channel_token_gate_init", 0.0)),
        ).to(self.device)

        self.raw_model = self.model
        if self.distributed:
            # find_unused_parameters is required and not merely defensive: a
            # step runs ONE route's frontend, so the other three frontends and
            # the other rate's patcher and decoder produce no gradient. Every
            # rank runs the same route, so the unused set is identical across
            # ranks and the reducer's bookkeeping stays consistent.
            self.model = DDP(
                self.model,
                device_ids=[info.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=True)

        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(), lr=float(tcfg.get("lr", 3e-4)),
            weight_decay=float(tcfg.get("weight_decay", 0.05)),
            betas=(0.9, 0.95))
        total_steps = max(1, self.epochs * len(self.schedule) // self.grad_accum)
        warmup = int(tcfg.get("warmup_epochs", 1)) * max(
            1, len(self.schedule) // self.grad_accum)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: self._lr_lambda(s, warmup, total_steps,
                                                      float(tcfg.get("min_lr_ratio", 0.01))))
        self.use_amp = (self.device.type == "cuda"
                        and tcfg.get("precision", "bf16") != "fp32")
        self.amp_dtype = (torch.bfloat16 if tcfg.get("precision", "bf16") == "bf16"
                          else torch.float16)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.use_amp and self.amp_dtype is torch.float16))

        self.epoch = 0
        self.global_step = 0
        self.best = float("inf")
        self.history: Dict[str, List] = {"train": [], "val": []}
        self.tb = None
        if self.is_main:
            try:
                from ..train.utils import TensorBoardWriter
                self.tb = TensorBoardWriter(os.path.join(out_dir, "tensorboard"))
            except Exception:                                 # noqa: BLE001
                self.tb = None

    @staticmethod
    def _lr_lambda(step, warmup, total, min_ratio):
        if warmup and step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    # -- files ------------------------------------------------------------- #
    def _write_startup(self):
        if not self.is_main:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "figures"), exist_ok=True)
        save_vocab(os.path.join(self.out_dir, "channel_vocab.json"))
        # Written here rather than in pretrain_main: this trainer is dispatched
        # before that function reaches its own copy of these, so leaving them to
        # it produced a run directory with no record of what produced it.
        try:
            from ..config import save_resolved
            save_resolved(self.cfg, os.path.join(self.out_dir,
                                                 "config_resolved.yaml"))
        except Exception as exc:                              # noqa: BLE001
            print(f"warning: could not write config_resolved.yaml ({exc})")
        try:
            from ..models.checkpoint import environment_info
            with open(os.path.join(self.out_dir, "environment.json"), "w") as f:
                json.dump(environment_info(), f, indent=2)
        except Exception as exc:                              # noqa: BLE001
            print(f"warning: could not write environment.json ({exc})")
        report = self.raw_model.parameter_report()
        mixture = self.schedule.realised_mixture()
        manifest = {
            "routes": {rid: {"n_channels": r.n_channels,
                             "window_samples": r.window_samples,
                             "sampling_rate": r.sampling_rate,
                             "patch_size": list(r.patch_size),
                             "n_tokens": r.n_tokens}
                       for rid, r in ROUTES.items()},
            "datasets": {d: {"route_id": PRETRAIN_DATASETS[d].route_id,
                             "n_windows": self.train_index.window_counts().get(d, 0)}
                         for d in self.schedule.dataset_ids},
            "target_weights": self.schedule.weights,
            "realised_mixture": mixture,
            "batch_size_by_route": self.batch_by_route,
            "steps_per_epoch": self.schedule.steps_per_epoch,
            "parameters": report,
            **self.raw_model.vocab_fingerprint(),
        }
        with open(os.path.join(self.out_dir, "dataset_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print("=" * 66)
        print("  EEG C1 multi-route pretraining")
        print(f"  total parameters        {report['total']:,}")
        print(f"  shared transformer      {report['shared_transformer']:,}")
        for rid in ROUTES:
            print(f"  frontend {rid:<10s}     {report[f'wavelet_frontend.{rid}']:,}")
        for rate in sorted(self.raw_model.patch_embed_by_rate):
            print(f"  patch_embed {rate:<7s}    {report[f'patch_embed.{rate}']:,}"
                  f"   decoder {report[f'reconstruction_head.{rate}']:,}")
        if "channel_encoder" in report:
            print(f"  channel encoder (C1)    {report['channel_encoder']:,}"
                  f"  + proj {report['channel_to_token']:,}")
        print("  routes:")
        for rid, r in ROUTES.items():
            print(f"    {r.describe()}")
        print("  target vs realised mixture (by step / by window):")
        for d in self.schedule.dataset_ids:
            print(f"    {d:<14s} target {self.schedule.weights[d]*100:5.1f}%   "
                  f"step {mixture['by_step'][d]*100:5.1f}%   "
                  f"window {mixture['by_window'][d]*100:5.1f}%")
        print(f"  channel vocab sha256    {vocab_payload()['channel_vocab_sha256'][:16]}")
        print(f"  steps/epoch {self.schedule.steps_per_epoch}  "
              f"epochs {self.epochs}  grad_accum {self.grad_accum}")
        print("=" * 66, flush=True)

    def _append_jsonl(self, name: str, row: Dict):
        if not self.is_main:
            return
        with open(os.path.join(self.out_dir, name), "a") as f:
            f.write(json.dumps(row) + "\n")

    # -- checkpointing ----------------------------------------------------- #
    def state_dict(self) -> Dict:
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss_masked_mse": self.best,
            "sampler": self.schedule.state_dict(),
            "history": self.history,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
            "config": self.cfg,
            **vocab_payload(),
        }

    def save(self, name: str):
        if not self.is_main:
            return
        path = os.path.join(self.out_dir, name)
        torch.save(self.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.raw_model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        self.epoch = int(ck.get("epoch", 0))
        self.global_step = int(ck.get("global_step", 0))
        self.best = float(ck.get("best_val_loss_masked_mse", float("inf")))
        self.history = ck.get("history", {"train": [], "val": []})
        if ck.get("sampler"):
            self.schedule.load_state_dict(ck["sampler"])
        rng = ck.get("rng") or {}
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"])
                                else rng["torch"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        recorded = ck.get("channel_vocab_sha256")
        current = vocab_payload()["channel_vocab_sha256"]
        if recorded and recorded != current:
            raise SystemExit(
                f"checkpoint was trained under channel vocabulary "
                f"{recorded[:16]} and this one is {current[:16]}. Every "
                f"embedding row would mean a different electrode. Check out the "
                f"commit that produced the checkpoint, or retrain.")
        print(f"resumed from {path}: epoch {self.epoch}, step {self.global_step}")

    # -- one epoch --------------------------------------------------------- #
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.schedule.set_epoch(self.epoch)
        acc = Accumulator()
        t0 = time.time()
        windows = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for i, batch in enumerate(self.loader):
            x = batch["x"].to(self.device, non_blocking=True)
            meta = {k: v.to(self.device) for k, v in batch["channel_meta"].items()}
            is_accum = (i + 1) % self.grad_accum != 0

            ctx = (torch.autocast(self.device.type, dtype=self.amp_dtype)
                   if self.use_amp else _nullcontext())
            with ctx:
                out = self.model(x, batch["route_id"], channel_meta=meta,
                                 mask_ratio=self.mask_ratio)
                loss, metrics = masked_reconstruction_loss(out, self.fold_kl)

            self.scaler.scale(loss / self.grad_accum).backward()
            grad_norm = float("nan")
            if not is_accum:
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), self.clip_grad))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            windows += x.shape[0]
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            if math.isfinite(grad_norm):
                metrics["grad_norm"] = grad_norm
            gate = self.raw_model.channel_gate_value()
            if gate is not None:
                metrics["channel_token_gate_tanh"] = gate
            alpha = out.get("fold_alpha")
            if alpha is not None:
                for s, a in enumerate(alpha.detach().float().cpu().tolist()):
                    metrics[f"fold_alpha_{s}"] = a
            acc.add(metrics, batch["route_id"], batch["dataset_id"])

            if self.is_main and (i % 50 == 0 or i + 1 == len(self.loader)):
                self._append_jsonl("metrics_step.jsonl", {
                    "epoch": self.epoch, "step": self.global_step,
                    "route_id": batch["route_id"],
                    "dataset_id": batch["dataset_id"], **metrics})
            self.schedule.start_step = i + 1
            if self.max_steps and self.global_step >= self.max_steps:
                break

        out = acc.mean()
        elapsed = max(1e-9, time.time() - t0)
        out["throughput_windows_per_s"] = windows / elapsed
        out["epoch_seconds"] = elapsed
        if self.device.type == "cuda":
            out["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated(
                self.device) / 1e6
        mixture = self.schedule.realised_mixture()
        for d, v in mixture["by_step"].items():
            out[f"mixture_step/{d}"] = v
        for d, v in mixture["by_window"].items():
            out[f"mixture_window/{d}"] = v
        return out

    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        if self.val_index is None:
            return {}
        self.model.eval()
        acc = Accumulator()
        it = ValIterator(self.val_index, self.batch_by_route,
                         getattr(self.info, "world_size", 1),
                         getattr(self.info, "rank", 0),
                         max_batches_per_dataset=max_batches)
        for batch in it:
            x = batch["x"].to(self.device, non_blocking=True)
            meta = {k: v.to(self.device) for k, v in batch["channel_meta"].items()}
            gen = _mask_generator(self.val_mask_seed, batch["dataset_id"],
                                  batch["mask_seed_index"])
            ctx = (torch.autocast(self.device.type, dtype=self.amp_dtype)
                   if self.use_amp else _nullcontext())
            with ctx:
                out = self.raw_model(x, batch["route_id"], channel_meta=meta,
                                     mask_ratio=self.mask_ratio,
                                     mask_generator=gen)
                _, metrics = masked_reconstruction_loss(out, self.fold_kl)
            acc.add(metrics, batch["route_id"], batch["dataset_id"])
        it.close()
        means = acc.mean()
        for k in ("loss_total", "loss_masked_mse"):
            if k in means:
                means[k] = _reduce_mean(means[k], self.distributed, self.device)
        return means

    # -- driver ------------------------------------------------------------ #
    def fit(self) -> int:
        self._write_startup()
        start = self.epoch
        for epoch in range(start, self.epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            row = {"epoch": epoch, "global_step": self.global_step,
                   **{f"train/{k}": v for k, v in train_metrics.items()},
                   **{f"val/{k}": v for k, v in val_metrics.items()}}
            self._append_jsonl("metrics_epoch.jsonl", row)
            if self.tb is not None:
                self.tb.add_scalars(
                    {k: float(v) for k, v in row.items()
                     if isinstance(v, (int, float)) and math.isfinite(v)},
                    self.global_step)
            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)
            if self.is_main:
                with open(os.path.join(self.out_dir, "history.json"), "w") as f:
                    json.dump(self.history, f, indent=2)
                print(f"epoch {epoch}: train mse "
                      f"{train_metrics.get('loss_masked_mse', float('nan')):.5f}  "
                      f"val mse {val_metrics.get('loss_masked_mse', float('nan')):.5f}  "
                      f"gate {train_metrics.get('channel_token_gate_tanh', 0):.4f}",
                      flush=True)

            self.schedule.start_step = 0
            self.epoch = epoch + 1
            # self.best is updated BEFORE latest.pth is written. Written the
            # other way round, latest.pth records the previous epoch's best, and
            # a resume from it then believes the best is worse than it is --
            # letting the next epoch overwrite best.pth with a worse checkpoint.
            v = val_metrics.get("loss_masked_mse")
            improved = v is not None and v < self.best
            if improved:
                self.best = v
            self.save("latest.pth")
            if improved:
                self.save("best.pth")
            if self.max_steps and self.global_step >= self.max_steps:
                break
        return 0


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
