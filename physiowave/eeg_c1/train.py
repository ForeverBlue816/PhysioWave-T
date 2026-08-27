"""
Training loop for the EEG C1 multi-route pretrainer.

The objective is unchanged from the legacy pretrainer:

    loss = spec_weight * masked MSE on the DETACHED clean folded-wavelet
           patches
         + raw_weight  * masked SmoothL1 on the DETACHED preprocessed EEG
           patches
         + fold_kl     * ScaleFold KL

    Under mask_before_frontend the masked patches are zeroed in the SIGNAL
    before the wavelet frontend runs, so nothing inside a masked patch can
    reach a visible token through the frontend.

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

#: Bin edges for the reconstruction-error histograms, log-spaced over |error|.
#: A histogram rather than per-batch quantiles because counts add exactly and
#: averaged quantiles do not: the mean of per-batch medians is not the median.
ERROR_BIN_EDGES = [0.0] + [10.0 ** (-3 + 0.125 * i) for i in range(41)]


class ErrorHistogram:
    """Counts of |prediction - target| per bin, split masked vs visible.

    The visible half is the control. A masked-reconstruction objective can be
    satisfied by a model that has learned to copy its input, and the way that
    shows is a visible error near zero while the masked error stays high. There
    is no way to see it from the loss, which only ever looks at masked tokens.
    """

    KEYS = ("spec_masked", "spec_visible", "raw_masked", "raw_visible")

    def __init__(self):
        n = len(ERROR_BIN_EDGES) - 1
        self.counts = {k: [0] * n for k in self.KEYS}
        self.sums = {k: 0.0 for k in self.KEYS}
        self.n = {k: 0 for k in self.KEYS}

    @torch.no_grad()
    def add(self, out: Dict):
        edges = torch.tensor(ERROR_BIN_EDGES, device=out["mask"].device)
        mask = out["mask"]
        for tag, pk, tk in (("spec", "pred_spec", "target_spec"),
                            ("raw", "pred_raw", "target_raw")):
            pred, target = out.get(pk), out.get(tk)
            if pred is None or target is None:
                continue
            err = (pred.float() - target.float()).abs()
            sel = mask.unsqueeze(-1).expand_as(err)
            valid = out.get("valid_tokens")
            if valid is not None:
                ok = valid.unsqueeze(-1).expand_as(err)
            else:
                ok = torch.ones_like(sel)
            for name, take in ((f"{tag}_masked", sel & ok),
                               (f"{tag}_visible", (~sel) & ok)):
                v = err[take]
                if v.numel() == 0:
                    continue
                idx = torch.bucketize(v, edges, right=False).clamp_(
                    1, len(ERROR_BIN_EDGES) - 1) - 1
                hist = torch.bincount(idx,
                                      minlength=len(ERROR_BIN_EDGES) - 1)
                counts = self.counts[name]
                for i, c in enumerate(hist.tolist()):
                    counts[i] += int(c)
                self.sums[name] += float(v.sum())
                self.n[name] += int(v.numel())

    def payload(self) -> Dict:
        return {
            "edges": ERROR_BIN_EDGES,
            "counts": self.counts,
            "mean_abs_error": {k: (self.sums[k] / self.n[k] if self.n[k] else 0.0)
                               for k in self.KEYS},
            "n": dict(self.n),
        }


@torch.no_grad()
def module_grad_norms(model) -> Dict[str, float]:
    """Gradient norm per branch, so a dead one is visible rather than inferred.

    A global norm hides which part produced it. With two decoders, a frontend
    run twice and a gated channel path that starts at zero, "the gradient is
    fine" is not a statement anyone should have to take on the aggregate.
    """
    groups: Dict[str, List[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        head = name.split(".")[0]
        if head in ("wavelet_frontends", "patch_embed_by_rate",
                    "reconstruction_heads", "raw_reconstruction_heads"):
            # Keep the route or rate: E128_512's frontend and E19_256's are
            # trained on different steps and averaging them says nothing.
            head = ".".join(name.split(".")[:2])
        groups.setdefault(head, []).append(p.grad.detach().float().reshape(-1))
    out = {}
    for k, v in groups.items():
        out[f"gradnorm/{k}"] = float(torch.cat(v).norm())
    return out


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
        # The objective's weights. Read from an `objective:` block, falling
        # back to `train:` keys so an existing config keeps working.
        ocfg = cfg.get("objective", {}) or {}
        def _obj(name, train_name, default):
            if name in ocfg:
                return ocfg[name]
            return tcfg.get(train_name, default)
        self.spec_weight = float(_obj("spec_weight", "spec_recon_weight", 1.0))
        self.raw_weight = float(_obj("raw_weight", "raw_recon_weight", 0.25))
        self.raw_beta = float(_obj("raw_beta", "raw_smooth_l1_beta", 0.5))
        self.fold_kl = float(_obj("fold_kl", "fold_kl", 1e-3))
        self.mask_before_frontend = bool(
            _obj("mask_before_frontend", "mask_before_frontend", True))
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
            mask_before_frontend=self.mask_before_frontend,
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
                  f"   spec decoder {report[f'reconstruction_head.{rate}']:,}"
                  f"   raw decoder {report[f'raw_reconstruction_head.{rate}']:,}")
        print(f"  downstream encoder      {report['downstream_encoder']:,}"
              f"   (+{report['pretraining_only']:,} pretraining-only:"
              f" both decoders and the mask token)")
        print(f"  objective: spec {self.spec_weight:g} x MSE"
              f"  + raw {self.raw_weight:g} x SmoothL1(beta={self.raw_beta:g})"
              f"  + fold_kl {self.fold_kl:g} x KL")
        print(f"  masking: {'signal before the frontend' if self.mask_before_frontend else 'tokens only, after the frontend'}"
              f"   targets: detached")
        if "channel_encoder" in report:
            print(f"  channel encoder (C1)    {report['channel_encoder']:,}"
                  f"  + proj {report['channel_to_token']:,}")
        print("  routes:")
        for rid, r in ROUTES.items():
            print(f"    {r.describe()}")
        counts = self.train_index.window_counts()
        total_w = max(1, sum(counts.values()))
        policy = getattr(self.schedule, "weight_policy", "explicit")
        print(f"  mixture policy: {policy}")
        print("    dataset          corpus%   step%  window%   passes/epoch")
        for d in self.schedule.dataset_ids:
            b = self.schedule.batch_by_route[PRETRAIN_DATASETS[d].route_id]
            seen = (mixture["by_step"][d] * self.schedule.steps_per_epoch
                    * b * self.schedule.num_replicas)
            # passes/epoch is the number that says whether a small corpus is
            # being read a hundred times to fill a quota. Under `proportional`
            # every row here is 1.0.
            passes = seen / max(1, counts.get(d, 0))
            print(f"    {d:<14s} {counts.get(d,0)/total_w*100:7.2f}% "
                  f"{mixture['by_step'][d]*100:6.2f}% "
                  f"{mixture['by_window'][d]*100:7.2f}%   {passes:6.2f}x")
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
        # STRICT, deliberately. A checkpoint from the single-decoder objective
        # has no raw_reconstruction_heads, and loading it loosely would resume
        # a run whose optimizer state, scheduler position and step count all
        # belong to a different objective -- reported as a continuation of it.
        # Resuming and initialising from are not the same operation.
        try:
            self.raw_model.load_state_dict(ck["model"])
        except RuntimeError as exc:
            missing = [k for k in self.raw_model.state_dict()
                       if k not in ck["model"]]
            raw_missing = [k for k in missing
                           if k.startswith("raw_reconstruction_heads.")]
            if raw_missing:
                raise SystemExit(
                    f"{path} has no raw reconstruction head, so it was written "
                    f"by the single-decoder objective and is not a resume of "
                    f"this one.\n\n"
                    f"  Its optimizer state, scheduler position and step count "
                    f"belong to a different loss, and continuing from them "
                    f"would be reported as one run when it is two.\n\n"
                    f"  To carry the representation over instead, export it and "
                    f"start a new run:\n"
                    f"    python scripts/export_eeg_pretrained_encoder.py "
                    f"--checkpoint {path} ...\n\n"
                    f"  ({len(raw_missing)} missing key(s), e.g. "
                    f"{raw_missing[0]})") from exc
            raise
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
        route_seconds: Dict[str, float] = {}
        route_windows: Dict[str, int] = {}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for i, batch in enumerate(self.loader):
            x = batch["x"].to(self.device, non_blocking=True)
            meta = {k: v.to(self.device) for k, v in batch["channel_meta"].items()}
            is_accum = (i + 1) % self.grad_accum != 0
            step_t0 = time.time()

            ctx = (torch.autocast(self.device.type, dtype=self.amp_dtype)
                   if self.use_amp else _nullcontext())
            with ctx:
                out = self.model(x, batch["route_id"], channel_meta=meta,
                                 mask_ratio=self.mask_ratio)
                loss, metrics = masked_reconstruction_loss(
                    out, spec_weight=self.spec_weight,
                    raw_weight=self.raw_weight, fold_kl=self.fold_kl,
                    raw_beta=self.raw_beta)

            self.scaler.scale(loss / self.grad_accum).backward()
            grad_norm = float("nan")
            branch_norms: Dict[str, float] = {}
            want_log = self.is_main and (i % 50 == 0 or i + 1 == len(self.loader))
            if not is_accum:
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(), self.clip_grad))
                # HERE, not after the step: zero_grad(set_to_none=True) is two
                # lines below and every gradient is None by then. Only on the
                # logged steps, because walking every parameter is nothing at
                # every fiftieth step and not nothing at every one.
                if want_log:
                    branch_norms = module_grad_norms(self.raw_model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            windows += x.shape[0]
            # Per route, so the mixture figure can show what each one COST as
            # well as what share it got. E128_512 draws 12 windows to
            # E19_256's 64 and each is 6.7x the tokens; share of steps says
            # nothing about where the hours went.
            route_seconds[batch["route_id"]] = route_seconds.get(
                batch["route_id"], 0.0) + (time.time() - step_t0)
            route_windows[batch["route_id"]] = route_windows.get(
                batch["route_id"], 0) + x.shape[0]

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

            if want_log:
                self._append_jsonl("metrics_step.jsonl", {
                    "epoch": self.epoch, "step": self.global_step,
                    "route_id": batch["route_id"],
                    "dataset_id": batch["dataset_id"],
                    **metrics, **branch_norms})
            self.schedule.start_step = i + 1
            if self.max_steps and self.global_step >= self.max_steps:
                break

        out = acc.mean()
        elapsed = max(1e-9, time.time() - t0)
        out["throughput_windows_per_s"] = windows / elapsed
        for rid, secs in route_seconds.items():
            out[f"route_seconds/{rid}"] = secs
            out[f"route_share_of_time/{rid}"] = secs / elapsed
        for rid, n in route_windows.items():
            out[f"route_windows/{rid}"] = float(n)
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
        hist = ErrorHistogram()
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
                _, metrics = masked_reconstruction_loss(
                    out, spec_weight=self.spec_weight,
                    raw_weight=self.raw_weight, fold_kl=self.fold_kl,
                    raw_beta=self.raw_beta)
            hist.add(out)
            # The control the loss cannot provide: error on the tokens the
            # model could SEE. A masked-reconstruction objective can be
            # satisfied by a model that has learned to copy its input, and the
            # way that shows is a near-zero visible error beside an unimproved
            # masked one. Nothing in the loss looks at visible tokens.
            with torch.no_grad():
                sel = out["mask"].unsqueeze(-1).expand_as(out["pred_spec"])
                vis = ~sel
                if out.get("valid_tokens") is not None:
                    vis = vis & out["valid_tokens"].unsqueeze(-1).expand_as(sel)
                if bool(vis.any()):
                    for tag, pk, tk in (("spec", "pred_spec", "target_spec"),
                                        ("raw", "pred_raw", "target_raw")):
                        e = (out[pk].float() - out[tk].float()).abs()[vis]
                        metrics[f"visible_{tag}_mae"] = float(e.mean())
            acc.add(metrics, batch["route_id"], batch["dataset_id"])
        it.close()
        means = acc.mean()
        if self.is_main:
            payload = hist.payload()
            payload["epoch"] = self.epoch
            payload["global_step"] = self.global_step
            with open(os.path.join(self.out_dir, "error_histogram.json"),
                      "w") as f:
                json.dump(payload, f)
            self._append_jsonl("error_histogram_by_epoch.jsonl", payload)
        for k in ("loss_total", "loss_masked_spec_mse",
                  "loss_masked_raw_smoothl1", "loss_masked_mse"):
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
