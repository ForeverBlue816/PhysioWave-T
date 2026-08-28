"""Supervised fine-tuning for the extension encoder (WAST + TARE + backbone).

The legacy ``finetune.py`` at the repository root builds
``model.BERTWaveletTransformer`` and cannot construct this model at all, so
until now the only supervised numbers in the project were the legacy
architecture's. This is the entry point that puts the token-efficient path on
the same downstream tasks, which is what makes the two comparable.

Data comes straight from the HDF5 that the dataset converters write
(``EMG/db5_finetune.py``, ``EMG/db6_finetune.py``, ``EEG/tuab_finetune.py``):

    data   (N, C, T) float32
    label  (N,)      int64, contiguous from 0

That is deliberately not the manifest path ``pretrain_main`` uses. Pretraining
mixes corpora and needs the registry to know what each one is; fine-tuning is
handed three files and a label column, and routing that through a manifest adds
a registry entry per downstream task without answering a question.

Channel metadata
----------------
TARE wants 3-D electrode coordinates. A forearm sEMG ring has no standard ones,
so the coordinates are left as the all-zero "unknown" row and TARE falls back to
its name branch, which for ``ch00..ch13`` amounts to a learned per-channel
embedding. That is weaker than TARE on a scalp montage and it is the honest
default: inventing cylinder coordinates for the ring would be a prior of ours,
not a measurement, and belongs in an ablation rather than in the main result.
``--channel-names`` and ``--channel-xyz`` supply real metadata when there is any.

Usage
-----
    torchrun --standalone --nproc_per_node=4 -m physiowave.train.finetune_main \\
        --config pretrain/semg \\
        --data-dir $SCRATCH/bio/emg/db6 --num-classes 8 \\
        --output-dir $FAST/yanlchen/runs/db6_wast
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..channels.tare import ChannelMeta
from ..config import load_config
from ..models.build import build_model
from ..models.checkpoint import load_checkpoint, save_checkpoint
from .utils import (
    AverageMeter,
    autocast_ctx,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    make_grad_scaler,
    pick_device,
    resolve_precision,
    set_seed,
    setup_distributed,
)

logger = logging.getLogger(__name__)

SELECTION_METRICS = ("loss", "acc", "balanced_acc", "kappa", "weighted_f1", "auroc")


PROGRESS_MODES = ("auto", "bar", "log", "none")


def resolve_progress(mode: str) -> str:
    """Turn ``auto`` into the mode that suits the stream we are writing to.

    torchrun leaves the workers' stdout and stderr alone unless ``--redirects``
    is passed (it defaults to ``0``), so a worker's ``isatty()`` really does
    report whether a human is watching: True under an interactive shell, False
    under sbatch, ``> log``, or ``tee``. That is the right thing to branch on,
    because a tqdm bar redraws with carriage returns and a log file records the
    whole run as one enormous line.

    Redirected runs get ``log`` rather than silence: the complaint a bar answers
    is "how far along is this", and a periodic line carrying the percentage and
    an ETA answers it in a form that survives being written to a file.
    """
    if mode != "auto":
        return mode
    return "bar" if sys.stderr.isatty() else "log"


def progress(iterable, desc: str, mode: str, is_main: bool):
    """Wrap a loader in a tqdm bar on rank 0, or leave it alone."""
    if not is_main or mode != "bar":
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, ncols=110, leave=False, mininterval=0.5)


def set_postfix(bar, **kw) -> None:
    if hasattr(bar, "set_postfix"):
        bar.set_postfix(**kw, refresh=False)


def fmt_eta(seconds: float) -> str:
    """``h:mm:ss`` for anything an epoch loop is likely to produce."""
    if not (seconds == seconds) or seconds < 0 or seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


class LabelledWindows(Dataset):
    """The (N, C, T) + (N,) HDF5 the converters write, held in memory.

    Read once into RAM rather than kept open per worker: the downstream files
    are a few GB, the windows are touched in random order every epoch, and an
    open h5py handle is not safe to share across forked workers.
    """

    def __init__(self, path: str) -> None:
        import h5py

        with h5py.File(path, "r") as f:
            self.data = np.asarray(f["data"][:], dtype=np.float32)
            self.labels = np.asarray(f["label"][:], dtype=np.int64)
        if len(self.data) != len(self.labels):
            raise ValueError(f"{path}: {len(self.data)} windows but {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.data[i]), int(self.labels[i])

    @property
    def num_channels(self) -> int:
        return self.data.shape[1]

    @property
    def window(self) -> int:
        return self.data.shape[2]

    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.labels, minlength=num_classes)


def build_channel_meta(num_channels: int, names: Optional[List[str]],
                       xyz_path: Optional[str], device: torch.device) -> ChannelMeta:
    """Channel metadata for TARE; unknown coordinates are the all-zero row."""
    if names is None:
        names = [f"ch{i:02d}" for i in range(num_channels)]
    if len(names) != num_channels:
        raise ValueError(f"{len(names)} channel names for {num_channels} channels")
    if xyz_path:
        xyz = torch.as_tensor(np.load(xyz_path), dtype=torch.float32)
        if xyz.shape != (num_channels, 3):
            raise ValueError(f"channel_xyz must be ({num_channels}, 3), got {tuple(xyz.shape)}")
    else:
        xyz = torch.zeros(num_channels, 3, dtype=torch.float32)
    return ChannelMeta(channel_names=names, channel_xyz=xyz.to(device))


def forward_logits(model: nn.Module, x: torch.Tensor, meta: ChannelMeta) -> torch.Tensor:
    out = model(x, meta)
    if "logits" not in out:
        raise RuntimeError(
            "the encoder produced no logits; model.num_classes must be set "
            "(pass --num-classes, which this entry point does)")
    return out["logits"]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, meta: ChannelMeta, device: torch.device,
             amp_dtype: torch.dtype, criterion: nn.Module, num_classes: int,
             desc: str = "eval", progress_mode: str = "none",
             is_main: bool = False) -> Dict[str, float]:
    """Metrics over the whole split, gathered across ranks."""
    from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                                 roc_auc_score)

    model.eval()
    loss_meter = AverageMeter()
    probs_all, target_all = [], []
    for x, y in progress(loader, desc, progress_mode, is_main):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device, amp_dtype):
            logits = forward_logits(model, x, meta)
            loss = criterion(logits.float(), y)
        loss_meter.update(loss.item(), y.numel())
        probs_all.append(F.softmax(logits.float(), dim=-1).cpu())
        target_all.append(y.cpu())

    probs = torch.cat(probs_all) if probs_all else torch.zeros(0, num_classes)
    target = torch.cat(target_all) if target_all else torch.zeros(0, dtype=torch.long)

    if dist.is_available() and dist.is_initialized():
        # Every rank sees a disjoint shard; the metrics below are not averages
        # of per-shard metrics, so the predictions have to be gathered before
        # anything is computed. Balanced accuracy in particular cannot be
        # recovered from per-rank values.
        sizes = [torch.zeros(1, dtype=torch.long, device=device)
                 for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, torch.tensor([len(target)], dtype=torch.long, device=device))
        sizes = [int(s.item()) for s in sizes]
        biggest = max(sizes)
        p_pad = torch.zeros(biggest, num_classes, device=device)
        t_pad = torch.zeros(biggest, dtype=torch.long, device=device)
        p_pad[:len(probs)] = probs.to(device)
        t_pad[:len(target)] = target.to(device)
        p_gather = [torch.zeros_like(p_pad) for _ in sizes]
        t_gather = [torch.zeros_like(t_pad) for _ in sizes]
        dist.all_gather(p_gather, p_pad)
        dist.all_gather(t_gather, t_pad)
        probs = torch.cat([p[:n] for p, n in zip(p_gather, sizes)]).cpu()
        target = torch.cat([t[:n] for t, n in zip(t_gather, sizes)]).cpu()

    y_true = target.numpy()
    y_prob = probs.numpy()
    y_pred = y_prob.argmax(1)
    present = np.unique(y_true)

    metrics = {
        "loss": loss_meter.avg,
        "acc": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                       if len(y_true) else 0.0,
    }
    # AUROC is one-vs-rest over the classes that actually occur. Dropping the
    # absent columns leaves rows that no longer sum to one, which sklearn
    # rejects, so they are renormalised -- without this a single unrepresented
    # class turns the whole metric into nan, which on a 53-class split is the
    # common case rather than the corner one.
    try:
        if len(present) >= 2:
            sub = y_prob[:, present]
            sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
            if len(present) == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, sub[:, 1],
                                                       labels=present))
            else:
                metrics["auroc"] = float(roc_auc_score(
                    y_true, sub, multi_class="ovr", average="macro", labels=present))
        else:
            metrics["auroc"] = float("nan")
    except ValueError as exc:                      # degenerate split
        logger.warning("AUROC unavailable: %s", exc)
        metrics["auroc"] = float("nan")
    if len(present) < num_classes:
        logger.warning("%d of %d classes absent from this split; AUROC is over the "
                       "%d present", num_classes - len(present), num_classes, len(present))
    return metrics


def train_one_epoch(model, loader, meta, device, amp_dtype, criterion, optimizer, scheduler,
                    scaler, grad_clip: float, epoch: int, log_every: int,
                    is_main: bool, progress_mode: str = "auto") -> Dict[str, float]:
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    bar = progress(loader, f"train {epoch}", progress_mode, is_main)
    total = len(loader)
    started = time.monotonic()
    for step, (x, y) in enumerate(bar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device, amp_dtype):
            logits = forward_logits(model, x, meta)
            loss = criterion(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), y.numel())
        acc_meter.update(float((logits.argmax(-1) == y).float().mean().item()), y.numel())
        set_postfix(bar, loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if is_main and progress_mode == "log" and log_every and step % log_every == 0:
            # The percentage and ETA are the part a bar would have given; they
            # are what makes a periodic line answer "how much longer" in a log
            # file, where a redrawing bar cannot go.
            done = step + 1
            eta = (time.monotonic() - started) / done * (total - done)
            logger.info("epoch %d step %d/%d (%3.0f%%) loss %.4f acc %.4f lr %.2e eta %s",
                        epoch, step, total, 100.0 * done / max(total, 1),
                        loss_meter.avg, acc_meter.avg,
                        optimizer.param_groups[0]["lr"], fmt_eta(eta))
    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pretrain/semg",
                   help="encoder config; the model block is reused, the pretraining "
                        "objectives are not")
    p.add_argument("--data-dir", help="directory holding train.h5 / val.h5 / test.h5")
    p.add_argument("--train-file")
    p.add_argument("--val-file")
    p.add_argument("--test-file")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pretrained", help="extension-format checkpoint to start from")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="train the classification head only (linear probe)")
    p.add_argument("--channel-names", nargs="+", default=None)
    p.add_argument("--channel-xyz", default=None,
                   help=".npy of shape (C, 3); without it TARE sees unknown coordinates")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"],
                   help="override the automatic pick; useful when the default "
                        "accelerator is the thing under suspicion")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50,
                   help="periodic in-epoch log lines; 0 disables. Ignored while a "
                        "progress bar is showing the same numbers.")
    p.add_argument("--progress", default="auto", choices=list(PROGRESS_MODES),
                   help="'auto' shows a tqdm bar on a terminal and periodic log lines "
                        "with a percentage and an ETA when the stream is redirected, "
                        "where a redrawing bar would become one enormous line")
    p.add_argument("--select-by", default="balanced_acc", choices=SELECTION_METRICS,
                   help="validation metric that decides best.pth; balanced accuracy by "
                        "default because downstream label sets are rarely balanced")
    p.add_argument("--patience", type=int, default=0,
                   help="stop when --select-by has not improved for this many epochs")
    p.add_argument("--min-delta", type=float, default=0.0)
    # 'extend' rather than the default 'store': the launch scripts pass a --set of
    # their own and EXTRA can carry another, and with 'store' the second silently
    # replaces the first -- an override that looks applied and is not.
    # default=None because argparse mutates a shared list default in place.
    p.add_argument("--set", nargs="*", action="extend", default=None,
                   help="config overrides, key=value; repeatable")
    args = p.parse_args(argv)
    args.set = args.set or []
    return args


def resolve_files(args) -> Tuple[str, Optional[str], Optional[str]]:
    if args.data_dir:
        j = lambda n: os.path.join(args.data_dir, f"{n}.h5")  # noqa: E731
        train = args.train_file or j("train")
        val = args.val_file or (j("val") if os.path.exists(j("val")) else None)
        test = args.test_file or (j("test") if os.path.exists(j("test")) else None)
    else:
        train, val, test = args.train_file, args.val_file, args.test_file
    if not train or not os.path.exists(train):
        raise SystemExit(f"ERROR: training file not found: {train}")
    return train, val, test


def main(argv=None) -> int:
    args = parse_args(argv)
    info = setup_distributed()
    device = torch.device(args.device) if args.device else pick_device()
    set_seed(args.seed + info.rank)
    logging.basicConfig(level=logging.INFO if info.is_main else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.progress = resolve_progress(args.progress)

    train_path, val_path, test_path = resolve_files(args)
    cfg = load_config(args.config, args.set)
    model_cfg = dict(cfg.get("model", {}) or {})
    model_cfg["num_classes"] = args.num_classes
    cfg["model"] = model_cfg

    if info.is_main:
        # Say this before the read, not after. LabelledWindows pulls the whole
        # file into RAM and every rank holds its own copy, so a few GB of
        # densely-strided windows is minutes of complete silence otherwise --
        # indistinguishable from a hung job, which is how an allocation gets
        # killed and rerun for no reason.
        sizes = " ".join(
            f"{os.path.basename(p)} {os.path.getsize(p) / 2**30:.2f} GiB"
            for p in (train_path, val_path, test_path)
            if p and os.path.exists(p))
        logger.info("loading %s into memory on each of %d rank(s): %s",
                    os.path.dirname(train_path) or ".", info.world_size, sizes)

    train_set = LabelledWindows(train_path)
    C, T = train_set.num_channels, train_set.window
    if info.is_main:
        counts = train_set.class_counts(args.num_classes)
        logger.info("train %d windows, %d channels, %d samples/window", len(train_set), C, T)
        logger.info("class counts: %s", counts.tolist())
        if (counts == 0).any():
            logger.warning("classes %s have no training windows",
                           np.flatnonzero(counts == 0).tolist())

    model = build_model(cfg).to(device)
    if args.pretrained:
        # strict=False on purpose: the classification head does not exist in a
        # pretraining checkpoint, and an encoder trained on a different channel
        # count keeps everything except the per-channel pieces.
        payload = load_checkpoint(args.pretrained, model=model, strict=False,
                                  restore_rng=False)
        if info.is_main:
            logger.info("loaded encoder from %s (epoch %s)", args.pretrained,
                        payload.get("epoch", "?"))
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
    if info.distributed:
        model = DDP(model, device_ids=[info.local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=True)
    core = model.module if hasattr(model, "module") else model
    if info.is_main:
        # State the block that is actually running. The components are dataclass
        # defaults rather than YAML entries, so reading the configs does not tell
        # you which ones are on, and an ablation that silently kept the defaults
        # would be indistinguishable from one that took effect.
        bb = getattr(getattr(core, "cfg", None), "backbone", None)
        if bb is not None:
            logger.info("backbone: depth=%d dim=%d | rope=%s norm=%s ffn=%s qk_norm=%s",
                        bb.depth, bb.embed_dim, bb.use_rope, bb.norm, bb.ffn, bb.qk_norm)
        total = sum(p.numel() for p in core.parameters())
        trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
        # Raw counts as well as millions: a linear probe leaves a couple of
        # thousand parameters trainable, which "0.00 M" reads as "none at all".
        logger.info("model %.2f M parameters (%s trainable, %.1f%%)",
                    total / 1e6, f"{trainable:,}", 100.0 * trainable / max(total, 1))

    meta = build_channel_meta(C, args.channel_names, args.channel_xyz, device)

    def loader_for(dataset, shuffle):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if info.distributed else None
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                          shuffle=(sampler is None and shuffle),
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                          drop_last=False), sampler

    train_loader, train_sampler = loader_for(train_set, True)
    val_loader = test_loader = None
    if val_path and os.path.exists(val_path):
        val_loader, _ = loader_for(LabelledWindows(val_path), False)
    if test_path and os.path.exists(test_path):
        test_loader, _ = loader_for(LabelledWindows(test_path), False)

    precision, amp_dtype = resolve_precision(args.precision, device)
    scaler = make_grad_scaler(device.type, precision == "fp16")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(core, args.lr, args.weight_decay)
    steps = max(len(train_loader), 1)
    warmup = args.warmup_epochs
    if warmup >= args.epochs:
        warmup = max(args.epochs // 10, 0)
        if info.is_main:
            logger.warning("warmup_epochs >= epochs; using %d so the cosine decay happens",
                           warmup)
    scheduler = build_scheduler(optimizer, warmup * steps, args.epochs * steps)

    os.makedirs(args.output_dir, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_score, best_epoch, stale = float("-inf"), -1, 0

    run_started = time.monotonic()
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr = train_one_epoch(model, train_loader, meta, device, amp_dtype, criterion,
                             optimizer, scheduler, scaler, args.grad_clip, epoch,
                             args.log_every, info.is_main, args.progress)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}}

        if val_loader is not None:
            va = evaluate(model, val_loader, meta, device, amp_dtype, criterion,
                          args.num_classes, f"val {epoch}", args.progress, info.is_main)
            row.update({f"val_{k}": v for k, v in va.items()})
            score = -va["loss"] if args.select_by == "loss" else va[args.select_by]
            improved = score > best_score + args.min_delta
            if improved:
                best_score, best_epoch, stale = score, epoch, 0
                if info.is_main:
                    save_checkpoint(os.path.join(args.output_dir, "best.pth"), core,
                                    optimizer, epoch=epoch, metrics=va,
                                    config={"args": vars(args), "config": args.config})
            else:
                stale += 1
            if info.is_main:
                done = epoch + 1
                eta = (time.monotonic() - run_started) / done * (args.epochs - done)
                logger.info("epoch %d/%d train_loss %.4f val_loss %.4f val_acc %.4f "
                            "val_bal %.4f val_auroc %.4f eta %s%s",
                            epoch, args.epochs - 1, tr["loss"], va["loss"],
                            va["acc"], va["balanced_acc"], va["auroc"], fmt_eta(eta),
                            "  *" if improved else "")
        history.append(row)
        if info.is_main:
            with open(os.path.join(args.output_dir, "history.json"), "w") as fh:
                json.dump(history, fh, indent=2)

        # Early stop: rank 0 decides, every rank obeys. A rank that leaves the
        # loop alone strands the others in the next all-reduce.
        if args.patience > 0 and val_loader is not None:
            flag = torch.zeros(1, device=device)
            if info.is_main and stale >= args.patience:
                logger.info("early stop at epoch %d: val %s flat for %d epochs (best %d)",
                            epoch, args.select_by, args.patience, best_epoch)
                flag[0] = 1.0
            if info.distributed:
                dist.broadcast(flag, src=0)
            if flag.item() > 0:
                break

    results: Dict[str, Any] = {"best_epoch": best_epoch, "select_by": args.select_by}
    if test_loader is not None:
        best = os.path.join(args.output_dir, "best.pth")
        if os.path.exists(best):
            load_checkpoint(best, model=core, strict=False, restore_rng=False)
            if info.distributed:
                for p in core.parameters():
                    dist.broadcast(p.data, src=0)
        te = evaluate(model, test_loader, meta, device, amp_dtype, criterion,
                      args.num_classes, "test", args.progress, info.is_main)
        results["test"] = te
        if info.is_main:
            logger.info("TEST acc %.4f bal %.4f kappa %.4f wf1 %.4f auroc %.4f",
                        te["acc"], te["balanced_acc"], te["kappa"], te["weighted_f1"],
                        te["auroc"])
    if info.is_main:
        with open(os.path.join(args.output_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("outputs in %s", args.output_dir)

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
