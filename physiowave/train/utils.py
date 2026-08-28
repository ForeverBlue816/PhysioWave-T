"""Training utilities: distributed setup, precision selection, meters, profiling."""

from __future__ import annotations

import contextlib
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG and, optionally, ask for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# --------------------------------------------------------------------------- #
# Distributed
# --------------------------------------------------------------------------- #
@dataclass
class DistInfo:
    """Resolved distributed context."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    distributed: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    """Initialise DDP from torchrun's environment, falling back to single process.

    Works unchanged for ``torchrun --nproc_per_node=N`` locally and for the
    multi-node Slurm launch in ``scripts/slurm/``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return DistInfo(rank, local_rank, world_size, device, True)

    device = pick_device()
    return DistInfo(0, 0, 1, device, False)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: float, info: DistInfo) -> float:
    if not info.distributed:
        return value
    t = torch.tensor([value], device=info.device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / info.world_size).item())


# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #
def resolve_precision(requested: str, device: torch.device) -> Tuple[str, torch.dtype]:
    """Pick the best available precision, falling back loudly.

    ``bf16`` needs Ampere or newer; on older GPUs it silently costs performance or
    is unsupported, so the request degrades to fp16 and then to fp32, and the
    substitution is logged.
    """
    req = (requested or "fp32").lower()
    if device.type == "cuda":
        if req == "bf16":
            if torch.cuda.is_bf16_supported():
                return "bf16", torch.bfloat16
            logger.warning("bf16 requested but unsupported on this GPU; using fp16.")
            return "fp16", torch.float16
        if req == "fp16":
            return "fp16", torch.float16
        return "fp32", torch.float32
    if req in ("bf16", "fp16"):
        logger.warning("%s requested on %s; autocast is disabled and fp32 is used.",
                       req, device.type)
    return "fp32", torch.float32


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    """Autocast context that is a no-op on fp32 / CPU / MPS."""
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def make_grad_scaler(device_type: str, enabled: bool):
    """A GradScaler that works on every torch we run on.

    The device-generic ``torch.amp.GradScaler`` only exists from torch 2.4.
    Leonardo's ``cineca-ai/4.3.0`` ships 2.2, where the scaler lives at
    ``torch.cuda.amp.GradScaler`` and takes no device argument -- so calling
    the new spelling there raises ``AttributeError`` before the first batch.
    The scaler is only ever enabled for fp16 on CUDA, which is the one device
    the old class supports, so the fallback loses nothing.
    """
    enabled = bool(enabled) and device_type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- #
# Meters and logging
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Running mean of a scalar."""

    def __init__(self) -> None:
        self.sum, self.count = 0.0, 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class MetricLogger:
    """Accumulates a dict of scalars and reports their means."""

    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, values: Dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self.meters.setdefault(k, AverageMeter()).update(v, n)

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self) -> None:
        self.meters.clear()


def check_finite(loss: torch.Tensor, step: int, extra: Optional[Dict[str, float]] = None) -> bool:
    """Return False (and log) if the loss is NaN or Inf."""
    if torch.isfinite(loss).all():
        return True
    logger.error("Non-finite loss at step %d: %s; extra=%s", step, loss.item(), extra or {})
    return False


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def peak_memory_mb(device: torch.device) -> float:
    """Peak allocated memory in MB (0.0 on backends that do not report it)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            return torch.mps.driver_allocated_memory() / (1024 ** 2)
        except Exception:
            return 0.0
    return 0.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def estimate_flops(model: torch.nn.Module, inputs: Tuple[Any, ...],
                   kwargs: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Forward FLOPs via ``torch.utils.flop_counter``; ``None`` if unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    try:
        counter = FlopCounterMode(display=False)
        # FlopCounterMode needs autograd active; benchmarks usually run under
        # no_grad, so grad is re-enabled just for the measurement pass.
        with torch.enable_grad(), counter:
            model(*inputs, **(kwargs or {}))
        return float(counter.get_total_flops())
    except Exception as exc:
        logger.warning("FLOP counting failed: %s", exc)
        return None


class Throughput:
    """Samples/second and tokens/second over a window of steps."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.samples = 0
        self.tokens = 0

    def update(self, samples: int, tokens: int = 0) -> None:
        self.samples += samples
        self.tokens += tokens

    def rates(self) -> Dict[str, float]:
        dt = max(time.time() - self.t0, 1e-9)
        return {"samples_per_sec": self.samples / dt, "tokens_per_sec": self.tokens / dt}

    def reset(self) -> None:
        self.__init__()


class TensorBoardWriter:
    """Thin TensorBoard wrapper that degrades to a no-op when unavailable."""

    def __init__(self, log_dir: Optional[str], enabled: bool = True) -> None:
        self.writer = None
        if not (enabled and log_dir):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except Exception as exc:
            logger.warning("TensorBoard unavailable (%s); continuing without it.", exc)

    def add_scalars(self, values: Dict[str, float], step: int, prefix: str = "") -> None:
        if self.writer is None:
            return
        for k, v in values.items():
            self.writer.add_scalar(f"{prefix}{k}" if prefix else k, v, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with weight decay disabled on norms, biases, embeddings and 1-D tensors.

    Decaying a LayerNorm gain or a learnable wavelet filter towards zero is not a
    regularisation you want: it would pull the filters away from a valid wavelet
    basis, so those tensors are put in the no-decay group.

    Embeddings and learned query/mask tokens are excluded for a sharper reason.
    They are identity, not weights: the backbone's slot embedding is what tells
    one channel from another, and when no channel metadata encoder is present it
    is the *only* thing that does. Decay shrinks it toward the one value that
    makes every channel identical -- regularising away the signal rather than the
    capacity to overfit it. This is the usual ViT convention (pos_embed and
    cls_token are no-decay) and it matters more here than it does there.
    """
    embedding_params = {
        id(p)
        for m in model.modules() if isinstance(m, torch.nn.Embedding)
        for p in m.parameters(recurse=False)
    }
    POSITIONAL = ("embed", "_query", "_token", "anchor")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lowered = name.lower()
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in lowered \
                or "dec_lo" in name or "dec_hi" in name \
                or id(p) in embedding_params \
                or any(tag in lowered for tag in POSITIONAL):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    """Linear warmup followed by cosine decay to ``min_ratio * lr``."""
    import math

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
