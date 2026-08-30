# -*- coding: utf-8 -*-
"""The protocol knobs that decide whether a downstream number means anything.

Three of them, each of which used to be able to go wrong silently:

* the config's ``train:`` block. Until ``resolve_hparams`` existed nothing in
  ``finetune_main`` read it, so ``batch_size: 32`` in ``eeg_c1_sleep.yaml``
  described a run at 64 and said so in a config a reader would quote.
* a frozen encoder's dropout. ``model.train()`` puts the whole module in train
  mode, so a linear probe was reading a fresh dropout sample of the
  representation every step rather than the representation.
* pooling. ``mean`` averages over electrodes AND time, which for an ERP is the
  signal divided by the number of patches.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from physiowave.eeg_c1.downstream import EEGC1Downstream
from physiowave.eeg_c1.routes import ROUTES
from physiowave.train.finetune_main import HPARAM_FALLBACKS, resolve_hparams


SLOTS = list(ROUTES["E64_256"].slots)


def build(pool="mean", freeze=False, num_classes=2, **kw):
    return EEGC1Downstream(
        in_channels=8, window_samples=512, sampling_rate=256, patch_samples=128,
        num_classes=num_classes, channel_names=SLOTS[:8], embed_dim=32, depth=1,
        num_heads=2, channel_embed_dim=8, pool=pool, freeze_encoder=freeze, **kw)


def namespace(**kw):
    base = {k: None for k in HPARAM_FALLBACKS}
    base["config"] = "finetune/eeg_c1_p300"
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# where a hyper-parameter comes from
# --------------------------------------------------------------------------- #
def test_the_config_train_block_is_read():
    args = namespace()
    rows = resolve_hparams(args, {"train": {"lr": 5e-4, "epochs": 100,
                                            "select_by": "auroc"}})
    assert args.lr == 5e-4 and args.epochs == 100 and args.select_by == "auroc"
    assert dict((k, s) for k, _, s in rows)["lr"].startswith("config:")


def test_the_command_line_beats_the_config():
    args = namespace(lr=2e-4)
    resolve_hparams(args, {"train": {"lr": 5e-4}})
    assert args.lr == 2e-4


def test_a_value_in_neither_falls_back_and_says_so():
    args = namespace()
    rows = resolve_hparams(args, {})
    assert args.weight_decay == HPARAM_FALLBACKS["weight_decay"]
    assert dict((k, s) for k, _, s in rows)["weight_decay"] == "builtin"


def test_an_unknown_selection_metric_is_refused():
    with pytest.raises(SystemExit):
        resolve_hparams(namespace(), {"train": {"select_by": "f_beta"}})


# --------------------------------------------------------------------------- #
# a frozen encoder is a frozen encoder
# --------------------------------------------------------------------------- #
def test_a_frozen_encoder_stays_in_eval_inside_model_train():
    model = build(freeze=True, dropout=0.5, head_dropout=0.5)
    model.train()
    assert not model.shared_transformer.training
    assert not model.wavelet_frontend.training
    # the head is what is being trained, and its dropout has to be live
    assert model.head_drop.training and model.head.training


def test_an_unfrozen_encoder_still_trains():
    model = build(freeze=False, dropout=0.5)
    model.train()
    assert model.shared_transformer.training


def test_a_frozen_probe_gives_the_same_features_twice():
    """The point of the eval override: a probe reads a fixed representation."""
    model = build(freeze=True, dropout=0.5, head_dropout=0.0)
    model.train()
    x = torch.randn(2, 8, 512)
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    with torch.no_grad():
        a = model(x, meta)["features"]
        b = model(x, meta)["features"]
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def test_time_pooling_keeps_the_time_axis():
    model = build(pool="time", probe_dim=16)
    x = torch.randn(2, 8, 512)
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    out = model(x, meta)
    # 512 / 128 = 4 patches, each projected to probe_dim and kept
    assert out["features"].shape == (2, 4 * 16)
    assert out["logits"].shape == (2, 2)


def test_mean_pooling_is_unchanged():
    model = build(pool="mean")
    x = torch.randn(2, 8, 512)
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    out = model(x, meta)
    assert out["features"].shape == (2, 32)


def test_time_pooling_is_sensitive_to_when_the_deflection_happens():
    """A mean over time cannot tell these apart; that is the whole objection."""
    model = build(pool="time", probe_dim=16).eval()
    meta = {"channel_ids": torch.arange(1, 9), "valid_channel_mask": torch.ones(8, dtype=torch.bool)}
    base = torch.zeros(1, 8, 512)
    early, late = base.clone(), base.clone()
    early[:, :, 0:128] = 3.0
    late[:, :, 384:512] = 3.0
    with torch.no_grad():
        a = model(early, meta)["features"]
        b = model(late, meta)["features"]
    assert not torch.allclose(a, b, atol=1e-4)


def test_an_unknown_pool_is_refused():
    with pytest.raises(ValueError):
        build(pool="attention")
