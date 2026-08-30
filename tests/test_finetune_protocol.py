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


# --------------------------------------------------------------------------- #
# the adaptive spatial filter, and the head that reads the time axis
# --------------------------------------------------------------------------- #
SLEEP_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4", "Fpz", "Fz", "Cz", "CPz",
               "Pz", "POz", "Oz"]


def build_free(pool="mean", freeze=False, num_classes=5, **kw):
    """A montage that is on no route, so it gets its own frontend."""
    return EEGC1Downstream(
        in_channels=2, window_samples=600, sampling_rate=100, patch_samples=50,
        num_classes=num_classes, channel_names=["Fpz-Cz", "Pz-Oz"],
        embed_dim=32, depth=1, num_heads=2, channel_embed_dim=8, pool=pool,
        freeze_encoder=freeze, **kw)


def test_a_scale_filter_keeps_the_montage():
    model = build(pool="time", spatial_filter="scale")
    assert model.raw_channels == 8 and model.in_channels == 8
    assert model.spatial_filter.gain.shape == (1, 8, 1)


def test_a_mix_filter_replaces_the_montage_with_its_named_outputs():
    model = build_free(pool="attn", spatial_filter="mix",
                       spatial_channels=SLEEP_NAMES)
    # the file still hands it two derivations; the frontend is built for 13
    assert model.raw_channels == 2
    assert model.in_channels == 13
    assert model.model_channel_names == SLEEP_NAMES
    out = model(torch.randn(2, 2, 600))
    assert out["logits"].shape == (2, 5)


def test_a_mix_looks_up_its_own_names_not_the_files():
    """Fpz-Cz's vocabulary row must not end up attached to a virtual Fz."""
    from channel_embedding import channel_ids_for

    model = build_free(pool="attn", spatial_filter="mix",
                       spatial_channels=SLEEP_NAMES)
    ids = model._model_name_meta(torch.device("cpu"))["channel_ids"]
    expected, _ = channel_ids_for(SLEEP_NAMES)
    assert ids.tolist() == list(expected)


def test_the_spatial_filter_trains_under_a_frozen_encoder():
    """EEGPT's probe optimiser is [chan_scale] + the linear layers."""
    model = build(pool="time", spatial_filter="scale", freeze=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("spatial_filter") for n in trainable)
    assert any(n.startswith("head") for n in trainable)
    assert not any(n.startswith("shared_transformer") for n in trainable)


def test_the_filter_stays_in_train_mode_when_the_encoder_is_frozen():
    model = build_free(pool="attn", spatial_filter="mix",
                       spatial_channels=SLEEP_NAMES, freeze=True)
    model.train()
    assert model.spatial_filter.training
    assert not model.shared_transformer.training


def test_the_attn_head_does_not_grow_with_the_number_of_positions():
    """Why staging gets it and P300 does not: 60 positions cannot be flattened."""
    short = build_free(pool="attn", probe_dim=16, head_depth=1)
    long_ = EEGC1Downstream(
        in_channels=2, window_samples=3000, sampling_rate=100, patch_samples=50,
        num_classes=5, channel_names=["Fpz-Cz", "Pz-Oz"], embed_dim=32, depth=1,
        num_heads=2, channel_embed_dim=8, pool="attn", probe_dim=16, head_depth=1)
    n = lambda m: sum(p.numel() for p in m.head.parameters())  # noqa: E731
    assert short.n_patches == 12 and long_.n_patches == 60
    assert n(short) == n(long_)


def test_the_flatten_head_does_grow_with_them():
    a = build(pool="time", probe_dim=16)
    b = EEGC1Downstream(
        in_channels=8, window_samples=1024, sampling_rate=256, patch_samples=128,
        num_classes=2, channel_names=SLOTS[:8], embed_dim=32, depth=1,
        num_heads=2, channel_embed_dim=8, pool="time", probe_dim=16)
    n = lambda m: sum(p.numel() for p in m.head.parameters())  # noqa: E731
    assert n(b) > n(a)


def test_max_norm_actually_constrains_the_rows():
    from physiowave.eeg_c1.heads import MaxNormLinear

    layer = MaxNormLinear(8, 3, max_norm=0.25)
    with torch.no_grad():
        layer.weight.fill_(10.0)
    layer(torch.randn(2, 8))
    assert torch.all(layer.weight.norm(p=2, dim=1) <= 0.25 + 1e-5)


def test_max_norm_zero_leaves_the_weights_alone():
    from physiowave.eeg_c1.heads import MaxNormLinear

    layer = MaxNormLinear(8, 3, max_norm=0.0)
    with torch.no_grad():
        layer.weight.fill_(10.0)
    layer(torch.randn(2, 8))
    assert torch.all(layer.weight == 10.0)


def test_channel_attention_pooling_is_not_a_mean():
    from physiowave.eeg_c1.heads import ChannelPool

    tokens = torch.randn(2, 6, 4, 32)
    pool = ChannelPool("attn", 32)
    with torch.no_grad():
        pool.query.copy_(tokens[0, 0, 0] * 4.0)     # a query that picks one out
        assert not torch.allclose(pool(tokens), tokens.mean(dim=1), atol=1e-3)


def test_a_mix_without_output_names_is_refused():
    with pytest.raises(ValueError):
        build_free(spatial_filter="mix")


# --------------------------------------------------------------------------- #
# how much of the recorded montage reaches a pretrained route
# --------------------------------------------------------------------------- #
def _p300_module():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "EEG", "physio_p300_finetune.py")
    spec = importlib.util.spec_from_file_location("p300_conv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def on_route(names, aliases=None):
    return EEGC1Downstream(
        in_channels=len(names), window_samples=512, sampling_rate=256,
        patch_samples=128, num_classes=2, channel_names=names,
        route_id="E64_256", embed_dim=32, depth=1, num_heads=2,
        channel_embed_dim=8, pool="time", slot_aliases=aliases)


def test_the_62_set_is_exactly_the_recorded_electrodes_the_route_has():
    conv = _p300_module()
    from channel_embedding import normalize_channel_name as norm

    slots = {norm(s) for s in ROUTES["E64_256"].slots}
    recorded = {norm(c) for c in conv.EEG_64}
    assert {norm(c) for c in conv.CHANNELS_62} == slots & recorded
    assert len(conv.CHANNELS_62) == 62


def test_58_leaves_four_pretrained_slots_empty():
    conv = _p300_module()
    a, b = on_route(conv.CHANNELS_58), on_route(conv.CHANNELS_62)
    assert int((a.slot_index >= 0).sum()) == 58
    assert int((b.slot_index >= 0).sum()) == 62


def test_all_64_is_refused_without_an_explicit_alias():
    """P9 is not TP9, and reading it as one has to be someone's decision."""
    conv = _p300_module()
    with pytest.raises(ValueError, match="P9"):
        on_route(conv.CHANNELS_64)


def test_an_alias_fills_the_route():
    conv = _p300_module()
    model = on_route(conv.CHANNELS_64, {"P9": "TP9", "P10": "TP10"})
    assert int((model.slot_index >= 0).sum()) == 64
    out = model(torch.randn(2, 64, 512),
                {"channel_ids": torch.arange(1, 65),
                 "valid_channel_mask": torch.ones(64, dtype=torch.bool)})
    assert out["logits"].shape == (2, 2)


def test_an_alias_onto_a_slot_that_does_not_exist_is_refused():
    conv = _p300_module()
    with pytest.raises(ValueError, match="not.*slots"):
        on_route(conv.CHANNELS_64, {"P9": "Nz", "P10": "TP10"})


def test_aliases_mean_nothing_off_a_route():
    with pytest.raises(ValueError, match="only means something on a route"):
        EEGC1Downstream(
            in_channels=2, window_samples=600, sampling_rate=100,
            patch_samples=50, num_classes=5, channel_names=["Fpz-Cz", "Pz-Oz"],
            embed_dim=32, depth=1, num_heads=2, channel_embed_dim=8,
            slot_aliases={"Fpz-Cz": "Fpz"})


def test_the_split_stage_subsets_the_cache_by_name():
    """58 vs 62 is a split, not another pass over 245 runs of EDF."""
    import numpy as np

    conv = _p300_module()
    cached = conv.EEG_64
    X = np.arange(len(cached) * 4, dtype=np.float32).reshape(1, len(cached), 4)
    out = conv._take_channels(X, cached, conv.CHANNELS_58, "test")
    assert out.shape == (1, 58, 4)
    for j, name in enumerate(conv.CHANNELS_58):
        assert np.array_equal(out[0, j], X[0, cached.index(name)])


def test_asking_the_cache_for_a_channel_it_lacks_is_refused():
    import numpy as np

    conv = _p300_module()
    X = np.zeros((1, 3, 4), dtype=np.float32)
    with pytest.raises(SystemExit, match="Cz"):
        conv._take_channels(X, ["Fp1", "Fp2", "Oz"], ["Fp1", "Cz"], "test")
