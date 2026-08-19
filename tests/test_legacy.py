"""Legacy compatibility: the original model still runs and still has its old shapes."""

from __future__ import annotations

import pytest
import torch

from physiowave.config import load_config
from physiowave.models.build import build_model
from physiowave.models.legacy import build_legacy_model, legacy_available, legacy_token_count

pytestmark = pytest.mark.skipif(not legacy_available(),
                                reason="legacy top-level modules not importable")


def _legacy(**kw):
    params = dict(in_channels=8, max_level=2, wave_kernel_size=16,
                  wavelet_names=["db4", "sym4"], patch_size=(1, 32),
                  embed_dim=32, depth=1, num_heads=4)
    params.update(kw)
    return build_legacy_model(**params)


def test_legacy_forward_shapes_unchanged():
    """Regression: with every new module off, the legacy shapes are exactly as before.

    ``forward_features`` returns ``[B, (J+1)*C/p_f * T/p_t, D]`` -- the sequence
    length that grows with the decomposition level, which is the behaviour WAST
    replaces (and which this test pins down so the replacement stays honest).
    """
    B, C, T, J, D = 2, 8, 256, 2, 32
    model = _legacy().eval()
    x = torch.randn(B, C, T)
    feats = model(x, task="features")
    expected_tokens = (J + 1) * C * (T // 32)
    assert feats.shape == (B, expected_tokens, D), (
        f"legacy feature shape changed: {tuple(feats.shape)}"
    )
    assert legacy_token_count(C, J, T, (1, 32))["num_tokens"] == expected_tokens


def test_legacy_pretrain_forward():
    B, C, T = 2, 8, 256
    model = _legacy().eval()
    pred, mask, target = model(torch.randn(B, C, T), task="pretrain", mask_ratio=0.5)
    assert pred.shape == target.shape
    assert mask.shape == pred.shape[:2]
    assert 0.4 < mask.float().mean().item() < 0.6


def test_legacy_classification_head():
    model = _legacy(task_type="classification", num_classes=4).eval()
    logits = model(torch.randn(2, 8, 256), task="downstream", task_name="classification")
    assert logits.shape == (2, 4)


def test_legacy_config_builds():
    cfg = load_config("model/legacy")
    cfg["model"]["legacy"].update({"in_channels": 8, "embed_dim": 32, "depth": 1,
                                   "num_heads": 4, "max_level": 2,
                                   "wavelet_names": ["db4"], "patch_size": [1, 32]})
    model = build_model(cfg)
    assert model(torch.randn(1, 8, 128), task="features").shape[-1] == 32


def test_legacy_token_inflation_is_linear_in_level():
    """The legacy token count grows with J; this is the baseline WAST is measured against."""
    counts = [legacy_token_count(19, J, 1024, (1, 64))["num_tokens"] for J in (1, 2, 3, 4)]
    assert counts == [2 * 19 * 16, 3 * 19 * 16, 4 * 19 * 16, 5 * 19 * 16]


def test_legacy_with_channel_id():
    """Rung 2 of the ablation ladder: legacy plus a learnable per-channel vector."""
    from physiowave.models.legacy import LegacyWithChannelID

    model = LegacyWithChannelID(in_channels=8, max_level=2, wave_kernel_size=16,
                                wavelet_names=["db4"], patch_size=(1, 32),
                                embed_dim=32, depth=1, num_heads=4)
    x = torch.randn(2, 8, 128)
    out = model(x, task="features")
    assert out.shape == (2, (2 + 1) * 8 * (128 // 32), 32)
    out.sum().backward()
    assert model.channel_id.grad is not None and model.channel_id.grad.abs().max() > 0

    # It is index-based, so it cannot accept a different montage size at all.
    with pytest.raises(AssertionError, match="montage-specific"):
        model(torch.randn(2, 12, 128), task="features")
