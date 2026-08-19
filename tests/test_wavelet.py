"""WAST / DWT tests: perfect reconstruction, critical sampling, boundaries, gradients."""

from __future__ import annotations

import math

import pytest
import torch

from physiowave.wavelet.dwt import (
    WaveletTransform1D,
    coeff_lengths,
    condition_number,
    dwt,
    get_filters,
    resolve_boundary_mode,
)
from physiowave.wavelet.wast import WAST, WASTConfig


def _stationary_signal(P: int, n: int = 8, cycles: float = 2.5, offset: float = 0.0,
                       dtype=torch.float64) -> torch.Tensor:
    t = torch.arange(P, dtype=dtype) / P
    base = torch.sin(2 * math.pi * cycles * t) + 0.3 * torch.sin(2 * math.pi * 7 * t)
    return (base + offset).unsqueeze(0).repeat(n, 1)


@pytest.mark.parametrize("mode", ["reflect", "symmetric", "periodization", "zero", "constant"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_dwt_roundtrip(mode, level):
    """Analysis followed by synthesis reproduces the input to numerical precision."""
    P = 64
    x = _stationary_signal(P)
    wt = WaveletTransform1D("bior4.4", level, mode, learnable=False).double()
    rec = wt.synthesis(wt.analysis(x), P)
    err = (rec - x).abs().max().item()
    assert err < 1e-9, f"mode={mode} level={level} round-trip error {err:.3e}"


@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize("P", [32, 64, 128])
def test_critical_sampling_length_conservation(level, P):
    """The multi-level coefficient budget equals the patch length exactly."""
    lens = coeff_lengths(P, level)
    assert sum(lens) == P, f"coeff_lengths({P}, {level}) sums to {sum(lens)}"
    wt = WaveletTransform1D("bior4.4", level, "reflect", learnable=False)
    coeffs = wt.analysis(torch.randn(4, P))
    assert [c.shape[-1] for c in coeffs] == lens
    assert sum(c.shape[-1] for c in coeffs) == P


def test_orthogonal_wavelet_falls_back_loudly(caplog):
    """A non-symmetric wavelet cannot use `reflect`; the fallback is logged."""
    with caplog.at_level("WARNING"):
        mode = resolve_boundary_mode("db4", "reflect")
    assert mode == "periodization"
    assert "periodization" in caplog.text
    assert resolve_boundary_mode("bior4.4", "reflect") == "reflect"


def test_boundary_artifact_roundtrip():
    """Edge reconstruction error must not exceed 3x the centre error for `reflect`.

    `zero` is measured and reported but is allowed to fail: it is not the default
    precisely because it asserts the signal vanishes just outside the patch.
    """
    P, edge = 64, 8
    x = _stationary_signal(P, offset=1.5)                 # baseline drift, as in real EEG
    report = {}
    for mode in ("reflect", "zero"):
        wt = WaveletTransform1D("bior4.4", 3, mode, learnable=False).double()
        err = (wt.synthesis(wt.analysis(x), P) - x) ** 2
        e = 0.5 * (err[:, :edge].mean() + err[:, -edge:].mean()).item()
        c = err[:, P // 2 - edge: P // 2 + edge].mean().item()
        report[mode] = (e, c, condition_number(P, "bior4.4", 3, mode))
    e, c, _ = report["reflect"]
    scale = float((x ** 2).mean())
    assert e <= 3 * c + 1e-12 * scale, (
        f"reflect edge MSE {e:.3e} exceeds 3x centre MSE {c:.3e}; report={report}"
    )
    print("boundary round-trip (edge_mse, center_mse, cond):", report)


def test_boundary_analysis_artifact():
    """Patch-local coefficients: `reflect` is closer to an infinite-context DWT.

    This is the measurement that actually justifies the default.  A patch is cut
    out of a longer signal with baseline drift; its finest detail band is compared
    against the band a whole-signal transform produces.  Zero extension asserts a
    step at each patch border, so its edge coefficients deviate far more.
    """
    P, S, level, edge = 64, 8, 2, 3
    T = P * S
    t = torch.arange(T, dtype=torch.float64) / P
    sig = (torch.sin(2 * math.pi * 2.5 * t) + 1.5
           + 0.05 * torch.randn(T, generator=torch.Generator().manual_seed(3), dtype=torch.float64))
    filt = get_filters("bior4.4", torch.float64)
    ref = dwt(sig.unsqueeze(0), filt, level, "periodization")[-1][0].view(S, P // 2)
    patches = sig.view(S, P)

    errs = {}
    for mode in ("reflect", "zero"):
        d1 = dwt(patches, filt, level, mode)[-1]
        e = (d1 - ref).abs()
        errs[mode] = 0.5 * (e[:, :edge].mean() + e[:, -edge:].mean()).item()
    assert errs["reflect"] < errs["zero"], (
        f"reflect should distort patch-edge coefficients less than zero: {errs}"
    )
    print("analysis-side edge deviation:", errs)


def test_wavelet_parameters_receive_gradient():
    """The wavelet filters are on the gradient path, not decorative."""
    m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=3))
    m.train()
    out = m(torch.randn(2, 4, 256))
    out["tokens"].pow(2).mean().backward()
    for name in ("dec_lo", "dec_hi"):
        g = getattr(m.wt, name).grad
        assert g is not None and g.abs().max() > 0, f"{name} received no gradient"
    assert m.subbands[0].depthwise.weight.grad.abs().max() > 0


def test_wavelet_parameters_change_output():
    """Perturbing the filters changes the tokens materially."""
    m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=3)).eval()
    x = torch.randn(2, 4, 256)
    with torch.no_grad():
        base = m(x)["tokens"].clone()
        m.wt.dec_lo.add_(0.05)
        m.wt.clear_cache()
        pert = m(x)["tokens"]
    rel = ((pert - base).norm() / base.norm()).item()
    assert rel > 1e-3, f"changing the wavelet changed the output by only {rel:.2e}"


def test_token_counts_and_compression():
    """WAST does not inflate tokens with decomposition level; report the ratio."""
    C, T, K = 19, 1024, 16
    prev = None
    for level in (1, 2, 3, 4):
        m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=level))
        rep = m.token_report(C, T, K)
        S = T // 64
        assert rep["N_wast"] == C * S, "WAST token count must not depend on the level"
        assert rep["N_new"] == K * S, "compressed token count must be K * S"
        assert rep["N_old_legacy"] == (level + 1) * C * S
        if prev is not None:
            assert rep["N_wast"] == prev, "token count grew with the decomposition level"
        prev = rep["N_wast"]
    print("compression at level 4:", rep)
    assert rep["compression_vs_legacy_with_compression"] > 1.0


def test_shapes_and_assertions():
    m = WAST(WASTConfig(patch_size=64, embed_dim=48, level=2))
    out = m(torch.randn(3, 7, 512))
    assert out["tokens"].shape == (3, 7, 8, 48)
    assert out["raw_patches"].shape == (3, 7, 8, 64)
    assert out["patch_scores"].shape == (3, 8)
    with pytest.raises(AssertionError):
        m(torch.randn(3, 7, 500))                          # T not a multiple of P
    with pytest.raises(AssertionError):
        m(torch.randn(3, 7))                               # not [B, C, T]


def test_fgm_scores_are_detached():
    """Masking decisions must never back-propagate into the wavelet filters."""
    m = WAST(WASTConfig(patch_size=64, embed_dim=32, level=2))
    out = m(torch.randn(2, 4, 256))
    assert not out["patch_scores"].requires_grad
