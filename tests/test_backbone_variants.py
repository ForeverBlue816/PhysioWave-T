"""The backbone's post-BERT components, and the switches that turn them off.

Each is a config option rather than a rewrite so the paper can ablate them.
The properties worth pinning are that the switches actually reach the modules,
that SwiGLU's 2/3 hidden scaling keeps the parameter count level with the GELU
MLP it replaces -- otherwise "better FFN" is really "bigger FFN" -- and that
RMSNorm reduces in fp32 under a half-precision input.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from physiowave.models.backbone import (        # noqa: E402
    BackboneConfig,
    FactorizedBackbone,
    RMSNorm,
    SwiGLU,
    make_ffn,
    make_norm,
)


def cfg(**kw) -> BackboneConfig:
    base = BackboneConfig(embed_dim=128, depth=2, num_heads=4, slot_heads=2, mlp_ratio=4.0)
    for k, v in kw.items():
        setattr(base, k, v)
    return base


def test_defaults_are_the_modern_block():
    c = cfg()
    assert (c.norm, c.ffn, c.qk_norm) == ("rmsnorm", "swiglu", True)


@pytest.mark.parametrize("norm,ffn,qk", [
    ("layernorm", "mlp", False),          # the original block
    ("rmsnorm", "mlp", False),
    ("layernorm", "swiglu", False),
    ("layernorm", "mlp", True),
    ("rmsnorm", "swiglu", True),          # the default
])
def test_every_combination_runs(norm, ffn, qk):
    m = FactorizedBackbone(cfg(norm=norm, ffn=ffn, qk_norm=qk)).eval()
    x = torch.randn(2, 4, 8, 128)
    with torch.no_grad():
        assert m(x).shape == x.shape


def test_switches_reach_the_modules():
    modern = FactorizedBackbone(cfg())
    legacy = FactorizedBackbone(cfg(norm="layernorm", ffn="mlp", qk_norm=False))
    assert isinstance(modern.blocks[0].n1, RMSNorm)
    assert isinstance(modern.blocks[0].mlp, SwiGLU)
    assert isinstance(modern.blocks[0].time_attn.q_norm, RMSNorm)
    assert isinstance(legacy.blocks[0].n1, torch.nn.LayerNorm)
    assert isinstance(legacy.blocks[0].mlp, torch.nn.Sequential)
    assert isinstance(legacy.blocks[0].time_attn.q_norm, torch.nn.Identity)


def test_swiglu_does_not_smuggle_in_extra_capacity():
    c = cfg()
    swiglu = sum(p.numel() for p in make_ffn(c).parameters())
    c.ffn = "mlp"
    mlp = sum(p.numel() for p in make_ffn(c).parameters())
    assert abs(swiglu - mlp) / mlp < 0.05, (swiglu, mlp)


def test_rmsnorm_matches_the_definition():
    n = RMSNorm(32)
    x = torch.randn(4, 7, 32) * 6
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + n.eps)
    assert torch.allclose(n(x), expected, atol=1e-6)


def test_rmsnorm_reduces_in_fp32_and_returns_the_input_dtype():
    n = RMSNorm(64)
    x = (torch.randn(2, 5, 64) * 100).to(torch.bfloat16)
    out = n(x)
    assert out.dtype == torch.bfloat16
    # bf16 has ~8 mantissa bits; reducing in it would drift far more than this
    ref = n(x.float())
    assert (out.float() - ref).abs().max() < 0.5


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="unknown norm"):
        make_norm("batchnorm", 8)
    with pytest.raises(ValueError, match="unknown ffn"):
        make_ffn(cfg(ffn="glu"))


def test_gradients_reach_every_parameter():
    m = FactorizedBackbone(cfg()).train()
    m(torch.randn(2, 4, 8, 128)).square().mean().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, dead
