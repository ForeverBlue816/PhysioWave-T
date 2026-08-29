"""Fine-tuning a C1 pretrained encoder.

The thing being guarded is that a downstream number means what it says. Three
ways it silently would not have:

  * the export dropped channel_token_gate, so the C1 mechanism was switched off
    in the experiment that exists to measure it;
  * a shape disagreement loaded nothing, trained from scratch, and logged
    "pretrained";
  * the legacy key migration renamed patch_embed out of existence on every
    save/load round trip, so the test metrics were computed with a freshly
    initialised patcher.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from channel_embedding import channel_ids_for, vocab_payload   # noqa: E402
from physiowave.eeg_c1.downstream import (DEFAULT_WAVELETS,    # noqa: E402
                                          EEGC1Downstream)
from physiowave.eeg_c1.model import MultiRouteEEGPretrainer    # noqa: E402
from physiowave.eeg_c1.routes import ROUTES                    # noqa: E402

SMALL = dict(embed_dim=64, depth=2, num_heads=4, channel_embed_dim=16,
             dropout=0.0, wavelet_names=list(DEFAULT_WAVELETS))


def p300_channels():
    src = open(os.path.join(ROOT, "EEG", "physio_p300_finetune.py")).read()
    body = re.search(r"^CHANNELS_58 = \[(.*?)\]", src, re.S | re.M).group(1)
    return re.findall(r"['\"]([^'\"]+)['\"]", body)


class _Meta:
    def __init__(self, names):
        self.channel_names = names
        self.channel_mask = None


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("pre")
    pre = MultiRouteEEGPretrainer(**SMALL)
    with torch.no_grad():
        pre.channel_token_gate.fill_(0.33)
    ck = d / "pre.pth"
    torch.save({"model": pre.state_dict(), "config": {"model": {}},
                "epoch": 4, "global_step": 1920}, ck)
    out = {}
    for route in ("E64_256", "E19_256"):
        p = d / f"enc_{route}.pth"
        r = subprocess.run(
            [sys.executable, "scripts/export_eeg_pretrained_encoder.py",
             "--checkpoint", str(ck), "--route", route, "--output", str(p)],
            cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        out[route] = str(p)
    out["gate"] = 0.33
    return out


# --- placement -------------------------------------------------------------- #
def test_a_montage_that_is_a_subset_of_a_route_goes_in_its_slots():
    ch = p300_channels()
    m = EEGC1Downstream(in_channels=len(ch), window_samples=512,
                        sampling_rate=256, patch_samples=128, num_classes=2,
                        channel_names=ch, route_id="E64_256", **SMALL)
    r = ROUTES["E64_256"]
    assert int((m.slot_index >= 0).sum()) == len(ch)

    # Every channel lands on the slot with its own name, and nowhere else.
    from channel_embedding import normalize_channel_name
    for slot_i, src_i in enumerate(m.slot_index.tolist()):
        if src_i >= 0:
            assert normalize_channel_name(ch[src_i]) == \
                normalize_channel_name(r.slots[slot_i])

    # The absent slots carry zero, not whatever was next in the array.
    x = torch.randn(2, len(ch), 512)
    placed = m._to_slots(x)
    assert placed.shape == (2, r.n_channels, 512)
    absent = (m.slot_index < 0)
    assert float(placed[:, absent].abs().sum()) == 0.0
    for slot_i, src_i in enumerate(m.slot_index.tolist()):
        if src_i >= 0:
            assert torch.equal(placed[:, slot_i], x[:, src_i])


def test_a_channel_the_route_does_not_have_is_refused_not_dropped():
    with pytest.raises(ValueError, match="not slots of"):
        EEGC1Downstream(in_channels=2, window_samples=512, sampling_rate=256,
                        patch_samples=128, num_classes=2,
                        channel_names=["Fpz-Cz", "Pz-Oz"],
                        route_id="E64_256", **SMALL)


def test_a_montage_off_every_route_gets_its_own_frontend():
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    assert m.slot_index is None
    assert m.route.n_channels == 2 and m.route.patch_t == 50
    out = m(torch.randn(2, 2, 3000), _Meta(["Fpz-Cz", "Pz-Oz"]))
    assert out["logits"].shape == (2, 5)
    assert out["features"].shape == (2, SMALL["embed_dim"])


# --- what transfers --------------------------------------------------------- #
def test_a_matching_route_transfers_the_frontend_and_the_patcher(exported):
    ch = p300_channels()
    m = EEGC1Downstream(in_channels=len(ch), window_samples=512,
                        sampling_rate=256, patch_samples=128, num_classes=2,
                        channel_names=ch, route_id="E64_256", **SMALL)
    rep = m.load_pretrained(exported["E64_256"])
    for prefix in ("shared_transformer.", "wavelet_frontend.", "patch_embed.",
                   "channel_encoder."):
        assert any(k.startswith(prefix) for k in rep["taken"]), prefix
    assert "channel_token_gate" in rep["taken"], \
        "the gate scales the whole C1 contribution and initialises to zero"
    assert float(m.channel_token_gate.detach()) == pytest.approx(exported["gate"])


def test_a_different_montage_transfers_the_transformer_and_says_so(exported):
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    rep = m.load_pretrained(exported["E19_256"])
    assert any(k.startswith("shared_transformer.") for k in rep["taken"])
    assert "channel_token_gate" in rep["taken"]
    # The frontend is per-electrode and the patcher per patch length: neither
    # is transferable here, and pretending otherwise is the failure.
    assert not any(k.startswith("wavelet_frontend.") for k in rep["taken"])
    assert not any(k.startswith("patch_embed.") for k in rep["taken"])
    assert "loaded: transformer" in m.describe_transfer(rep)


def test_the_weights_actually_land(exported):
    """A report that says "taken" and a tensor that did not move is the bug."""
    ch = p300_channels()
    kw = dict(in_channels=len(ch), window_samples=512, sampling_rate=256,
              patch_samples=128, num_classes=2, channel_names=ch,
              route_id="E64_256", **SMALL)
    torch.manual_seed(0)
    fresh = EEGC1Downstream(**kw)
    torch.manual_seed(0)
    loaded = EEGC1Downstream(**kw)
    loaded.load_pretrained(exported["E64_256"])

    sd_f, sd_l = fresh.state_dict(), loaded.state_dict()
    moved = [k for k in sd_f
             if sd_f[k].shape == sd_l[k].shape and not torch.equal(sd_f[k], sd_l[k])]
    assert any(k.startswith("shared_transformer.") for k in moved)
    assert "channel_token_gate" in moved
    # And the head, which nothing supplies, did not.
    assert not any(k.startswith("head.") for k in moved)


# --- the ways it could lie --------------------------------------------------- #
def test_a_shape_disagreement_is_refused_not_ignored(exported, tmp_path):
    ch = p300_channels()
    wrong = dict(SMALL, embed_dim=128, num_heads=4)
    m = EEGC1Downstream(in_channels=len(ch), window_samples=512,
                        sampling_rate=256, patch_samples=128, num_classes=2,
                        channel_names=ch, route_id="E64_256", **wrong)
    with pytest.raises(SystemExit, match="do not fit"):
        m.load_pretrained(exported["E64_256"])


def test_a_checkpoint_with_no_transformer_is_refused(tmp_path):
    bad = tmp_path / "bad.pth"
    torch.save({"model": {"head.weight": torch.zeros(2, 64)}}, bad)
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    with pytest.raises(SystemExit, match="no transformer weights"):
        m.load_pretrained(str(bad))


def test_a_different_channel_vocabulary_is_refused(exported, tmp_path):
    ck = torch.load(exported["E19_256"], map_location="cpu", weights_only=False)
    ck["channel_vocab_sha256"] = "0" * 64
    p = tmp_path / "relabelled.pth"
    torch.save(ck, p)
    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    with pytest.raises(SystemExit, match="different electrode"):
        m.load_pretrained(str(p))


def test_the_legacy_migration_does_not_rename_a_patcher_this_model_owns():
    """patch_embed. -> legacy_patch_embed. was unconditional.

    So the C1 downstream encoder's patcher was renamed out of existence on
    every save/load round trip: reported as remapped, loaded as missing, and
    the test metrics computed with a freshly initialised patcher while the log
    said the checkpoint had loaded.
    """
    from physiowave.models.checkpoint import migrate_state_dict

    m = EEGC1Downstream(in_channels=2, window_samples=3000, sampling_rate=100,
                        patch_samples=50, num_classes=5,
                        channel_names=["Fpz-Cz", "Pz-Oz"], **SMALL)
    out, report = migrate_state_dict(m.state_dict(), m)
    assert not report.missing, f"a round trip lost {report.missing[:6]}"
    assert not report.remapped, f"a round trip renamed {report.remapped[:6]}"
    assert any(k.startswith("patch_embed.") for k in out)
