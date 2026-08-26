"""
The EEG C1 multi-route pretraining path.

Numbered to the acceptance list, so a failure names the requirement it broke.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from channel_embedding import (CHANNEL_TO_ID, CHANNEL_VOCAB, PAD_ID,  # noqa: E402
                               channel_ids_for, normalize_channel_name,
                               vocab_sha256)
from model import BERTWaveletTransformer                             # noqa: E402
from physiowave.eeg_c1.data import (CorpusIndex, RouteSchedule,      # noqa: E402
                                    ShardInfo)
from physiowave.eeg_c1.model import (MultiRouteEEGPretrainer,        # noqa: E402
                                     masked_reconstruction_loss)
from physiowave.eeg_c1.preprocess import (map_to_slots, place_on_slots,  # noqa: E402
                                          split_subjects, window_signal,
                                          zscore_windows)
from physiowave.eeg_c1.routes import (DOWNSTREAM_ONLY,               # noqa: E402
                                      PRETRAIN_DATASETS, ROUTES,
                                      SLOTS_32, default_sampling_weights)


def tiny(**kw):
    kw.setdefault("embed_dim", 64)
    kw.setdefault("depth", 1)
    kw.setdefault("num_heads", 4)
    kw.setdefault("channel_embed_dim", 16)
    return MultiRouteEEGPretrainer(**kw)


def meta_for(route_id, n_valid=None):
    r = ROUTES[route_id]
    ids, _ = channel_ids_for(r.slots)
    valid = torch.ones(r.n_channels, dtype=torch.bool)
    if n_valid is not None:
        valid[n_valid:] = False
        ids = [i if valid[k] else PAD_ID for k, i in enumerate(ids)]
    return {"channel_ids": torch.tensor(ids, dtype=torch.long),
            "valid_channel_mask": valid}


# --- 1 ---------------------------------------------------------------------- #
def test_1_c0_pretrain_unchanged_without_channel_embedding():
    """C0 keeps the behaviour it had before the channel code existed."""
    def build():
        torch.manual_seed(0)
        return BERTWaveletTransformer(
            in_channels=4, max_level=3, patch_size=(1, 128), embed_dim=64,
            depth=1, num_heads=4, scale_fold="dynamic", task_type="pretrain",
            mask_ratio=0.5)

    x = torch.randn(2, 4, 1024)
    m = build().eval()
    torch.manual_seed(7)
    a = m(x, task="pretrain", mask_ratio=0.5)
    m2 = build().eval()
    torch.manual_seed(7)
    b = m2(x, task="pretrain", mask_ratio=0.5)
    assert torch.equal(a[1], b[1])
    assert torch.allclose(a[0], b[0], atol=1e-6)
    assert m.channel_encoder is None
    with pytest.raises(ValueError):
        m(x, task="pretrain", channel_meta={"channel_ids": torch.zeros(4).long()})


# --- 2 ---------------------------------------------------------------------- #
def test_2_c1_pretrain_consumes_channel_metadata():
    """The legacy pretrain path accepts channel_meta instead of refusing it."""
    torch.manual_seed(0)
    m = BERTWaveletTransformer(
        in_channels=4, max_level=3, patch_size=(1, 128), embed_dim=64, depth=1,
        num_heads=4, scale_fold="dynamic", task_type="pretrain", mask_ratio=0.5,
        channel_encoding="id", channel_injection="token", channel_embed_dim=16)
    meta = {"channel_ids": torch.tensor([2, 3, 4, 5])}
    pred, mask, target = m(torch.randn(2, 4, 1024), task="pretrain",
                           mask_ratio=0.5, channel_meta=meta)
    assert pred.shape == target.shape == (2, 32, 128)
    assert mask.shape == (2, 32)

    out = tiny()(torch.randn(2, 19, 1024), "E19_256",
                 channel_meta=meta_for("E19_256"), mask_ratio=0.5)
    assert out["pred"].shape == (2, 152, 128)


# --- 3 ---------------------------------------------------------------------- #
def test_3_c1_injects_at_token_site_only():
    """Not into the waveform, and not into the fold's scale logits."""
    torch.manual_seed(0)
    m = tiny().eval()
    x = torch.randn(2, 19, 1024)
    meta = meta_for("E19_256")

    with torch.no_grad():
        # A big gate: if the code reached the frontend at all, the folded
        # spectrogram would move with it.
        m.channel_token_gate.fill_(2.0)
        spec_with = m(x, "E19_256", channel_meta=meta,
                      mask_generator=torch.Generator().manual_seed(3))["spec"]
        spec_bare = m.wavelet_frontends["E19_256"](x)
    assert torch.allclose(spec_with, spec_bare, atol=1e-6), \
        "the channel code changed the wavelet/ScaleFold output"

    # It must reach the tokens, though.
    with torch.no_grad():
        m.channel_token_gate.fill_(0.0)
        t0 = m(x, "E19_256", channel_meta=meta,
               mask_generator=torch.Generator().manual_seed(3))["tokens"]
        m.channel_token_gate.fill_(2.0)
        t1 = m(x, "E19_256", channel_meta=meta,
               mask_generator=torch.Generator().manual_seed(3))["tokens"]
    assert not torch.allclose(t0, t1), "the token site received nothing"

    assert not hasattr(m, "channel_to_scale")
    with pytest.raises(ValueError):
        tiny(channel_injection="fold")
    with pytest.raises(ValueError):
        tiny(channel_encoding="signed")


# --- 4 ---------------------------------------------------------------------- #
def test_4_gate_has_finite_nonzero_gradient_at_first_step():
    torch.manual_seed(0)
    m = tiny()
    out = m(torch.randn(2, 19, 1024), "E19_256",
            channel_meta=meta_for("E19_256"), mask_ratio=0.5)
    loss, _ = masked_reconstruction_loss(out)
    loss.backward()
    g = m.channel_token_gate.grad
    assert g is not None and torch.isfinite(g).all() and float(g) != 0.0
    assert m.channel_encoder.id_embed.weight.grad is not None


# --- 5 ---------------------------------------------------------------------- #
@pytest.mark.parametrize("route_id,expected", [
    ("E19_256", 152), ("E32_512", 256), ("E64_256", 512), ("E128_512", 1024)])
def test_5_token_counts_per_route(route_id, expected):
    r = ROUTES[route_id]
    assert r.n_tokens == expected
    assert r.patches_per_channel == 8
    assert r.window_samples == int(4.0 * r.sampling_rate)
    assert r.patch_size == (1, int(0.5 * r.sampling_rate))
    m = tiny()
    out = m(torch.randn(1, r.n_channels, r.window_samples), route_id,
            channel_meta=meta_for(route_id), mask_ratio=0.5)
    assert out["pred"].shape == (1, expected, r.patch_t)
    assert out["mask"].shape == (1, expected)


# --- 6 ---------------------------------------------------------------------- #
def test_6_four_wavelet_experts_are_independent():
    m = tiny()
    assert len(m.wavelet_frontends) == 4
    seen = {}
    for rid, front in m.wavelet_frontends.items():
        ids = {id(p) for p in front.parameters()}
        for other, prev in seen.items():
            assert not (ids & prev), f"{rid} shares parameters with {other}"
        seen[rid] = ids


# --- 7 ---------------------------------------------------------------------- #
def test_7_transformer_is_shared_not_copied():
    m = tiny()
    names = [n for n, _ in m.named_parameters()]
    assert sum(n.startswith("shared_transformer.") for n in names) > 0
    for rid in ROUTES:
        assert not any(n.startswith(f"wavelet_frontends.{rid}.encoder")
                       for n in names)
    shared = {id(p) for p in m.shared_transformer.parameters()}
    for front in m.wavelet_frontends.values():
        assert not (shared & {id(p) for p in front.parameters()})
    # One encoder's worth of blocks, not four.
    assert len([n for n in names if ".blocks." in n and
                n.startswith("shared_transformer.")]) > 0


# --- 8 ---------------------------------------------------------------------- #
def test_8_same_rate_routes_share_patcher_and_decoder():
    m = tiny()
    assert sorted(m.patch_embed_by_rate) == ["256", "512"]
    assert sorted(m.reconstruction_heads) == ["256", "512"]
    assert ROUTES["E19_256"].rate_key == ROUTES["E64_256"].rate_key == "256"
    assert ROUTES["E32_512"].rate_key == ROUTES["E128_512"].rate_key == "512"
    assert (m.patch_embed_by_rate[ROUTES["E19_256"].rate_key]
            is m.patch_embed_by_rate[ROUTES["E64_256"].rate_key])
    assert (m.reconstruction_heads[ROUTES["E32_512"].rate_key]
            is m.reconstruction_heads[ROUTES["E128_512"].rate_key])


# --- 9 ---------------------------------------------------------------------- #
def test_9_padding_channels_affect_nothing():
    torch.manual_seed(0)
    m = tiny().eval()
    meta = meta_for("E32_512", n_valid=26)
    x = torch.randn(2, 32, 2048)
    x[:, 26:] = 0.0
    x2 = x.clone()
    x2[:, 26:] = torch.randn(2, 6, 2048) * 99.0

    def run(t):
        return m(t, "E32_512", channel_meta=meta, mask_ratio=0.5,
                 mask_generator=torch.Generator().manual_seed(11))

    with torch.no_grad():
        a, b = run(x), run(x2)
    assert torch.equal(a["mask"], b["mask"])
    sel = a["mask"]
    assert torch.allclose(a["pred"][sel], b["pred"][sel], atol=1e-6)
    la, _ = masked_reconstruction_loss(a)
    lb, _ = masked_reconstruction_loss(b)
    assert torch.allclose(la, lb, atol=1e-7)
    # Padded slots are never selected, which is how they leave the loss.
    assert int(a["mask"].reshape(2, 32, 8)[:, 26:].sum()) == 0
    # And the code for a padded slot is zero.
    with torch.no_grad():
        code = m.channel_encoder(meta)
    assert torch.allclose(code[26:], torch.zeros_like(code[26:]))


# --- 10 --------------------------------------------------------------------- #
def test_10_tdbrain_26_maps_into_32_slots():
    tdbrain = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
               "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3",
               "Pz", "P4", "P8", "O1", "Oz", "O2"]
    assert len(tdbrain) == 26
    mp = map_to_slots(tdbrain, SLOTS_32)
    assert int(mp.valid.sum()) == 26
    assert len(mp.empty_slots) == 6
    assert not mp.unmatched_sources and not mp.unknown_names
    # Placed by NAME: every filled slot holds the row whose name matches it.
    x = np.arange(26 * 10, dtype=np.float32).reshape(26, 10)
    placed = place_on_slots(x, mp, 32)
    for src, slot in zip(mp.matrix_rows, mp.slot_of_row):
        assert SLOTS_32[slot] == tdbrain[src]
        assert np.array_equal(placed[slot], x[src])
    for j in range(32):
        if not mp.valid[j]:
            assert np.all(placed[j] == 0)


# --- 11 --------------------------------------------------------------------- #
def test_11_hbn_129_to_128_is_by_name_and_traceable():
    from EEG.preprocess_pretrain_corpus import HBN_NON_SCALP
    assert "E129" in HBN_NON_SCALP
    names = [f"E{i}" for i in range(1, 130)]
    drop = [n for n in names if n.strip().upper() in
            {s.upper() for s in HBN_NON_SCALP}]
    assert drop == ["E129"], "the removed channel must be named, not positional"
    assert len([n for n in names if n not in drop]) == 128
    # A 129-channel file with no identifiable reference must fail, not be
    # trimmed by position.
    anon = [f"Ch{i}" for i in range(1, 130)]
    assert not [n for n in anon if n.strip().upper() in
                {s.upper() for s in HBN_NON_SCALP}]


# --- 12 --------------------------------------------------------------------- #
class _Args:
    """The subset of the CLI namespace the adapters and registry read."""
    allow_upsample_faced = False
    mains_hz = None
    no_notch = False
    registry = None


def test_12_upsampling_is_allowed_only_at_a_corpus_native_rate():
    """FACED ships 31 of 123 subjects at 250 Hz; that is acquisition, not a
    derivative. A rate the registry does not list stays refused, which is the
    only thing separating those subjects from the 250 Hz preprocessed release.
    """
    from EEG.preprocess_pretrain_corpus import upsample_allowed

    assert upsample_allowed("faced", 250.0, _Args())
    assert not upsample_allowed("faced", 128.0, _Args())
    assert not upsample_allowed("faced", 200.0, _Args())
    # PhysioNetMI is 160 Hz throughout against a 256 Hz route.
    assert upsample_allowed("physionet_mi", 160.0, _Args())
    # TDBRAIN records at 500 and lists nothing: a 250 Hz TDBRAIN file is wrong.
    assert not upsample_allowed("tdbrain", 250.0, _Args())

    forced = _Args()
    forced.allow_upsample_faced = True
    assert upsample_allowed("faced", 128.0, forced)
    assert not upsample_allowed("tdbrain", 128.0, forced)   # FACED-only escape


# --- 12b -------------------------------------------------------------------- #
def test_12b_registry_supplies_mains_and_m3cv_is_not_notched_twice():
    """A wrong mains frequency is silent in both directions, so it comes from a
    checked-in table rather than whichever shell script launched the job.
    """
    from EEG.preprocess_pretrain_corpus import resolved_mains

    assert resolved_mains("faced", _Args())[0] == 50.0
    assert resolved_mains("tdbrain", _Args())[0] == 50.0
    assert resolved_mains("physionet_mi", _Args())[0] == 60.0
    assert resolved_mains("hbn", _Args())[0] == 60.0
    assert resolved_mains("hgd", _Args())[0] == 50.0
    assert resolved_mains("tueg", _Args())[0] == 60.0

    # M3CV is published already notched at 49-51 Hz. Filtering it again would
    # be a second stopband on an already-empty one.
    freq, why = resolved_mains("m3cv", _Args())
    assert freq is None
    assert "already notched" in why

    explicit = _Args()
    explicit.mains_hz = 60.0
    assert resolved_mains("m3cv", explicit) == (60.0, "--mains-hz")

    off = _Args()
    off.no_notch = True
    assert resolved_mains("faced", off)[0] is None


# --- 12c -------------------------------------------------------------------- #
def test_12c_physionet_channel_padding_resolves_to_electrodes():
    """PhysioNet's EDF+ pads labels to four characters with periods. Before this
    was handled, all 64 channels landed as UNK, per-file slot coverage was 0 of
    64, and every recording in the corpus failed the coverage gate.
    """
    from channel_embedding import CHANNEL_TO_ID, normalize_channel_name

    for raw, want in (("Fc5.", "FC5"), ("C3..", "C3"), ("Cz..", "Cz"),
                      ("Iz..", "Iz"), ("Fp1.", "Fp1"), ("Af7.", "AF7"),
                      ("Po7.", "PO7"), ("T10.", "T10")):
        got = normalize_channel_name(raw)
        assert got == want, f"{raw!r} -> {got!r}, expected {want!r}"
        assert got in CHANNEL_TO_ID

    # The wrappers that were already handled must keep working.
    assert normalize_channel_name("EEG Fp1-REF") == "Fp1"
    assert normalize_channel_name("T3") == "T7"
    assert normalize_channel_name("E128") == "E128"


# --- 12d -------------------------------------------------------------------- #
def test_12d_bids_subject_identity_is_taken_from_the_label():
    """Subject identity decides the train/val split, so a wrong answer here is a
    subject leaking across it that nothing downstream can detect. The earlier
    FACED adapter used the parent directory, which in BIDS is always "eeg".
    """
    from EEG.preprocess_pretrain_corpus import _bids_subject

    cases = {
        "HBN/R3/sub-NDARAB123/eeg/sub-NDARAB123_task-rest_eeg.set": "sub-NDARAB123",
        "faced/sub-101/eeg/sub-101_task-emotion_eeg.edf": "sub-101",
        "hgd/sub-14/eeg/sub-14_task-motor_eeg.set": "sub-14",
        "tdbrain/sub-19681349/ses-1/eeg/x.edf": "sub-19681349",
    }
    for path, want in cases.items():
        assert _bids_subject(path) == want
    assert _bids_subject("nobids/S001R03.edf") is None
    # Distinct subjects must not collapse onto one id.
    ids = {_bids_subject(p) for p in cases}
    assert len(ids) == len(cases)


# --- 12e -------------------------------------------------------------------- #
def test_12e_streaming_adapters_shard_by_subject():
    """--array=0-63 must give each task 1/64 of the corpus, not all of it.

    TUEG shards a file list it builds up front. The other six adapters stream
    Recordings, and nothing applied --shard to them: every task of an array read
    the whole corpus and raced its siblings for the same output paths. The check
    now runs inside each adapter, before the file is opened -- filtering after
    the read would be correct and useless, since each of 64 tasks would still
    decode all of HBN and discard 63/64 of it.
    """
    from EEG.preprocess_pretrain_corpus import owns, subject_shard

    class Sharded:
        shard = None

    subjects = [f"sub-{i:05d}" for i in range(2000)]

    # A subject belongs to exactly one task, for any partition.
    for total in (2, 4, 16, 64):
        buckets = [0] * total
        for sub in subjects:
            idx = subject_shard(sub, total)
            assert 0 <= idx < total
            buckets[idx] += 1
        assert sum(buckets) == len(subjects)
        assert all(b > 0 for b in buckets), f"an empty task at N={total}"
        # Roughly balanced: no task may carry more than 3x the mean.
        assert max(buckets) < 3 * (len(subjects) / total)

    # owns() partitions the corpus and never duplicates or drops a subject.
    seen = []
    for idx in range(8):
        a = Sharded()
        a.shard = (idx, 8)
        seen += [s for s in subjects if owns(s, a)]
    assert sorted(seen) == sorted(subjects)

    # No --shard means one process owns everything.
    assert all(owns(s, Sharded()) for s in subjects[:50])

    # The two sharding paths must agree, or a subject lands on two tasks.
    assert subject_shard("aaaaaaaa", 16) == subject_shard("aaaaaaaa", 16)


# --- 12f -------------------------------------------------------------------- #
def test_12f_proportional_sampling_sees_every_window_once():
    """"As much data as a dataset has" -- not a quota a small corpus repeats to.

    The mixture weights are probabilities over STEPS, and a step draws
    batch_by_route[route] windows: 64 on E19_256, 12 on E128_512. Weighting
    steps by window count directly would hand E19_256 five times the windows
    its share of the corpus warrants, so the weights divide by the route's batch
    size and the WINDOW shares come out equal to the corpus shares.
    """
    from physiowave.eeg_c1.data import (DEFAULT_BATCH_BY_ROUTE, CorpusIndex,
                                        RouteSchedule)

    counts = {"tueg": 13_711_294, "physionet_mi": 39_000, "m3cv": 400_000,
              "faced": 1_100_000, "tdbrain": 90_000, "hbn": 4_500_000,
              "hgd": 47_000}
    index = CorpusIndex([
        ShardInfo(f"/x/{d}.h5", d, PRETRAIN_DATASETS[d].route_id, n)
        for d, n in counts.items()])
    total = sum(counts.values())

    sched = RouteSchedule(index, weights="proportional", seed=42, num_replicas=4)
    assert sched.weight_policy == "proportional"
    mix = sched.realised_mixture()

    for d, n in counts.items():
        # Window share tracks corpus share.
        assert abs(mix["by_window"][d] - n / total) < 0.01, d
        # And that means one pass, which is the requirement.
        b = DEFAULT_BATCH_BY_ROUTE[PRETRAIN_DATASETS[d].route_id]
        seen = mix["by_step"][d] * sched.steps_per_epoch * b * 4
        assert 0.85 < seen / n < 1.20, f"{d} seen {seen/n:.2f}x per epoch"

    # The step shares are deliberately NOT the corpus shares -- HBN's small
    # micro-batch buys it more steps for the same number of windows.
    assert mix["by_step"]["hbn"] > mix["by_step"]["tueg"]
    assert counts["hbn"] < counts["tueg"]

    # null means proportional.
    assert RouteSchedule(index, weights=None, seed=42).weight_policy == "proportional"

    # The old behaviour is still reachable, and still repeats small corpora.
    bal = RouteSchedule(index, weights="balanced", seed=42, num_replicas=4)
    bmix = bal.realised_mixture()
    b = DEFAULT_BATCH_BY_ROUTE[PRETRAIN_DATASETS["physionet_mi"].route_id]
    pmi = bmix["by_step"]["physionet_mi"] * bal.steps_per_epoch * b * 4
    assert bmix["by_window"]["physionet_mi"] > 20 * (counts["physionet_mi"] / total)
    assert pmi > 0

    with pytest.raises(SystemExit):
        RouteSchedule(index, weights="whatever-that-means", seed=42)


# --- 13 --------------------------------------------------------------------- #
def test_13_deap_is_never_a_pretraining_dataset():
    assert "deap" not in PRETRAIN_DATASETS
    assert "deap" in DOWNSTREAM_ONLY
    assert "deap" not in default_sampling_weights()
    for name in DOWNSTREAM_ONLY:
        assert name not in PRETRAIN_DATASETS
    assert set(PRETRAIN_DATASETS) == {
        "tueg", "faced", "tdbrain", "physionet_mi", "m3cv", "hbn", "hgd"}


# --- 14 --------------------------------------------------------------------- #
def test_14_subject_split_has_no_leak():
    subs = [f"s{i:03d}" for i in range(37)]
    train, val = split_subjects(subs, 0.1, 42)
    assert not (set(train) & set(val))
    assert set(train) | set(val) == set(subs)
    assert len(val) >= 1
    assert split_subjects(subs, 0.1, 42) == (train, val)
    assert split_subjects(subs, 0.1, 43) != (train, val)


# --- 15 --------------------------------------------------------------------- #
def _fake_index(n=5000):
    return CorpusIndex([ShardInfo(f"/x/{d}.h5", d, s.route_id, n)
                        for d, s in PRETRAIN_DATASETS.items()])


def test_15_route_sampler_mixture_and_ddp_agreement():
    index = _fake_index()
    lengths = {d: 5000 for d in PRETRAIN_DATASETS}

    # `balanced` is the policy that means P(route)=1/4, and it still does.
    single = RouteSchedule(index, weights="balanced", steps_per_epoch=4000,
                           seed=42)
    single.set_epoch(0)
    plan = single.plan()
    by_route = {}
    for d in plan:
        rid = PRETRAIN_DATASETS[d].route_id
        by_route[rid] = by_route.get(rid, 0) + 1
    for rid, n in by_route.items():
        assert abs(n / len(plan) - 0.25) < 0.03, f"{rid} drew {n/len(plan):.3f}"

    # The DEFAULT is proportional, and these seven datasets are all the same
    # size -- so each takes an equal share of WINDOWS, and therefore an unequal
    # share of steps, since a route's micro-batch ranges from 64 down to 12.
    prop = RouteSchedule(index, steps_per_epoch=4000, seed=42)
    assert prop.weight_policy == "proportional"
    windows = prop.realised_mixture()["by_window"]
    for d in PRETRAIN_DATASETS:
        assert abs(windows[d] - 1.0 / len(PRETRAIN_DATASETS)) < 0.02, d
    steps = prop.realised_mixture()["by_step"]
    assert steps["hbn"] > 3 * steps["tueg"], "batch size was not divided out"

    ranks = [RouteSchedule(index, steps_per_epoch=500, seed=42,
                           num_replicas=4, rank=r) for r in range(4)]
    for s in ranks:
        s.set_epoch(0)
    steps = [list(s.steps(lengths)) for s in ranks]
    for i in range(500):
        assert len({t[i][0] for t in steps}) == 1, "ranks disagreed on the route"
        assert len({t[i][1] for t in steps}) == 1
        picks = [x for t in steps for x in t[i][2]]
        assert len(set(picks)) == len(picks), "two ranks got the same window"


# --- 16 --------------------------------------------------------------------- #
def test_16_resume_continues_the_sampling_sequence():
    index = _fake_index()
    lengths = {d: 5000 for d in PRETRAIN_DATASETS}
    full = RouteSchedule(index, steps_per_epoch=200, seed=42, num_replicas=2,
                         rank=0)
    full.set_epoch(3)
    baseline = list(full.steps(lengths))

    resumed = RouteSchedule(index, steps_per_epoch=200, seed=42, num_replicas=2,
                            rank=0)
    resumed.load_state_dict({"epoch": 3, "seed": 42, "steps_per_epoch": 200,
                             "start_step": 80})
    assert list(resumed.steps(lengths)) == baseline[80:]
    assert len(resumed) == 120

    other = RouteSchedule(index, steps_per_epoch=200, seed=42)
    other.set_epoch(4)
    assert other.plan() != full.plan()


# --- 17 --------------------------------------------------------------------- #
@pytest.mark.parametrize("route_id", list(ROUTES))
def test_17_unpatchify_inverts_patchify(route_id):
    r = ROUTES[route_id]
    spec = torch.randn(3, r.n_channels, r.window_samples)
    patches = MultiRouteEEGPretrainer.patchify(spec, r.patch_t)
    assert patches.shape == (3, r.n_tokens, r.patch_t)
    back = MultiRouteEEGPretrainer.unpatchify(patches, r.n_channels, r.patch_t)
    assert back.shape == spec.shape
    assert torch.equal(spec, back)


# --- 18 --------------------------------------------------------------------- #
REQUIRED_FIGURES = [
    "fig_dataset_routes", "fig_pretraining_convergence", "fig_route_convergence",
    "fig_mask_reconstruction", "fig_mask_examples_by_dataset",
    "fig_mask_statistics", "fig_wavelet_frequency_response",
    "fig_scale_fold_weights", "fig_channel_embedding",
]


@pytest.mark.slow
def test_18_all_required_svgs_generate_from_a_smoke_checkpoint(tmp_path):
    from physiowave.eeg_c1.entry import build_smoke_corpus

    corpus = build_smoke_corpus(str(tmp_path / "smoke_corpus"), subjects=3,
                                recordings=1, windows=2)
    run_dir = tmp_path / "run"
    cmd = [sys.executable, "-m", "physiowave.train.pretrain_main",
           "--config", "pretrain/eeg_c1_moe", "--output-dir", str(run_dir),
           "--set", f"data.manifest_train={corpus['train']}",
           f"data.manifest_val={corpus['val']}",
           "model.embed_dim=48", "model.depth=1", "model.num_heads=4",
           "model.channel_embed_dim=16", "train.epochs=1",
           "train.warmup_epochs=0", "train.precision=fp32",
           "train.grad_accumulation_steps=1", "train.steps_per_epoch=4",
           "train.batch_size_by_route.E19_256=1",
           "train.batch_size_by_route.E32_512=1",
           "train.batch_size_by_route.E64_256=1",
           "train.batch_size_by_route.E128_512=1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert (run_dir / "best.pth").is_file()

    r = subprocess.run(
        [sys.executable, "scripts/visualize_eeg_pretraining.py",
         "--run-dir", str(run_dir), "--checkpoint", "best.pth",
         "--split", "val", "--format", "svg"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]

    for name in REQUIRED_FIGURES:
        svg = run_dir / "figures" / f"{name}.svg"
        assert svg.is_file(), f"{name}.svg was not generated"
        text = svg.read_text()
        assert len(text) > 0
        # Matplotlib writes the XML declaration first, which is correct for a
        # standalone SVG; the requirement's "starts with <svg" is read as "is a
        # non-empty SVG document".
        assert text.startswith("<?xml") or text.startswith("<svg")
        assert "<svg" in text
        assert (run_dir / "figure_data" / f"{name}.npz").is_file()
        meta = json.loads(
            (run_dir / "figure_metadata" / f"{name}.json").read_text())
        for key in ("checkpoint_sha256", "global_step", "mask_seed",
                    "plotting_script_git_commit", "generated_utc"):
            assert key in meta


# --- vocabulary and schema, supporting the above ---------------------------- #
def test_channel_vocabulary_is_append_only():
    """The ids Sleep-EDF and PhysioP300 were trained under must not move."""
    assert CHANNEL_VOCAB[0] == "<pad>" and CHANNEL_VOCAB[1] == "<unk>"
    for name, expected in (("Fpz-Cz", 2), ("Pz-Oz", 3)):
        assert CHANNEL_TO_ID[name] == expected
    # The PhysioP300 monopolar block starts where it always did.
    assert CHANNEL_TO_ID["Fp1"] == 22
    assert CHANNEL_TO_ID["Iz"] == 85
    assert len(CHANNEL_VOCAB) == len(set(CHANNEL_VOCAB))
    assert len(vocab_sha256()) == 64


def test_channel_name_normalisation():
    assert normalize_channel_name("EEG Fp1-REF") == "Fp1"
    assert normalize_channel_name("EEG T3-LE") == "T7"      # old nomenclature
    assert normalize_channel_name("fp2") == "Fp2"
    assert normalize_channel_name("E17") == "E17"
    ids, unknown = channel_ids_for(["EEG Fp1-REF", "nonsense"])
    assert ids[0] == CHANNEL_TO_ID["Fp1"] and ids[1] == 1
    assert unknown == ["nonsense"]


def test_every_dataset_slot_list_matches_its_route():
    for spec in PRETRAIN_DATASETS.values():
        assert len(spec.slots) == spec.route.n_channels
        _, unknown = channel_ids_for(spec.slots)
        assert not unknown, f"{spec.dataset_id} has out-of-vocabulary slots"


def test_windowing_drops_the_incomplete_tail():
    x = np.random.randn(4, 1024 * 2 + 500).astype(np.float32)
    w, starts = window_signal(x, 1024, 1024)
    assert w.shape == (2, 4, 1024)
    assert starts.tolist() == [0, 1024]
    short, _ = window_signal(np.random.randn(4, 100).astype(np.float32), 1024,
                             1024)
    assert short.shape[0] == 0


def test_zscore_leaves_padded_channels_exactly_zero():
    w = (np.random.randn(3, 8, 256).astype(np.float32) + 5.0) * 20
    valid = np.array([True] * 6 + [False] * 2)
    w[:, ~valid] = 0.0
    z = zscore_windows(w, valid, 1e-6, 20.0)
    assert np.all(z[:, ~valid] == 0.0)
    assert abs(float(z[:, valid].mean())) < 1e-4
    assert abs(float(z[:, valid].std()) - 1.0) < 1e-2
    assert float(np.abs(z).max()) <= 20.0


# --------------------------------------------------------------------------- #
# TUEG: identity, sharding, and the gates that decide whether a corpus is usable
# --------------------------------------------------------------------------- #

def test_tueg_identity_comes_from_the_filename():
    from EEG.preprocess_pretrain_corpus import tueg_identity

    root = "/corpus/TUEG_v2.0.2"
    p = f"{root}/edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf"
    ident = tueg_identity(p, root)
    assert ident == {"subject": "aaaaaaaa", "session": "s001",
                     "montage": "01_tcp_ar", "rule": "filename"}

    # v1.x had no bucket directory; the filename rule does not care.
    p1 = f"{root}/edf/aaaaaaab/s003_2011/02_tcp_le/aaaaaaab_s003_t002.edf"
    assert tueg_identity(p1, root)["subject"] == "aaaaaaab"
    assert tueg_identity(p1, root)["montage"] == "02_tcp_le"
    assert tueg_identity(p1, root)["rule"] == "filename"

    # A file that does not follow the convention falls back and SAYS it did,
    # so an --inspect run shows whether the fallback is carrying the corpus.
    p2 = f"{root}/edf/000/aaaaaaac/s001_2015/01_tcp_ar/oddly_named.edf"
    fb = tueg_identity(p2, root)
    assert fb["rule"] == "path"
    assert fb["subject"] == "aaaaaaac"


def test_tueg_sharding_is_by_subject_and_covers_everything():
    from EEG.preprocess_pretrain_corpus import shard_files, tueg_identity

    root = "/corpus"
    files = [f"{root}/edf/000/aaaaaa{c}{d}/s{s:03d}_2015/01_tcp_ar/"
             f"aaaaaa{c}{d}_s{s:03d}_t000.edf"
             for c in "abcde" for d in "fghij" for s in (1, 2, 3)]
    assert len(files) == 75

    class Args:
        shard = None

    n = 7
    seen, owners = [], {}
    for i in range(n):
        Args.shard = (i, n)
        kept = shard_files(files, Args, root)
        seen.extend(kept)
        for f in kept:
            owners.setdefault(tueg_identity(f, root)["subject"], set()).add(i)

    assert sorted(seen) == sorted(files), "sharding lost or duplicated a file"
    for subject, tasks in owners.items():
        assert len(tasks) == 1, f"{subject} was split across tasks {tasks}"


def test_subject_split_side_is_independent_of_shard_count():
    """The whole reason the split is hashed rather than shuffled."""
    from physiowave.eeg_c1.preprocess import subject_split_side

    subs = [f"aaaaaa{i:03d}" for i in range(500)]
    sides = {s: subject_split_side(s, 0.1, 42) for s in subs}
    # Re-deciding one subject at a time, in any order, gives the same answer.
    for s in reversed(subs):
        assert subject_split_side(s, 0.1, 42) == sides[s]
    val = [s for s in subs if sides[s] == "val"]
    assert 0.05 < len(val) / len(subs) < 0.16
    assert not (set(val) & {s for s in subs if sides[s] == "train"})
    # A different seed is a different partition.
    assert [subject_split_side(s, 0.1, 7) for s in subs] != list(sides.values())


def test_auxiliary_channels_are_dropped_without_failing_the_corpus():
    """TUEG's EKG/PHOTIC/IBI rows are 21% of a file and are not a naming bug."""
    from physiowave.eeg_c1.routes import SLOTS_19

    tuh = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
           "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF",
           "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
           "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF",
           "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF",
           "EEG EKG1-REF", "PHOTIC-REF", "IBI", "BURSTS", "SUPPR"]
    mp = map_to_slots(tuh, SLOTS_19)

    # Every scalp electrode lands, old nomenclature included.
    assert int(mp.valid.sum()) == 19, f"empty: {mp.empty_slots}"
    assert not mp.empty_slots
    # The five auxiliary rows are dropped and named.
    assert len(mp.unmatched_sources) == 5
    assert {"IBI", "BURSTS", "SUPPR"} <= set(mp.unmatched_sources)
    # T3/T4/T5/T6 are the same electrodes as T7/T8/P7/P8, so they land there.
    placed = {SLOTS_19[j]: tuh[i]
              for i, j in zip(mp.matrix_rows, mp.slot_of_row)}
    assert placed["T7"] == "EEG T3-REF"
    assert placed["P8"] == "EEG T6-REF"
    # No placed channel is UNK; the gate that matters is slot coverage.
    ids, _ = channel_ids_for(SLOTS_19)
    assert 1 not in ids


def test_a_montage_that_cannot_be_named_leaves_slots_empty():
    """The failure the coverage gate exists to catch, as distinct from the above."""
    from physiowave.eeg_c1.routes import SLOTS_19

    anonymous = [f"Ch{i}" for i in range(1, 25)]
    mp = map_to_slots(anonymous, SLOTS_19)
    assert int(mp.valid.sum()) == 0
    assert len(mp.empty_slots) == 19
    assert len(mp.unmatched_sources) == 24


def test_manifest_merge_refuses_a_split_leak(tmp_path):
    """The check that is only possible once every shard's manifest is present."""
    root = tmp_path / "corpus"
    d = root / "tueg"
    d.mkdir(parents=True)
    row = {"path": str(d / "a.h5"), "dataset_id": "tueg",
           "route_id": "E19_256", "n_windows": 4, "subjects": ["aaaaaaaa"]}
    (d / "manifest_train.0000.jsonl").write_text(json.dumps(row) + "\n")
    leaked = dict(row, path=str(d / "b.h5"))
    (d / "manifest_val.0001.jsonl").write_text(json.dumps(leaked) + "\n")

    r = subprocess.run(
        [sys.executable, "scripts/build_eeg_c1_manifest.py",
         "--corpus-root", str(root), "--datasets", "tueg", "--allow-missing"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1
    assert "both splits" in r.stderr
    assert "aaaaaaaa" in r.stderr


def test_a_sparse_recording_is_skipped_not_fatal(tmp_path):
    """A file carrying a third of the montage fails alone, not the whole run.

    TUEG holds recordings from 21 to 41 channels, and some of the small ones do
    not carry the 10-20 nineteen at all. Those produce windows that are more
    mask than measurement; the aggregate empty-slot gate averages them away, so
    the per-recording floor is what actually keeps them out.
    """
    from EEG.preprocess_pretrain_corpus import process_recording, Recording
    from physiowave.eeg_c1.preprocess import PreprocessConfig, PreprocessError
    from physiowave.eeg_c1.routes import SLOTS_19

    route = ROUTES["E19_256"]
    cfg = PreprocessConfig(notch_hz=60.0)

    class Args:
        min_slot_coverage = 0.75

    sparse = Recording(
        recording_id="sparse", subject_id="aaaaaaaz",
        data=np.random.randn(6, 256 * 30) * 25e-6,
        channel_names=["EEG FP1-REF", "EEG FP2-REF", "EEG C3-REF",
                       "EEG C4-REF", "EEG O1-REF", "EEG O2-REF"],
        sampling_rate=256.0, unit="V", mains_hz=60.0)
    with pytest.raises(PreprocessError) as exc:
        process_recording(sparse, "tueg", cfg, SLOTS_19, route,
                          str(tmp_path), {}, args_ref=Args())
    assert "6 of 19" in str(exc.value)

    # The full montage passes the same floor.
    full = Recording(
        recording_id="full", subject_id="aaaaaaay",
        data=np.random.randn(19, 256 * 30) * 25e-6,
        channel_names=[f"EEG {c.upper()}-REF" for c in SLOTS_19],
        sampling_rate=256.0, unit="V", mains_hz=60.0)
    entry = process_recording(full, "tueg", cfg, SLOTS_19, route,
                              str(tmp_path), {}, args_ref=Args())
    assert entry is not None and entry["n_windows"] > 0
    assert entry["placed_total"] == 19 and entry["placed_unk"] == 0
    assert not entry["empty_slots"]
