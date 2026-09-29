"""
The ECG C1 multi-route pretraining path.

What is ECG-specific is tested here; what is shared with EEG (the objective,
the schedule, resume) is tested in tests/test_eeg_c1_*.py and only its wiring
to the ECG registry is exercised again.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from channel_embedding import CHANNEL_VOCAB, vocab_sha256  # noqa: E402
from physiowave.ecg_c1.leads import (  # noqa: E402
    ECG_LEAD_VOCAB,
    LEADS_12,
    LIMB_LEADS,
    UNK_ID,
    ecg_vocab_sha256,
    lead_id,
    normalize_lead,
)
from physiowave.ecg_c1.model import ECG_WAVELETS, MultiRouteECGPretrainer  # noqa: E402
from physiowave.ecg_c1.preprocess import (  # noqa: E402
    ECGPreprocessConfig,
    derive_limb_leads,
    highpass,
    map_leads,
    normalise_windows,
    process_ecg_record,
    strip_zero_padding,
)
from physiowave.ecg_c1.routes import (  # noqa: E402
    DOWNSTREAM_ONLY,
    LEAD_GROUPS,
    PRETRAIN_DATASETS,
    ROUTES,
)
from physiowave.eeg_c1.model import (  # noqa: E402
    MultiRouteEEGPretrainer,
    masked_reconstruction_loss,
)
from physiowave.eeg_c1.objective import resume_incompatibilities  # noqa: E402

PY = sys.executable
PREP = os.path.join(ROOT, "ECG", "preprocess_ecg_corpus.py")
MERGE = os.path.join(ROOT, "scripts", "build_eeg_c1_manifest.py")


def tiny(**kw):
    kw.setdefault("embed_dim", 64)
    kw.setdefault("depth", 1)
    kw.setdefault("num_heads", 4)
    kw.setdefault("channel_embed_dim", 16)
    kw.setdefault("dropout", 0.0)
    return MultiRouteECGPretrainer(**kw)


def meta_for(route_id):
    from physiowave.ecg_c1.leads import lead_ids_for
    r = ROUTES[route_id]
    ids, _ = lead_ids_for(r.slots)
    return {"channel_ids": torch.tensor(ids, dtype=torch.long),
            "valid_channel_mask": torch.ones(r.n_channels, dtype=torch.bool)}


# --------------------------------------------------------------------------- #
# Vocabulary and registry
# --------------------------------------------------------------------------- #

def test_ecg_vocab_is_separate_from_eeg():
    """Adding ECG leads must not move the EEG vocabulary's hash."""
    assert "aVR" not in CHANNEL_VOCAB and "V1" not in CHANNEL_VOCAB
    assert ecg_vocab_sha256() != vocab_sha256()
    assert ECG_LEAD_VOCAB[:2] == ["<pad>", "<unk>"]
    assert ECG_LEAD_VOCAB[2:14] == list(LEADS_12)


@pytest.mark.parametrize("raw,canon", [
    ("DI", "I"), ("DII", "II"), ("DIII", "III"), ("AVR", "aVR"),
    ("avl", "aVL"), ("aVF", "aVF"), ("v1", "V1"), ("Lead II", "II"),
    ("lead_III", "III"), ("MLII", "MLII"), ("V6", "V6")])
def test_lead_aliases(raw, canon):
    assert normalize_lead(raw) == canon
    assert lead_id(raw) != UNK_ID


def test_unknown_lead_is_unk_not_guessed():
    assert lead_id("ECG") == UNK_ID
    assert lead_id("Pleth") == UNK_ID


def test_routes_and_registry():
    r12, r1 = ROUTES["L12_500"], ROUTES["L1_250"]
    assert (r12.window_samples, r12.patch_t, r12.patches_per_channel) == (5000, 250, 20)
    assert (r1.window_samples, r1.patch_t, r1.patches_per_channel) == (2500, 125, 20)
    assert tuple(r12.slots) == LEADS_12
    assert [LEADS_12[i] for i in LEAD_GROUPS["L12_500"][0]] == list(LIMB_LEADS)
    for spec in PRETRAIN_DATASETS.values():
        assert spec.route_id in ROUTES
    assert not set(DOWNSTREAM_ONLY) & set(PRETRAIN_DATASETS)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def test_standard_model_is_about_30m():
    m = MultiRouteECGPretrainer(embed_dim=512, depth=9, num_heads=8, dropout=0.0)
    rep = m.parameter_report()
    assert 29e6 < rep["total"] < 31e6, rep["total"]
    assert set(m.wavelet_frontends) == {"L12_500", "L1_250"}
    assert set(m.patch_embed_by_rate) == {"500", "250"}


def test_frontend_starts_from_the_ecg_wavelets():
    import pywt

    from wavelet_modules import load_wavelet_kernel
    m = tiny()
    bank = m.wavelet_frontends["L12_500"].decomp.selector.wavelet_filters
    assert [f.wave_init for f in bank] == list(ECG_WAVELETS)
    for f in bank:
        lo, _ = load_wavelet_kernel(f.wave_init, 16, "pad")
        assert len(pywt.Wavelet(f.wave_init).dec_lo) <= 16
        assert torch.allclose(f.low_filter.weight[0, 0], lo)


def _mask(m, route_id, ratio=0.7, seed=0, batch=3):
    r = ROUTES[route_id]
    # The input is fixed too: frequency-guided masking scores the tokens, so a
    # different window is a different mask by design.
    x = torch.randn(batch, r.n_channels, r.window_samples,
                    generator=torch.Generator().manual_seed(1000 + seed))
    g = torch.Generator().manual_seed(seed)
    out = m(x, route_id, channel_meta=meta_for(route_id), mask_ratio=ratio,
            mask_generator=g)
    return out, out["mask"].reshape(batch, r.n_channels, r.patches_per_channel)


def test_limb_leads_are_masked_as_one_unit():
    m = tiny().eval()
    out, mask = _mask(m, "L12_500")
    limb = mask[:, :6, :]
    # Every time patch: all six limb leads masked, or none of them.
    assert bool((limb.all(dim=1) | (~limb).all(dim=1)).all())
    P = ROUTES["L12_500"].patches_per_channel
    assert int(limb[:, 0, :].sum(dim=1).unique()) == int(P * 0.7)
    assert int(mask[:, 6:, :].sum(dim=(1, 2)).unique()) == int(6 * P * 0.7)
    # And the overall ratio is the configured one.
    assert abs(mask.float().mean().item() - 0.7) < 0.02
    loss, metrics = masked_reconstruction_loss(out)
    assert torch.isfinite(loss)
    loss.backward()


def test_precordial_leads_are_masked_independently():
    m = tiny().eval()
    _, mask = _mask(m, "L12_500", seed=3, batch=8)
    prec = mask[:, 6:, :]
    # Not column masking: some time patch has precordials both masked and not.
    assert bool((prec.any(dim=1) & ~prec.all(dim=1)).any())


def test_group_masking_off_is_the_eeg_selection_exactly():
    m = tiny(lead_group_masking=False).eval()
    tokens = torch.randn(2, 240, 64)
    m._mask_route = "L12_500"
    a = m._select_mask(tokens, 0.7, None, torch.Generator().manual_seed(5))
    b = MultiRouteEEGPretrainer._select_mask(
        m, tokens, 0.7, None, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_single_lead_route_masks_per_token_and_trains():
    m = tiny()
    out, mask = _mask(m, "L1_250", batch=4)
    assert int(mask.sum(dim=(1, 2)).unique()) == int(20 * 0.7)
    loss, _ = masked_reconstruction_loss(out)
    loss.backward()
    assert m.wavelet_frontends["L1_250"].decomp.selector.wavelet_filters[0] \
        .low_filter.weight.grad is not None


def test_validation_mask_is_reproducible():
    m = tiny().eval()
    _, a = _mask(m, "L12_500", seed=11)
    _, b = _mask(m, "L12_500", seed=11)
    assert torch.equal(a, b)


def test_encode_accepts_a_longer_window():
    m = tiny().eval()
    x = torch.randn(1, 12, 2 * 5000)
    with torch.no_grad():
        z = m.encode(x, "L12_500", channel_meta=meta_for("L12_500"))
    assert z.shape == (1, 12 * 40, 64)


def test_changing_lead_group_masking_refuses_a_resume():
    old = {"model": {"lead_group_masking": True}}
    new = {"model": {"lead_group_masking": False}}
    assert any(k == "model.lead_group_masking"
               for k, _, _ in resume_incompatibilities(old, new))


# --------------------------------------------------------------------------- #
# Signal pipeline
# --------------------------------------------------------------------------- #

def test_limb_leads_derived_exactly():
    rng = np.random.default_rng(0)
    i, ii = rng.normal(size=100), rng.normal(size=100)
    leads = {"I": i, "II": ii}
    made = derive_limb_leads(leads)
    assert set(made) == {"III", "aVR", "aVL", "aVF"}
    np.testing.assert_allclose(leads["III"], ii - i)
    np.testing.assert_allclose(leads["aVR"] + leads["aVL"] + leads["aVF"],
                               0.0, atol=1e-12)


def test_eight_lead_file_fills_twelve_slots():
    x = np.random.default_rng(1).normal(size=(8, 50))
    names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    m = map_leads(x, names, LEADS_12)
    assert m.valid.all() and set(m.derived) == {"III", "aVR", "aVL", "aVF"}


def test_code_names_map_by_name_not_position():
    names = ["V6", "DI", "DII", "DIII", "AVR", "AVL", "AVF",
             "V1", "V2", "V3", "V4", "V5"]
    x = np.arange(12, dtype=float)[:, None] * np.ones((12, 5))
    m = map_leads(x, names, LEADS_12)
    assert m.valid.all()
    assert m.placed[LEADS_12.index("V6"), 0] == 0.0     # row 0 of the source
    assert m.placed[LEADS_12.index("I"), 0] == 1.0


def test_strip_zero_padding():
    x = np.zeros((12, 4096))
    x[:, 48:4048] = 1.0
    assert strip_zero_padding(x).shape == (12, 4000)


def test_shared_normalisation_keeps_lead_ratios():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(1, 12, 500))
    w[:, 3] *= 4.0                        # one lead four times the others
    out = normalise_windows(w, np.ones(12, bool),
                            ECGPreprocessConfig(clip_sigma=0))
    sd = out[0].std(axis=-1)
    assert sd[3] / np.median(np.delete(sd, 3)) == pytest.approx(4.0, rel=0.15)
    per = normalise_windows(w, np.ones(12, bool),
                            ECGPreprocessConfig(normalization="window_per_lead"))
    assert np.allclose(per[0].std(axis=-1), 1.0, atol=1e-3)


def test_highpass_has_no_edge_transient_on_a_10s_record():
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    sig = np.sin(2 * np.pi * 10 * t)
    y = highpass(sig[None] + 2.0, fs, 0.5)[0]         # a 2 mV DC offset
    assert np.max(np.abs(y - sig)) < 0.05


def _ecg(fs, seconds, leads=12, seed=0):
    sys.path.insert(0, os.path.join(ROOT, "ECG"))
    from preprocess_ecg_corpus import synthetic_ecg
    return synthetic_ecg(leads, fs, seconds, np.random.default_rng(seed))


def test_record_to_one_window_at_route_rate():
    cfg = ECGPreprocessConfig()
    x = _ecg(400.0, 10.0) * 10.0                       # in units of 1e-4 V
    out = process_ecg_record(x, ["DI", "DII", "DIII", "AVR", "AVL", "AVF",
                                 "V1", "V2", "V3", "V4", "V5", "V6"],
                             400.0, "1e-4V", ROUTES["L12_500"], 60.0, cfg)
    assert out.windows.shape == (1, 12, 5000)
    assert out.windows.dtype == np.float32
    assert abs(float(np.sqrt((out.windows ** 2).mean())) - 1.0) < 1e-3


def test_qc_drops_flat_nonfinite_and_implausible_windows():
    cfg = ECGPreprocessConfig(stride_seconds=10.0)
    x = _ecg(500.0, 40.0)                              # four windows
    x[3, 5000:10000] = 0.0                             # flat lead, window 1
    x[0, 12000] = np.nan                               # window 2
    x[7, 16000] = 80.0                                 # 80 mV spike, window 3
    out = process_ecg_record(x, list(LEADS_12), 500.0, "mV",
                             ROUTES["L12_500"], None, cfg)
    assert out.n_candidate_windows == 4
    assert out.windows.shape[0] == 1
    assert out.qc == {"non_finite": 1, "flat_lead": 1, "amplitude": 1}


def test_unit_error_is_caught_by_the_amplitude_check():
    """A file in uV read as mV is 1000x too large -- QC must say so."""
    x = _ecg(500.0, 10.0) * 1000.0
    out = process_ecg_record(x, list(LEADS_12), 500.0, "mV",
                             ROUTES["L12_500"], None, ECGPreprocessConfig())
    assert out.windows.shape[0] == 0 and out.qc["amplitude"] == 1


def test_continuous_record_is_capped_reproducibly():
    cfg = ECGPreprocessConfig(max_windows_per_record=5)
    x = _ecg(250.0, 300.0, leads=1)
    a = process_ecg_record(x, ["patch1"], 250.0, "mV", ROUTES["L1_250"],
                           60.0, cfg, record_key="p00001_s03")
    b = process_ecg_record(x, ["patch1"], 250.0, "mV", ROUTES["L1_250"],
                           60.0, cfg, record_key="p00001_s03")
    assert a.windows.shape == (5, 1, 2500) and a.n_candidate_windows == 30
    np.testing.assert_array_equal(a.starts_seconds, b.starts_seconds)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def _write_wfdb(dirpath, name, sig_mv, fs, names):
    import wfdb
    os.makedirs(dirpath, exist_ok=True)
    wfdb.wrsamp(name, fs=fs, units=["mV"] * sig_mv.shape[0],
                sig_name=list(names), p_signal=sig_mv.T, fmt=["16"] * sig_mv.shape[0],
                write_dir=dirpath)


def test_wfdb_read_from_directory_and_from_zip(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "ECG"))
    import preprocess_ecg_corpus as prep
    sig = _ecg(500.0, 10.0)
    rec_dir = tmp_path / "raw" / "files" / "p1000" / "p10000032" / "s40689238"
    _write_wfdb(str(rec_dir), "40689238", sig, 500, LEADS_12)
    root = str(tmp_path / "raw")
    ad = prep.MimicIVECG()
    keys = ad.list_keys(root)
    assert keys == ["files/p1000/p10000032/s40689238/40689238"]
    a = ad.read(root, keys[0], {})
    assert a.subject_id == "p10000032" and a.fs == 500.0
    np.testing.assert_allclose(a.data, sig, atol=2e-3)

    zpath = str(tmp_path / "mimic.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(rec_dir):
            zf.write(rec_dir / fn, f"mimic-iv-ecg-1.0/{keys[0].rsplit('/', 1)[0]}/{fn}")
    zkeys = ad.list_keys(zpath)
    assert len(zkeys) == 1
    b = ad.read(zpath, zkeys[0], {})
    np.testing.assert_allclose(b.data, a.data)
    assert b.subject_id == "p10000032"
    assert b.record_id == "mimic-iv-ecg-1.0/" + keys[0]


def test_code_reader(tmp_path):
    """Padding stripped wherever it is, the exam_id-0 row skipped, patients joined."""
    import h5py
    sys.path.insert(0, os.path.join(ROOT, "ECG"))
    import preprocess_ecg_corpus as prep
    tr = np.zeros((3, 4096, 12), dtype=np.float32)
    tr[0] = _ecg(400.0, 10.24).T                           # fills all 4096
    tr[1, 581:3515] = _ecg(400.0, 7.335, seed=1).T          # the common 7.3 s
    # row 2 stays all zeros: the published files' extra exam_id-0 row
    with h5py.File(tmp_path / "exams_part0.hdf5", "w") as f:
        f["tracings"] = tr
        f["exam_id"] = np.array([101, 202, 0])
    with open(tmp_path / "exams.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exam_id", "age", "patient_id", "trace_file"])
        w.writerows([[101, 60, 7, "exams_part0.hdf5"],
                     [202, 61, 7, "exams_part0.hdf5"]])
    ad = prep.CODE()
    keys = ad.list_keys(str(tmp_path))
    assert keys == ["exams_part0.hdf5#0", "exams_part0.hdf5#1"]
    ctx = {}
    r0, r1 = (ad.read(str(tmp_path), k, ctx) for k in keys)
    assert r0.data.shape == (12, 4096) and r1.data.shape == (12, 2934)
    assert r0.subject_id == r1.subject_id == "code15_7"
    route = ROUTES["L12_500"]
    long_ = process_ecg_record(r0.data, r0.lead_names, r0.fs, r0.unit, route,
                               60.0, ECGPreprocessConfig())
    short = process_ecg_record(r1.data, r1.lead_names, r1.fs, r1.unit, route,
                               60.0, ECGPreprocessConfig())
    assert long_.windows.shape == (1, 12, 5000)
    # A 7.3 s exam holds no 10 s window: not a failure, and counted as such.
    assert short.n_candidate_windows == 0


def test_mimic_file_order_is_mapped_by_name():
    """MIMIC-IV-ECG stores aVF before aVL; slot order must not follow the file."""
    x = np.arange(12, dtype=float)[:, None] * np.ones((12, 5))
    names = ["I", "II", "III", "aVR", "aVF", "aVL",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    m = map_leads(x, names, LEADS_12)
    assert m.placed[LEADS_12.index("aVL"), 0] == 5.0
    assert m.placed[LEADS_12.index("aVF"), 0] == 4.0


def test_medalcare_reader_takes_one_variant_and_the_torso_as_subject(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "ECG"))
    import preprocess_ecg_corpus as prep
    run = tmp_path / "MedalCare-XL" / "WP2_largeDataset_Noise" / "sinus" / "train" / "run_S65"
    run.mkdir(parents=True)
    sig = _ecg(500.0, 10.0)
    for v in ("raw", "noise", "filtered"):
        np.savetxt(run / f"000014_{v}.csv", sig, delimiter=",")
    (run / "siginfo.csv").write_text("info1,info2\nx.mat,sinus/train/run_S65/run_000001\n")
    mac = tmp_path / "__MACOSX" / "MedalCare-XL"
    mac.mkdir(parents=True)
    (mac / "._000014_noise.csv").write_text("junk")
    ad = prep.MedalCareXL()
    keys = ad.list_keys(str(tmp_path))
    assert len(keys) == 1 and keys[0].endswith("000014_noise.csv")
    rec = ad.read(str(tmp_path), keys[0], {})
    assert rec.subject_id == "medalcare_run_S65" and rec.data.shape == (12, 5000)
    np.testing.assert_allclose(rec.data, sig, atol=1e-6)


def test_heedb_patient_ids_come_from_metadata(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "ECG"))
    import preprocess_ecg_corpus as prep
    site = tmp_path / "ECG" / "I0006"
    wf = site / "WFDB" / "2012"
    (site / "metadata").mkdir(parents=True)
    for i, fs in enumerate((500, 250, 500)):
        _write_wfdb(str(wf), f"rec{i}", _ecg(float(fs), 10.0, seed=i), fs, LEADS_12)
    with open(site / "metadata" / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["BDSPPatientID", "FileName", "FileID"])
        w.writerow(["P1", "WFDB/2012/rec0.hea", "1"])
        w.writerow(["P1", "WFDB/2012/rec1.hea", "2"])
    ad = prep.HEEDB()
    keys = ad.list_keys(str(tmp_path))
    subj = [ad.read(str(tmp_path), k, {}).subject_id for k in keys]
    assert subj == ["heedb_P1", "heedb_P1", "heedb_rec2"]
    r1 = ad.read(str(tmp_path), keys[1], {})
    assert r1.fs == 250.0
    out = process_ecg_record(r1.data, r1.lead_names, r1.fs, r1.unit,
                             ROUTES["L12_500"], 60.0, ECGPreprocessConfig())
    assert out.windows.shape == (1, 12, 5000)


def test_code2_refuses_rather_than_guessing(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "ECG"))
    import preprocess_ecg_corpus as prep
    with pytest.raises(SystemExit, match="no reader"):
        prep.ADAPTERS["code2"].list_keys(str(tmp_path))


# --------------------------------------------------------------------------- #
# The preprocessing driver: tasks, resume, merge
# --------------------------------------------------------------------------- #

def _prep(*args):
    r = subprocess.run([PY, PREP, *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.slow
def test_tasks_resume_and_merge(tmp_path):
    out = str(tmp_path / "corpus" / "mimic_iv_ecg")
    common = ["--dataset", "mimic_iv_ecg", "--smoke-test", "--out-dir", out,
              "--smoke-records", "24", "--windows-per-shard", "4",
              "--val-fraction", "0.3"]
    _prep(*common, "--task", "0/2")
    _prep(*common, "--task", "1/2")
    assert os.path.isfile(os.path.join(out, "manifest_train.0000.jsonl"))
    assert os.path.isfile(os.path.join(out, "subjects_val.0001.txt"))
    again = _prep(*common, "--task", "0/2")
    assert "0 to do" in again
    r = subprocess.run([PY, MERGE, "--modality", "ecg", "--corpus-root",
                        str(tmp_path / "corpus"), "--allow-missing",
                        "--check-shards"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [json.loads(ln) for ln in
            open(tmp_path / "corpus" / "merged" / "manifest_train.jsonl")]
    assert sum(r["n_windows"] for r in rows) > 0
    assert all("subjects" not in r for r in rows)


def test_downstream_sets_are_refused():
    r = subprocess.run([PY, PREP, "--dataset", "ptbxl", "--smoke-test",
                        "--out-dir", "/nonexistent"], capture_output=True,
                       text=True)
    assert r.returncode == 1 and "downstream" in r.stderr


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_trainer_smoke_end_to_end(tmp_path):
    from physiowave.train.pretrain_main import main
    out = str(tmp_path / "run")
    rc = main(["--config", "pretrain/ecg_c1_moe", "--smoke-test",
               "--max-steps", "3", "--output-dir", out])
    assert rc == 0
    for name in ("latest.pth", "best.pth", "channel_vocab.json",
                 "metrics_epoch.jsonl", "dataset_manifest.json"):
        assert os.path.isfile(os.path.join(out, name)), name
    ck = torch.load(os.path.join(out, "latest.pth"), map_location="cpu",
                    weights_only=False)
    assert ck["channel_vocab_sha256"] == ecg_vocab_sha256()
    row = json.loads(open(os.path.join(out, "metrics_epoch.jsonl")).readline())
    assert "val/route/L1_250/loss_total" in row
    assert "val/route/L12_500/loss_total" in row
