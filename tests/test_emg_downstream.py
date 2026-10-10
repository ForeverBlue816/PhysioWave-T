"""
sEMG C1 downstream: the EPN-612 / GRABMyo / NinaPro DB2 converters and the
sEMG encoder transfer.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "EMG"), os.path.join(ROOT, "ECG"),
          os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

PY = sys.executable
GESTURES = ["noGesture", "waveIn", "waveOut", "pinch", "open", "fist"]


def _semg(C, T, rng):
    return rng.normal(0, 1, (C, T)) * (0.5 + rng.random((C, 1)))


def _run(task, raw, out, *extra):
    from emg_downstream_prep import main
    assert main(["--task", task, "--raw-dir", str(raw), "--out-dir", str(out),
                 "--jobs", "1", *extra]) == 0
    import h5py
    return ({s: h5py.File(out / f"{s}.h5", "r") for s in ("train", "val", "test")},
            json.load(open(out / "split.json")))


def _epn_zip(path, n_per_gesture=12):
    rng = np.random.default_rng(0)

    def samples(labelled):
        d, k = {}, 0
        for g in GESTURES:
            for _ in range(n_per_gesture):
                k += 1
                emg = {f"ch{c}": (rng.integers(-60, 60, 1000)).tolist() for c in range(1, 9)}
                s = {"emg": emg, "startPointforGestureExecution": 200}
                if labelled:
                    s["gestureName"] = g
                d[f"idx_{k}"] = s
        return d

    with zipfile.ZipFile(path, "w") as z:
        for part in ("trainingJSON", "testingJSON"):
            for u in ("user1", "user2"):
                body = {"trainingSamples": samples(True),
                        "testingSamples": samples(part == "trainingJSON")}
                z.writestr(f"EMG-EPN612 Dataset/{part}/{u}/{u}.json", json.dumps(body))


def test_epn612_paper_and_user_independent_protocols(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _epn_zip(raw / "EMG-EPN612 Dataset.zip")
    f, info = _run("epn612", raw, tmp_path / "paper")
    sp = info["splits"]
    # train: training users' trainingSamples (2 x 72) + test users' first 10/gesture (2 x 60)
    assert sp["train"]["windows"] == 2 * 72 + 2 * 60
    assert sp["val"]["windows"] == 2 * 72                   # training users' testingSamples
    assert sp["test"]["windows"] == 2 * 6 * 2               # test users' remaining 2/gesture
    assert f["train"]["data"].shape[1:] == (8, 1000)
    assert f["train"].attrs["sampling_rate"] == 200.0
    assert abs(float(f["train"]["data"][:20].std()) - 1.0) < 0.05

    f2, info2 = _run("epn612_xuser", raw, tmp_path / "xuser")
    assert not any(info2["subject_overlap"].values())       # nobody in two splits
    assert info2["splits"]["test"]["windows"] == 2 * 72     # test users' labelled samples


def test_grabmyo_inter_session_split(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    rng = np.random.default_rng(1)
    names = ([f"F{k}" for k in range(1, 17)] + ["U1"] + [f"W{k}" for k in range(1, 7)]
             + ["U2", "U3"] + [f"W{k}" for k in range(7, 13)] + ["U4"])
    for sess in (1, 2, 3):
        for part in (1, 2):
            d = tmp_path / "raw" / f"Session{sess}" / f"session{sess}_participant{part}"
            d.mkdir(parents=True, exist_ok=True)
            for g in (1, 17):
                for t in (1, 7):
                    wfdb.wrsamp(f"session{sess}_participant{part}_gesture{g}_trial{t}",
                                fs=2048, units=["mV"] * 32, sig_name=names,
                                p_signal=_semg(32, 2048 * 5, rng).T * 0.05,
                                fmt=["16"] * 32, write_dir=str(d))
    f, info = _run("grabmyo", tmp_path / "raw", tmp_path / "out")
    sp = info["splits"]
    assert f["train"]["data"].shape[1:] == (28, 2000)
    assert [c.decode() for c in f["train"]["channel_names"][:]][16] == "W1"
    assert sp["test"]["segments"] == 2 * 2 * 2              # session 3: 2 subj x 2 gestures x 2 trials
    assert sp["val"]["segments"] == 2 * 2 * 2               # trial 7 of sessions 1-2
    assert sp["train"]["segments"] == 2 * 2 * 2
    assert sp["train"]["windows"] == 9 * sp["train"]["segments"]   # 5 s, 1 s windows, 0.5 s stride
    assert set(np.unique(f["test"]["label"][:]).tolist()) == {0, 16}


def _db2_zip(path, subject, rng):
    import scipy.io
    fs, seg, rest = 2000, 5 * 2000, 3 * 2000
    emg, lab, rep = [], [], []
    for m in (41, 42):                                       # exercise D labels
        for r in range(1, 7):
            emg.append(_semg(12, rest, rng).T * 1e-5); lab += [0] * rest; rep += [0] * rest
            emg.append(_semg(12, seg, rng).T * 5e-5); lab += [m] * seg; rep += [r] * seg
    buf = io.BytesIO()
    scipy.io.savemat(buf, {"emg": np.concatenate(emg).astype(np.float32),
                           "restimulus": np.array(lab)[:-1, None].astype(np.int8),
                           "rerepetition": np.array(rep)[:-1, None].astype(np.int8)})
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"DB2_s{subject}/S{subject}_E3_A1.mat", buf.getvalue())


def test_ninapro_db2_repetition_split_and_rest(tmp_path):
    rng = np.random.default_rng(2)
    raw = tmp_path / "raw"
    raw.mkdir()
    for s in (1, 2):
        _db2_zip(raw / f"DB2_s{s}.zip", s, rng)
    f, info = _run("ninapro_db2", raw, tmp_path / "out", "--db2-exercises", "D")
    assert not any(info["segment_overlap"].values())
    seg = {s: [x.decode() if isinstance(x, bytes) else x for x in f[s]["segment"][:]]
           for s in f}
    reps = lambda side: {int(x.rsplit("/r", 1)[1]) for x in seg[side]}   # noqa: E731
    assert reps("test") == {2, 5}
    assert reps("train") | reps("val") <= {1, 3, 4, 6}
    labels = np.concatenate([f[s]["label"][:] for s in f])
    counts = np.bincount(labels, minlength=50)
    assert counts[41] > 0 and counts[42] > 0
    assert counts[0] <= max(counts[41], counts[42])          # rest subsampled
    assert f["train"]["data"].shape[1:] == (12, 2000)


def _emg_encoder(tmp_path):
    from physiowave.train.pretrain_main import main as pretrain
    out = tmp_path / "pre"
    assert pretrain(["--config", "pretrain/emg_c1_moe", "--smoke-test",
                     "--max-steps", "2", "--output-dir", str(out)]) == 0
    import yaml
    m = yaml.safe_load(open(out / "config_resolved.yaml"))["model"]
    enc = tmp_path / "enc.pth"
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "export_eeg_pretrained_encoder.py"),
                        "--checkpoint", str(out / "latest.pth"), "--output", str(enc)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return enc, m


@pytest.mark.slow
def test_emg_encoder_transfers_and_probe_trains_the_fresh_frontend(tmp_path):
    from physiowave.emg_c1.downstream import EMGC1Downstream
    enc, m = _emg_encoder(tmp_path)
    kw = dict(in_channels=8, window_samples=1000, sampling_rate=200, patch_samples=25,
              num_classes=6, channel_names=[f"myo{k}" for k in range(1, 9)],
              embed_dim=m["embed_dim"], depth=m["depth"], num_heads=m["num_heads"],
              channel_embed_dim=m["channel_embed_dim"])
    model = EMGC1Downstream(**kw)
    rep = model.load_pretrained(str(enc))
    assert "gate yes" in model.describe_transfer(rep)
    assert not any(k.startswith("wavelet_frontend.") for k in rep["taken"])   # no route
    meta = type("M", (), {"channel_names": kw["channel_names"], "channel_mask": None})()
    assert model(torch.randn(2, 8, 1000), meta)["logits"].shape == (2, 6)

    probe = EMGC1Downstream(**kw, freeze_encoder=True, freeze_scope="pretrained")
    probe.load_pretrained(str(enc))
    trainable = {n.split(".")[0] for n, p in probe.named_parameters() if p.requires_grad}
    assert {"wavelet_frontend", "patch_embed", "head"} <= trainable
    assert "shared_transformer" not in trainable

    ck = torch.load(enc, map_location="cpu", weights_only=False)
    from physiowave.ecg_c1.leads import ecg_vocab_payload
    ck.update(ecg_vocab_payload())
    torch.save(ck, tmp_path / "ecg_like.pth")
    with pytest.raises(SystemExit):
        EMGC1Downstream(**kw).load_pretrained(str(tmp_path / "ecg_like.pth"))


def test_emg_collector_marks_the_validation_choice(tmp_path):
    for name, lr, val, acc in [("epn612_ft", 1e-4, .90, .95), ("epn612_ft_lr3e-5", 3e-5, .92, .93),
                               ("epn612_scratch", 1e-4, .85, .86)]:
        (tmp_path / name).mkdir()
        json.dump({"best_epoch": 3, "select_by": "acc", "best_val": val,
                   "hparams": {"lr": lr},
                   "test": {"acc": acc, "balanced_acc": acc, "macro_f1": acc, "kappa": acc}},
                  open(tmp_path / name / "results.json", "w"))
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "collect_emg_downstream.py"),
                        "--root", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0
    # the 3e-5 run has the better validation score, so it is the one compared
    assert "pretrained - scratch (Acc, each chosen on validation): +0.0700" in r.stdout
    assert "0.9450" in r.stdout                              # the v1 paper row
