"""A shard whose header survived a write its data did not.

HDF5 lays the object header down before any chunk, so a preprocessing task
killed part-way through leaves a file at the final path whose shape and
attributes read back perfectly and whose windows raise "inflate() failed". A
metadata-only check passes it, the manifest carries it, and the first training
epoch that touches that window takes every rank down. These tests pin the three
places that has to be caught.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _corrupt_last_chunk(path):
    """Overwrite one chunk's bytes in place, leaving the file's length alone.

    Truncating the file instead can cost HDF5 the chunk index and make it
    unopenable, which is a different and much louder failure. This reproduces
    the quiet one.
    """
    import h5py
    with h5py.File(path, "r") as f:
        n = int(f["data"].shape[0])
        info = f["data"].id.get_chunk_info(n - 1)
        off, size = info.byte_offset, info.size
    with open(path, "r+b") as fh:
        fh.seek(off + 8)
        fh.write(b"\xff" * max(1, size - 16))
    return n


@pytest.fixture
def corpus_with_one_bad_shard(tmp_path):
    import glob
    from physiowave.eeg_c1.entry import build_smoke_corpus

    root = str(tmp_path / "corpus")
    build_smoke_corpus(root, subjects=3, recordings=1, windows=4)
    shards = sorted(glob.glob(os.path.join(root, "*", "shards", "*.h5")))
    victim = shards[len(shards) // 2]
    n = _corrupt_last_chunk(victim)
    return root, victim, n


def _merge(root, *extra):
    return subprocess.run(
        [sys.executable, "scripts/build_eeg_c1_manifest.py",
         "--corpus-root", root, "--allow-missing", *extra],
        cwd=ROOT, capture_output=True, text=True)


def test_a_corrupt_chunk_still_reports_its_shape(corpus_with_one_bad_shard):
    """The premise. If this ever fails, the metadata check was enough."""
    import h5py
    _, victim, n = corpus_with_one_bad_shard
    with h5py.File(victim, "r") as f:
        assert int(f["data"].shape[0]) == n            # header intact
        with pytest.raises(OSError):
            f["data"][n - 1]                           # data is not


def test_meta_level_misses_it_and_full_level_catches_it(corpus_with_one_bad_shard):
    root, victim, _ = corpus_with_one_bad_shard
    lenient = _merge(root, "--check-shards", "--check-level", "meta", "--jobs", "2")
    assert lenient.returncode == 0, lenient.stderr[-2000:]

    strict = _merge(root, "--check-shards", "--check-level", "full", "--jobs", "2")
    assert strict.returncode == 1
    assert victim in strict.stderr
    assert "unreadable" in strict.stderr


def test_ends_level_catches_a_bad_tail(corpus_with_one_bad_shard):
    """Truncation takes the tail, so the cheap level has to read it."""
    root, victim, _ = corpus_with_one_bad_shard
    r = _merge(root, "--check-shards", "--check-level", "ends", "--jobs", "2")
    assert r.returncode == 1
    assert victim in r.stderr


def test_drop_unreadable_writes_manifests_without_it(corpus_with_one_bad_shard):
    root, victim, _ = corpus_with_one_bad_shard
    r = _merge(root, "--check-shards", "--check-level", "full",
               "--drop-unreadable", "--jobs", "2")
    assert r.returncode == 0, r.stderr[-2000:]

    for split in ("train", "val"):
        rows = [json.loads(l) for l
                in open(os.path.join(root, "merged", f"manifest_{split}.jsonl"))]
        assert all(row["path"] != victim for row in rows)

    report = os.path.join(root, "merged", "unreadable_shards.jsonl")
    listed = [json.loads(l) for l in open(report)]
    assert [row["path"] for row in listed] == [victim]

    # The summary must describe the manifests that were written, not the ones
    # that would have been written had every shard been readable.
    summary = json.load(open(os.path.join(root, "merged", "corpus_summary.json")))
    assert summary["total_train_windows"] == sum(
        json.loads(l)["n_windows"]
        for l in open(os.path.join(root, "merged", "manifest_train.jsonl")))


def test_the_read_error_names_the_shard(corpus_with_one_bad_shard):
    """A 16-rank traceback that does not say which of 95,000 files is useless."""
    from physiowave.eeg_c1.data import CorpusIndex, EEGWindowDataset

    root, victim, n = corpus_with_one_bad_shard
    _merge(root, "--check-shards", "--check-level", "meta", "--jobs", "2")
    for split in ("train", "val"):
        path = os.path.join(root, "merged", f"manifest_{split}.jsonl")
        index = CorpusIndex.from_manifest(path)
        hit = [s for s in index.shards if s.path == victim]
        if hit:
            ds = EEGWindowDataset(index, hit[0].dataset_id)
            i = [k for k, s in enumerate(ds.shards) if s.path == victim][0]
            last = int(ds.offsets[i]) + ds.shards[i].n_windows - 1
            with pytest.raises(OSError) as exc:
                ds[last]
            assert victim in str(exc.value)
            assert "build_eeg_c1_manifest" in str(exc.value)
            return
    pytest.fail("the corrupted shard reached neither manifest")


def test_write_shard_publishes_atomically(tmp_path):
    """A write that dies part-way must not leave a file at the final path."""
    from physiowave.eeg_c1 import preprocess as pp
    from physiowave.eeg_c1.routes import ROUTES

    route = ROUTES["E19_256"]
    path = str(tmp_path / "shard.h5")
    windows = np.zeros((2, route.n_channels, route.window_samples), np.float32)
    kw = dict(route=route, dataset_id="tueg",
              channel_names=["x"] * route.n_channels,
              channel_ids=list(range(route.n_channels)),
              valid=np.ones(route.n_channels, bool),
              subject_ids=["s"] * 2, recording_ids=["r"] * 2,
              window_starts=[0.0, 4.0], source_rate=256.0)

    # An un-serialisable provenance fails at the LAST statement inside the open
    # file, so every dataset is on disk and the publish is the only thing that
    # does not happen -- the same shape as a task killed near the end.
    with pytest.raises(TypeError):
        pp.write_shard(path, windows, provenance={"bad": object()}, **kw)
    assert not os.path.exists(path), "a failed write left a file at the final path"
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")], \
        "the abandoned temporary was left behind"

    row = pp.write_shard(path, windows, provenance={}, **kw)
    assert os.path.isfile(path) and row["n_windows"] == 2
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


def test_resume_does_not_count_a_bad_shard_as_done(corpus_with_one_bad_shard):
    """Otherwise every later --resume accepts it too, and it never gets rebuilt."""
    import glob
    import h5py
    from EEG.preprocess_pretrain_corpus import _shard_data_is_readable

    root, victim, n = corpus_with_one_bad_shard
    with h5py.File(victim, "r") as f:
        assert not _shard_data_is_readable(f, n)

    good = [p for p in sorted(glob.glob(os.path.join(root, "*", "shards", "*.h5")))
            if p != victim]
    with h5py.File(good[0], "r") as f:
        assert _shard_data_is_readable(f, int(f["data"].shape[0]))
