"""Deciding whether a finished run may be skipped.

"The result file exists" is the wrong test, and quietly so. Two ways it failed
here in one afternoon:

  * an evaluation-sharding fix landed between two runs of the same sweep, and
    the older result -- identical hyper-parameters, different code path -- sat
    in the table looking current while the paired delta blamed the difference
    on the channel flags;
  * `rm -rf $PW_CKPT_ROOT/...` with PW_CKPT_ROOT unset in the interactive shell
    expanded to a path that does not exist, removed nothing, and exited 0, so a
    deliberate attempt to delete the stale result reported success.

The runner now asks what produced the result, not whether it is there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECK = os.path.join(_HERE, "scripts", "check_run_current.py")

SWEEP = ["EPOCHS=10", "WARMUP_EPOCHS=1", "BATCH_SIZE=32", "LR=3e-4",
         "MIN_LR=1e-6", "WEIGHT_DECAY=1e-2", "DROPOUT=0.1", "HEAD_DROPOUT=0.1",
         "LABEL_SMOOTHING=0.1", "FOLD_KL=1e-3", "SELECT_BY=balanced_acc",
         "CHANNEL_ENCODING=id", "CHANNEL_INJECTION=token", "SEED=42"]


def _result(**over):
    r = {
        "result_schema_version": 2,
        "per_class_support": [2259, 1175, 5156, 1776, 2051],
        "provenance": {
            "channel_encoding": "id", "channel_injection": "token", "seed": 42,
            "resolved_model_config": {
                "epochs": 10, "warmup_epochs": 1, "batch_size": 32, "lr": 0.0003,
                "weight_decay": 0.01, "dropout": 0.1, "head_dropout": 0.1,
                "label_smoothing": 0.1, "fold_kl": 0.001,
                "select_by": "balanced_acc"},
        },
    }
    for k, v in over.items():
        if k in ("channel_encoding", "channel_injection", "seed"):
            r["provenance"][k] = v
        elif k in r:
            r[k] = v
        else:
            r["provenance"]["resolved_model_config"][k] = v
    return r


def _check(tmp_path, result, args=SWEEP):
    p = tmp_path / "test_results.json"
    if result is not None:
        p.write_text(json.dumps(result))
    out = subprocess.run([sys.executable, _CHECK, str(p), *args],
                         capture_output=True, text=True)
    return out.returncode, out.stderr


def test_a_matching_result_may_be_skipped(tmp_path):
    rc, _ = _check(tmp_path, _result())
    assert rc == 0


def test_a_missing_result_is_run(tmp_path):
    rc, _ = _check(tmp_path, None)
    assert rc == 1


def test_an_older_evaluation_path_is_stale_despite_matching_config(tmp_path):
    """The case that actually happened: every hyper-parameter agreed.

    Only the code differed, so nothing in `provenance` could show it. The
    schema version is the lever that can.
    """
    stale = _result()
    del stale["result_schema_version"]                 # v1 results carry none
    stale["per_class_support"] = [558, 295, 1284, 457, 511]   # one rank's quarter
    rc, err = _check(tmp_path, stale)
    assert rc == 1
    assert "result_schema_version" in err and "3105" in err


@pytest.mark.parametrize("field,value", [
    ("epochs", 20), ("batch_size", 64), ("select_by", "kappa"),
    ("dropout", 0.2), ("fold_kl", 0.0),
    ("channel_encoding", "signed"), ("channel_injection", "dual"), ("seed", 43),
])
def test_any_changed_field_makes_it_stale(tmp_path, field, value):
    rc, err = _check(tmp_path, _result(**{field: value}))
    assert rc == 1 and field.upper() in err


def test_equal_floats_written_differently_still_match(tmp_path):
    """3e-4 and 0.0003 are the same learning rate.

    A string comparison would re-run the entire sweep on every invocation, and
    the check would be quietly useless rather than loudly wrong.
    """
    rc, _ = _check(tmp_path, _result(lr=0.0003),
                   [a if not a.startswith("LR=") else "LR=0.00030" for a in SWEEP])
    assert rc == 0


def test_a_result_without_provenance_is_stale(tmp_path):
    r = _result()
    del r["provenance"]
    rc, err = _check(tmp_path, r)
    assert rc == 1 and "provenance" in err


def test_an_unreadable_result_is_stale_not_a_crash(tmp_path):
    p = tmp_path / "test_results.json"
    p.write_text("{not json")
    out = subprocess.run([sys.executable, _CHECK, str(p), *SWEEP],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "unreadable" in out.stderr


def test_every_sweep_variable_the_runners_pass_is_checkable():
    """A hyper-parameter applied to a run but absent from the check is a hole.

    _SWEEP_VARS is turned into the check's arguments by the runners, so any
    name there must either be compared or be listed as uncheckable on purpose.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("chk", _CHECK)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)

    import re
    for runner in ("EEG/run_sleep_channel_ablation.sh",
                   "EEG/run_p300_channel_ablation.sh"):
        src = open(os.path.join(_HERE, runner)).read()
        m = re.search(r"_SWEEP_VARS=\(([^)]*)\)", src)
        assert m, f"{runner} no longer defines _SWEEP_VARS"
        names = m.group(1).split()
        unknown = [n for n in names
                   if n not in chk.FIELDS and n not in chk.UNCHECKABLE]
        assert not unknown, (
            f"{runner} passes {unknown} to every run, but check_run_current.py "
            f"neither compares them nor lists them in UNCHECKABLE")
