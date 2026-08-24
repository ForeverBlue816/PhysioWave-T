#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Is a finished run's result the one this sweep would produce?

    python scripts/check_run_current.py <test_results.json> NAME=VALUE ...

Exit 0  -- it matches, the runner may skip it
Exit 1  -- it is stale or unreadable, and says on stderr which field differs

A sweep runner has to decide whether a directory that already holds a result
may be skipped. "The file exists" is the wrong test, and quietly so: a result
written before a bug fix, or under a different EPOCHS, sits in the table looking
exactly like a current one, and the paired delta attributes the difference to
whatever the sweep was varying. It happened here twice -- an evaluation-sharding
fix landed between two runs of the same sweep, and a `rm -rf $PW_CKPT_ROOT/...`
with PW_CKPT_ROOT unset removed nothing and reported success, so the stale row
survived a deliberate attempt to delete it.

Every NAME is a shell variable the runner sets; they are resolved to their
places in the result's own `provenance` block. Floats compare with a relative
tolerance, since 3e-4 and 0.0003 are the same learning rate.

Only the standard library, so this runs in whichever environment is active.
"""

from __future__ import annotations

import json
import math
import os
import sys

#: A result written by an older evaluation path is not comparable to a current
#: one however well its hyper-parameters match, and that is exactly the case a
#: config comparison cannot see. Version 2 means val and test were scored on the
#: whole set rather than on one rank's shard.
MIN_RESULT_SCHEMA = 2

#: shell name -> where it lives in test_results.json's provenance.
#: "cfg" is provenance.resolved_model_config; "top" is provenance itself.
FIELDS = {
    "EPOCHS": ("cfg", "epochs"),
    "WARMUP_EPOCHS": ("cfg", "warmup_epochs"),
    "BATCH_SIZE": ("cfg", "batch_size"),
    "LR": ("cfg", "lr"),
    "WEIGHT_DECAY": ("cfg", "weight_decay"),
    "DROPOUT": ("cfg", "dropout"),
    "HEAD_DROPOUT": ("cfg", "head_dropout"),
    "LABEL_SMOOTHING": ("cfg", "label_smoothing"),
    "FOLD_KL": ("cfg", "fold_kl"),
    "SELECT_BY": ("cfg", "select_by"),
    "IN_CHANNELS": ("cfg", "in_channels"),
    "CHANNEL_ENCODING": ("top", "channel_encoding"),
    "CHANNEL_INJECTION": ("top", "channel_injection"),
    "SEED": ("top", "seed"),
    "GIT_COMMIT": ("top", "git_commit"),
    "METADATA_HASH": ("top", "metadata_hash"),
}
#: MIN_LR and the rest are not recorded in resolved_model_config, so they cannot
#: be checked here. Naming them keeps a silent no-op from looking like a check.
UNCHECKABLE = ("MIN_LR",)


def same(want: str, got) -> bool:
    if got is None:
        return False
    if isinstance(got, bool):
        return str(got).lower() == want.strip().lower()
    if isinstance(got, (int, float)):
        try:
            w = float(want)
        except ValueError:
            return False
        return math.isclose(w, float(got), rel_tol=1e-9, abs_tol=1e-12)
    return str(got) == want


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    path, pairs = sys.argv[1], sys.argv[2:]

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 1                       # nothing there; the runner runs it
    try:
        with open(path) as f:
            result = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  stale: {path} is unreadable ({exc})", file=sys.stderr)
        return 1

    schema = result.get("result_schema_version", 1)
    if schema < MIN_RESULT_SCHEMA:
        print(f"  stale: {path}", file=sys.stderr)
        print(f"    result_schema_version {schema} < {MIN_RESULT_SCHEMA} -- it "
              f"was produced by an older evaluation path.", file=sys.stderr)
        if result.get("per_class_support"):
            print(f"    scored on {sum(result['per_class_support'])} windows, "
                  f"which for a v1 result is one rank's share of the test set.",
                  file=sys.stderr)
        return 1

    prov = result.get("provenance")
    if not prov:
        print(f"  stale: {path} records no provenance -- it predates it, so "
              f"there is no way to tell what produced it", file=sys.stderr)
        return 1
    cfg = prov.get("resolved_model_config", {})

    differences = []
    for pair in pairs:
        if "=" not in pair:
            continue
        name, want = pair.split("=", 1)
        if name in UNCHECKABLE or name not in FIELDS:
            continue
        where, key = FIELDS[name]
        got = (cfg if where == "cfg" else prov).get(key)
        if not same(want, got):
            differences.append(f"{name}: recorded {got!r}, this sweep wants {want!r}")

    if differences:
        print(f"  stale: {path}", file=sys.stderr)
        for d in differences:
            print(f"    {d}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
