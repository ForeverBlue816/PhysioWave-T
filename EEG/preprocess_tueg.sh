#!/bin/bash
# ============================================================================
# TUEG -> the EEG C1 pretraining corpus (route E19_256).
#
#   bash EEG/preprocess_tueg.sh                    # everything, one process
#   SHARD=3/64 bash EEG/preprocess_tueg.sh         # one shard of an array
#
# TUEG is the largest corpus in the mixture and the only one that has to be
# processed in parallel: tens of thousands of EDF files, each read, filtered,
# resampled from whatever rate it was recorded at, and written as its own HDF5.
#
# LOOK BEFORE YOU RUN. The adapter's assumptions about layout and channel
# naming have been verified against a synthetic tree with TUH's conventions, not
# against this copy of the corpus. One command checks them, opens headers only
# and takes seconds:
#
#   INSPECT=200 bash EEG/preprocess_tueg.sh
#
# Read the report before launching the array. What matters in it:
#   * identity rule should be 'filename' for every file. Any 'path' means the
#     filenames do not follow <subject>_s<NNN>_t<NNN>.edf here and the subject
#     ids -- and therefore the train/val split -- are coming from a fallback.
#   * slots filled should be 19 of 19. Anything less is an electrode this
#     adapter cannot name, and it would be silently masked out of every window.
#   * unmatched names should be only EKG / PHOTIC / IBI / BURSTS / SUPPR.
#
# SHARDING is by SUBJECT, not by file: a subject's recordings must all be
# handled by one task, because the train/val side is decided per subject and two
# tasks disagreeing about a subject is the leak the split exists to prevent.
# --split-mode hash follows automatically from --shard.
#
# RESUMABLE. A shard whose HDF5 already exists is reused, so a task that hits
# the walltime loses at most the recording it was in. Resubmit the same array.
#
# DISK. Each window is 19 x 1024 float32 = 78 KB before gzip, ~40 KB after. The
# run prints a projection from the first few hundred recordings before it
# commits to the rest; MAX_RECORDINGS caps the corpus if that number is larger
# than the filesystem can take.
#
# ENVIRONMENT VARIABLES:
#   TUEG_ROOT       raw corpus     (/leonardo_scratch/large/.../TUEG_v2.0.2)
#   DATA_ROOT       corpus root    ($PW_DATA_EEG/eeg_c1_corpus)
#   OUT_DIR         this dataset   ($DATA_ROOT/tueg)
#   SHARD           "I/N"          (unset = one process does everything)
#   INSPECT         N              (report and exit, process nothing)
#   MAINS_HZ                       (60 -- TUH is recorded in Philadelphia)
#   MAX_RECORDINGS  cap            (unset)
#   STRIDE_SECONDS  window stride  (unset = no overlap)
#   RESUME          1 to reuse existing shards (1)
#   VAL_FRACTION / SPLIT_SEED      (0.10 / 42)
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Paths only: reading EDF needs mne, which lives in the preparation venv, and
# activating the training venv here would replace it.
PW_VARS_ONLY=1 source "$(pwd)/scripts/cineca_env.sh"

TUEG_ROOT="${TUEG_ROOT:-/leonardo_scratch/large/userexternal/ychen003/TUH_EEG/TUEG_v2.0.2}"
DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/eeg_c1_corpus}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/tueg}"
MAINS_HZ="${MAINS_HZ:-60}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
RESUME="${RESUME:-1}"
SHARD="${SHARD:-}"
INSPECT="${INSPECT:-}"
MAX_RECORDINGS="${MAX_RECORDINGS:-}"
STRIDE_SECONDS="${STRIDE_SECONDS:-}"

if [[ ! -d "${TUEG_ROOT}" ]]; then
    echo "ERROR: no TUEG at ${TUEG_ROOT}" >&2
    echo "       Set TUEG_ROOT. This script never downloads the corpus -- TUEG" >&2
    echo "       is behind a data use agreement and fetching it on your behalf" >&2
    echo "       would be agreeing to that on your behalf." >&2
    exit 1
fi

if ! python -c "import mne" 2>/dev/null; then
    echo "ERROR: mne is not importable in $(python -c 'import sys;print(sys.prefix)')." >&2
    echo "       Reading EDF needs it, and it lives in the preparation venv:" >&2
    echo "         source \$HOME/pwprep/bin/activate" >&2
    echo "         PW_VARS_ONLY=1 source scripts/cineca_env.sh" >&2
    exit 1
fi

ARGS=(--dataset tueg --root "${TUEG_ROOT}" --out-dir "${OUT_DIR}"
      --mains-hz "${MAINS_HZ}" --val-fraction "${VAL_FRACTION}"
      --split-seed "${SPLIT_SEED}")
[[ -n "${INSPECT}" ]]        && ARGS+=(--inspect "${INSPECT}")
[[ -n "${SHARD}" ]]          && ARGS+=(--shard "${SHARD}")
[[ -n "${MAX_RECORDINGS}" ]] && ARGS+=(--max-recordings "${MAX_RECORDINGS}")
[[ -n "${STRIDE_SECONDS}" ]] && ARGS+=(--stride-seconds "${STRIDE_SECONDS}")
[[ "${RESUME}" == "1" ]]     && ARGS+=(--resume)

echo "============================================================"
echo "  TUEG -> E19_256  (19 x 1024 @ 256 Hz, 152 tokens)"
echo "  raw   ${TUEG_ROOT}"
echo "  out   ${OUT_DIR}"
echo "  mains ${MAINS_HZ} Hz   val ${VAL_FRACTION} (seed ${SPLIT_SEED})"
[[ -n "${SHARD}" ]]   && echo "  shard ${SHARD} (by subject)"
[[ -n "${INSPECT}" ]] && echo "  INSPECT ${INSPECT} -- reporting only, nothing written"
echo "============================================================"

python EEG/preprocess_pretrain_corpus.py "${ARGS[@]}"
_rc=$?

if [[ ${_rc} -eq 0 && -z "${INSPECT}" ]]; then
    echo ""
    echo "When every shard has finished, merge the manifests:"
    echo "  python scripts/build_eeg_c1_manifest.py --corpus-root ${DATA_ROOT} \\"
    echo "      --datasets tueg --allow-missing --check-shards"
fi
exit "${_rc}"
