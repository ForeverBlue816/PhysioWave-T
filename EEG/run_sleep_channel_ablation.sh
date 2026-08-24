#!/bin/bash

# ============================================================================
# Sleep-EDF channel-embedding ablation
# ============================================================================
# Six variants against the current strong baseline. Only the channel metadata,
# its encoder, and where the code is injected change; everything else -- the
# waveform, the preprocessing, the split, the schedule, the backbone -- is held
# fixed, and a test pins the legacy parameters to the same values at one seed.
#
#   C0  none    none    the baseline this is measured against
#   C1  id      token   EEGPT-style learned channel-name embedding
#   C2  signed  token   the derivation's geometry instead of its name
#   C3  signed  fold    the same code, biasing only the scale choice
#   C4  signed  dual    both injection sites
#   C5  hybrid  dual    whether a name and a geometry say different things
#
# Plumbing check -- two runs, on the same four-GPU path the real ones take:
#
#   VARIANTS=C0,C4 SEEDS=42 FOLDS=0 NUM_GPUS=4 EPOCHS=2 \
#       bash EEG/run_sleep_channel_ablation.sh
#
# Not five variants times three seeds on one GPU. That is fifteen runs and a
# couple of hours, and one process does not exercise the multi-rank path.
#
# Full (six variants x ten folds x three seeds = 180 runs -- read the note on
# cost below before starting it):
#
#   VARIANTS=C0,C1,C2,C3,C4,C5 SEEDS=42,43,44 FOLDS=0,1,2,3,4,5,6,7,8,9 \
#       NUM_GPUS=4 bash EEG/run_sleep_channel_ablation.sh
#
# COST. At the Sleep-EDF defaults a fold is roughly 20 minutes on four GPUs, so
# the full matrix is on the order of sixty GPU-hours. This script does not
# submit it for you and does not fan out; it runs what you ask for, in order,
# and skips anything already finished. Start with the smoke.
#
# WHAT IS HELD FIXED, and how:
#   * one HDF5 per fold, shared by every variant -- the data cannot differ
#   * SEED is passed to finetune.py, which seeds the sampler, so a given
#     (fold, seed) sees the same batch order in every variant
#   * every legacy module is constructed and initialised before any channel
#     module exists, so the same seed gives every variant the same backbone
#   * every hyper-parameter other than the channel flags comes from
#     EEG/finetune_sleep.sh's defaults, untouched here
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

VARIANTS="${VARIANTS:-C0,C1,C2,C3,C4}"
SEEDS="${SEEDS:-42}"
FOLDS="${FOLDS:-0}"
EPOCHS="${EPOCHS:-}"
NUM_GPUS="${NUM_GPUS:-4}"

DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/sleep_edf_channel}"
SWEEP_ROOT="${SWEEP_ROOT:-${PW_CKPT_ROOT}/sleep_channel_ablation}"
CACHE_DIR="${CACHE_DIR:-${PW_DATA_EEG}/sleep_edf/cache}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# encoding and injection for each variant. Kept as one table so a row cannot be
# defined in two places and drift.
variant_encoding() {
    case "$1" in
        C0) echo "none" ;;   C1) echo "id" ;;     C2) echo "signed" ;;
        C3) echo "signed" ;; C4) echo "signed" ;; C5) echo "hybrid" ;;
        *)  echo "" ;;
    esac
}
variant_injection() {
    case "$1" in
        C0) echo "none" ;;  C1) echo "token" ;; C2) echo "token" ;;
        C3) echo "fold" ;;  C4) echo "dual" ;;  C5) echo "dual" ;;
        *)  echo "" ;;
    esac
}

IFS=',' read -r -a _variants <<< "${VARIANTS}"
IFS=',' read -r -a _seeds    <<< "${SEEDS}"
IFS=',' read -r -a _folds    <<< "${FOLDS}"

for v in "${_variants[@]}"; do
    [[ -n "$(variant_encoding "${v}")" ]] || {
        echo "ERROR: unknown variant '${v}' (C0..C5)" >&2; exit 1; }
done

echo "============================================================"
echo "  Sleep-EDF channel-embedding ablation"
echo "  variants=${VARIANTS}  seeds=${SEEDS}  folds=${FOLDS}"
echo "  gpus=${NUM_GPUS}  epochs=${EPOCHS:-<script default>}"
echo "  data=${DATA_ROOT}"
echo "  out =${SWEEP_ROOT}"
echo "============================================================"

# -- data: one split per fold, built once and shared ----------------------- #
# The decode cache is reused, so this is file assembly and not preprocessing.
# Building per variant would be both slow and wrong: the variants have to see
# the same windows in the same order.
for k in "${_folds[@]}"; do
    d="${DATA_ROOT}/fold${k}"
    if [[ -f "${d}/train.h5" && -f "${d}/val.h5" && -f "${d}/test.h5" ]]; then
        echo "fold ${k}: split present, reused"
        continue
    fi
    echo "fold ${k}: building split from ${CACHE_DIR}"
    [[ "${DRY_RUN}" == "1" ]] && continue
    python EEG/sleep_edf_finetune.py --stage split --split eegpt-fold --fold "${k}" \
        --cache-dir "${CACHE_DIR}" --out-dir "${d}" || {
            echo "fold ${k}: split failed" >&2; exit 1; }
done

# -- runs ------------------------------------------------------------------ #
total=0; ran=0; skipped=0; failed=()
for k in "${_folds[@]}"; do
  for v in "${_variants[@]}"; do
    for s in "${_seeds[@]}"; do
      total=$((total + 1))
      enc="$(variant_encoding "${v}")"
      inj="$(variant_injection "${v}")"
      out="${SWEEP_ROOT}/fold${k}/${v}/seed${s}"

      # A run counts as finished only if it wrote its result file. A directory
      # left by a job that died mid-epoch has a checkpoint and no result, and
      # skipping on the directory alone would silently drop it from the mean.
      if [[ "${FORCE}" != "1" && -s "${out}/test_results.json" ]]; then
          skipped=$((skipped + 1)); continue
      fi

      echo ""
      echo "----- fold ${k}  ${v} (${enc}/${inj})  seed ${s} -----"
      if [[ "${DRY_RUN}" == "1" ]]; then
          echo "  DATA_DIR=${DATA_ROOT}/fold${k} OUTPUT_DIR=${out} \\"
          echo "  CHANNEL_ENCODING=${enc} CHANNEL_INJECTION=${inj} SEED=${s} \\"
          echo "  NUM_GPUS=${NUM_GPUS}${EPOCHS:+ EPOCHS=${EPOCHS}} bash EEG/finetune_sleep.sh"
          continue
      fi
      mkdir -p "${out}"
      # Built as an array and handed to `env`, not written as a command prefix.
      # Bash decides which words are assignments while PARSING, before any
      # expansion, so a conditional prefix like ${EPOCHS:+EPOCHS="${EPOCHS}"}
      # is not an assignment: it expands to the string `EPOCHS=2`, which then
      # becomes the COMMAND NAME. Every run would die with
      # "EPOCHS=2: command not found" the moment EPOCHS was set -- that is, in
      # exactly the smoke run this script tells you to start with.
      _env=(DATA_DIR="${DATA_ROOT}/fold${k}" OUTPUT_DIR="${out}"
            CHANNEL_ENCODING="${enc}" CHANNEL_INJECTION="${inj}"
            SEED="${s}" NUM_GPUS="${NUM_GPUS}")
      [[ -n "${EPOCHS}" ]] && _env+=(EPOCHS="${EPOCHS}")
      if env "${_env[@]}" \
         bash EEG/finetune_sleep.sh 2>&1 | tee "${out}/run.log"; then
          ran=$((ran + 1))
      else
          echo "fold ${k} ${v} seed ${s}: FAILED" >&2
          failed+=("fold${k}/${v}/seed${s}")
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "  ${ran} run(s), ${skipped} skipped as already finished, of ${total}"
[[ ${#failed[@]} -gt 0 ]] && echo "  ${#failed[@]} failed: ${failed[*]}"
echo "  python scripts/collect_sleep_channel_ablation.py ${SWEEP_ROOT}"
echo "============================================================"

[[ "${DRY_RUN}" == "1" ]] || \
    python scripts/collect_sleep_channel_ablation.py "${SWEEP_ROOT}" || true
