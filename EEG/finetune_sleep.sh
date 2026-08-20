#!/bin/bash

# ============================================================================
# Sleep-EDF 5-class sleep staging
# ============================================================================
# Trains the folded wavelet transformer on Sleep-EDFx Sleep Cassette, on the
# preprocessing EEGPT uses, so the result sits next to their published row.
#
# Usage:
#   1. pip install braindecode
#   2. python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf
#   3. bash EEG/finetune_sleep.sh
#
# The comparison, stated up front so it is not read as more than it is:
#
#   EEGPT (NeurIPS'24)   BalAcc 0.6917 +- 0.0069   Kappa 0.6857 +- 0.0019
#
# That number is a *pretrained* encoder, frozen, with a four-layer transformer
# decoder head trained on top, scored on the validation fold it also selects
# on, averaged over ten folds. This script trains from scratch with every
# parameter free and -- under `--split holdout` -- scores a test set that
# nothing selected on. Beating it from scratch would be a real result;
# not beating it is the expected starting point and is the argument for
# pretraining, not against the architecture.
#
# Memory: after the need_weights fix in CrossScaleCAFFN, the whole model's
# fp32 activations at T=3000 are 0.66 GiB at batch 16, 1.27 at 32 and 2.51 at
# 64, against a 64 GiB card. Before it, the cross-scale attention alone wanted
# 17.2 GiB per decomposition level at batch 64 and the run died in step 1.
# BATCH_SIZE=64 is safe; 32 is the default only because more steps per epoch
# suits the 20-epoch budget.
#
# Architecture notes:
#   in_channels 2    Fpz-Cz and Pz-Oz, the two EEG derivations Sleep-EDF has
#   patch_size 50    0.5 s at 100 Hz, the timescale of a sleep spindle
#                    (0.5-2 s) and a K-complex (0.5-1.5 s). 3000 samples give
#                    60 time patches, so the folded model sees 2 x 60 = 120
#                    tokens where the unfolded one would see 480.
#   wave_init pad    every wavelet here is at most 16 taps and is zero-padded
#                    to the kernel rather than stretched. Stretching moves the
#                    half-band cutoff -- sym4 fitted to 16 taps by
#                    interpolation cuts at 0.203*pi instead of 0.5*pi and
#                    fails |H_lo|^2+|H_hi|^2 = 2 by 1.999 out of 2 -- so the
#                    "wavelet initialisation" would not be a filter bank.
#                    coif3 and bior4.4 are absent for that reason: coif3 is 18
#                    taps and does not fit, bior4.4 is biorthogonal and is not
#                    power-complementary by construction.
#
# Comments sit on their own lines on purpose: a `#` after a trailing `\` does
# not comment out anything, it breaks the continuation.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

[[ "${PW_ALLOW_NO_GPU:-0}" == "1" ]] || pw_require_gpu || exit 1
pw_require_python_deps || exit 1

NUM_GPUS="${NUM_GPUS:-4}"
TASK="${TASK:-sleep_edf}"
DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/${TASK}}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.h5}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.h5}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/test.h5}"

OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_${TASK}}"
pw_check_output_dir "${OUTPUT_DIR}" || exit 1

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
    [[ -f "${f}" ]] || {
        echo "ERROR: missing ${f}" >&2
        echo "       build it with: python EEG/sleep_edf_finetune.py --out-dir ${DATA_DIR}" >&2
        exit 1
    }
done

PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
if [[ -n "${PRETRAINED_MODEL}" && -f "${PRETRAINED_MODEL}" ]]; then
    PRETRAINED_ARG=(--pretrained_path "${PRETRAINED_MODEL}")
else
    [[ -n "${PRETRAINED_MODEL}" ]] && \
        echo "WARNING: no checkpoint at ${PRETRAINED_MODEL}; training from scratch." >&2
    PRETRAINED_ARG=()
fi
mkdir -p "${OUTPUT_DIR}"

echo "Sleep-EDF fine-tuning: data=${DATA_DIR} out=${OUTPUT_DIR}"

if [[ "${WARMUP_EPOCHS:-3}" -ge "${EPOCHS:-20}" ]]; then
    _w=$(( ${EPOCHS:-20} / 10 )); [[ "${_w}" -lt 1 ]] && _w=0
    echo "WARNING: WARMUP_EPOCHS >= EPOCHS; using ${_w} (the cosine decay would never start)." >&2
    WARMUP_EPOCHS="${_w}"
fi

# shellcheck disable=SC2086
"${PW_TORCHRUN[@]}" --standalone --nproc_per_node="${NUM_GPUS}" finetune.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --test_file "${TEST_FILE}" \
  ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
  --in_channels "${IN_CHANNELS:-2}" \
  --max_level "${MAX_LEVEL:-3}" \
  --wave_kernel_size "${WAVE_KERNEL_SIZE:-16}" \
  --wavelet_names ${WAVELET_NAMES:-sym4 sym5 db6 sym8 db8} \
  --wave_init_mode "${WAVE_INIT_MODE:-pad}" \
  --use_separate_channel \
  --patch_size "${PATCH_SIZE:-50}" \
  --embed_dim "${EMBED_DIM:-384}" \
  --depth "${DEPTH:-6}" \
  --num_heads "${NUM_HEADS:-6}" \
  --mlp_ratio "${MLP_RATIO:-4.0}" \
  --dropout "${DROPOUT:-0.1}" \
  --norm "${NORM:-rmsnorm}" \
  --ffn "${FFN:-swiglu}" \
  ${QK_NORM:+--qk_norm} \
  --scale_fold "${SCALE_FOLD:-dynamic}" \
  ${FOLD_PATCH_LEN:+--fold_patch_len "${FOLD_PATCH_LEN}"} \
  --fold_synthesis "${FOLD_SYNTHESIS:-3}" \
  ${FOLD_SYNTHESIS_NORM:+--fold_synthesis_norm} \
  ${FOLD_SHARE_CHANNELS:+--fold_share_channels} \
  ${FOLD_SHRINKAGE:+--fold_shrinkage} \
  --fold_scale_dropout "${FOLD_SCALE_DROPOUT:-0.0}" \
  --fold_gamma "${FOLD_GAMMA:-0.1}" \
  --fold_kl "${FOLD_KL:-1e-3}" \
  --use_pos_embed \
  --pos_embed_type 2d \
  --batch_size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-3e-4}" \
  --min_lr "${MIN_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-1e-2}" \
  --grad_clip "${GRAD_CLIP:-1.0}" \
  --use_amp \
  --num_workers "${NUM_WORKERS:-8}" \
  --world_size "${NUM_GPUS}" \
  --scheduler "${SCHEDULER:-cosine}" \
  --warmup_epochs "${WARMUP_EPOCHS:-3}" \
  --num_classes "${NUM_CLASSES:-5}" \
  --pooling mean \
  --head_dropout "${HEAD_DROPOUT:-0.1}" \
  --head_hidden_dim "${HEAD_HIDDEN_DIM:-512}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.1}" \
  --select_by "${SELECT_BY:-balanced_acc}" \
  --patience "${PATIENCE:-0}" \
  --min_delta "${MIN_DELTA:-0.0}" \
  --seed "${SEED:-42}" \
  --output_dir "${OUTPUT_DIR}" \
  ${EXTRA:-}

echo "Done. Results in ${OUTPUT_DIR}"
echo "  test_results.json   the number to report"
echo "  best_model.pth      inspect with scripts/inspect_checkpoint.py"
