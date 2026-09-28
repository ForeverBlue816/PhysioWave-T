#!/bin/bash
# =============================================================================
# EEGPT benchmarks (BCIC-IV-2a, BCIC-IV-2b, KaggleERN) on the C1 encoder.
#
#   sbatch --export=ALL,TASK=bcic2a,PRETRAINED=$HOME/hf_release/eeg_c1_encoder.pth \
#          scripts/slurm/cineca_eegpt_bench.sbatch
#
# Runs MODES side by side, one GPU each, on one node: by default ft (the
# pretrained encoder, fine-tuned), scratch (the same model from random
# initialisation -- the control), and probe (encoder frozen -- the mode shaped
# like EEGPT's published rows). The three finish in roughly the time of one.
#
# All three tasks:
#
#   for t in bcic2a bcic2b kaggleern; do
#     sbatch --export=ALL,TASK=$t,PRETRAINED=$HOME/hf_release/eeg_c1_encoder.pth \
#            scripts/slurm/cineca_eegpt_bench.sbatch
#   done
#
# Two encoders side by side (best vs last epoch): add TAG=best / TAG=final so
# their runs land in different directories.
#
# The data must be DOWNLOADED first, on the login node -- compute nodes have no
# internet. Converting it into a split happens here, on first use. See
# EEG/finetune_eegpt_bench.sh.
# =============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240000
#SBATCH --job-name=physiowave_eegpt_bench
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

TASK="${TASK:?set TASK=bcic2a|bcic2b|kaggleern}"
FOLD="${FOLD:-0}"
MODES="${MODES:-ft scratch probe}"
read -r -a MODE_LIST <<< "${MODES}"
export TASK FOLD NUM_GPUS=1 PROGRESS="${PROGRESS:-auto}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================================"
echo "  EEGPT benchmark ${TASK} fold ${FOLD}, modes: ${MODES}  (job ${SLURM_JOB_ID:-?})"
echo "  encoder ${PRETRAINED:-<none>}"
pw_print_layout
echo "============================================================"

# The split is built ONCE, before the modes start, rather than by each of them
# at the same moment into the same directory.
# A plain case block, not $(case ... in x) ...): the unbalanced ")" of a case
# pattern inside a command substitution is a syntax error on older bash.
case "${TASK}" in
    bcic2a) DATA_SUB=bcic2a ;;
    bcic2b) DATA_SUB=bcic2b ;;
    *)      DATA_SUB=ern ;;
esac
DATA_DIR="${DATA_DIR:-${PW_DATA_EEG}/${DATA_SUB}_f${FOLD}}"
if [[ ! -f "${DATA_DIR}/test.h5" ]]; then
    case "${TASK}" in
        bcic2a)    python EEG/bcic_iv2_finetune.py --dataset 2a \
                       --raw-dir "${RAW_DIR:-${PW_DATA_EEG}/bcic_iv2a}" \
                       --out-dir "${DATA_DIR}" --fold "${FOLD}" ;;
        bcic2b)    python EEG/bcic_iv2_finetune.py --dataset 2b \
                       --raw-dir "${RAW_DIR:-${PW_DATA_EEG}/bcic_iv2b}" \
                       --out-dir "${DATA_DIR}" --fold "${FOLD}" ;;
        kaggleern) python EEG/kaggle_ern_finetune.py \
                       --raw-dir "${RAW_DIR:-${PW_DATA_EEG}/kaggle_ern}" \
                       --out-dir "${DATA_DIR}" --fold "${FOLD}" ;;
    esac || { echo "ERROR: building the split failed" >&2; exit 1; }
fi
export DATA_DIR

pids=()
for i in "${!MODE_LIST[@]}"; do
    mode="${MODE_LIST[$i]}"
    log="${PW_CKPT_ROOT}/eegpt_bench/${TASK}_f${FOLD}_${mode}${TAG:+_${TAG}}.log"
    mkdir -p "$(dirname "${log}")"
    echo "  starting ${mode} on GPU ${i}  -> ${log}"
    CUDA_VISIBLE_DEVICES="${i}" MODE="${mode}" \
        bash EEG/finetune_eegpt_bench.sh > "${log}" 2>&1 &
    pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  ${MODE_LIST[$i]}: done"
    else
        echo "  ${MODE_LIST[$i]}: FAILED -- see its log" >&2
        rc=1
    fi
done

echo ""
python scripts/collect_eegpt_bench.py --root "${PW_CKPT_ROOT}/eegpt_bench" || true
exit ${rc}
