#!/bin/bash
# =============================================================================
# PhysioWave downstream fine-tuning on CINECA Leonardo.
#
#   sbatch --export=ALL,MODALITY=emg,TASK=db5,IN_CHANNELS=16,NUM_CLASSES=53 \
#          scripts/slurm/cineca_finetune.sbatch
#
# Dispatches to the modality's launch script, which already takes every
# hyper-parameter from the environment. Paths come from scripts/cineca_env.sh.
#
# One node: fine-tuning datasets are small enough that a second node spends
# more time in the all-reduce than it saves in compute.
# =============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=physiowave_finetune
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

MODALITY="${MODALITY:-emg}"
case "${MODALITY}" in
    emg|semg)    SCRIPT="EMG/finetune_emg.sh" ;;
    ecg)         SCRIPT="${MULTILABEL:+ECG/shaoxing_multilabel.sh}"
                 SCRIPT="${SCRIPT:-ECG/finetune_ecg.sh}" ;;
    eeg)         SCRIPT="EEG/finetune_eeg.sh" ;;
    *) echo "ERROR: unknown MODALITY '${MODALITY}' (emg|ecg|eeg)" >&2; exit 1 ;;
esac

export NUM_GPUS="${NUM_GPUS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / NUM_GPUS))}"
# Keep the HF cache off $HOME: transformers is imported through the module's
# site-packages and migrates its cache on first use, and $HOME is 50 GB.
export HF_HOME="${HF_HOME:-${PW_FAST_ROOT}/hf}"
mkdir -p "${HF_HOME}"

OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/finetune_${MODALITY}_${TASK:-default}_${SLURM_JOB_ID}}"
export OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "  PhysioWave fine-tuning -- ${SCRIPT}"
echo "  task=${TASK:-<script default>}  epochs=${EPOCHS:-<script default>}"
echo "  gpus=${NUM_GPUS}  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
pw_print_layout
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "============================================================"

bash "${SCRIPT}"

echo "Done. Outputs in ${OUTPUT_DIR}"
