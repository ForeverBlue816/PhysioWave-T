#!/bin/bash
# ============================================================================
# EEG C1 multi-route pretraining, as a batch job.
#
#   sbatch scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch
#
# Overrides go through --export, same as the other sbatch scripts here. Note the
# commas: --export takes a comma-separated list, so quote any value containing
# one:
#
#   sbatch --export=ALL,EPOCHS=80,"BATCH_SIZE_BY_ROUTE=E19_256=32,E128_512=8" \
#          scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch
#
# RESUMABLE. OUTPUT_DIR carries no job id and the trainer writes latest.pth
# every epoch with the optimizer, scheduler, AMP scaler, sampler position, RNG
# state and history in it, so RESUME=auto continues the sequence the interrupted
# run would have drawn next rather than restarting the epoch.
#
#   sbatch --export=ALL,RESUME=auto scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch
#
# The corpora are NOT preprocessed here. That step needs mne and belongs under
# $HOME/pwprep; this job reads the manifests it produced.
# ============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=physiowave_eeg_c1_moe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"

# sbatch defaults to --export=ALL, so this job inherits the submitting shell's
# environment -- including a PW_VENV left pinned to $HOME/pwprep by the
# preprocessing step, which is exactly the shell someone submits this from.
if [[ -n "${PW_VENV:-}" && "${PW_VENV}" != "${VENV:-${HOME}/pw}" \
      && "${PW_ALLOW_INHERITED_VENV:-0}" != "1" ]]; then
    echo "NOTE: ignoring inherited PW_VENV=${PW_VENV}; using ${VENV:-${HOME}/pw}." >&2
    unset PW_VENV
fi

# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

export NUM_GPUS="${NUM_GPUS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / NUM_GPUS))}"

# Large caches and checkpoints belong on $FAST, never in $HOME (50 GB quota).
export DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/eeg_c1_corpus}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe}"

echo "============================================================"
echo "  EEG C1 multi-route pretraining  (job ${SLURM_JOB_ID})"
echo "  gpus=${NUM_GPUS}  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  epochs=${EPOCHS:-<script default>}  resume=${RESUME:-<none>}"
pw_print_layout
echo "  DATA_ROOT  =${DATA_ROOT}"
echo "  OUTPUT_DIR =${OUTPUT_DIR}   (resubmit with RESUME=auto to continue)"
echo "============================================================"

bash EEG/pretrain_eeg_c1_moe.sh
_rc=$?

echo ""
echo "============================================================"
echo "  Figures:"
echo "    python scripts/visualize_eeg_pretraining.py \\"
echo "        --run-dir ${OUTPUT_DIR} --checkpoint best.pth --split val --format svg"
echo "  Resume:"
echo "    sbatch --export=ALL,RESUME=auto \\"
echo "        scripts/slurm/cineca_eeg_c1_moe_pretrain.sbatch"
echo "============================================================"
exit "${_rc}"
