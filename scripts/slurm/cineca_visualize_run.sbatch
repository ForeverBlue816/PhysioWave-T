#!/bin/bash
# ============================================================================
# Draw a C1 pretraining run's figures and collect them into the repository.
#
#   sbatch --export=ALL,MODALITY=ecg scripts/slurm/cineca_visualize_run.sbatch
#   sbatch --export=ALL,MODALITY=emg scripts/slurm/cineca_visualize_run.sbatch
#   sbatch --export=ALL,RUN_DIR=/path/to/run,NAME=my_run scripts/slurm/cineca_visualize_run.sbatch
#
# then, on a login node (compute nodes cannot reach GitHub):
#   git add docs/runs/<NAME> && git commit -m "<NAME>: figures" && git push
#
# A job rather than a login-node command because the figures run the model on
# validation windows and read the corpus -- a few minutes of CPU, which is the
# kind of load a login node kills. CPU only: the visualiser does not use a GPU.
#
#   MODALITY    eeg | ecg | emg: RUN_DIR defaults to $PW_CKPT_ROOT/pretrain_<m>_c1_moe
#   RUN_DIR     the run directory (holds best.pth, metrics_epoch.jsonl)
#   NAME        docs/runs/<NAME>                     (<m>_c1_moe)
#   CHECKPOINT  best.pth (lowest val total loss); latest.pth for the last epoch
#   RECON_WINDOWS  windows per route the reconstruction figures choose from (4)
# ============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --job-name=physiowave_visualize
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}" || exit 1
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"

MODALITY="${MODALITY:-ecg}"
RUN_DIR="${RUN_DIR:-${PW_CKPT_ROOT}/pretrain_${MODALITY}_c1_moe}"
NAME="${NAME:-${MODALITY}_c1_moe}"
CHECKPOINT="${CHECKPOINT:-best.pth}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "visualising ${RUN_DIR} (${CHECKPOINT}) -> docs/runs/${NAME}"
[[ -f "${RUN_DIR}/${CHECKPOINT}" ]] || { echo "ERROR: no ${RUN_DIR}/${CHECKPOINT}" >&2; exit 1; }

python scripts/visualize_eeg_pretraining.py --run-dir "${RUN_DIR}" \
    --checkpoint "${CHECKPOINT}" --format png \
    --recon-windows "${RECON_WINDOWS:-4}" --threads "${OMP_NUM_THREADS}" || exit 1
bash scripts/collect_pretrain_run.sh "${RUN_DIR}" "${NAME}"
