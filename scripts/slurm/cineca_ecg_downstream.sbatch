#!/bin/bash
# ============================================================================
# ECG C1 downstream: one task, all three modes in parallel, one GPU each.
#
#   sbatch --export=ALL,TASK=ptbxl    scripts/slurm/cineca_ecg_downstream.sbatch
#   sbatch --export=ALL,TASK=cpsc2018 scripts/slurm/cineca_ecg_downstream.sbatch
#   sbatch --export=ALL,TASK=chapman  scripts/slurm/cineca_ecg_downstream.sbatch
#   sbatch --export=ALL,TASK=ptbxl_super scripts/slurm/cineca_ecg_downstream.sbatch
#
# Builds the split first if it is not there (ECG/ecg_downstream_prep.py, CPU
# only, no network: the raw data comes from scripts/download_ecg_downstream.sh
# on a login node), then runs MODES on GPUs 0, 1, 2:
#   ft       pretrained encoder, fine-tuned          (the row to report)
#   probe    pretrained encoder frozen, head only
#   scratch  same model, random initialisation       (the control)
# and finally prints the table of every finished run
# (scripts/collect_ecg_downstream.py).
#
#   PRETRAINED  $PW_CKPT_ROOT/pretrain_ecg_c1_moe/best.pth
#   MODES       "ft probe scratch"
#   TAG, LR, EPOCHS, SEED, SET   passed to ECG/finetune_ecg_c1.sh
# ============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240000
#SBATCH --job-name=physiowave_ecg_downstream
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

TASK="${TASK:?set TASK=ptbxl|ptbxl_super|cpsc2018|chapman}"
MODES="${MODES:-ft probe scratch}"
read -r -a MODE_LIST <<< "${MODES}"
ECG_ROOT="${ECG_ROOT:-/leonardo_scratch/large/userexternal/ychen003/bio/ecg}"
case "${TASK}" in
    ptbxl|ptbxl_super) RAW="${ECG_ROOT}/downstream/raw/ptbxl" ;;
    cpsc2018)          RAW="${ECG_ROOT}/downstream/raw/cpsc2018" ;;
    chapman)           RAW="${ECG_ROOT}/downstream/raw/chapman" ;;
    *) echo "ERROR: unknown TASK ${TASK}" >&2; exit 1 ;;
esac
export TASK ECG_ROOT NUM_GPUS=1 PROGRESS="${PROGRESS:-log}"
export DATA_DIR="${DATA_DIR:-${ECG_ROOT}/downstream/c1/${TASK}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "============================================================"
echo "  ECG C1 downstream ${TASK}, modes: ${MODES}  (job ${SLURM_JOB_ID:-?})"
echo "  encoder ${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_ecg_c1_moe/best.pth}"
echo "  raw ${RAW}  ->  split ${DATA_DIR}"
echo "============================================================"

if [[ ! -f "${DATA_DIR}/split.json" ]]; then
    [[ -f "${RAW}/.complete" ]] || {
        echo "ERROR: ${RAW} is not downloaded (no .complete). On a login node:" >&2
        echo "  bash scripts/download_ecg_downstream.sh ${TASK%_super}" >&2
        exit 1; }
    python ECG/ecg_downstream_prep.py --task "${TASK}" --raw-dir "${RAW}" \
        --out-dir "${DATA_DIR}" --jobs "$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))" \
        || { echo "ERROR: building the split failed" >&2; exit 1; }
fi

pids=()
for i in "${!MODE_LIST[@]}"; do
    mode="${MODE_LIST[$i]}"
    log="${PW_CKPT_ROOT}/ecg_downstream/${TASK}_${mode}${TAG:+_${TAG}}.log"
    mkdir -p "$(dirname "${log}")"
    echo "  starting ${mode} on GPU ${i}  -> ${log}"
    CUDA_VISIBLE_DEVICES="${i}" MODE="${mode}" \
        bash ECG/finetune_ecg_c1.sh > "${log}" 2>&1 &
    pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  ${MODE_LIST[$i]}: done"
    else echo "  ${MODE_LIST[$i]}: FAILED -- see its log" >&2; rc=1; fi
done

echo ""
python scripts/collect_ecg_downstream.py --root "${PW_CKPT_ROOT}/ecg_downstream" \
    --json "${PW_CKPT_ROOT}/ecg_downstream/summary.json" \
    | tee "${PW_CKPT_ROOT}/ecg_downstream/summary.txt" || true
exit ${rc}
