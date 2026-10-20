#!/bin/bash
# ============================================================================
# EEG C1 downstream on Sleep-EDF and PhysioNet P300: every task x mode in ONE
# job, one node, four GPUs (EEG/finetune_eeg_c1.sh). GPU i takes runs i, i+4,
# ... of the TASKS x MODES list, then the table of every finished run
# (scripts/collect_eeg_downstream.py: per mode, the run with the best
# VALIDATION score is the one compared).
#
#   sbatch scripts/slurm/cineca_eeg_downstream.sbatch
#
#   TASKS       "sleep_edf physio_p300"
#   MODES       "ft@1e-4 ft@3e-5 probe@1e-3 scratch"   MODE or MODE@LR:
#                 ft       pretrained encoder fine-tuned, layer decay 0.75
#                 probe    pretrained encoder frozen; frontend, pool, head train
#                 scratch  same model, random initialisation (config lr)
#   EPOCHS      20 (the configs carry EEGPT's 40 / 100)
#   PRETRAINED  $PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth
#   TAG, SEED, SET, EXTRA, LAYER_DECAY   passed through
#
# The splits are built beforehand (they need mne, the P300 download a login
# node); a missing split stops that run with the command that builds it. A run
# whose results.json has a test block is skipped: resubmit to finish what the
# walltime cut off. FORCE=1 reruns everything.
# ============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400000
#SBATCH --job-name=physiowave_eeg_downstream
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"
TASKS="${TASKS:-sleep_edf physio_p300}"
MODES="${MODES:-ft@1e-4 ft@3e-5 probe@1e-3 scratch}"
N_GPUS="${N_GPUS:-4}"
OUT_ROOT="${OUT_ROOT:-${PW_CKPT_ROOT}/eeg_downstream}"
export EPOCHS="${EPOCHS:-20}" NUM_GPUS=1 PROGRESS="${PROGRESS:-log}" OUT_ROOT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NUM_WORKERS="${NUM_WORKERS:-4}"
RUNS=()
for t in ${TASKS}; do for m in ${MODES}; do RUNS+=("${t}:${m}"); done; done
mkdir -p "${OUT_ROOT}"
echo "============================================================"
echo "  EEG C1 downstream  (job ${SLURM_JOB_ID:-?}, 1 node, ${N_GPUS} GPUs)"
echo "  tasks   ${TASKS}"
echo "  modes   ${MODES}   epochs ${EPOCHS}"
echo "  encoder ${PRETRAINED:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe/best.pth}"
echo "  output  ${OUT_ROOT}"
echo "============================================================"

done_already() {   # results.json with a test block
    python - "$1/results.json" <<'PY' 2>/dev/null
import json, sys
sys.exit(0 if "test" in json.load(open(sys.argv[1])) else 1)
PY
}
worker() {
    local gpu="$1" i spec task token mode lr tag out rc=0
    for ((i = gpu; i < ${#RUNS[@]}; i += N_GPUS)); do
        spec="${RUNS[$i]}"; task="${spec%%:*}"; token="${spec#*:}"
        mode="${token%%@*}"; lr=""; [[ "${token}" == *@* ]] && lr="${token#*@}"
        tag="${lr:+lr${lr}}${TAG:+${lr:+_}${TAG}}"
        out="${OUT_ROOT}/${task}_${mode}${tag:+_${tag}}"
        if [[ "${FORCE:-0}" != "1" ]] && done_already "${out}"; then
            echo "[gpu ${gpu}] ${task} ${token}: done, skipped"; continue
        fi
        echo "[gpu ${gpu}] ${task} ${token}: start $(date +%H:%M)"
        if CUDA_VISIBLE_DEVICES="${gpu}" TASK="${task}" MODE="${mode}" LR="${lr}" \
                TAG="${tag}" OUTPUT_DIR="${out}" bash EEG/finetune_eeg_c1.sh \
                > "${out}.log" 2>&1; then
            echo "[gpu ${gpu}] ${task} ${token}: done $(date +%H:%M)"
        else
            echo "[gpu ${gpu}] ${task} ${token}: FAILED -- see ${out}.log"; rc=1
        fi
    done
    return ${rc}
}
pids=()
for ((g = 0; g < N_GPUS && g < ${#RUNS[@]}; g++)); do worker "${g}" & pids+=($!); done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
python scripts/collect_eeg_downstream.py --root "${OUT_ROOT}" \
    --markdown "${OUT_ROOT}/summary.md" | tee "${OUT_ROOT}/summary.txt"
exit ${rc}
