#!/bin/bash
# ============================================================================
# Multimodal benchmark preprocessing: MPDB and SeizeIT2 -> aligned windows.
#
#   sbatch scripts/slurm/cineca_fusion_prep.sbatch                       # both
#   sbatch --export=ALL,TASKS=seizeit2,SUBJECTS='sub-0(0[1-9]|[1-3][0-9])' \
#          scripts/slurm/cineca_fusion_prep.sbatch                       # a subset first
#
# Reads $FUSION_ROOT/raw/<task> (scripts/download_fusion_data.sh, login node)
# and writes $FUSION_ROOT/c1/<task>/{train,val,test}.h5 + split.json.
# MPDB needs mne in the venv (pip install mne, on a login node).
#
#   TASKS        "mpdb seizeit2"
#   SUBJECTS     seizeit2: regex over sub-XXX (default all)
#   BG_PER_HOUR  seizeit2 background windows per recorded hour (30)
#   JOBS         worker processes (16)
# ============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --job-name=physiowave_fusion_prep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
# shellcheck disable=SC1091
source "${PROJECT_DIR}/scripts/cineca_env.sh"
cd "${PROJECT_DIR}"

FUSION_ROOT="${FUSION_ROOT:-/leonardo_scratch/large/userexternal/ychen003/bio/fusion}"
TASKS="${TASKS:-mpdb seizeit2}"
JOBS="${JOBS:-16}"
rc=0
for task in ${TASKS}; do
    raw="${FUSION_ROOT}/raw/${task}"; out="${FUSION_ROOT}/c1/${task}"
    if [[ ! -d "${raw}" ]]; then
        echo "${task}: ${raw} missing -- on a login node: bash scripts/download_fusion_data.sh ${task}"
        rc=1; continue
    fi
    echo "==> ${task}: ${raw} -> ${out}  ($(date +%H:%M))"
    case "${task}" in
        mpdb)
            python Fusion/mpdb_prep.py --raw-dir "${raw}" --out-dir "${out}" || rc=1 ;;
        seizeit2)
            python Fusion/seizeit2_prep.py --raw-dir "${raw}" --out-dir "${out}" \
                --jobs "${JOBS}" --bg-per-hour "${BG_PER_HOUR:-30}" \
                ${SUBJECTS:+--subjects "${SUBJECTS}"} || rc=1 ;;
        *) echo "unknown task ${task}"; rc=1 ;;
    esac
done
exit ${rc}
