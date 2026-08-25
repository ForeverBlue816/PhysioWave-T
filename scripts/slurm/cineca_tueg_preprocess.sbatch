#!/bin/bash
# ============================================================================
# TUEG preprocessing, as a SLURM array.
#
#   INSPECT first, on a login node, and read the report:
#     INSPECT=200 bash EEG/preprocess_tueg.sh
#
#   then:
#     sbatch --array=0-63 scripts/slurm/cineca_tueg_preprocess.sbatch
#
# The array index IS the shard index and --array's length is the shard count,
# so the two cannot drift apart. 64 tasks is a starting point; TUEG's per-file
# cost is dominated by reading and resampling, so tasks scale nearly linearly.
#
# CPU ONLY. No GPU is requested: this is EDF decoding, filtering and resampling.
# Asking for one would queue behind GPU work for no reason.
#
# RESUMABLE. A shard whose HDF5 exists is reused, so resubmitting the same array
# after a walltime finishes what is left. Sharding is by subject, so tasks never
# touch each other's files and the train/val side of a subject is the same in
# every task.
#
# AFTER THE ARRAY, merge -- and only then is the corpus usable:
#   python scripts/build_eeg_c1_manifest.py --corpus-root $DATA_ROOT \
#       --datasets tueg --allow-missing --check-shards
# ============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --job-name=physiowave_tueg_prep
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"

# The preparation venv, not the training one: this step needs mne and needs no
# CUDA. PW_VARS_ONLY inside preprocess_tueg.sh keeps cineca_env.sh from
# replacing it.
if [[ -f "${PWPREP:-${HOME}/pwprep}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PWPREP:-${HOME}/pwprep}/bin/activate"
else
    echo "ERROR: no preparation venv at ${PWPREP:-${HOME}/pwprep}." >&2
    echo "       It is the one with mne; training's venv does not have it." >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

# The array bounds are the shard count, so they cannot disagree with it.
N_SHARDS="${N_SHARDS:-$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))}"
if [[ "${N_SHARDS}" -lt 1 ]]; then
    echo "ERROR: could not determine the shard count. Submit with --array." >&2
    exit 1
fi
export SHARD="${SLURM_ARRAY_TASK_ID:-0}/${N_SHARDS}"

echo "============================================================"
echo "  TUEG preprocessing  (job ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?})"
echo "  shard ${SHARD}   cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "  python $(python -c 'import sys;print(sys.prefix)')"
echo "============================================================"

bash EEG/preprocess_tueg.sh
_rc=$?

echo ""
echo "============================================================"
echo "  task ${SLURM_ARRAY_TASK_ID:-?} exit ${_rc}"
echo "  When every task has finished:"
echo "    python scripts/build_eeg_c1_manifest.py \\"
echo "        --corpus-root \${DATA_ROOT:-\${PW_DATA_EEG}/eeg_c1_corpus} \\"
echo "        --datasets tueg --allow-missing --check-shards"
echo "============================================================"
exit "${_rc}"
