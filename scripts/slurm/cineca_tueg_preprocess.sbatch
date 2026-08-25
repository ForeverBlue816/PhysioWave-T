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
# THE PARTITION IS NOT boost_usr_prod, and that is deliberate. Every other
# sbatch here is on Booster because every other one trains. This job decodes
# EDF, filters and resamples: no GPU is touched. Booster nodes are billed by
# GPU-hour and come with four A100s attached, so a CPU-only array there burns
# the project's GPU budget to run scipy -- 64 tasks would hold 256 idle A100s.
#
# dcgp_usr_prod is Leonardo's CPU partition. If this project has no DCGP
# allocation the submission will be rejected, and the fallbacks in order of
# preference are:
#
#   sbatch --partition=lrd_all_serial --cpus-per-task=4 --array=0-63 ...
#       the serial partition; smaller tasks, more of them, no allocation needed
#
#   sbatch --partition=boost_usr_prod --gres=gpu:1 --cpus-per-task=8 ...
#       last resort. It works and it wastes a GPU per task; use a short array.
#
# Check what the project actually has before submitting:  saldo -b
#
# CORES ARE USED, not merely held. The filtering and resampling are
# single-threaded, so without --jobs a task decodes on one core and idles the
# rest. JOBS defaults to the cores the task was given.
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
#SBATCH -p dcgp_usr_prod
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
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

# One thread per worker process, not per task: numpy would otherwise start
# SLURM_CPUS_PER_TASK threads inside each of JOBS processes and oversubscribe
# the node by that product.
export JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# The array bounds are the shard count, so they cannot disagree with it.
N_SHARDS="${N_SHARDS:-$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))}"
if [[ "${N_SHARDS}" -lt 1 ]]; then
    echo "ERROR: could not determine the shard count. Submit with --array." >&2
    exit 1
fi
export SHARD="${SLURM_ARRAY_TASK_ID:-0}/${N_SHARDS}"

echo "============================================================"
echo "  TUEG preprocessing  (job ${SLURM_ARRAY_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?})"
echo "  shard ${SHARD}   cpus ${SLURM_CPUS_PER_TASK:-?}   workers ${JOBS}"
echo "  partition ${SLURM_JOB_PARTITION:-?}"
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
