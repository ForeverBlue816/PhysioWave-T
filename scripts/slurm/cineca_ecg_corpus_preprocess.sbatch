#!/bin/bash
# ============================================================================
# ECG corpus preprocessing, as a SLURM array. One dataset per submission.
#
#   INSPECT first, on a login node, and read the amplitude column:
#     DATASET=mimic_iv_ecg INSPECT=40 bash ECG/preprocess_ecg_corpus.sh
#
#   then, sized to the corpus:
#     sbatch --export=ALL,DATASET=mimic_iv_ecg      --array=0-3  scripts/slurm/cineca_ecg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=code15            --array=0-3  scripts/slurm/cineca_ecg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=icentia11k        --array=0-7  scripts/slurm/cineca_ecg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=heedb             --array=0-15 scripts/slurm/cineca_ecg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=medalcare_xl      --array=0-0  scripts/slurm/cineca_ecg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=georgia           --array=0-0  scripts/slurm/cineca_ecg_corpus_preprocess.sbatch
#     sbatch --export=ALL,DATASET=norwegian_athlete --array=0-0  scripts/slurm/cineca_ecg_corpus_preprocess.sbatch
#
# WHERE THOSE SIZES COME FROM. Measured on real records on a laptop core: a
# 10 s 12-lead record takes ~7 ms to read out of the zip and preprocess
# (~150/s/worker), a 70-minute Icentia11k segment ~40 ms. At 15 workers a task
# that is ~2,000 records/s: MIMIC-IV-ECG's 800k is minutes, HEEDB's 11.6 M
# about 1.5 h on one task. Lustre reads will be slower than a laptop SSD by
# some factor nobody has measured yet -- the arrays above leave room for 5-10x
# inside the 4 h walltime, and a task that runs out resumes where it stopped.
#
# TASK 0 LISTS THE CORPUS and the others wait for its record_list.txt, so the
# tree is walked once. For HEEDB that walk is the slow part; running
#   DATASET=heedb LIST_ONLY=1 bash ECG/preprocess_ecg_corpus.sh
# on a login node first takes it out of the array's walltime.
#
# RESUMABLE. A unit of records is done when its part file exists, so a task
# that hits the walltime is resubmitted with the SAME --array and picks up at
# the first unit without one. Keep RECORDS_PER_UNIT the same across attempts.
#
# AFTER THE ARRAY, merge every dataset that has finished:
#   python scripts/build_eeg_c1_manifest.py --modality ecg \
#       --corpus-root $ECG_ROOT/ecg_c1_corpus --allow-missing --check-shards --jobs 16
# ============================================================================
#SBATCH -A iscrb_wearusfm
#SBATCH -p boost_usr_prod
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --job-name=physiowave_ecg_prep
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -uo pipefail

DATASET="${DATASET:-}"
if [[ -z "${DATASET}" ]]; then
    echo "ERROR: name the corpus:" >&2
    echo "  sbatch --export=ALL,DATASET=mimic_iv_ecg --array=0-15 \\" >&2
    echo "         scripts/slurm/cineca_ecg_corpus_preprocess.sbatch" >&2
    exit 1
fi
export DATASET

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
cd "${PROJECT_DIR}"
# The preparation venv, as for EEG: numpy, scipy, h5py and wfdb, no torch.
if [[ -f "${PWPREP:-${HOME}/pwprep}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PWPREP:-${HOME}/pwprep}/bin/activate"
fi

# One worker per core leaves nothing for h5py's gzip and the zip reads; the
# units are independent, so this is the whole parallelism of a task.
export JOBS="${JOBS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 1 ))}"
export OMP_NUM_THREADS=1
export ECG_ROOT="${ECG_ROOT:-/leonardo_scratch/large/userexternal/ychen003/bio/ecg}"

echo "task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1} of ${DATASET}, ${JOBS} worker(s)"
bash ECG/preprocess_ecg_corpus.sh
