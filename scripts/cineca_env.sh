# shellcheck shell=bash
# =============================================================================
# PhysioWave -- shared environment for CINECA Leonardo.
#
# This file is *sourced*, never executed:
#
#     PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
#     source "${PROJECT_DIR}/scripts/cineca_env.sh"
#
# Fixed layout on Leonardo
# ------------------------
#   code         $HOME/PhysioWave-T           backed up, 50 GB quota
#   venv         $HOME/pw                     built on top of the cineca-ai module
#   checkpoints  $FAST/yanlchen/runs          permanent, project-shared, low latency
#   caches       $FAST/yanlchen/cache         SSL operators + preprocessed windows
#   manifests    $FAST/yanlchen/manifests
#   data         $SCRATCH/bio/{eeg,ecg,emg}   HDF5 training corpora
#
# WARNING: $SCRATCH is *temporary* -- files are purged 40 days after creation.
# The corpora under $SCRATCH/bio are reproducible from the raw archives, but
# nothing else may live there.  Anything you cannot rebuild belongs in $FAST.
#
# Every path below is an environment variable with a default, so a single
# `sbatch --export=ALL,PW_CKPT_ROOT=...` overrides it without editing this file.
# =============================================================================

# --------------------------------------------------------------------------- #
# 0. Where are we?
#
# $FAST only exists on Leonardo.  Off-cluster the module/venv steps are skipped
# and every path falls back to a repository-relative default, so the same launch
# scripts work on a laptop without a second code path.
# --------------------------------------------------------------------------- #
PW_ON_CINECA=0
[[ -n "${FAST:-}${PW_FAST:-}" ]] && PW_ON_CINECA=1
export PW_ON_CINECA

PROJECT_DIR="${PROJECT_DIR:-${HOME}/PhysioWave-T}"
[[ -d "${PROJECT_DIR}" ]] || PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --------------------------------------------------------------------------- #
# 1. Modules
#
# profile/deeplrn + cineca-ai provide a PyTorch built against Leonardo's A100s
# and its NCCL/InfiniBand stack.  Two things must NOT be done here:
#
#   * `export PYTHONPATH=""` -- cineca-ai injects its site-packages through
#     PYTHONPATH.  Clearing it makes `import torch` fail on the compute node.
#   * `module load cuda/... gcc/...` on top -- cineca-ai already pins a matching
#     toolchain; loading another CUDA silently mixes runtimes.
# --------------------------------------------------------------------------- #
PW_CINECA_AI="${PW_CINECA_AI:-cineca-ai/4.3.0}"

# Is a module with this name (any version) already in the environment?
# $LOADEDMODULES is the colon-separated list the module system maintains.
pw_module_is_loaded() {
    local want="${1%%/*}" entry
    local IFS=':'
    for entry in ${LOADEDMODULES:-}; do
        [[ "${entry%%/*}" == "${want}" ]] && return 0
    done
    return 1
}

if [[ "${PW_ON_CINECA}" -eq 1 ]]; then
    if ! type -t module >/dev/null 2>&1; then
        # Non-login shells do not always have the module function defined.
        # shellcheck disable=SC1091
        [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    fi

    # Loading a module that is already loaded is not a no-op: it re-resolves the
    # whole dependency chain and aborts on the first version conflict (typically
    # bzip2, pulled in by python/3.11). So load only what is missing, and let
    # PW_SKIP_MODULES=1 bypass the step entirely for an already-set-up shell.
    if [[ "${PW_SKIP_MODULES:-0}" == "1" ]]; then
        :
    elif pw_module_is_loaded "${PW_CINECA_AI}"; then
        :
    else
        pw_module_is_loaded profile || module load profile/deeplrn
        module load "${PW_CINECA_AI}"
    fi

    # ----------------------------------------------------------------------- #
    # 2. Virtualenv (created with --system-site-packages on top of cineca-ai)
    # ----------------------------------------------------------------------- #
    PW_VENV="${PW_VENV:-${VENV:-${HOME}/pw}}"
    if [[ "${VIRTUAL_ENV:-}" == "${PW_VENV}" ]]; then
        :                                   # already active; re-sourcing stacks PATH
    elif [[ -f "${PW_VENV}/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${PW_VENV}/bin/activate"
    else
        echo "ERROR: virtualenv not found at ${PW_VENV}." >&2
        echo "       Create it on the login node with the cineca-ai module loaded:" >&2
        echo "         module load profile/deeplrn ${PW_CINECA_AI}" >&2
        echo "         python -m venv ${PW_VENV} --system-site-packages" >&2
        return 1 2>/dev/null || exit 1
    fi
else
    PW_VENV="${PW_VENV:-${VIRTUAL_ENV:-<none>}}"
fi

# --------------------------------------------------------------------------- #
# 3. Storage roots
#
# $SCRATCH is the documented Leonardo variable in some profiles and
# $CINECA_SCRATCH in others; accept either rather than silently falling back to
# $HOME, whose 50 GB quota a single pretraining run would blow through.
# --------------------------------------------------------------------------- #
if [[ "${PW_ON_CINECA}" -eq 1 ]]; then
    PW_SCRATCH="${PW_SCRATCH:-${SCRATCH:-${CINECA_SCRATCH:-}}}"
    PW_FAST="${PW_FAST:-${FAST:-}}"
    if [[ -z "${PW_SCRATCH}" ]]; then
        echo "ERROR: neither \$SCRATCH nor \$CINECA_SCRATCH is set. Set PW_SCRATCH=/path." >&2
        return 1 2>/dev/null || exit 1
    fi
    PW_FAST_ROOT="${PW_FAST_ROOT:-${PW_FAST}/yanlchen}"
    PW_DATA_ROOT="${PW_DATA_ROOT:-${PW_SCRATCH}/bio}"
else
    # Off-cluster: reuse the repository layout that run_tpami.sh and .gitignore
    # already assume, rather than inventing a parallel one.
    PW_SCRATCH="${PW_SCRATCH:-${PROJECT_DIR}}"
    PW_FAST="${PW_FAST:-${PROJECT_DIR}}"
    PW_FAST_ROOT="${PW_FAST_ROOT:-${PROJECT_DIR}}"
    PW_DATA_ROOT="${PW_DATA_ROOT:-${PROJECT_DIR}/data}"
    PW_CKPT_ROOT="${PW_CKPT_ROOT:-${PROJECT_DIR}/outputs}"
fi

# Checkpoints, caches and manifests: permanent, project-shared, fast I/O.
PW_CKPT_ROOT="${PW_CKPT_ROOT:-${PW_FAST_ROOT}/runs}"
PW_CACHE_ROOT="${PW_CACHE_ROOT:-${PW_FAST_ROOT}/cache}"
PW_SSL_CACHE="${PW_SSL_CACHE:-${PW_CACHE_ROOT}/ssl}"
PW_PREP_CACHE="${PW_PREP_CACHE:-${PW_CACHE_ROOT}/preprocessed}"
PW_MANIFEST_DIR="${PW_MANIFEST_DIR:-${PW_FAST_ROOT}/manifests}"

# Training corpora: one directory per modality, each holding per-dataset
# subdirectories of HDF5 windows (e.g. $PW_DATA_EEG/tueg/*.h5).
PW_DATA_EEG="${PW_DATA_EEG:-${PW_DATA_ROOT}/eeg}"
PW_DATA_ECG="${PW_DATA_ECG:-${PW_DATA_ROOT}/ecg}"
PW_DATA_EMG="${PW_DATA_EMG:-${PW_DATA_ROOT}/emg}"

mkdir -p "${PW_CKPT_ROOT}" "${PW_SSL_CACHE}" "${PW_PREP_CACHE}" "${PW_MANIFEST_DIR}" \
         "${PW_DATA_EEG}" "${PW_DATA_ECG}" "${PW_DATA_EMG}"

export PROJECT_DIR PW_VENV PW_SCRATCH PW_FAST PW_FAST_ROOT
export PW_CKPT_ROOT PW_CACHE_ROOT PW_SSL_CACHE PW_PREP_CACHE PW_MANIFEST_DIR
export PW_DATA_ROOT PW_DATA_EEG PW_DATA_ECG PW_DATA_EMG

# --------------------------------------------------------------------------- #
# 4. Launcher
#
# Always `python -m torch.distributed.run`, never the `torchrun` console script.
#
# torch.distributed.run spawns its workers with sys.executable. The `torchrun`
# on PATH belongs to the cineca-ai environment, so its sys.executable is the
# *module's* python -- and the workers then cannot see anything pip installed
# into the venv (pywt, for one), even though the parent shell has the venv
# active. Invoking the module form makes the venv's python the parent, and the
# workers inherit its site-packages.
#
# On the login node `torchrun` is not on PATH at all; on a compute node it is.
# Preferring it would therefore fail only on the machine that has GPUs.
# --------------------------------------------------------------------------- #
PW_TORCHRUN=("${PYTHON:-python}" -m torch.distributed.run)

# Fail before the allocation is spent, not inside a worker: every module the
# training entry points import at top level, checked in the venv's interpreter.
pw_require_python_deps() {
    local missing
    missing="$("${PYTHON:-python}" - <<'PYEOF'
import importlib.util as u
need = ["torch", "numpy", "scipy", "h5py", "pywt", "sklearn", "yaml", "tqdm"]
print(" ".join(m for m in need if u.find_spec(m) is None))
PYEOF
)"
    if [[ -n "${missing}" ]]; then
        echo "ERROR: missing Python packages in $("${PYTHON:-python}" -c 'import sys;print(sys.prefix)'): ${missing}" >&2
        echo "       pip install --no-cache-dir --no-deps ${missing}" >&2
        echo "       (pywt is the PyWavelets distribution; sklearn is scikit-learn)" >&2
        return 1
    fi
    return 0
}

# Refuse to start training on a login node.  finetune.py and pretrain_main both
# initialise NCCL, which needs a GPU; without one the job dies deep inside
# init_process_group instead of here, and CINECA frowns on login-node compute.
pw_require_gpu() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        return 0                      # inside a Slurm allocation (sbatch or srun)
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        return 0                      # a GPU is actually visible
    fi
    cat >&2 <<'MSG'
ERROR: no GPU and no Slurm allocation -- this looks like a login node.

  Training initialises NCCL, which requires a GPU. Get an allocation first:

    # interactive, for a smoke test
    srun --nodes=1 --gpus=1 --ntasks-per-node=1 --cpus-per-task=8 \
         -A iscrb_wearusfm -p boost_usr_prod --time=0:30:00 --pty /bin/bash

    # or submit a batch job
    sbatch scripts/slurm/cineca_pretrain.sbatch

  Set PW_ALLOW_NO_GPU=1 to override (CPU-only debugging).
MSG
    return 1
}

# --------------------------------------------------------------------------- #
# 5. Helpers
# --------------------------------------------------------------------------- #

# Directory holding the corpora for one modality.  `emg` and `semg` are the same
# data: the registry calls it semg, the on-disk layout the user asked for is emg.
pw_data_dir() {
    case "$1" in
        eeg)        echo "${PW_DATA_EEG}" ;;
        ecg)        echo "${PW_DATA_ECG}" ;;
        emg|semg)   echo "${PW_DATA_EMG}" ;;
        *)          echo "ERROR: unknown modality '$1'" >&2; return 1 ;;
    esac
}

# Print `data.roots.<id>=<dir>/<id>` for each dataset id given, so a caller can
# splice them straight into `--set`.
pw_roots_for() {
    local modality="$1"; shift
    local dir; dir="$(pw_data_dir "${modality}")" || return 1
    local id
    for id in "$@"; do
        echo "data.roots.${id}=${dir}/${id}"
    done
}

pw_print_layout() {
    cat <<LAYOUT
  code        ${PROJECT_DIR}
  venv        ${PW_VENV}
  checkpoints ${PW_CKPT_ROOT}
  caches      ${PW_SSL_CACHE} , ${PW_PREP_CACHE}
  manifests   ${PW_MANIFEST_DIR}
  data        ${PW_DATA_EEG} , ${PW_DATA_ECG} , ${PW_DATA_EMG}
LAYOUT
}
