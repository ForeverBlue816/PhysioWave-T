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
#
# Two jobs in one file, and you often want only the first
# ------------------------------------------------------
#   1. export the PW_* paths
#   2. load the cineca-ai module and activate ${PW_VENV}, for training
#
# The data preparation scripts need (1) and nothing from (2) -- no CUDA, no
# torch -- and (2) actively gets in their way, because activating ${PW_VENV}
# replaces whatever environment they were being run from. So:
#
#     PW_VARS_ONLY=1 source scripts/cineca_env.sh
#
# exports the paths and touches neither the modules nor the virtualenv, and can
# be sourced before or after `source $HOME/pwprep/bin/activate` without
# disturbing it. PW_SKIP_MODULES=1 is the narrower version: skip the module
# load, still activate ${PW_VENV}.
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

if [[ "${PW_ON_CINECA}" -eq 1 && "${PW_VARS_ONLY:-0}" == "1" ]]; then
    # Paths only. Nothing here loads a module or activates a virtualenv, so
    # whatever environment is active stays active -- which is the point: the
    # preparation scripts run under $HOME/pwprep and want PW_DATA_EEG, not
    # torch.
    PW_VENV="${PW_VENV:-${VIRTUAL_ENV:-<none>}}"
elif [[ "${PW_ON_CINECA}" -eq 1 ]]; then
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
        if ! module load "${PW_CINECA_AI}" 2>&1; then
            # module prints its own error and returns non-zero, and the script
            # used to carry straight on. ${PW_VENV} is built with
            # --system-site-packages on top of this module, so without it the
            # venv activates fine and then `import torch` fails somewhere much
            # later -- typically inside torchrun, after the job is allocated.
            echo "" >&2
            echo "WARNING: ${PW_CINECA_AI} did not load." >&2
            echo "  ${PW_VENV} is a venv built --system-site-packages on top of it," >&2
            echo "  so torch comes from the module. Without it the venv still" >&2
            echo "  activates and the failure surfaces later, usually as an" >&2
            echo "  ImportError inside torchrun once the job is running." >&2
            echo "" >&2
            echo "  Find what is actually available and pin it:" >&2
            echo "      module avail cineca-ai" >&2
            echo "      export PW_CINECA_AI=cineca-ai/<version>" >&2
            echo "" >&2
            echo "  Preparation and download need no CUDA, so this is harmless" >&2
            echo "  for those steps. Fix it before training." >&2
            echo "" >&2
        fi
    fi

    # ----------------------------------------------------------------------- #
    # 2. Virtualenv (created with --system-site-packages on top of cineca-ai)
    # ----------------------------------------------------------------------- #
    PW_VENV="${PW_VENV:-${VENV:-${HOME}/pw}}"
    if [[ "${VIRTUAL_ENV:-}" == "${PW_VENV}" ]]; then
        :                                   # already active; re-sourcing stacks PATH
    elif [[ -f "${PW_VENV}/bin/activate" ]]; then
        # Replacing a *different* virtualenv is silent otherwise, and the only
        # visible trace is the prompt changing. That has cost real time twice
        # on the P300 preparation: `source $HOME/pwprep/bin/activate` followed
        # by this script leaves pwprep inactive, and the next command fails on
        # `import mne` with no hint about why. This script has to win -- the
        # training launchers source it precisely to get ${PW_VENV} -- so say
        # what happened rather than change who wins.
        if [[ -n "${VIRTUAL_ENV:-}" ]]; then
            echo "NOTE: replacing the active virtualenv ${VIRTUAL_ENV} with ${PW_VENV}." >&2
            echo "      If you wanted the other one, source it AFTER this script:" >&2
            echo "        source scripts/cineca_env.sh          # first, for PW_DATA_*" >&2
            echo "        source ${VIRTUAL_ENV}/bin/activate    # second" >&2
        fi
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

# Catch an OUTPUT_DIR that came from a variable the caller's shell had not set.
#
# `OUTPUT_DIR=$PW_CKPT_ROOT/run bash EMG/finetune_emg.sh` expands PW_CKPT_ROOT in
# the *caller's* shell, which happens before this file is sourced by the launcher
# -- so a shell that has not sourced it produces `/run`, and the failure is a
# permission error from mkdir at the filesystem root, several lines away from the
# cause. Fail here instead, with the fix in the message.
pw_check_output_dir() {
    local dir="${1:-}"
    [[ -n "${dir}" ]] || return 0
    local parent
    parent="$(dirname "${dir}")"
    if [[ "${parent}" == "/" ]]; then
        cat >&2 <<MSG
ERROR: OUTPUT_DIR is '${dir}' -- directly under the filesystem root.

  This is almost always an unset variable: OUTPUT_DIR=\$PW_CKPT_ROOT/name is
  expanded by your shell before this script runs, and \$PW_CKPT_ROOT is empty
  unless you have sourced the environment yourself:

    source scripts/cineca_env.sh

  Then re-run. Or leave OUTPUT_DIR unset and let the launcher choose it.
MSG
        return 1
    fi
    if [[ ! -d "${parent}" ]]; then
        echo "ERROR: OUTPUT_DIR '${dir}' -- parent '${parent}' does not exist" >&2
        return 1
    fi
    return 0
}

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
