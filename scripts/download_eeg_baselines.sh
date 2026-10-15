#!/bin/bash
# ============================================================================
# The released EEGPT and LaBraM weights, for the SeizeIT2 EEG-only comparison
# (physiowave/fusion/baselines.py). Run on the login node (compute nodes have
# no internet); each file is checked against the SHA-256 it was verified with.
#
#   bash scripts/download_eeg_baselines.sh
#
#   -> $PW_CKPT_ROOT/baselines/eegpt/eegpt_mcae_58chs_4s_large4E.safetensors
#   -> $PW_CKPT_ROOT/baselines/labram/labram-base.pth
#
# LaBraM-Base: the authors' file, from their repository (MIT) at a fixed
# commit. EEGPT large (eegpt_mcae_58chs_4s_large4E): the authors release it on
# figshare (figshare.com/s/e37df4f8a907a866df4b), whose download page does not
# serve scripts; this takes the same checkpoint converted to safetensors from
# huggingface.co/eeg-telecom-paris/eegpt-large-official at a fixed revision --
# a third-party mirror. Its target encoder loads strictly into the official
# EEGTransformer and gives bit-identical outputs; to use the authors' file
# instead, download the .ckpt in a browser and pass EEGPT_CKPT=<path> to the
# job (both formats load).
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PW_CKPT_ROOT:-}" ]]; then
    PW_CKPT_ROOT="$(set +u; source "${HERE}/cineca_env.sh" >/dev/null 2>&1; echo "${PW_CKPT_ROOT}")"
fi
[[ -n "${PW_CKPT_ROOT}" ]] || { echo "ERROR: PW_CKPT_ROOT is not set" >&2; exit 1; }
OUT="${PW_CKPT_ROOT}/baselines"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

fetch() {   # url dest sha256
    local url="$1" dest="$2" sum="$3" got
    mkdir -p "$(dirname "${dest}")"
    if [[ -f "${dest}" && "$(sha256 "${dest}")" == "${sum}" ]]; then
        echo "==> ${dest}: present, checksum ok"; return 0
    fi
    echo "==> ${url}"
    curl -fL --retry 5 --retry-delay 10 -C - -sS -o "${dest}.part" "${url}"
    got="$(sha256 "${dest}.part")"
    if [[ "${got}" != "${sum}" ]]; then
        echo "ERROR: ${dest}: checksum mismatch (got ${got:0:16}..., want ${sum:0:16}...)" >&2
        rm -f "${dest}.part"; return 1
    fi
    mv "${dest}.part" "${dest}"
    echo "    ${dest}: $(du -h "${dest}" | cut -f1), checksum ok"
}

fetch "https://huggingface.co/eeg-telecom-paris/eegpt-large-official/resolve/11e0075c388e282a2762496f7f3775652c9e3db9/weights.safetensors" \
      "${OUT}/eegpt/eegpt_mcae_58chs_4s_large4E.safetensors" \
      c45c31cb42f68f8b9c630a7739a8084ae5dd6a6608a0d732a5165f180c755a96
fetch "https://raw.githubusercontent.com/935963004/LaBraM/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints/labram-base.pth" \
      "${OUT}/labram/labram-base.pth" \
      7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c
echo "done: ${OUT}"
