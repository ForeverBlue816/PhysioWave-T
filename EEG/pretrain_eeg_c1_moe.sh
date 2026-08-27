#!/bin/bash
# ============================================================================
# EEG C1 multi-route pretraining -- the recommended entry point.
#
#   bash EEG/pretrain_eeg_c1_moe.sh
#
# Four wavelet frontends over seven corpora, one shared RoPE Transformer, C1
# channel-name embedding injected at the token site. This is NOT the WAST/TARE
# path: it shares no module with it, and configs/pretrain/eeg.yaml is a
# different run.
#
# BEFORE THE FIRST RUN the corpora have to be preprocessed, once per dataset:
#
#   python EEG/preprocess_pretrain_corpus.py --dataset tueg \
#       --root ${PW_DATA_EEG}/tueg --out-dir ${DATA_ROOT}/tueg --mains-hz 60
#
# and their manifests merged into ${DATA_ROOT}/merged. That step needs mne and
# belongs under $HOME/pwprep, not in the training venv.
#
# ENVIRONMENT VARIABLES, all optional:
#
#   NUM_GPUS              GPUs per node                              (4)
#   NNODES                nodes                            ($SLURM_NNODES or 1)
#   CONFIG                which objective          (pretrain/eeg_c1_moe = full)
#   EPOCHS                                                          (50)
#   MAX_STEPS             stop after this many optimizer steps      (unset)
#   BATCH_SIZE_BY_ROUTE   "E19_256=64,E32_512=48,E64_256=24,E128_512=12"
#   GRAD_ACCUMULATION                                               (4)
#   LR / WEIGHT_DECAY / MASK_RATIO / SEED
#   DATA_ROOT             holds merged/manifest_{train,val}.jsonl
#   OUTPUT_DIR            checkpoints and figures
#   RESUME                'auto' to continue OUTPUT_DIR/latest.pth
#   VIS_EVERY_EPOCHS      snapshot cadence for the fixed-sample figures (5)
#
# These are VALUES passed straight through, not `${VAR:+--flag}` presence
# tests: MASK_RATIO=0 is a non-empty string and that idiom would read it as on.
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source "$(pwd)/scripts/cineca_env.sh"

pw_require_python_deps || exit 1

NUM_GPUS="${NUM_GPUS:-4}"          # GPUs per node
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
# Which objective. pretrain/eeg_c1_moe is the full one; the two ablations
# differ from it in exactly the raw_weight and mask_before_frontend lines.
CONFIG="${CONFIG:-pretrain/eeg_c1_moe}"
EPOCHS="${EPOCHS:-50}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MASK_RATIO="${MASK_RATIO:-0.5}"
SEED="${SEED:-42}"
VIS_EVERY_EPOCHS="${VIS_EVERY_EPOCHS:-5}"
DATA_ROOT="${DATA_ROOT:-${PW_DATA_EEG}/eeg_c1_corpus}"
OUTPUT_DIR="${OUTPUT_DIR:-${PW_CKPT_ROOT}/pretrain_eeg_c1_moe}"
BATCH_SIZE_BY_ROUTE="${BATCH_SIZE_BY_ROUTE:-}"
MAX_STEPS="${MAX_STEPS:-}"
RESUME="${RESUME:-}"

MANIFEST_TRAIN="${MANIFEST_TRAIN:-${DATA_ROOT}/merged/manifest_train.jsonl}"
MANIFEST_VAL="${MANIFEST_VAL:-${DATA_ROOT}/merged/manifest_val.jsonl}"

# Say what is missing once, here, rather than as a stack trace after the model
# has been built on every rank.
for f in "${MANIFEST_TRAIN}" "${MANIFEST_VAL}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: no manifest at ${f}" >&2
        echo "       Preprocess the corpora first, one run per dataset:" >&2
        echo "         python EEG/preprocess_pretrain_corpus.py --dataset <id> \\" >&2
        echo "             --root <raw> --out-dir ${DATA_ROOT}/<id> --mains-hz <50|60>" >&2
        echo "       then merge the per-dataset manifests into ${DATA_ROOT}/merged." >&2
        echo "       Nothing here falls back to synthetic data; --smoke-test is" >&2
        echo "       the explicit way to run without a corpus." >&2
        exit 1
    fi
done

OVERRIDES=(
  "data.manifest_train=${MANIFEST_TRAIN}"
  "data.manifest_val=${MANIFEST_VAL}"
  "train.epochs=${EPOCHS}"
  "train.grad_accumulation_steps=${GRAD_ACCUMULATION}"
  "train.lr=${LR}"
  "train.weight_decay=${WEIGHT_DECAY}"
  "train.vis_every_epochs=${VIS_EVERY_EPOCHS}"
  "model.mask_ratio=${MASK_RATIO}"
  "seed=${SEED}"
)

# "E19_256=64,E32_512=48" -> one dotted override per route.
if [[ -n "${BATCH_SIZE_BY_ROUTE}" ]]; then
    IFS=',' read -r -a _pairs <<< "${BATCH_SIZE_BY_ROUTE}"
    for _p in "${_pairs[@]}"; do
        [[ "${_p}" == *=* ]] || { echo "ERROR: BATCH_SIZE_BY_ROUTE entry '${_p}' is not ROUTE=N" >&2; exit 1; }
        OVERRIDES+=("train.batch_size_by_route.${_p}")
    done
fi

EXTRA=()
[[ -n "${MAX_STEPS}" ]] && EXTRA+=(--max-steps "${MAX_STEPS}")
[[ -n "${RESUME}" ]]    && EXTRA+=(--resume "${RESUME}")

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "  EEG C1 multi-route pretraining"
echo "  config=${CONFIG}"
echo "  nodes=${NNODES} x ${NUM_GPUS} gpu = $((NNODES * NUM_GPUS)) rank(s)"
echo "  epochs=${EPOCHS}  grad_accum=${GRAD_ACCUMULATION}"
echo "  lr=${LR}  wd=${WEIGHT_DECAY}  mask_ratio=${MASK_RATIO}  seed=${SEED}"
echo "  train manifest ${MANIFEST_TRAIN}"
echo "  val   manifest ${MANIFEST_VAL}"
echo "  out            ${OUTPUT_DIR}"
[[ -n "${BATCH_SIZE_BY_ROUTE}" ]] && echo "  batch/route    ${BATCH_SIZE_BY_ROUTE}"
[[ -n "${RESUME}" ]] && echo "  resume         ${RESUME}"
echo "============================================================"

# The full command, recorded next to the checkpoints, so a run can be reproduced
# from its own output directory rather than from shell history.
# --standalone brings up its own rendezvous on localhost, which is right for
# one node and wrong for four: every node would elect itself rank 0 and the
# job would run as N independent one-node trainings that never all-reduce.
# Above one node, c10d against a named endpoint is what joins them.
if [[ "${NNODES}" -gt 1 ]]; then
    if [[ -z "${MASTER_ADDR:-}" ]]; then
        echo "ERROR: NNODES=${NNODES} but MASTER_ADDR is unset. The sbatch" >&2
        echo "       sets it from the nodelist; exporting it is how the ranks" >&2
        echo "       find each other." >&2
        exit 1
    fi
    RDZV=(--nnodes="${NNODES}" --nproc_per_node="${NUM_GPUS}"
          --node_rank="${SLURM_NODEID:-0}"
          --rdzv_id="${RDZV_ID:-${SLURM_JOB_ID:-0}}"
          --rdzv_backend=c10d
          --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT:-29500}")
else
    RDZV=(--standalone --nproc_per_node="${NUM_GPUS}")
fi

CMD=("${PW_TORCHRUN[@]}" "${RDZV[@]}"
     -m physiowave.train.pretrain_main
     --config "${CONFIG}"
     --output-dir "${OUTPUT_DIR}"
     "${EXTRA[@]}"
     --set "${OVERRIDES[@]}")

# One node writes this. Four nodes racing on the same path leaves whichever
# finished last, and the file is meant to record the run rather than a node.
if [[ "${SLURM_NODEID:-0}" == "0" ]]; then
    printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/train_command.txt"
    echo >> "${OUTPUT_DIR}/train_command.txt"
    printf '%s\n' "$(cat "${OUTPUT_DIR}/train_command.txt")"
fi

"${CMD[@]}"
_rc=$?

if [[ ${_rc} -eq 0 ]]; then
    echo ""
    echo "Figures:"
    echo "  python scripts/visualize_eeg_pretraining.py \\"
    echo "      --run-dir ${OUTPUT_DIR} --checkpoint best.pth --split val --format svg"
fi
exit "${_rc}"
