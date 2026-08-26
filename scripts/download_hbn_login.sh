#!/bin/bash
# ============================================================================
# HBN, downloaded from a LOGIN NODE, surviving being killed.
#
#   nohup bash scripts/download_hbn_login.sh > ~/hbn.log 2>&1 &
#   tail -f ~/hbn.log
#
#   bash scripts/download_hbn_login.sh R4 R10 R11      # just these
#
# WHY NOT A BATCH JOB. The compute nodes on this system have no route to the
# public internet, so `sbatch`-ing an aws transfer produces "Could not connect
# to the endpoint URL" and an empty directory -- which is exactly what R4, R10
# and R11 got while other releases were being fetched interactively. Downloads
# have to happen on a login node here.
#
# WHICH MEANS THIS WILL BE KILLED. A login node SIGKILLs a process that runs too
# long or holds too much -- R5 died at 164 of 224 GB, exit 137. So this is built
# around being killed rather than around avoiding it:
#
#   * Low concurrency, so the resident set stays small and the killer looks
#     elsewhere for longer.
#   * `aws s3 sync`, which resumes: a killed transfer loses the file in flight,
#     nothing more.
#   * An outer loop that reruns after a kill. The killer takes the aws process,
#     not this shell, so the loop survives and starts the next attempt.
#   * A per-release completeness check against S3's own byte count, so "done"
#     means done rather than "the last attempt did not crash".
#
# RUN IT UNDER nohup so a disconnected session does not take it down, and leave
# it. It is resumable, so rerunning after anything at all is safe.
#
# ENVIRONMENT:
#   EEG_ROOT        corpus root
#   AWS_CONCURRENCY transfers in flight (default 3; lower survives longer)
#   MAX_ROUNDS      attempts per release (default 40)
# ============================================================================

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
EEG_ROOT="${EEG_ROOT:-/leonardo_scratch/large/userexternal/ychen003/bio/eeg}"
export EEG_ROOT
export AWS_CONCURRENCY="${AWS_CONCURRENCY:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"

RELEASES="$*"
[[ -z "${RELEASES}" ]] && RELEASES="R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11"

if ! aws --version >/dev/null 2>&1; then
    echo "ERROR: the AWS CLI does not run. HBN is S3-only." >&2
    echo "  source \$HOME/pwprep/bin/activate && pip install awscli" >&2
    exit 1
fi

s3_bytes() {
    aws s3 ls --recursive --summarize --no-sign-request \
        "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_$1/" 2>/dev/null \
        | awk '/Total Size/ {print $3}'
}

_STAT_FLAG=""
if stat -c%s "${BASH_SOURCE[0]}" >/dev/null 2>&1; then _STAT_FLAG="-c%s"
elif stat -f%z "${BASH_SOURCE[0]}" >/dev/null 2>&1; then _STAT_FLAG="-f%z"
fi

local_bytes() {
    [[ -z "${_STAT_FLAG}" ]] && { echo 0; return; }
    find -L "$1" -type f -print0 2>/dev/null \
        | xargs -0 stat "${_STAT_FLAG}" 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

human() { awk -v b="$1" 'BEGIN {printf "%.1f GB", b/1073741824}'; }

echo "============================================================"
echo "  HBN via login node   $(date '+%F %H:%M:%S')"
echo "  releases    ${RELEASES}"
echo "  into        ${EEG_ROOT}/HBN/raw"
echo "  concurrency ${AWS_CONCURRENCY}   max rounds ${MAX_ROUNDS}"
echo "  pid         $$"
echo "============================================================"

_failed=""
for r in ${RELEASES}; do
    dest="${EEG_ROOT}/HBN/raw/${r}"
    mkdir -p "${dest}"

    want=$(s3_bytes "${r}")
    if [[ -z "${want}" || "${want}" -eq 0 ]]; then
        echo ""
        echo "### ${r}: cannot read the remote size -- skipping for now"
        _failed="${_failed} ${r}"
        continue
    fi

    echo ""
    echo "############################################################"
    echo "### ${r}   target $(human "${want}")   $(date '+%H:%M:%S')"
    echo "############################################################"

    round=1
    while [[ "${round}" -le "${MAX_ROUNDS}" ]]; do
        have=$(local_bytes "${dest}")
        if [[ "${have}" -ge "${want}" ]]; then
            echo "    complete: $(human "${have}") / $(human "${want}")"
            break
        fi
        pct=$(awk -v a="${have}" -v b="${want}" 'BEGIN {printf "%.1f", 100*a/b}')
        echo "    round ${round}: $(human "${have}") / $(human "${want}") (${pct}%)  $(date '+%H:%M:%S')"

        bash scripts/download_eeg_pretrain_corpora.sh hbn "${r}"
        rc=$?

        after=$(local_bytes "${dest}")
        if [[ "${after}" -le "${have}" ]] && [[ "${rc}" -ne 0 ]]; then
            # No progress AND an error: retrying immediately just repeats it.
            echo "    no progress this round (exit ${rc}); pausing 60s" >&2
            sleep 60
        fi
        round=$(( round + 1 ))
    done

    have=$(local_bytes "${dest}")
    if [[ "${have}" -ge "${want}" ]]; then
        echo "    ${r} DONE"
    else
        echo "    ${r} still short after ${MAX_ROUNDS} round(s)" >&2
        _failed="${_failed} ${r}"
    fi
done

echo ""
echo "============================================================"
echo "  finished $(date '+%F %H:%M:%S')"
for r in ${RELEASES}; do
    d="${EEG_ROOT}/HBN/raw/${r}"
    printf "  %-4s %12s\n" "${r}" "$(human "$(local_bytes "${d}")")"
done
if [[ -n "${_failed}" ]]; then
    echo ""
    echo "  incomplete:${_failed}"
    echo "  Rerun -- everything already on disk is skipped:"
    echo "    nohup bash scripts/download_hbn_login.sh${_failed} > ~/hbn.log 2>&1 &"
else
    echo ""
    echo "  every release complete. Confirm against the bucket:"
    echo "    bash scripts/download_eeg_pretrain_corpora.sh verify-hbn"
fi
echo "============================================================"
