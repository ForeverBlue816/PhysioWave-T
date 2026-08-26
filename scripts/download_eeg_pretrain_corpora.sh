#!/bin/bash
# ============================================================================
# Fetch the six remaining C1 pretraining corpora into one layout.
#
#   bash scripts/download_eeg_pretrain_corpora.sh layout
#   bash scripts/download_eeg_pretrain_corpora.sh physionet_mi
#   bash scripts/download_eeg_pretrain_corpora.sh hbn R3
#   bash scripts/download_eeg_pretrain_corpora.sh hbn all
#
# ONE DATASET PER INVOCATION, NAMED EXPLICITLY. There is no "download
# everything": these total roughly 2 TB, most of it HBN, and a command that
# starts two terabytes of transfer because it was run without arguments is a
# command that gets run without arguments.
#
# NOTHING HERE NEEDS npm. FACED, M3CV and HGD come straight from
# data.nemar.org over HTTPS, resumably; nemar-cli is a node package that adds
# nothing, and a login node without npm is the normal case.
#
# TDBRAIN IS NOT DOWNLOADED HERE and cannot be. It needs an ORCID login and an
# accepted data use agreement, and accepting an agreement is not something a
# script does on your behalf. `tdbrain` prints what to do instead.
#
# HBN is on open S3, but the "Not for Commercial Use" releases are excluded
# from the main model -- R1-R11 are the standard ones and are what `all` fetches.
#
# SIZES, so nothing starts a transfer that will not fit:
#   PhysioNetMI  1.9 GB zip / 3.4 GB unpacked
#   FACED        22.7 GB          M3CV  ~30 GB          HGD  25.1 GB
#   TDBRAIN      reserve 130 GB unpacked
#   HBN          1.875 TB for R1-R11 (103,120,140,230,224,91,245,157,185,160,220)
#
# AFTER DOWNLOADING, inspect before processing. Every adapter has been checked
# against synthetic trees with these corpora's conventions, not against your
# copy:
#   DATASET=<id> INSPECT=40 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh
# ============================================================================

set -uo pipefail

EEG_ROOT="${EEG_ROOT:-/leonardo_scratch/large/userexternal/ychen003/bio/eeg}"

usage() {
    sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo "datasets: layout physionet_mi faced m3cv hgd hbn tdbrain"
}

need() {
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "ERROR: $1 is not on PATH. $2" >&2
    return 1
}

# `aws` on PATH is not the same as a working AWS CLI. PyPI carries an abandoned
# Python-2 package literally named `aws`, and `pip install aws` puts its
# entry point at the same bin/aws an awscli install would use -- so it is found,
# it is executed, and it dies on a print statement from 2012. Run it before
# trusting it.
have_working_aws() {
    command -v aws >/dev/null 2>&1 || return 1
    aws --version >/dev/null 2>&1
}

aws_repair_hint() {
    cat >&2 <<'MSG'

  The `aws` on PATH is the abandoned Python-2 package of that name, not the
  AWS CLI. They install to the same bin/aws. To replace it:

      pip uninstall -y aws
      pip install awscli
      aws --version          # should print aws-cli/1.x or /2.x

MSG
}

layout() {
    mkdir -p "${EEG_ROOT}"/{FACED,TDBRAIN,PhysioNetMI,M3CV,HBN,HGD}/{raw,processed,manifests}
    echo "layout under ${EEG_ROOT}:"
    ls -d "${EEG_ROOT}"/*/ 2>/dev/null
    echo ""
    echo "'processed' and 'manifests' are there for per-corpus scratch. The"
    echo "pipeline itself writes shards and manifests into"
    echo "  \${DATA_ROOT:-${EEG_ROOT}}/eeg_c1_corpus/<dataset>/"
    echo "which is where TUEG already is and where the merge step reads from."
}

case "${1:-}" in

layout) layout ;;

physionet_mi)
    # Open access, no agreement. 64 channels at 160 Hz, EDF+.
    dest="${EEG_ROOT}/PhysioNetMI/raw"; mkdir -p "${dest}"
    if have_working_aws; then
        echo "==> S3 (faster than the HTTP mirror)"
        aws s3 sync --no-sign-request \
            s3://physionet-open/eegmmidb/1.0.0/ "${dest}/eegmmidb-1.0.0"
    else
        command -v aws >/dev/null 2>&1 && aws_repair_hint
        need wget "Install it, or repair the AWS CLI as above." || exit 1
        echo "==> HTTP mirror (wget; the AWS CLI is unusable here)"
        ( cd "${dest}" && wget -r -N -c -np -nH --cut-dirs=3 \
            https://physionet.org/files/eegmmidb/1.0.0/ )
    fi
    echo ""
    echo "downloaded $(find "${dest}" -name '*.edf' 2>/dev/null | wc -l) .edf file(s)"
    echo "expected 1526 (109 subjects x 14 runs)"
    ;;

faced|m3cv|hgd)
    # NEMAR serves one zip per published version straight over HTTPS:
    #   https://data.nemar.org/<accession>/<version>.zip
    # nemar-cli is a node package and adds nothing here, so it is not used --
    # a login node with no npm is the normal case, not a broken one.
    #
    # VERSIONS ARE PINNED, and verified against nemar.org on 2026-08-26. A
    # dataset that gets republished moves to a new version and this URL 404s
    # rather than silently fetching something else; NEMAR_VERSION overrides.
    case "$1" in
        faced) acc=nm000112; dir=FACED; ver=v1.1.3; approx="22.7 GB" ;;
        m3cv)  acc=nm000166; dir=M3CV;  ver=v1.0.0; approx="21.1 GB" ;;
        hgd)   acc=nm000172; dir=HGD;   ver=v1.0.3; approx="25.1 GB" ;;
    esac
    ver="${NEMAR_VERSION:-${ver}}"
    # NEMAR_BASE_URL exists so this branch can be exercised without pulling
    # 21 GB; it is not something a real run sets.
    url="${NEMAR_BASE_URL:-https://data.nemar.org}/${acc}/${ver}.zip"
    dest="${EEG_ROOT}/${dir}/raw"; mkdir -p "${dest}"
    zip="${dest}/${dir}_${acc}_${ver}.zip"

    echo "==> ${dir} ${acc} ${ver}  (~${approx})"
    echo "    ${url}"
    echo "    -> ${zip}"
    echo ""

    # An archive already on disk and sound is not re-fetched. `curl -C -` on a
    # COMPLETE file is an error in some versions rather than a no-op, so a
    # second run of this script would otherwise fail on the one thing that
    # already succeeded -- and re-running is exactly what someone does after an
    # interrupted 22 GB transfer.
    _have_it=0
    if [[ -s "${zip}" ]]; then
        echo "==> ${zip} exists; checking whether it is complete"
        if python "$(dirname "${BASH_SOURCE[0]}")/unpack_archive.py" "${zip}" \
                >/dev/null 2>&1; then
            echo "    complete and sound -- skipping the download"
            _have_it=1
        else
            echo "    incomplete or unreadable -- resuming the download"
        fi
    fi

    if [[ "${_have_it}" -eq 0 ]]; then
        # -C - resumes a partial file. At this size on a shared login node an
        # interrupted transfer is likely, and restarting 22 GB from zero because
        # the flag was missing is the kind of thing that costs an afternoon.
        if command -v curl >/dev/null 2>&1; then
            curl -fL -C - --retry 5 --retry-delay 10 \
                 --connect-timeout 20 -o "${zip}" "${url}"
            _rc=$?
            # 33: server refused the range. Start over rather than leave a file
            # that is half of one transfer and half of another.
            if [[ "${_rc}" -eq 33 ]]; then
                echo "    server refused a ranged request; restarting whole" >&2
                curl -fL --retry 5 --retry-delay 10 \
                     --connect-timeout 20 -o "${zip}" "${url}"
                _rc=$?
            fi
        elif command -v wget >/dev/null 2>&1; then
            wget -c -O "${zip}" "${url}"
            _rc=$?
        else
            echo "ERROR: neither curl nor wget is available." >&2
            exit 1
        fi
        if [[ "${_rc}" -ne 0 ]]; then
            echo "" >&2
            echo "ERROR: download failed (exit ${_rc})." >&2
            echo "  If it was a 404, the published version has moved. Check" >&2
            echo "    https://nemar.org/dataset/${acc}" >&2
            echo "  and re-run with the version it names:" >&2
            echo "    NEMAR_VERSION=v1.2.3 bash $0 $1" >&2
            exit 1
        fi
    fi

    echo ""
    echo "==> checking the archive before unpacking"
    # NOT unzip: Info-ZIP rejects these as possible zip bombs, and the
    # environment variable it suggests would force a truncated download open
    # just as readily as a sound one.
    python "$(dirname "${BASH_SOURCE[0]}")/unpack_archive.py" "${zip}" \
        --extract-to "${dest}" || exit 1

    if [[ "$1" == "faced" ]]; then
        echo ""
        echo "FACED note: 92 subjects are 1000 Hz and 31 are 250 Hz, in this"
        echo "same release. Both are accepted -- the 250 Hz ones are the real"
        echo "acquisition, and every shard written from one records"
        echo "upsampled_from_hz in its provenance. A rate that is NEITHER is"
        echo "refused, because that is the preprocessed derivative."
    fi
    if [[ "$1" == "m3cv" ]]; then
        echo ""
        echo "M3CV note: this release is already cleaned (band-pass, 49-51 Hz"
        echo "notch, ICA, 1000 -> 250 Hz). The registry marks it"
        echo "notch_already_applied, so no second notch is run. Do not describe"
        echo "it as raw 1000 Hz data."
    fi
    ;;

hbn)
    if ! have_working_aws; then
        if command -v aws >/dev/null 2>&1; then
            echo "ERROR: the AWS CLI on PATH does not run." >&2
            aws_repair_hint
        else
            echo "ERROR: aws is not on PATH. pip install awscli" >&2
        fi
        echo "  HBN is S3-only -- there is no HTTP mirror to fall back to." >&2
        exit 1
    fi
    rel="${2:-}"
    if [[ -z "${rel}" ]]; then
        echo "ERROR: name a release, or 'all' for R1-R11 (1.875 TB)." >&2
        echo "  bash $0 hbn R3      # 140 GB, enough to validate the adapter" >&2
        echo "  bash $0 hbn all     # every standard release" >&2
        exit 1
    fi
    fetch_one() {
        local r="$1"
        local dest="${EEG_ROOT}/HBN/raw/${r}"
        mkdir -p "${dest}"
        echo "==> HBN ${r} -> ${dest}"
        # sync, not `cp --recursive`. At 100-245 GB per release a transfer WILL
        # be interrupted, and cp restarts every file from the beginning; sync
        # skips what is already there with a matching size. That is the
        # difference between resuming a release and re-fetching it.
        aws s3 sync "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_${r}" \
            "${dest}" --no-sign-request
        local rc=$?
        local n
        n=$(find -L "${dest}" -type f \( -iname '*.set' -o -iname '*.fdt' \
            -o -iname '*.edf' -o -iname '*.mff' \) 2>/dev/null | wc -l)
        echo "    ${r}: exit ${rc}, $(du -sh "${dest}" 2>/dev/null | cut -f1) "\
             "in ${n} recording file(s)"
        return "${rc}"
    }
    if [[ "${rel}" == "all" ]]; then
        for i in $(seq 1 11); do fetch_one "R${i}"; done
    else
        fetch_one "${rel}"
    fi
    echo ""
    echo "HBN note: files report 129 EEG channels. The adapter removes the"
    echo "NAMED vertex reference to reach 128 and writes what it removed into"
    echo "every shard's provenance. It refuses to drop a row by position, so a"
    echo "release whose reference is labelled differently fails loudly rather"
    echo "than quietly deleting whichever channel came last."
    ;;

tdbrain)
    cat <<'MSG'
TDBRAIN is not downloaded by this script.

It requires an ORCID login and an accepted Data Use Agreement, and the DUA also
forbids redistribution -- so the transfer has to be yours, made under your own
acceptance of it.

  1. https://brainclinics.com/resources/tdbrain-dataset
  2. Sign in with ORCID, accept the DUA.
  3. Unpack into  $EEG_ROOT/TDBRAIN/raw  (reserve ~130 GB).

Then, as with every other corpus:

  DATASET=tdbrain INSPECT=40 VERIFY_POWERLINE=1 bash EEG/preprocess_eeg_corpus.sh

Read the coverage line in that report before launching anything. TDBRAIN
records 26 electrodes against E32_512's 32 slots, so the expected per-file
coverage is 26 of 32 (81%) and the expected empty-slot rate is 18.8% -- both
inside the gates. The six unrecorded slots stay zero with valid_channel_mask
False. Nothing interpolates them: a spatially interpolated channel is a smooth
function of its neighbours, and the model would learn the interpolation.
MSG
    ;;

*) usage; exit 1 ;;
esac
