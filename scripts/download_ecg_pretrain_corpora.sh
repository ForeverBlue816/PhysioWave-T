#!/bin/bash
# ============================================================================
# Fetch the ECG C1 pretraining corpora into one layout under $ECG_ROOT.
#
#   bash scripts/download_ecg_pretrain_corpora.sh status        # what is there
#   bash scripts/download_ecg_pretrain_corpora.sh all           # every open one
#   bash scripts/download_ecg_pretrain_corpora.sh mimic_iv_ecg  # just one
#
#   nohup bash scripts/download_ecg_pretrain_corpora.sh all > ~/ecg_download.log 2>&1 &
#
# `all` fetches the six OPEN corpora, smallest first, then says what the two
# gated ones need. Every step is resumable and idempotent: a finished corpus
# carries a .complete marker and is skipped, a partial download continues from
# where it stopped, and rerunning `all` after an interruption is the way to
# continue it.
#
# RUN IT ON A LOGIN NODE. Compute nodes here have no route to the internet.
# A login node has also killed a long transfer before (HBN, at 164 of 224 GB),
# and a process the node kills is being stopped on purpose -- so this does NOT
# loop to restart after one. If it is killed, rerun it later, or move the big
# ones (Icentia11k 188 GB, CODE-15% 46 GB, MIMIC-IV-ECG 34 GB) through the
# data-transfer service CINECA provides for this.
#
# LAYOUT -- what ECG/preprocess_ecg_corpus.sh expects:
#
#   $ECG_ROOT/MIMIC-IV-ECG/raw/<zip>        read in place, not unpacked
#   $ECG_ROOT/CODE-15/raw/exams_part*.hdf5, exams.csv
#   $ECG_ROOT/MedalCare-XL/raw/...          unpacked
#   $ECG_ROOT/NorwegianAthlete/raw/...      unpacked
#   $ECG_ROOT/Georgia/raw/...               .hea + .mat
#   $ECG_ROOT/Icentia11k/raw/<zip>          read in place (1.1 TB unpacked)
#   $ECG_ROOT/HEEDB/raw/...                 credentialed, see `heedb`
#   $ECG_ROOT/CODE-II/raw/...               restricted, see `code2`
#   $ECG_ROOT/ecg_c1_corpus/<dataset>/      written by preprocessing
#
# SIZES (verified against the servers, 2026-09-29), and what stays on disk:
#   Norwegian Athlete   1.8 MB zip                  -> 3.2 MB unpacked
#   Georgia             ~1.25 GB, 20,688 files over HTTP (no zip, no open S3)
#   MedalCare-XL        9.3 GB zip                  -> 28.2 GB unpacked
#   MIMIC-IV-ECG        36.3 GB zip, kept zipped    (90.4 GB if unpacked)
#   CODE-15%            46.3 GB in 18 zips          -> ~68 GB of hdf5
#   Icentia11k          202.2 GB zip, kept zipped   (1.1 TB if unpacked)
#   about 335 GB in all once done; `all` checks free space before it starts.
# Every source answers ranged requests, so an interrupted file resumes.
#
# ENVIRONMENT:
#   ECG_ROOT    download root (/leonardo_scratch/large/userexternal/ychen003/bio/ecg)
#   KEEP_ZIPS   1: keep CODE-15% / MedalCare / athlete zips after unpacking
#   UNPACK      1: also unpack MIMIC-IV-ECG (90 GB) instead of reading the zip
#   SKIP_SPACE_CHECK  1: do not refuse to start when free space looks short
#   WITH_HEEDB  1: `all` also syncs HEEDB (needs bdsp.io credentials)
#   HEEDB_S3_URI      override HEEDB's S3 access point
#   AWS_PROFILE       the AWS profile holding the bdsp.io credentials
# ============================================================================

set -uo pipefail

ECG_ROOT="${ECG_ROOT:-/leonardo_scratch/large/userexternal/ychen003/bio/ecg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNPACK_PY="${HERE}/unpack_archive.py"
PYTHON="${PYTHON:-python}"

OPEN_DATASETS="norwegian_athlete georgia medalcare_xl mimic_iv_ecg code15 icentia11k"

# --------------------------------------------------------------------------- #
# Sources. Pinned versions: a republished dataset moves to a new URL and the
# old one 404s rather than silently fetching something else.
# --------------------------------------------------------------------------- #
# PhysioNet's get-zip endpoints serve the whole-project zip with its real
# name in Content-Disposition; the name is given here so the file on disk has
# it too.
MIMIC_URL="${MIMIC_URL:-https://physionet.org/content/mimic-iv-ecg/get-zip/1.0/}"
MIMIC_ZIP="${MIMIC_ZIP:-mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip}"
ICENTIA_URL="${ICENTIA_URL:-https://physionet.org/content/icentia11k-continuous-ecg/get-zip/1.0/}"
ICENTIA_ZIP="${ICENTIA_ZIP:-icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip}"
ATHLETE_URL="${ATHLETE_URL:-https://physionet-open.s3.amazonaws.com/norwegian-athlete-ecg/norwegian-athlete-ecg-1.0.0.zip}"
# Challenge 2021 has no zip and no open S3 prefix; the Challenge 2020 tarball
# on Google Cloud is gone (NoSuchBucket). Recursive HTTP is what remains.
GEORGIA_HTTP="${GEORGIA_HTTP:-https://physionet.org/files/challenge-2021/1.0.3/training/georgia/}"
CODE15_BASE="${CODE15_BASE:-https://zenodo.org/api/records/4916206/files}"
CODE15_PARTS="${CODE15_PARTS:-18}"
MEDALCARE_URL="${MEDALCARE_URL:-https://zenodo.org/api/records/8068944/files/MedalCare-XL.zip/content}"
# HEEDB's controlled-access S3 access point, from the AWS Open Data registry.
# It needs the credentials bdsp.io issues after its agreement is signed.
HEEDB_S3_URI="${HEEDB_S3_URI:-s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-ecg-accesspoint/ECG/}"

usage() {
    # The header comment, up to the first line of code.
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
        "${BASH_SOURCE[0]}"
    echo "targets: status all layout ${OPEN_DATASETS} heedb code2"
}

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

have_working_aws() {
    command -v aws >/dev/null 2>&1 || return 1
    aws --version >/dev/null 2>&1
}

raw_dir() {
    case "$1" in
        mimic_iv_ecg)      echo "${ECG_ROOT}/MIMIC-IV-ECG/raw" ;;
        code15)            echo "${ECG_ROOT}/CODE-15/raw" ;;
        medalcare_xl)      echo "${ECG_ROOT}/MedalCare-XL/raw" ;;
        norwegian_athlete) echo "${ECG_ROOT}/NorwegianAthlete/raw" ;;
        georgia)           echo "${ECG_ROOT}/Georgia/raw" ;;
        icentia11k)        echo "${ECG_ROOT}/Icentia11k/raw" ;;
        heedb)             echo "${ECG_ROOT}/HEEDB/raw" ;;
        code2)             echo "${ECG_ROOT}/CODE-II/raw" ;;
        *) die "unknown dataset $1" ;;
    esac
}

#: GB each corpus needs on disk once this script is done with it.
need_gb() {
    case "$1" in
        norwegian_athlete) echo 1 ;;
        georgia)           echo 2 ;;
        medalcare_xl)      echo 38 ;;
        mimic_iv_ecg)      echo 37 ;;
        code15)            echo 75 ;;
        icentia11k)        echo 204 ;;
        *)                 echo 0 ;;
    esac
}

is_done()   { [[ -f "$(raw_dir "$1")/.complete" ]]; }
mark_done() { date -u +%FT%TZ > "$(raw_dir "$1")/.complete"; }

free_gb() {
    local d="$1"
    while [[ ! -d "${d}" ]]; do d="$(dirname "${d}")"; done
    df -Pk "${d}" | awk 'NR==2 {printf "%d", $4 / 1048576}'
}

check_space() {
    local want="$1" have
    have="$(free_gb "${ECG_ROOT}")"
    if [[ "${have}" -lt "${want}" ]]; then
        if [[ "${SKIP_SPACE_CHECK:-0}" == "1" ]]; then
            warn "need ~${want} GB under ${ECG_ROOT}, ${have} GB free; continuing (SKIP_SPACE_CHECK=1)"
        else
            die "need ~${want} GB under ${ECG_ROOT} and only ${have} GB is free. Free some, or SKIP_SPACE_CHECK=1 if a quota rather than df is what limits you."
        fi
    fi
}

#: Content-Length after redirects, or empty if the server does not say.
remote_size() {
    curl -sIL --connect-timeout 20 "$1" 2>/dev/null \
        | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n}'
}

local_size() {
    [[ -f "$1" ]] || { echo 0; return; }
    if stat -c%s "$1" >/dev/null 2>&1; then stat -c%s "$1"; else stat -f%z "$1"; fi
}

#: fetch URL DEST -- resumable, and complete only when the size matches.
fetch() {
    local url="$1" dest="$2" want have rc
    mkdir -p "$(dirname "${dest}")"
    want="$(remote_size "${url}")"
    have="$(local_size "${dest}")"
    if [[ -n "${want}" && "${have}" == "${want}" ]]; then
        echo "    ${dest##*/}: complete (${have} bytes)"
        return 0
    fi
    echo "    ${url}"
    echo "    -> ${dest}  (${have} of ${want:-?} bytes on disk)"
    if command -v curl >/dev/null 2>&1; then
        curl -fL -C - --retry 5 --retry-delay 10 --connect-timeout 20 \
             -o "${dest}" "${url}"
        rc=$?
        if [[ ${rc} -eq 33 ]]; then
            # The server refused a ranged request: start over rather than
            # leave a file that is half of one transfer and half of another.
            warn "server refused to resume; restarting ${dest##*/}"
            curl -fL --retry 5 --retry-delay 10 --connect-timeout 20 \
                 -o "${dest}" "${url}"
            rc=$?
        fi
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "${dest}" "${url}"
        rc=$?
    else
        die "neither curl nor wget is available"
    fi
    [[ ${rc} -eq 0 ]] || { warn "download of ${url} failed (exit ${rc})"; return 1; }
    have="$(local_size "${dest}")"
    if [[ -n "${want}" && "${have}" != "${want}" ]]; then
        warn "${dest##*/} is ${have} bytes, the server says ${want}"
        return 1
    fi
    return 0
}

#: A zip that opens, has a sane compression ratio and a readable directory.
zip_ok() { "${PYTHON}" "${UNPACK_PY}" "$1" >/dev/null 2>&1; }

unpack() {
    local zip="$1" dest="$2"
    say "unpacking ${zip##*/}"
    "${PYTHON}" "${UNPACK_PY}" "${zip}" --extract-to "${dest}" || return 1
    if [[ "${KEEP_ZIPS:-0}" != "1" ]]; then
        rm -f "${zip}"
    fi
}

# --------------------------------------------------------------------------- #
# One function per corpus. Each returns 0 only when the corpus is complete.
# --------------------------------------------------------------------------- #

get_norwegian_athlete() {
    local d; d="$(raw_dir norwegian_athlete)"
    local zip="${d}/${ATHLETE_URL##*/}"
    fetch "${ATHLETE_URL}" "${zip}" || return 1
    unpack "${zip}" "${d}" || return 1
    local n; n=$(find "${d}" -name '*.hea' | wc -l)
    echo "    ${n} record header(s) (expected 28)"
    [[ ${n} -ge 28 ]]
}

get_georgia() {
    local d; d="$(raw_dir georgia)"
    mkdir -p "${d}"
    command -v wget >/dev/null 2>&1 || die "Georgia needs wget (recursive HTTP)"
    say "HTTP ${GEORGIA_HTTP}"
    echo "    ~20,700 small files; -N skips the ones already here on a rerun"
    # --cut-dirs=5 drops files/challenge-2021/1.0.3/training/georgia, so
    # g1..g11 land directly under raw/.
    ( cd "${d}" && wget -q -r -N -c -np -nH --cut-dirs=5 -R 'index.html*' \
          "${GEORGIA_HTTP}" ) || return 1
    local n; n=$(find "${d}" -name '*.hea' | wc -l)
    echo "    ${n} record header(s) (expected 10,344)"
    [[ ${n} -ge 10000 ]]
}

get_medalcare_xl() {
    local d; d="$(raw_dir medalcare_xl)"
    local zip="${d}/MedalCare-XL.zip"
    fetch "${MEDALCARE_URL}" "${zip}" || return 1
    unpack "${zip}" "${d}" || return 1
    local n; n=$(find "${d}" -name '*_noise.csv' -not -path '*__MACOSX*' | wc -l)
    echo "    ${n} signal(s) in the 'noise' rendering (expected ~16,848)"
    [[ ${n} -ge 16000 ]]
}

get_mimic_iv_ecg() {
    local d; d="$(raw_dir mimic_iv_ecg)"
    local zip="${d}/${MIMIC_ZIP}"
    fetch "${MIMIC_URL}" "${zip}" || return 1
    zip_ok "${zip}" || { warn "${zip} does not open as a zip"; return 1; }
    if [[ "${UNPACK:-0}" == "1" ]]; then
        KEEP_ZIPS=1 unpack "${zip}" "${d}" || return 1
    else
        echo "    kept zipped: ECG/preprocess_ecg_corpus.sh reads records out of it"
    fi
}

get_code15() {
    local d; d="$(raw_dir code15)"
    local base="${CODE15_BASE}"
    mkdir -p "${d}"
    fetch "${base}/exams.csv/content" "${d}/exams.csv" || return 1
    local i zip h5
    for ((i = 0; i < CODE15_PARTS; i++)); do
        h5="${d}/exams_part${i}.hdf5"
        zip="${d}/exams_part${i}.zip"
        if [[ -s "${h5}" && ! -f "${zip}" ]]; then
            echo "    exams_part${i}.hdf5 present"
            continue
        fi
        fetch "${base}/exams_part${i}.zip/content" "${zip}" || return 1
        unpack "${zip}" "${d}" || return 1
    done
    local n; n=$(find "${d}" -name 'exams_part*.hdf5' | wc -l)
    echo "    ${n} of ${CODE15_PARTS} part file(s)"
    [[ ${n} -eq ${CODE15_PARTS} ]]
}

get_icentia11k() {
    local d; d="$(raw_dir icentia11k)"
    local zip="${d}/${ICENTIA_ZIP}"
    fetch "${ICENTIA_URL}" "${zip}" || return 1
    zip_ok "${zip}" || { warn "${zip} does not open as a zip"; return 1; }
    echo "    kept zipped (1.1 TB unpacked): records are read out of it"
}

heedb() {
    local d; d="$(raw_dir heedb)"
    mkdir -p "${d}"
    if ! have_working_aws || ! aws sts get-caller-identity >/dev/null 2>&1; then
        cat <<MSG
HEEDB is credentialed, and its agreement is yours to sign:
  1. On bdsp.io: an account with credentialed status, the CITI "Data or
     Specimens Only Research" course, and the BDSP Credentialed Health Data
     Use Agreement, then request access to HEEDB.
  2. bdsp.io issues AWS credentials for its S3 access point. Put them in a
     profile (aws configure --profile bdsp).
  3. AWS_PROFILE=bdsp bash scripts/download_ecg_pretrain_corpora.sh heedb
     syncs ${HEEDB_S3_URI}
     into ${d}. If bdsp.io gives you a different location, pass it as
     HEEDB_S3_URI=...
HEEDB is ~11.6 M ECGs; its size is not published. Expect on the order of a
terabyte, and check the quota under ${ECG_ROOT} first.
MSG
        return 2
    fi
    say "S3 ${HEEDB_S3_URI} -> ${d}"
    aws s3 sync --only-show-errors "${HEEDB_S3_URI}" "${d}"
}

code2() {
    cat <<MSG
CODE-II is not public: requests go to the Telehealth Network of Minas Gerais
(see the CODE-II paper, arXiv 2511.15632). Its distribution format is not
published -- the paper describes 2-4 tracings of 7-12 s per exam at
300-1000 Hz in a custom format -- so there is no reader for it yet.
Put the files in
  $(raw_dir code2)
and write one Adapter for them in ECG/preprocess_ecg_corpus.py (leads by
name, rate, unit, patient id); everything after reading is shared.
MSG
    return 2
}

run_one() {
    local ds="$1" rc
    if is_done "${ds}"; then
        say "${ds}: complete ($(cat "$(raw_dir "${ds}")/.complete")) -- skipping"
        return 0
    fi
    mkdir -p "$(raw_dir "${ds}")"
    say "${ds} -> $(raw_dir "${ds}")"
    "get_${ds}"
    rc=$?
    if [[ ${rc} -eq 0 ]]; then
        mark_done "${ds}"
        say "${ds}: complete"
    else
        warn "${ds}: not complete; rerun to continue"
    fi
    return ${rc}
}

status() {
    printf '%-18s %-10s %10s  %s\n' dataset state on-disk path
    local ds d state size
    for ds in ${OPEN_DATASETS} heedb code2; do
        d="$(raw_dir "${ds}")"
        if is_done "${ds}"; then state=complete
        elif [[ -d "${d}" ]] && [[ -n "$(ls -A "${d}" 2>/dev/null)" ]]; then state=partial
        else state=absent; fi
        size="$(du -sh "${d}" 2>/dev/null | cut -f1)"
        printf '%-18s %-10s %10s  %s\n' "${ds}" "${state}" "${size:--}" "${d}"
    done
    echo "free under ${ECG_ROOT}: $(free_gb "${ECG_ROOT}") GB"
}

main() {
    local target="${1:-}"
    case "${target}" in
        ""|-h|--help|help) usage; return 0 ;;
        status) status ;;
        layout)
            local ds
            for ds in ${OPEN_DATASETS} heedb code2; do mkdir -p "$(raw_dir "${ds}")"; done
            status ;;
        all)
            local ds want=0 failed=()
            for ds in ${OPEN_DATASETS}; do
                is_done "${ds}" || want=$(( want + $(need_gb "${ds}") ))
            done
            mkdir -p "${ECG_ROOT}"
            check_space "${want}"
            say "fetching into ${ECG_ROOT}: ${OPEN_DATASETS} (~${want} GB still to place)"
            for ds in ${OPEN_DATASETS}; do
                run_one "${ds}" || failed+=("${ds}")
            done
            if [[ "${WITH_HEEDB:-0}" == "1" ]]; then
                # Opt in: HEEDB is on the order of a terabyte, and `all`
                # should not start that because credentials happen to exist.
                heedb || failed+=(heedb)
            fi
            echo ""
            status
            echo ""
            [[ "${WITH_HEEDB:-0}" == "1" ]] \
                || echo "HEEDB and CODE-II are gated: bash $0 heedb / bash $0 code2 say what they need."
            if [[ ${#failed[@]} -gt 0 ]]; then
                warn "incomplete: ${failed[*]} -- rerun \`bash $0 all\` to continue"
                return 1
            fi ;;
        heedb) heedb ;;
        code2) code2 ;;
        norwegian_athlete|georgia|medalcare_xl|mimic_iv_ecg|code15|icentia11k)
            mkdir -p "${ECG_ROOT}"
            is_done "${target}" || check_space "$(need_gb "${target}")"
            run_one "${target}" ;;
        *) usage; die "unknown target '${target}'" ;;
    esac
}

main "$@"
