#!/bin/bash
# ============================================================================
# Fetch the ECG C1 pretraining corpora into one layout under $ECG_ROOT.
#
#   bash scripts/download_ecg_pretrain_corpora.sh status        # what is there
#   bash scripts/download_ecg_pretrain_corpora.sh all           # all seven
#   bash scripts/download_ecg_pretrain_corpora.sh mimic_iv_ecg  # just one
#
#   nohup bash scripts/download_ecg_pretrain_corpora.sh all > ~/ecg_download.log 2>&1 &
#
# `all` fetches all seven corpora, smallest first. Every step is resumable
# and idempotent: a finished corpus
# carries a .complete marker and is skipped, a partial download continues from
# where it stopped, and rerunning `all` after an interruption is the way to
# continue it.
#
# RUN IT ON A LOGIN NODE. Compute nodes here have no route to the internet.
# A login node has also killed a long transfer before (HBN, at 164 of 224 GB),
# and a process the node kills is being stopped on purpose -- so this does NOT
# loop to restart after one. If it is killed, rerun it later, or move the big
# ones (PulseDB 388 GB, Icentia11k 202 GB, CODE-15% 46 GB) through the
# data-transfer service CINECA provides for this.
#
# LAYOUT -- what ECG/preprocess_ecg_corpus.sh expects:
#
#   $ECG_ROOT/MIMIC-IV-ECG/raw/<zip>        read in place, not unpacked
#   $ECG_ROOT/CODE-15/raw/exams_part*.hdf5, exams.csv
#   $ECG_ROOT/SPH/raw/records/A*.h5, metadata.csv
#   $ECG_ROOT/Georgia/raw/...               .hea + .mat
#   $ECG_ROOT/MedalCare-XL/raw/...          unpacked
#   $ECG_ROOT/Icentia11k/raw/<zip>          read in place (1.1 TB unpacked)
#   $ECG_ROOT/PulseDB/raw/PulseDB_{MIMIC,Vital}.zip.NNN
#                                           the Box pieces, read in place
#   $ECG_ROOT/ecg_c1_corpus/<dataset>/      written by preprocessing
#
# SIZES (verified against the servers, 2026-09-29), and what stays on disk:
#   Georgia             ~1.25 GB, 20,688 files over HTTP (no zip, no open S3)
#   SPH                 2.28 GB tar (not gzipped, despite its name), unpacked
#   MedalCare-XL        9.3 GB zip                  -> 28.2 GB unpacked
#   MIMIC-IV-ECG        36.3 GB zip, kept zipped    (90.4 GB if unpacked)
#   CODE-15%            46.3 GB in 18 zips          -> ~68 GB of hdf5
#   Icentia11k          202.2 GB zip, kept zipped   (1.1 TB if unpacked)
#   PulseDB             388.4 GB in 26 Box pieces, kept as they are
#                       (~636 GB if unpacked; almost all of it PPG and ABP)
#   about 730 GB in all once done; `all` checks free space before it starts.
# Every source answers ranged requests, so an interrupted file resumes.
#
# ENVIRONMENT:
#   ECG_ROOT    download root (/leonardo_scratch/large/userexternal/ychen003/bio/ecg)
#   KEEP_ZIPS   1: keep the CODE-15% / MedalCare / SPH archives after unpacking
#   PULSEDB_HALVES    "MIMIC Vital" (default), or just one of the two. The
#                     VitalDB half is CC BY-NC-SA 4.0 (non-commercial,
#                     share-alike); the MIMIC half is ODbL.
#   UNPACK      1: also unpack MIMIC-IV-ECG (90 GB) instead of reading the zip
#   SKIP_SPACE_CHECK  1: do not refuse to start when free space looks short
# ============================================================================

set -uo pipefail

ECG_ROOT="${ECG_ROOT:-/leonardo_scratch/large/userexternal/ychen003/bio/ecg}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNPACK_PY="${HERE}/unpack_archive.py"
PW_REPO="$(cd "${HERE}/.." && pwd)"
export PW_REPO
PYTHON="${PYTHON:-python}"

OPEN_DATASETS="georgia sph medalcare_xl mimic_iv_ecg code15 icentia11k pulsedb"

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
# Challenge 2021 has no zip and no open S3 prefix; the Challenge 2020 tarball
# on Google Cloud is gone (NoSuchBucket). Recursive HTTP is what remains.
GEORGIA_HTTP="${GEORGIA_HTTP:-https://physionet.org/files/challenge-2021/1.0.3/training/georgia/}"
CODE15_BASE="${CODE15_BASE:-https://zenodo.org/api/records/4916206/files}"
CODE15_PARTS="${CODE15_PARTS:-18}"
MEDALCARE_URL="${MEDALCARE_URL:-https://zenodo.org/api/records/8068944/files/MedalCare-XL.zip/content}"
# SPH, figshare collection 5779802 (CC0). ndownloader redirects to an S3 URL
# signed for ten seconds; curl -L re-asks each time, so resuming works.
SPH_RECORDS_URL="${SPH_RECORDS_URL:-https://ndownloader.figshare.com/files/32630684}"
SPH_RECORDS_SIZE="${SPH_RECORDS_SIZE:-2281799680}"
SPH_METADATA_URL="${SPH_METADATA_URL:-https://ndownloader.figshare.com/files/34793152}"
PULSEDB_HALVES="${PULSEDB_HALVES:-MIMIC Vital}"
# PulseDB's pieces, from the project README: name, bytes, Box link. Each half
# is ONE ZIP64 archive cut into 15 GB pieces -- they concatenate to an ordinary
# zip, and preprocessing reads them as one without doing so.
PULSEDB_PARTS="${PULSEDB_PARTS:-
PulseDB_MIMIC.zip.001 15728640000 https://rutgers.box.com/shared/static/7l8n3tn9tr0602tdss1x7e3uliahlibp.001
PulseDB_MIMIC.zip.002 15728640000 https://rutgers.box.com/shared/static/zco48rvz5dog72970679foen6hct15c8.002
PulseDB_MIMIC.zip.003 15728640000 https://rutgers.box.com/shared/static/x22qpmelx6sz3wgkm5qyc0eis429361f.003
PulseDB_MIMIC.zip.004 15728640000 https://rutgers.box.com/shared/static/xj25sqnluiz6s4z8tzzm5phk00ohp6e8.004
PulseDB_MIMIC.zip.005 15728640000 https://rutgers.box.com/shared/static/dxus2lsoop02chaspnwipwrf0g4wmenr.005
PulseDB_MIMIC.zip.006 15728640000 https://rutgers.box.com/shared/static/rts6sj441laenm2sy1qcemg7ke4om3j6.006
PulseDB_MIMIC.zip.007 15728640000 https://rutgers.box.com/shared/static/vor4hjllld7a0c3nzef8uptbb4ut3koo.007
PulseDB_MIMIC.zip.008 15728640000 https://rutgers.box.com/shared/static/a2qg2p4ebyrooji3z88djlokji65tlf3.008
PulseDB_MIMIC.zip.009 15728640000 https://rutgers.box.com/shared/static/uh6kbiuqgnib5wakiv6o35gkpusyamc7.009
PulseDB_MIMIC.zip.010 15728640000 https://rutgers.box.com/shared/static/h6eyhkkx48pf3ce3th1clwj43hn98j5c.010
PulseDB_MIMIC.zip.011 15728640000 https://rutgers.box.com/shared/static/e93dp94hxpkas45yc59n289s2wvkafgi.011
PulseDB_MIMIC.zip.012 15728640000 https://rutgers.box.com/shared/static/iuvyuw7dmlxvbjvt53dj49wqn3gelqni.012
PulseDB_MIMIC.zip.013 15728640000 https://rutgers.box.com/shared/static/qxx6tjz8c3778601ib3icu6o1rranmc7.013
PulseDB_MIMIC.zip.014 15728640000 https://rutgers.box.com/shared/static/ip2ninwqj8437l9fyffjprnk90ptnx9k.014
PulseDB_MIMIC.zip.015 15728640000 https://rutgers.box.com/shared/static/yrtbo0lg8mjhaw624iw9bbhk1obbocwd.015
PulseDB_MIMIC.zip.016 9372260162 https://rutgers.box.com/shared/static/wmzndowgfa5xi3tvtqahxkld3ngdyjds.016
PulseDB_Vital.zip.001 15728640000 https://rutgers.box.com/shared/static/vtxoksmn7emeaxypb2prywgwscuefoqa.001
PulseDB_Vital.zip.002 15728640000 https://rutgers.box.com/shared/static/euzkek7c3xoy62jisheuxqar7z5y8xig.002
PulseDB_Vital.zip.003 15728640000 https://rutgers.box.com/shared/static/49lngo0benxfjw193jnqz9tctlyb3qam.003
PulseDB_Vital.zip.004 15728640000 https://rutgers.box.com/shared/static/jf4fwgkmhry20mf5tcg9t0wxvky64um0.004
PulseDB_Vital.zip.005 15728640000 https://rutgers.box.com/shared/static/2lgxysbskfuapsaan4jypvmm8316fdkc.005
PulseDB_Vital.zip.006 15728640000 https://rutgers.box.com/shared/static/x27ktb4qsx43razwo4tjmxq9v1ro0x3y.006
PulseDB_Vital.zip.007 15728640000 https://rutgers.box.com/shared/static/q0t36fikgf3pimhvnerwwnovfr0umtp8.007
PulseDB_Vital.zip.008 15728640000 https://rutgers.box.com/shared/static/ihckx2g0f981g5yz2x8v5rgwndl6yebw.008
PulseDB_Vital.zip.009 15728640000 https://rutgers.box.com/shared/static/y8j14h8tvi5b3du8nap9dnura1omfrk6.009
PulseDB_Vital.zip.010 1499811157 https://rutgers.box.com/shared/static/fu0m9tx33jkxywq32shh0g8dg3not15u.010
}"
usage() {
    # The header comment, up to the first line of code.
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
        "${BASH_SOURCE[0]}"
    echo "targets: status all layout ${OPEN_DATASETS}"
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
        georgia)           echo "${ECG_ROOT}/Georgia/raw" ;;
        sph)               echo "${ECG_ROOT}/SPH/raw" ;;
        pulsedb)           echo "${ECG_ROOT}/PulseDB/raw" ;;
        icentia11k)        echo "${ECG_ROOT}/Icentia11k/raw" ;;
        *) die "unknown dataset $1" ;;
    esac
}

#: GB each corpus needs on disk once this script is done with it.
need_gb() {
    case "$1" in
        georgia)           echo 2 ;;
        sph)               echo 5 ;;
        pulsedb)           echo 390 ;;
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
    # A one-byte ranged GET, and the total from its Content-Range. HEAD would
    # be the obvious probe, and Box answers it 404 and figshare's signed S3
    # URL 403 -- a GET is what they are signed for.
    local n
    n="$(curl -sL -r 0-0 -D - -o /dev/null --connect-timeout 20 "$1" 2>/dev/null \
        | tr -d '\r' | awk 'tolower($1)=="content-range:" {sub(/.*\//, "", $3); n=$3} END {print n}')"
    if [[ -z "${n}" ]]; then
        n="$(curl -sIL --connect-timeout 20 "$1" 2>/dev/null \
            | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n}')"
    fi
    echo "${n}"
}

local_size() {
    [[ -f "$1" ]] || { echo 0; return; }
    if stat -c%s "$1" >/dev/null 2>&1; then stat -c%s "$1"; else stat -f%z "$1"; fi
}

#: fetch URL DEST [BYTES] -- resumable, and complete only when the size
#: matches: BYTES when the size is known in advance, else what the server says.
fetch() {
    local url="$1" dest="$2" want="${3:-}" have rc
    mkdir -p "$(dirname "${dest}")"
    [[ -n "${want}" ]] || want="$(remote_size "${url}")"
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

get_sph() {
    local d; d="$(raw_dir sph)"
    local tar="${d}/records.tar"
    mkdir -p "${d}"
    fetch "${SPH_METADATA_URL}" "${d}/metadata.csv" || return 1
    local n; n=$(find "${d}/records" -name 'A*.h5' 2>/dev/null | wc -l)
    if [[ ${n} -lt 25770 ]]; then
        # Published as records.tar.gz, but it is a plain tar: `tar xzf`
        # would refuse it.
        fetch "${SPH_RECORDS_URL}" "${tar}" "${SPH_RECORDS_SIZE}" || return 1
        say "unpacking records.tar"
        tar -xf "${tar}" -C "${d}" || return 1
        [[ "${KEEP_ZIPS:-0}" == "1" ]] || rm -f "${tar}"
        n=$(find "${d}/records" -name 'A*.h5' | wc -l)
    fi
    echo "    ${n} record(s) (expected 25,770)"
    [[ ${n} -eq 25770 ]]
}

get_pulsedb() {
    local d; d="$(raw_dir pulsedb)"
    mkdir -p "${d}"
    local name bytes url half h want_half
    while read -r name bytes url; do
        [[ -n "${name}" ]] || continue
        half="${name#PulseDB_}"; half="${half%%.*}"
        want_half=0
        for h in ${PULSEDB_HALVES}; do [[ "${h}" == "${half}" ]] && want_half=1; done
        [[ ${want_half} -eq 1 ]] || continue
        fetch "${url}" "${d}/${name}" "${bytes}" || return 1
    done <<< "${PULSEDB_PARTS}"
    # Every piece at the right size is necessary, not sufficient: the
    # archive's directory lives in its last piece and has to read back.
    for half in ${PULSEDB_HALVES}; do
        "${PYTHON}" - "${d}" "PulseDB_${half}.zip" <<'PYEOF' || return 1
import os, sys, zipfile
sys.path.insert(0, os.path.join(os.environ["PW_REPO"], "ECG"))
from preprocess_ecg_corpus import _open_archive
root, name = sys.argv[1], sys.argv[2]
with _open_archive(os.path.join(root, name)) as fh, zipfile.ZipFile(fh) as zf:
    n = sum(1 for i in zf.infolist() if i.filename.endswith(".mat"))
print(f"    {name}: {n} subject file(s) in the archive directory")
sys.exit(0 if n else 1)
PYEOF
    done
    echo "    read in place: ECG/preprocess_ecg_corpus.sh inflates one subject at a time"
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
    for ds in ${OPEN_DATASETS}; do
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
            for ds in ${OPEN_DATASETS}; do mkdir -p "$(raw_dir "${ds}")"; done
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
            echo ""
            status
            echo ""
            if [[ ${#failed[@]} -gt 0 ]]; then
                warn "incomplete: ${failed[*]} -- rerun \`bash $0 all\` to continue"
                return 1
            fi ;;
        georgia|sph|medalcare_xl|mimic_iv_ecg|code15|icentia11k|pulsedb)
            mkdir -p "${ECG_ROOT}"
            is_done "${target}" || check_space "$(need_gb "${target}")"
            run_one "${target}" ;;
        *) usage; die "unknown target '${target}'" ;;
    esac
}

main "$@"
