#!/bin/bash
# ============================================================================
# Fetch KaggleERN (BCI Challenge @ NER 2015) and lay it out for the converter.
#
#   bash EEG/download_kaggle_ern.sh $PW_DATA_EEG/kaggle_ern
#
# TWO THINGS ONLY YOU CAN DO, once, before this works:
#
#   1. Accept the competition rules while logged in to Kaggle:
#        https://www.kaggle.com/c/inria-bci-challenge/rules
#      Without it the API answers 403, which reads like a bad token.
#   2. Give the API a token: Kaggle -> Settings -> API -> "Create New Token"
#      downloads kaggle.json. Put it at ~/.kaggle/kaggle.json, chmod 600.
#      (Or export KAGGLE_USERNAME and KAGGLE_KEY instead.)
#
# The kaggle CLI goes into a venv of its own, run with PYTHONPATH unset, for
# the reason scripts/upload_hf_release.sh does: the cineca-ai module puts its
# site-packages on PYTHONPATH ahead of any venv, so a package installed into
# the training venv can import the module's copy of its dependencies instead.
#
# Result, the layout EEG/kaggle_ern_finetune.py and EEGPT both expect:
#
#   <dest>/TrainLabels.csv  true_labels.csv  SampleSubmission.csv
#   <dest>/train/Data_S02_Sess01.csv ...     16 subjects x 5 sessions
#   <dest>/test/Data_S01_Sess01.csv  ...     10 subjects x 5 sessions
# ============================================================================

set -uo pipefail

DEST="${1:-${PW_DATA_EEG:-.}/kaggle_ern}"
COMP="inria-bci-challenge"
VENV="${KAGGLE_VENV:-${HOME}/kaggleenv}"
KAGGLE_BIN="${KAGGLE_BIN:-${VENV}/bin/kaggle}"

mkdir -p "${DEST}" || exit 1
DEST="$(cd "${DEST}" && pwd)"
echo "KaggleERN -> ${DEST}"

# --- the CLI, in its own venv ------------------------------------------------ #
if [[ "${KAGGLE_BIN}" == "${VENV}/bin/kaggle" ]]; then
    if [[ ! -x "${VENV}/bin/python" ]]; then
        echo "  creating ${VENV} (no system site-packages)"
        env -u PYTHONPATH python3 -m venv "${VENV}" || exit 1
    fi
    env -u PYTHONPATH "${VENV}/bin/pip" install -q -U pip kaggle || {
        echo "ERROR: could not install the kaggle CLI (outbound HTTPS?)" >&2; exit 1; }
fi

if [[ -z "${KAGGLE_USERNAME:-}" && ! -f "${HOME}/.kaggle/kaggle.json" \
      && ! -f "${HOME}/.config/kaggle/kaggle.json" ]]; then
    cat >&2 <<EOF
ERROR: no Kaggle credentials.
  Kaggle -> Settings -> API -> Create New Token, then:
      mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  and accept the rules at https://www.kaggle.com/c/${COMP}/rules
EOF
    exit 1
fi

# --- download ---------------------------------------------------------------- #
ARCHIVE="${DEST}/${COMP}.zip"
if [[ ! -s "${ARCHIVE}" ]]; then
    echo "  downloading ${COMP} (a few GB)"
    if ! env -u PYTHONPATH "${KAGGLE_BIN}" competitions download -c "${COMP}" -p "${DEST}"; then
        cat >&2 <<EOF
ERROR: the download failed. A 403 almost always means the competition rules
       have not been accepted for this account:
           https://www.kaggle.com/c/${COMP}/rules
       A 401 means the token is wrong or expired.
EOF
        exit 1
    fi
fi

# --- unpack ------------------------------------------------------------------ #
# The archive holds train.zip and test.zip, and whether those put their CSVs in
# a train/ folder or at the top level is not something to rely on. So every
# Data_S*.csv is found wherever it lands and moved to the folder its archive
# names -- the files are named identically in both, and which one a file came
# from is the only thing that says whether its labels are in TrainLabels.csv.
python3 - "${ARCHIVE}" "${DEST}" <<'PY'
import glob, os, shutil, sys, tempfile, zipfile

archive, dest = sys.argv[1], sys.argv[2]
work = tempfile.mkdtemp(dir=dest)
with zipfile.ZipFile(archive) as z:
    z.extractall(work)
for inner in glob.glob(os.path.join(work, "**", "*.zip"), recursive=True):
    with zipfile.ZipFile(inner) as z:
        z.extractall(inner[:-4])
    os.remove(inner)
for name in ("TrainLabels.csv", "true_labels.csv", "SampleSubmission.csv",
             "ChannelsLocation.csv"):
    hits = glob.glob(os.path.join(work, "**", name), recursive=True)
    if hits:
        shutil.move(hits[0], os.path.join(dest, name))
for split in ("train", "test"):
    os.makedirs(os.path.join(dest, split), exist_ok=True)
    roots = [d for d in glob.glob(os.path.join(work, "**", split), recursive=True)
             if os.path.isdir(d)]
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "Data_S*.csv"), recursive=True):
            shutil.move(f, os.path.join(dest, split, os.path.basename(f)))
shutil.rmtree(work)
PY
[[ $? -eq 0 ]] || { echo "ERROR: unpacking ${ARCHIVE} failed" >&2; exit 1; }

# --- verify ------------------------------------------------------------------ #
n_train=$(ls "${DEST}"/train/Data_S*.csv 2>/dev/null | wc -l | tr -d ' ')
n_test=$(ls "${DEST}"/test/Data_S*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "  train sessions ${n_train} (expect 80)   test sessions ${n_test} (expect 50)"
rc=0
for f in TrainLabels.csv true_labels.csv; do
    if [[ -f "${DEST}/${f}" ]]; then
        echo "  ${f}  $(($(wc -l < "${DEST}/${f}") - 1)) rows"
    else
        echo "  MISSING ${f}" >&2
        rc=1
    fi
done
if [[ ! -f "${DEST}/true_labels.csv" ]]; then
    cat >&2 <<EOF

  true_labels.csv holds the TEST subjects' labels, and EEGPT scores against
  it. It was not in this download. Look on the competition's data page for a
  labels file released after the competition, and put it at
      ${DEST}/true_labels.csv
  (a 'label' column, one row per test feedback, in SampleSubmission.csv order,
  or an IdFeedBack column). Without it the converter refuses to build a test set.
EOF
fi
[[ "${n_train}" -gt 0 && "${n_test}" -gt 0 ]] || rc=1
[[ ${rc} -eq 0 ]] && rm -f "${ARCHIVE}" && echo "  done; archive removed"
exit ${rc}
