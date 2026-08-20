#!/bin/bash

# ============================================================================
# Fetch Sleep-EDFx Sleep Cassette straight from PhysioNet.
# ============================================================================
# EEG/sleep_edf_finetune.py normally lets MNE download each recording as it
# needs it: one request per file, serial, no resume. That is fine on a laptop
# and painful on a login node with a wall clock and ~8 GB to move.
#
# This fetches the same files with wget -- resumable, restartable, and
# parallelisable across shells -- and puts them where MNE looks, so the
# preparation script finds them and downloads nothing.
#
# It then verifies them, because MNE will not. mne/datasets/sleep_physionet's
# _fetch_one returns a pre-existing file immediately:
#
#     destination = op.join(path, fname)
#     if op.isfile(destination) and not force_update:
#         return destination, False
#
# The SHA1 it knows is only consulted when pooch actually downloads, so a
# truncated file placed here by hand is read as a short recording rather than
# reported. The check below uses MNE's own SHA1SUMS, the same list it would
# have compared against had it done the download itself.
#
# Usage:
#   export MNE_DATA=$SCRATCH/mne_data
#   bash EEG/download_sleep_edf.sh
#   python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf
#
# Run it on a login node -- compute nodes have no route out.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
MNE_ROOT="${MNE_DATA:-$HOME/mne_data}"
DEST="${MNE_ROOT}/physionet-sleep-data"

command -v wget >/dev/null || { echo "ERROR: wget not found" >&2; exit 1; }

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    if ! curl -sI --max-time 15 https://physionet.org >/dev/null 2>&1; then
        cat >&2 <<MSG
ERROR: cannot reach physionet.org.

  Compute nodes on Leonardo have no route to the internet. Run this on a
  login node, then run the training where the GPUs are.
MSG
        exit 1
    fi
fi

# MNE raises rather than creating MNE_DATA itself, so both levels are made here.
mkdir -p "${DEST}"
echo "Destination: ${DEST}"
echo "Source:      ${BASE}"
echo

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    # -np             do not walk up to the parent directory
    # -nd             flat output; MNE wants one directory, not a subtree
    # -c -N           resume partial files, skip ones not newer on the server
    # -A '*.edf'      the checksum and index files are not what MNE reads
    # --tries         PhysioNet drops long connections often enough to matter
    wget -r -np -nd -c -N --tries=5 --waitretry=10 -q --show-progress \
         -A '*.edf' -P "${DEST}" "${BASE}"
fi

psg=$(find "${DEST}" -name '*-PSG.edf' | wc -l | tr -d ' ')
hyp=$(find "${DEST}" -name '*-Hypnogram.edf' | wc -l | tr -d ' ')
echo
echo "In ${DEST}: ${psg} PSG, ${hyp} hypnogram files"
[[ "${psg}" -gt 0 ]] || { echo "ERROR: nothing downloaded" >&2; exit 1; }

echo
echo "Verifying against MNE's SHA1SUMS (this reads every byte; give it a minute)..."
python - "${DEST}" <<'PYEOF'
import hashlib
import os
import sys

dest = sys.argv[1]
try:
    import mne.datasets.sleep_physionet as sp
except ImportError:
    raise SystemExit("  mne is not installed, so the files cannot be verified.\n"
                     "  Install it (pip install braindecode) and rerun, or accept\n"
                     "  the risk that a truncated file becomes a short recording.")

sums = os.path.join(os.path.dirname(sp.__file__), "SHA1SUMS")
if not os.path.isfile(sums):
    raise SystemExit(f"  MNE has no SHA1SUMS at {sums}; cannot verify.")

expected = {}
with open(sums) as f:
    for line in f:
        parts = line.strip().split("  ")
        if len(parts) == 2 and parts[1].endswith(".edf"):
            expected[os.path.basename(parts[1])] = parts[0]

ok = bad = unknown = 0
for name in sorted(os.listdir(dest)):
    if not name.endswith(".edf"):
        continue
    want = expected.get(name)
    if want is None:
        unknown += 1
        continue
    h = hashlib.sha1()
    with open(os.path.join(dest, name), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() == want:
        ok += 1
    else:
        bad += 1
        print(f"  CORRUPT {name}")

print(f"  {ok} verified, {bad} corrupt, {unknown} not in MNE's list")
if bad:
    raise SystemExit("  Delete the corrupt files and rerun this script to refetch them.")
PYEOF

echo
echo "Next:  python EEG/sleep_edf_finetune.py --out-dir \$PW_DATA_EEG/sleep_edf"
