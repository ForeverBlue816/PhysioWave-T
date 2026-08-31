"""What is actually on disk: the caches, and the splits already built."""
import glob, os, sys
import numpy as np

base = sys.argv[1]                      # $PW_DATA_EEG
for d in sorted(glob.glob(os.path.join(base, "erpbci*", "cache", "c*"))):
    npz = sorted(glob.glob(os.path.join(d, "sub*.npz")))
    if not npz:
        print(f"{d:<60} EMPTY"); continue
    with np.load(npz[0]) as z:
        names = "in the file" if "channel_names" in z.files else "NOT in the file"
        c = z["data"].shape[1]
    print(f"{d:<60} {len(npz)} subjects, {c} channels, names {names}")

import h5py
for h5 in sorted(glob.glob(os.path.join(base, "*", "train.h5"))):
    with h5py.File(h5, "r") as f:
        c = f["data"].shape[1]
        fs = f.attrs.get("sampling_rate", "MISSING")
        ch = [x.decode() for x in f["channel_names"][:3]] if "channel_names" in f else "?"
    print(f"{h5:<60} {f'{c} ch':>7}, sampling_rate {fs}, first {ch}")
