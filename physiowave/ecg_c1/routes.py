"""
The routes and the corpus of the ECG C1 pretraining run.

Same idea as physiowave.eeg_c1.routes: a route is the (lead count, sampling
rate) shape a batch arrives in, chosen by ``route_id`` carried in the data. It
is not learned and it is not a mixture of experts -- whether a recording is a
12-lead resting ECG or a single-lead ambulatory patch is known before the model
sees it.

Two routes, because the corpus has exactly two shapes in it:

    route      leads  rate   window         patch         tokens
    L12_500    12     500    10 s = 5000    0.5 s = 250   12 x 20 = 240
    L1_250      1     250    10 s = 2500    0.5 s = 125    1 x 20 =  20

L12_500 is every 12-lead corpus. 500 Hz is the native rate of MIMIC-IV-ECG,
Georgia, MedalCare-XL, the Norwegian athletes and HEEDB, and it is what the
repository's ECG fine-tuning already feeds the model; CODE-15% and CODE-II are
recorded at 400 Hz and are resampled up, which each shard records in its
provenance. L1_250 is Icentia11k, a single chest-patch lead at its native
250 Hz -- putting it on the 12-lead route would mean eleven padded rows in
every window, which is eleven twelfths of the compute spent on zeros.

TEN SECONDS, because that is what a resting ECG is: MIMIC-IV-ECG, Georgia,
MedalCare-XL, the athletes and HEEDB store 10 s records, so a record is one
window and nothing is cut or padded. A 0.5 s patch is the EEG route's patch
length too, which keeps the token count per second identical across the two
modalities; it is a judgement, not a measurement.

WHAT TEN SECONDS COSTS. CODE-15% is not a 10 s corpus: measured over ~13,000
of its exams, 55-59% are 2934 samples at 400 Hz (7.3 s) and only 37-41% are
10.24 s. A 7.3 s exam holds no 10 s window, so under this setting roughly
three fifths of CODE-15% contributes nothing -- preprocessing counts them as
``records_shorter_than_window``. CODE-II's 7-12 s tracings lose their short
ones the same way, and Georgia's 52 five-second records are dropped.
``WINDOW_SECONDS = 5.0`` keeps every one of them (a 10 s record then gives two
windows) at the price of half the rhythm context per window. It is one
number, below, and it requires re-preprocessing every corpus: the shards
store the window, and the model refuses a window of the wrong length.

As on the EEG side, the rates share nothing but the Transformer: each route has
its own wavelet frontend (the filters are per-lead), each RATE its own patch
embedding and decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..eeg_c1.routes import Route
from .leads import LEADS_12, LIMB_LEADS

#: Icentia11k's one lead. Its own vocabulary entry -- see leads.py for why it
#: is not called lead I.
SLOTS_PATCH: Tuple[str, ...] = ("patch1",)

#: The window every route uses, and preprocessing's default. See above.
WINDOW_SECONDS = 10.0
PATCH_SECONDS = 0.5

ROUTES: Dict[str, Route] = {
    r.route_id: r
    for r in (
        Route("L12_500", 12, 500, window_seconds=WINDOW_SECONDS,
              patch_seconds=PATCH_SECONDS, slots=LEADS_12),
        Route("L1_250", 1, 250, window_seconds=WINDOW_SECONDS,
              patch_seconds=PATCH_SECONDS, slots=SLOTS_PATCH),
    )
}
ROUTE_IDS: Tuple[str, ...] = tuple(ROUTES)
RATE_KEYS: Tuple[str, ...] = tuple(
    sorted({r.rate_key for r in ROUTES.values()}, key=int))

#: Per route, the slot indices that are masked TOGETHER at a time patch. The
#: limb leads are six views of one two-dimensional frontal-plane vector, so any
#: two determine the rest exactly; masking them independently would leave the
#: model half its masked limb patches to compute rather than infer.
LEAD_GROUPS: Dict[str, Tuple[Tuple[int, ...], ...]] = {
    "L12_500": (tuple(LEADS_12.index(n) for n in LIMB_LEADS),),
    "L1_250": (),
}


@dataclass(frozen=True)
class ECGDatasetSpec:
    """One pretraining corpus: where it goes, and what it takes to get it."""

    dataset_id: str
    route_id: str
    native_rate: Optional[float]
    native_leads: int
    #: Mains frequency where the recording was made, for the notch. None for
    #: simulated data, which has no mains to remove.
    mains_hz: Optional[float]
    #: open | credentialed | restricted -- what the download script can do.
    access: str
    #: Directory under the ECG download root.
    raw_dir: str
    #: A cap on windows per record, for the continuous corpora. None keeps all.
    default_max_windows_per_record: Optional[int] = None
    notes: str = ""

    @property
    def route(self) -> Route:
        return ROUTES[self.route_id]


PRETRAIN_DATASETS: Dict[str, ECGDatasetSpec] = {
    d.dataset_id: d
    for d in (
        ECGDatasetSpec("mimic_iv_ecg", "L12_500", 500.0, 12, 60.0, "open",
                       "MIMIC-IV-ECG",
                       notes="~800k 10 s 12-lead ECGs, ~160k subjects, Beth "
                             "Israel (US); files store aVF before aVL"),
        ECGDatasetSpec("code15", "L12_500", 400.0, 12, 60.0, "open",
                       "CODE-15",
                       notes="345,779 exams, 233,770 patients; 400 -> 500 Hz, "
                             "zero padding stripped; ~59% are 7.3 s; Brazil"),
        ECGDatasetSpec("medalcare_xl", "L12_500", 500.0, 12, None, "open",
                       "MedalCare-XL",
                       notes="16,842 SIMULATED 12-lead ECGs from 13 torso "
                             "models; the 'noise' variant; no mains"),
        ECGDatasetSpec("norwegian_athlete", "L12_500", 500.0, 12, 50.0, "open",
                       "NorwegianAthlete",
                       notes="28 athletes; Norway, 50 Hz. EACH LEAD IS "
                             "RESCALED to full int16 range in the files, so "
                             "its inter-lead amplitudes are not physical"),
        ECGDatasetSpec("georgia", "L12_500", 500.0, 12, 60.0, "open",
                       "Georgia",
                       notes="10,344 ECGs, PhysioNet/CinC Challenge 2021; no "
                             "patient id; 52 are 5 s"),
        ECGDatasetSpec("heedb", "L12_500", None, 12, 60.0, "credentialed",
                       "HEEDB",
                       notes="Harvard-Emory ECG Database, ~11.6M ECGs, ~2.2M "
                             "patients; 250 or 500 Hz per record; bdsp.io "
                             "credentialed access"),
        ECGDatasetSpec("icentia11k", "L1_250", 250.0, 1, 60.0, "open",
                       "Icentia11k", default_max_windows_per_record=16,
                       notes="11,000 patients, up to 2 weeks each; capped "
                             "windows per segment"),
        ECGDatasetSpec("code2", "L12_500", None, 12, 60.0, "restricted",
                       "CODE-II",
                       notes="~2.7M exams; access by request only; its file "
                             "format is not public, so it has no reader yet"),
    )
}
DATASET_IDS: Tuple[str, ...] = tuple(PRETRAIN_DATASETS)

#: Never pretrained on: the repository's ECG fine-tuning benchmarks
#: (ECG/ptbxl_finetune.py, ECG/cpsc_multilabel.py, ECG/shaoxing_multilabel.py).
#: Named rather than merely absent, so preprocessing and the merge refuse them
#: by name and a test pins it.
DOWNSTREAM_ONLY: Tuple[str, ...] = ("ptbxl", "cpsc2018", "chapman_shaoxing")

#: Per-rank micro-batch. A 12-lead window is 240 tokens and a single-lead one
#: is 20, so one batch size would either starve L1_250 or exhaust memory on
#: L12_500. These keep tokens per step within ~15% of each other; the trainer's
#: banner prints the window share each dataset actually gets.
DEFAULT_BATCH_BY_ROUTE: Dict[str, int] = {
    "L12_500": 96,
    "L1_250": 1024,
}
