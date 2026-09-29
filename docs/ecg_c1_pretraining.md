# ECG C1 multi-route pretraining

The EEG C1 model — per-route wavelet frontend with a dynamic ScaleFold, one
shared RoPE Transformer, C1 channel-name embedding at the token site — trained
on ECG. About 30 M parameters (512 wide, 9 deep, 8 heads: 30.08 M in total,
28.64 M in the encoder that fine-tuning inherits).

## From download to a running job

```bash
# 1. download (login node; resumable, rerun to continue)
bash scripts/download_ecg_pretrain_corpora.sh status
nohup bash scripts/download_ecg_pretrain_corpora.sh all > ~/ecg_download.log 2>&1 &

# 2. look at each corpus before processing it -- the amplitude column is the unit check
DATASET=mimic_iv_ecg INSPECT=40 bash ECG/preprocess_ecg_corpus.sh

# 3. preprocess, one array per corpus (see the sbatch header for sizes)
sbatch --export=ALL,DATASET=mimic_iv_ecg --array=0-15 scripts/slurm/cineca_ecg_corpus_preprocess.sbatch

# 4. merge whatever has finished
python scripts/build_eeg_c1_manifest.py --modality ecg \
    --corpus-root $ECG_ROOT/ecg_c1_corpus --allow-missing --check-shards --jobs 16

# 5. train (RESUME=auto to continue)
sbatch scripts/slurm/cineca_ecg_c1_moe_pretrain.sbatch
```

`ECG_ROOT` defaults to `/leonardo_scratch/large/userexternal/ychen003/bio/ecg`,
next to the EEG corpora. Scratch is purged after 40 days; the corpus is
rebuildable from the raw downloads, checkpoints are not and go to `$FAST`.

A smoke run needs no data at all:

```bash
python -m physiowave.train.pretrain_main --config pretrain/ecg_c1_moe --smoke-test --max-steps 24
```

## Routes

| route     | leads | rate   | window | patch | tokens | corpora |
|-----------|-------|--------|--------|-------|--------|---------|
| `L12_500` | 12    | 500 Hz | 10 s   | 0.5 s | 240    | MIMIC-IV-ECG, CODE-15%, MedalCare-XL, Norwegian athletes, Georgia, HEEDB, CODE-II |
| `L1_250`  | 1     | 250 Hz | 10 s   | 0.5 s | 20     | Icentia11k |

As on the EEG side, the route is a property of the recording, carried in the
data, never learned. Each route has its own wavelet frontend (the filters are
per lead), each sampling rate its own patch embedding and decoders, and
everything else is shared. CODE is recorded at 400 Hz and HEEDB partly at
250 Hz; both are resampled to 500, and each shard records it.

**What 10 s costs.** Most corpora store 10 s records, so a record is one
window. CODE-15% does not: about 59% of its exams are 7.3 s and give no 10 s
window. Preprocessing reports them as `records_shorter_than_window`, and
Georgia's 52 five-second records are dropped the same way.
`WINDOW_SECONDS = 5.0` in `physiowave/ecg_c1/routes.py` keeps all of them,
and a 10 s record then gives two windows. The price is half the rhythm
context per window, and every corpus must be preprocessed again.

## What differs from EEG, and why

**Wavelets.** The frontend starts from db4, db6, sym4, sym8 and coif2, the
families used in the ECG literature for QRS and P/T-wave decomposition. All
are 8–16 taps, so the `pad` initialisation places each one in the 16-tap
kernel exactly. coif3, the original default, has 18 taps and would not fit.
The filter bank is shared across levels and runs at full rate, so the family
sets each filter's starting shape, not an octave per level.

**Limb leads are masked together.** I, II, III, aVR, aVL and aVF are
projections of one frontal-plane vector, and any two of them determine the
other four exactly. With per-token masking at 0.70, a masked limb-lead patch
has two or more of its siblings visible about 47% of the time, so
reconstructing it is arithmetic, not inference. A time patch of the limb group
is therefore masked as one unit, and V1–V6 are masked per token. Each part is
masked at the configured ratio, so the overall ratio does not change.
`model.lead_group_masking: false` is the EEG behaviour exactly.

**Normalisation keeps inter-lead amplitude.** Each window has each lead's mean
removed and is then divided by one standard deviation taken over all its
leads. Per-lead z-scoring, which EEG uses, would erase the ratios between
leads. In ECG those ratios carry diagnostic information: axis, voltage
criteria and inversions.

**Its own lead vocabulary.** The ECG leads have their own vocabulary
(`physiowave/ecg_c1/leads.py`) with its own hash. Adding them to
`CHANNEL_VOCAB` would have changed the EEG vocabulary's hash and made every
existing EEG checkpoint unloadable.

**Preprocessing** (`physiowave/ecg_c1/preprocess.py`):

1. Convert to mV and remove DC.
2. Notch at the corpus's mains frequency (none for simulated data).
3. 0.5 Hz high-pass, zero-phase, in second-order sections, mirror-padded.
4. Polyphase resample, then place leads on slots by name. III, aVR, aVL and
   aVF are derived from I and II when a file stores only the eight independent
   leads.

No band-pass is applied. A window is dropped if it contains a non-finite
sample, has a flat lead, or exceeds 25 mV, and drops are counted per reason.

## Corpora

| id | leads @ rate | access | notes |
|----|--------------|--------|-------|
| `mimic_iv_ecg` | 12 @ 500 | open (PhysioNet) | read from the 36 GB zip, not unpacked. Files store aVF before aVL; leads are placed by name |
| `code15` | 12 @ 400 | open (Zenodo) | zero padding stripped; the extra `exam_id 0` row per file skipped; values are mV despite the README's "1e-4 V"; patient from exams.csv |
| `medalcare_xl` | 12 @ 500 | open (Zenodo) | simulated. One rendering of three (`noise`, via `--medalcare-variant`); the subject is the torso model `run_SXX` |
| `norwegian_athlete` | 12 @ 500 | open (PhysioNet) | 28 records. Each lead is rescaled to full int16 range in the files, so its amplitudes are not physical |
| `georgia` | 12 @ 500 | open (PhysioNet Challenge 2021) | recursive HTTP, since there is no zip or open S3 copy. No patient id |
| `heedb` | 12 @ 250/500 | credentialed (bdsp.io) | synced from its S3 access point with your credentials; patient from `metadata.csv` |
| `icentia11k` | 1 @ 250 | open (PhysioNet) | read from the 202 GB zip; 16 random windows per 70-minute segment |
| `code2` | 12 | restricted | by request. Its format is not public, so there is no reader until the files are in hand |

Never pretrained on: `ptbxl`, `cpsc2018` and `chapman_shaoxing`, the
repository's ECG fine-tuning benchmarks. Preprocessing and the merge refuse
them by name.

The split is by subject hash (5% validation), so every array task puts a given
patient on the same side. Georgia and the athletes carry no patient id, so
each of their records is its own subject.

The adapters were checked against real files on 2026-09-29:

- a MIMIC-IV-ECG record, read from a zip;
- Georgia `.mat` records, including a 5 s one;
- an Icentia11k segment, read from a zip;
- all 28 athletes;
- CODE-15%'s `exams_part17`.

MedalCare-XL and HEEDB were checked against their documented layouts only.
Run `INSPECT` before an array on either.
