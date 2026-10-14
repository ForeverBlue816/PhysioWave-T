# Multimodal fusion on the three C1 encoders

Three independently pretrained encoders (EEG `pretrain_eeg_c1_moe`, ECG
`pretrain_ecg_c1_moe`, sEMG `pretrain_emg_c1_moe`, epoch 38) each hand the
fusion a few **summary tokens**; **RALF** (Reliability-Aware Latent Fusion)
fuses them with per-sample reliability, hard masks for missing modalities and
a residual of reliability-weighted unimodal logits. Two tasks share the
encoders and the fusion and keep their own heads:

| task | data | classes | split |
|---|---|---|---|
| `mpdb` | MPDB: 35 drivers, 59-ch EEG + chest ECG + 4 limb sEMG, 4 s windows around the simulator events | 5 behaviours (smooth, acceleration, deceleration, lane change, turning) | by subject, 25 / 5 / 5 |
| `seizeit2` | SeizeIT2: 125 patients, 2 behind-the-ear EEG + ECG + deltoid EMG, 4 s windows | seizure vs background | by patient, 85 / 15 / 25 |

The comparison is three fusion cores under one training loop and one test
matrix:

| `model.fusion` | what it is |
|---|---|
| `ralf` | 8 fusion tokens, two cross-attention layers over the summary tokens; a reliability per modality and sample gates attention and weights the unimodal residual; absent modalities are masked out of the attention |
| `concat` | mean-pool each modality, concatenate, classify; absence is a zero vector |
| `weighted_logit` | the earlier scheme: a logit per modality and one global learned weight per modality, renormalised over what is present |

and three encoder settings, `model.encoder_mode`: `frozen` (what pretraining
supplied stays fixed; the montage adapters, pools, fusion and heads train),
`finetune` (layer-wise decay 0.75 over the pretrained core), `scratch` (no
encoder loaded, the control).

## Commands (Leonardo)

```bash
cd ~/PhysioWave-T && git pull && source $HOME/pw/bin/activate
pip install mne                      # once; the MPDB release is EEGLAB .set files

# 1. data (login node)
bash scripts/download_fusion_data.sh mpdb_raw            # 23 GB, two .rar; the authors' epochs: mpdb
SUBJECTS='sub-0(0[1-9]|[1-3][0-9])' bash scripts/download_fusion_data.sh seizeit2   # 39 patients first
bash scripts/download_fusion_data.sh seizeit2            # all 125, 97.5 GB

# 2. windows (compute node, CPU)
sbatch scripts/slurm/cineca_fusion_prep.sbatch
#    a subset: sbatch --export=ALL,TASKS=seizeit2,SUBJECTS='sub-0(0[1-9]|[1-3][0-9])' scripts/slurm/cineca_fusion_prep.sbatch

# 3. the comparison: five runs on four GPUs, one job
sbatch scripts/slurm/cineca_fusion_c1.sbatch
#    MPDB alone first: sbatch --export=ALL,TASKS=mpdb scripts/slurm/cineca_fusion_c1.sbatch

# 4. the table: $PW_CKPT_ROOT/fusion/summary.{txt,md,json}; to share it
bash scripts/sync_results.sh
```

If the login node has no `unrar`, `7z` or `bsdtar`, the MPDB step prints the
three lines that put rarlab's `unrar` in `$HOME/bin`.

## Data

### MPDB (`Fusion/mpdb_raw_prep.py`; the authors' epochs via `Fusion/mpdb_prep.py`)

The source is the RAW release (figshare 22193119, 23 GB): per subject four
experiments (brake, throttle, turn, change), each a continuous EEGLAB file
per device -- EEG&ECG (59 electrodes and the chest ECG on one Neuracle
amplifier) and EMG (four limb channels on a NeuSen WM), 1 kHz, with the
simulator's markers and their times. The marker codes of the paper's
Table 2 give the five classes: smooth 133; acceleration 135, 137 (the
release's code for overtaking), 143; deceleration 139, 141, 145; lane change
129, 131; turning 125, 127.

Windows are 4 s, from 0.5 s before the marker, so the onset sits where the
authors' 2 s epochs put it and the window is the EEG encoder's pretraining
length. An EMG window is paired with the EEG/ECG window of the same event:
within an experiment the two devices' marker sequences are matched in order
(their LCS when one device missed a trigger) and their inter-event intervals
compared; a pair whose intervals disagree by more than 50 ms is refused and
the window keeps EEG/ECG without EMG (`present_emg` False). `split.json`
records per experiment the pairing counts and the devices' clock offset.
Each record is filtered whole with its modality's pretraining pipeline, then
cut, resampled (EEG 256 Hz, ECG 250 Hz, EMG 2000 Hz) and normalised as that
corpus was. No ICA and no epoch rejection: quality is the reliability
estimator's problem, which is the point.

The authors' preprocessed release (figshare 22192831: `EEG_n / ECG_n / EMG_n.set`,
2 s epochs, ICA-pruned) carries no timestamps and its three files differ in
length, so it can only be aligned by marker order with an ambiguity the
fusion would inherit. `Fusion/mpdb_prep.py` still reads it (`TASKS=mpdb_release`)
for comparison; it is not the paper's data.

The sEMG of the pedal leg can carry the label directly. That is why every
modality subset is reported, not only the complete one.

### SeizeIT2 (`Fusion/seizeit2_prep.py`)

OpenNeuro ds005873 (CC0), BIDS: per run EEG, ECG and EMG EDFs at 256 Hz and
an `events.tsv` (bckg / impd / sz_*). 4 s windows. Seizure windows every
second across each seizure (overlap at least 2 s); background windows on a
4 s grid at least 30 s from any seizure and clear of impedance checks,
subsampled to 30 per recorded hour (seed per run). Each run is filtered
whole (EEG: detrend, 50 Hz notch, 0.5 Hz; ECG: notch, 0.5 Hz; EMG: notch
with harmonics, 20 Hz), then windows are cut, resampled per modality and
normalised. ECG/EMG missing from a run (3% of the data) is kept as real
absence. `--bg-per-hour`, `--guard`, `--sz-stride` change the sampling;
`--subjects` restricts to a regex for a first pass.

### What transfers

| modality | montage | route | what loads |
|---|---|---|---|
| EEG | MPDB's 59 electrodes | E64_256 if every name is one of its slots, else routeless | core (+ frontend on a route) |
| EEG | SeizeIT2's two bte channels | routeless (names not in the vocabulary: a new embedding row each, seeded from `<unk>`, trained as an adapter) | core |
| ECG | one chest derivation, named `patch1` | L1_250 | core + frontend |
| sEMG | 4 limb channels / 1 deltoid | routeless | core |

`encoders.json` in every run directory records, per task and modality, the
route and every key taken or skipped. The second task's encoder of a modality
shares the first task's pretrained core (`shares_core_with`), so one
Transformer per modality trains on both tasks and only the montage adapters
differ.

## Training (`physiowave/fusion/train.py`, `configs/fusion/c1_ralf.yaml`)

Per batch of one task (the two tasks' batches are interleaved in a shuffled
order, every batch of each once per epoch):

```
L = CE(fused)                                    task
  + 0.2 * sum_m CE(unimodal_m)                   on items that have m
  + 1.0 * MSE(reliability_m, 1 - corruption_m)   RALF; corruption applied with a known severity
  + 1.0 * symKL(fused | random subset, fused | all present)
```

The corruption (`physiowave.pretrain.corruption`: noise, drift, saturation,
bad channel, temporal dropout) hits each modality's raw window with
probability 0.5 before the encoders. The subset view drops present
modalities with probability 0.3 each, keeping at least one. SeizeIT2's class
imbalance is met with square-root inverse-frequency class weights.

| setting | value |
|---|---|
| fusion width | 256; 4 summary tokens per modality; 8 fusion tokens, 2 layers, 8 heads |
| epochs / batch | 30 / 32, patience 8 |
| lr | 1e-4 (fusion, pools, adapters, heads); `finetune`: the core at lr with layer decay 0.75 |
| selection | validation, complete modalities, no corruption: mean of MPDB balanced accuracy and SeizeIT2 AUPRC |

## The test matrix

The test set is scored once, at the selected checkpoint, under: complete;
each modality alone; each pair; and each of eeg~noise, eeg~bad channel,
eeg~drift, ecg~saturation, ecg~noise, emg~dropout, emg~noise at severity
0.7 on top of the complete set. A condition an item cannot meet (its only
present modality excluded) drops it from that condition; the count is in
`results.json`. `scripts/collect_fusion.py` prints one block per task with
the headline metric (MPDB balanced accuracy, SeizeIT2 AUPRC) per run and
condition.

## Files

| file | what |
|---|---|
| `scripts/download_fusion_data.sh` | MPDB (figshare, .rar) and SeizeIT2 (OpenNeuro S3) |
| `Fusion/mpdb_raw_prep.py`, `Fusion/mpdb_prep.py`, `Fusion/seizeit2_prep.py` | raw -> `{train,val,test}.h5` + `split.json` |
| `physiowave/fusion/prep_common.py` | rates, windows, per-modality filtering and normalisation, the HDF5 layout, an EDF reader |
| `physiowave/fusion/{data,encoders,model,train}.py` | dataset, per-task C1 encoders, the three cores, the entry point |
| `configs/fusion/c1_ralf.yaml` | the run's settings |
| `scripts/slurm/cineca_fusion_prep.sbatch`, `scripts/slurm/cineca_fusion_c1.sbatch` | preprocessing; the five-run comparison |
| `scripts/collect_fusion.py` | the table |
| `tests/test_fusion_c1.py` | both preparations on synthetic files, the three cores, an end-to-end run on smoke encoders |

The earlier legacy path (`physiowave/models/fusion.py`, `configs/fusion/ralf.yaml`,
`physiowave/train/fusion_main.py`) is kept for its tests; it ran on synthetic
data with the pre-C1 encoders and is not what the paper reports.
