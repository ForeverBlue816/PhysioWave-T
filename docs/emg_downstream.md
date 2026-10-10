# sEMG C1 downstream evaluation

The pretrained sEMG C1 encoder (`docs/emg_c1_pretraining.md`) is evaluated
the same way as ECG (`docs/ecg_downstream.md`). There is one fixed split per
task, nothing is averaged over folds, and every task runs in three modes:

| mode | what it is |
|---|---|
| **ft** | pretrained encoder, fine-tuned; the main row |
| **probe** | pretrained parts frozen |
| **scratch** | random initialisation; the control |

The test set is scored once, at the checkpoint chosen on validation. Where
several learning rates run, the table compares, for each mode, the run with
the best *validation* score.

## Commands (Leonardo)

```bash
cd ~/PhysioWave-T && git pull && source $HOME/pw/bin/activate

# 1. data (login node, ~35 GB, resumable)
bash scripts/download_emg_downstream.sh all
bash scripts/download_emg_downstream.sh status

# 2. one job: 1 node, 4 GPUs, 6 h. Builds each split if needed, then runs, per task:
#    ft at 1e-4 and 3e-5, probe at 1e-3, scratch at 1e-4.
sbatch scripts/slurm/cineca_emg_downstream.sbatch

# 3. results: $PW_CKPT_ROOT/emg_downstream/summary.{txt,md,json}; to share them
bash scripts/sync_results.sh
```

## Tasks

The paper's methods section should state the following details.

| task | data | classes | protocol |
|---|---|---|---|
| `epn612` | EMG-EPN-612: 612 users, Myo armband, 8 ch, 200 Hz | 6: noGesture, waveIn, waveOut, pinch, open, fist | **the paper's (v1) protocol, user-dependent**; see below |
| `epn612_xuser` | same | 6 | **user-independent**; see below |
| `grabmyo` | GRABMyo 1.1.0: 43 participants, 3 sessions (days 1, 8, 29), 28 electrodes (16 forearm + 12 wrist), 2048 Hz | 17: 16 gestures + rest | **inter-session**; see below |
| `grabmyo_xsubj` | same | 17 | **subject-independent**: participants split 70/15/15 (seed 42), all sessions |
| `ninapro_db2` | NinaPro DB2: 40 subjects, 12 Delsys electrodes, 2 kHz | 50: 49 movements of exercises B (17), C (23) and D (9), plus rest | **repetition split**; see below |

### `epn612`: the paper's (v1) protocol, user-dependent

This is the protocol of `EMG/epn_finetune.py`.

- The 306 training users' `trainingSamples` train, and their `testingSamples`
  validate.
- Each of the 306 testing users contributes its first 10 `trainingSamples`
  per gesture to training and its other 15 to test.

Test users are therefore seen during training.

### `epn612_xuser`: user-independent

- Training users train, except 10% of them (seed 42), who validate.
- The testing users' labelled samples are the test set. They are never
  trained on.

### `grabmyo`: inter-session

- Sessions 1–2 train; trial 7 of those sessions validates.
- Session 3, recorded three weeks later, tests.

### `ninapro_db2`: repetition split

- This follows Atzori et al. (2014): repetitions 1, 3, 4 and 6 train, and 2
  and 5 test. Every subject appears on both sides.
- 10% of the training (subject, movement, repetition) segments validate.
  Validation takes whole segments, so no validation window overlaps a
  training window.
- Labels are the refined `restimulus` and `rerepetition`.
- Rest is subsampled, per subject and exercise, to the mean window count of
  one movement.
- `--db2-exercises` (default `BCD`) selects a subset of exercises.

### Preprocessing

Preprocessing is the pretraining corpus's:

1. convert to mV;
2. remove DC;
3. notch at the mains frequency (60 Hz for EPN and GRABMyo, 50 Hz for DB2),
   with the harmonics below Nyquist;
4. 20 Hz high-pass;
5. resample;
6. cut windows;
7. scale each window by one factor shared across its channels.

No quality filter is applied to a benchmark. Windows are as follows:

- **EPN-612** stays at 200 Hz. Each sample gives one window of 5 s, which is
  40 patches of 0.125 s.
- **GRABMyo and DB2** are resampled to 2000 Hz and cut into 1 s windows with a
  0.5 s stride, the pretraining window. A window must lie inside a single
  label.

### What transfers

None of the benchmark montages is a pretraining route. So the downstream model
builds its own wavelet frontend and patch embedding and loads the pretrained
channel encoder, gate and Transformer (`physiowave/emg_c1/downstream.py`).

The electrode names are not in the pretraining vocabulary, so every channel
gets the same unknown code. Channels are still told apart by the 2-D position
encoding.

The probe uses `freeze_scope: pretrained`: it freezes what pretraining
supplied and trains the head along with the montage's new frontend.

## Training and scoring

`physiowave/train/finetune_main.py` with `configs/finetune/emg_c1_<task>.yaml`:

| setting | value |
|---|---|
| model | 384 / 6 / 6, the pretraining encoder |
| pooling | mean pool |
| epochs | 30 |
| warmup | 10% |
| weight decay | 0.05 |
| batch size | 64 |
| early stopping | patience 10 |
| checkpoint selection | validation accuracy |
| reported metrics | accuracy, balanced accuracy, macro F1, Cohen's kappa |

## The paper's (v1) number

The table prints PhysioWave v1's EPN-612 accuracy (94.5%) beside the C1
`epn612` rows. The split is the same, but v1 used max-abs normalisation,
1024-sample inputs and its own training recipe. The number is indicative, not
like-for-like.

## Files

| file | what |
|---|---|
| `scripts/download_emg_downstream.sh` | EPN-612 (Zenodo, parallel ranged), GRABMyo (open S3), DB2 (ninapro.hevs.ch, parallel ranged) |
| `EMG/emg_downstream_prep.py` | raw → `{train,val,test}.h5` + `split.json` (reads EPN's and DB2's zips in place) |
| `configs/finetune/emg_c1_*.yaml` | per-task settings |
| `EMG/finetune_emg_c1.sh` | one run (TASK, MODE) |
| `scripts/slurm/cineca_emg_downstream.sbatch` | everything in one job, 1 node, 4 GPUs, 6 h |
| `scripts/collect_emg_downstream.py` | the table |
| `physiowave/emg_c1/downstream.py` | the sEMG downstream model |
