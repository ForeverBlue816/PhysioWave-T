# EEGPT benchmarks: BCIC-IV-2a, BCIC-IV-2b, KaggleERN

The three downstream tasks from EEGPT (NeurIPS 2024) Table 4, run on the C1
pretrained encoder.

| task | classes | channels | epoch | EEGPT (linear probe, fold mean) |
|---|---|---|---|---|
| BCIC-IV-2a | 4 (L/R hand, feet, tongue) | 22 | cue + 0..4 s, 1024 @ 256 Hz | BAC 0.5846, κ 0.4462, wF1 0.5715 |
| BCIC-IV-2b | 2 (L/R hand) | 3 (bipolar C3/Cz/C4) | cue + 0..4 s, 1024 @ 256 Hz | BAC 0.7212, κ 0.4426, AUROC 0.8059 |
| KaggleERN | 2 (error / correct feedback) | 56 (or `--channels eegpt19`) | feedback −0.7..+1.3 s, 512 @ 256 Hz | BAC 0.5837, κ 0.1882, AUROC 0.6621 |

## Run it

On the **login node** (compute nodes have no internet):

```bash
cd ~/PhysioWave-T && git pull
source scripts/cineca_env.sh

python EEG/download_eegpt_benchmarks.py --dataset 2a --dest $PW_DATA_EEG/bcic_iv2a
python EEG/download_eegpt_benchmarks.py --dataset 2b --dest $PW_DATA_EEG/bcic_iv2b
bash   EEG/download_kaggle_ern.sh $PW_DATA_EEG/kaggle_ern     # needs a Kaggle token
```

Then one job per task. Each runs two modes in parallel on one node — `ft`
(pretrained encoder, every parameter fine-tuned) and `scratch` (same model,
random init: the control). The difference between them is the measurement.

```bash
ENC=$PW_CKPT_ROOT/pretrain_eeg_c1_moe/best.pth     # or an exported eeg_c1_encoder.pth
for t in bcic2a bcic2b kaggleern; do
  sbatch --export=ALL,TASK=$t,PRETRAINED=$ENC scripts/slurm/cineca_eegpt_bench.sbatch
done
```

The split is built inside the job on first use. Results:

```bash
python scripts/collect_eegpt_bench.py --root $PW_CKPT_ROOT/eegpt_bench
```

Comparing two encoders (best vs last epoch): add `TAG=best` / `TAG=final` to the
`--export` list, and `MODES=ft` on the second so the scratch control is not
rerun.

Training: 30 epochs, batch 64, AdamW (wd 0.01), lr 2.5e-4 with linear warmup
over the first 10% of steps and a cosine to 1% by the last, label smoothing
0.1. Checkpoint selected on κ (2a) or AUROC (2b, KaggleERN) on the validation
subjects, then scored once on the test subjects.

## What is reproduced, and what is not

Reproduced from EEGPT's `downstream/` code: the epochs, the 0–38 Hz filter on
BCIC, Euclidean alignment per session then common average reference, the
channel sets, their four KaggleERN folds, and the leave-one-subject-out shape of
their BCIC folds.

Different, on purpose (details in `EEG/eegpt_bench_common.py`):

- **Amplitude normalisation is per session, not per trial.** Per-trial scaling
  divides away the C3/C4 power asymmetry motor imagery is decided on, and the
  ERN's amplitude. EA sets the scale on BCIC; a session z-score on KaggleERN.
- **Polyphase resampling**, not nearest-neighbour stretching of each epoch.
- **Test subjects select nothing.** EEGPT validates on the subjects it reports.
  Here validation is a separate subject set, so these numbers are pessimistic
  relative to theirs.
- **One fold, not a mean over folds.** `FOLD=k` picks which.
- **Full fine-tuning, 30 epochs.** EEGPT trains a linear probe on a frozen
  encoder for 100 epochs.
- **The wavelet frontend, patcher and classification head are fresh**; the
  channel embedding, its gate and the shared transformer come from pretraining.

## Data notes

- BCIC comes from BNCI Horizon 2020 (001-2014, 004-2014): the competition
  recordings, openly downloadable, with evaluation-session labels in the files.
- KaggleERN needs a Kaggle account that has accepted the competition rules.
  The download has no `true_labels.csv`, so the 10 Kaggle test subjects cannot
  be scored. The converter then builds train/val/test from the 16 labelled
  subjects in EEGPT's fold shape: the 4 subjects their fold leaves out are the
  test set, 2 of their 12 training subjects validate, 10 train. Subject-
  disjoint and valid, but not EEGPT's test subjects — their KaggleERN row is a
  reference, not a comparison on the same data. If a `true_labels.csv` turns
  up, drop it in the raw directory and rebuild the split; `--test-split auto`
  switches to their test set.
