# EEGPT benchmarks: BCIC-IV-2a, BCIC-IV-2b, KaggleERN

The three downstream tasks from EEGPT (NeurIPS 2024) Table 4, on the C1
pretrained encoder, with the preprocessing of **our pretraining**, not EEGPT's.

| task | classes | channels | epoch | EEGPT (linear probe, fold mean) |
| --- | --- | --- | --- | --- |
| BCIC-IV-2a | 4 (L/R hand, feet, tongue) | 22 | cue + 0..4 s, 1024 @ 256 Hz | BAC 0.5846, κ 0.4462, wF1 0.5715 |
| BCIC-IV-2b | 2 (L/R hand) | 3 (bipolar C3/Cz/C4) | cue + 0..4 s, 1024 @ 256 Hz | BAC 0.7212, κ 0.4426, AUROC 0.8059 |
| KaggleERN | 2 (error / correct feedback) | 56 | feedback −0.7..+1.3 s, 512 @ 256 Hz | BAC 0.5837, κ 0.1882, AUROC 0.6621 |

## Run it — one command, on the login node

```bash
cd ~/PhysioWave-T && git pull
bash scripts/run_eegpt_bench.sh
```

It downloads whatever is missing (BCIC from BNCI Horizon; KaggleERN via the
Kaggle API, which needs `~/.kaggle/access_token`), skips any task whose data it
cannot get, and submits one job per task. Each job builds its split, then runs
two models side by side on one node:

- `ft` — the pretrained encoder, every parameter fine-tuned
- `scratch` — the same model from random initialisation: the control

`ft − scratch` is the measurement. Results land in
`$PW_CKPT_ROOT/eegpt_bench/summary.txt`, rewritten by each job as it finishes.

Options: `PRETRAINED=<path>` (default `pretrain_eeg_c1_moe/best.pth`),
`TASKS="bcic2a bcic2b"`, `FOLD=k`, `DRY_RUN=1`. A second encoder, without
rerunning the control:

```bash
PRETRAINED=$PW_CKPT_ROOT/pretrain_eeg_c1_moe/latest.pth TAG=final MODES=ft \
    bash scripts/run_eegpt_bench.sh
```

## Preprocessing: the pretraining pipeline

The same `physiowave.eeg_c1.preprocess` functions and defaults that built the
pretraining corpus, applied to each continuous run or session before epoching:

```text
microvolts → linear detrend → 50 Hz notch (+ harmonic) → 0.5 Hz high-pass
→ polyphase resample to 256 Hz → epoch → per-window, per-channel z-score, clip ±20
```

So, unlike EEGPT: no 0–38 Hz band-pass, no Euclidean alignment, no common
average, no per-trial min-max. The encoder saw none of those, and fine-tuning
should reuse what it learned rather than first adapt to a new input
distribution. Split directories are named by preprocessing version
(`$PW_DATA_EEG/eegpt_bench/<task>_f<k>_pretrain-v1`), so a split built by an
earlier pipeline is never picked up.

The per-window z-score removes each channel's absolute power, but the
information these tasks rely on survives it: a band-power classifier on 2a
subject 1, trained on one session and tested on the other, scores 0.569 with it
and 0.573 without (4 classes, chance 0.25).

## Training

30 epochs, batch 64, AdamW (wd 0.01), lr 2.5e-4, linear warmup over the first
10% of steps then a cosine to 1% by the last, label smoothing 0.1. Selection on
κ (2a) or AUROC (2b, KaggleERN) on validation subjects; the test subjects are
scored once, by the selected checkpoint. The wavelet frontend, patcher and
classification head are fresh; the channel embedding, its gate and the shared
transformer come from pretraining.

## How it relates to EEGPT's numbers

Kept from EEGPT: the tasks — epochs, classes, channels, and the cross-subject
shape of their folds. Different: the preprocessing (above); full fine-tuning
for 30 epochs rather than a 100-epoch linear probe on a frozen encoder; a test
set nothing selects on (they validate on the subjects they report); one fold
rather than a mean over folds. Their rows are printed as a reference, not as a
like-for-like comparison.

KaggleERN specifically: the Kaggle download has no `true_labels.csv`, so the
ten Kaggle test subjects cannot be scored. Train, val and test then all come
from the 16 labelled subjects, in EEGPT's fold shape — the 4 their fold leaves
out are the test set, 2 of their 12 validate, 10 train. Their KaggleERN row is
on different subjects. If a `true_labels.csv` turns up, put it in the raw
directory and delete the split; the converter switches to their test set.
