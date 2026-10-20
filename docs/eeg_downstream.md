# EEG C1 downstream: Sleep-EDF and PhysioNet P300

The pretrained EEG C1 encoder in three modes on two tasks, one job:

| mode | what trains |
|---|---|
| `ft` | everything, layer-wise lr decay 0.75 over the Transformer |
| `probe` | what pretraining did not supply: the montage's fresh frontend, the pool, the head (`freeze_scope: pretrained`) |
| `scratch` | everything, from random initialisation (the control) |

Neither montage sits on a pretraining route (Sleep-EDF: Fpz-Cz and Pz-Oz at
100 Hz; P300: 64 electrodes at 256 Hz with 512-sample windows), so in both the
frontend is built fresh and what transfers is the channel embedding, its gate
and the Transformer. A probe that also froze the fresh frontend would freeze
random filters.

```bash
sbatch scripts/slurm/cineca_eeg_downstream.sbatch     # -> $PW_CKPT_ROOT/eeg_downstream
```

Defaults: `TASKS="sleep_edf physio_p300"`, `MODES="ft@1e-4 ft@3e-5 probe@1e-3
scratch"` (scratch at the config's lr: 4e-4 / 5e-4), `EPOCHS=20` (the configs
carry EEGPT's 40 / 100). Per mode, the table (`summary.txt`,
`scripts/collect_eeg_downstream.py`) marks the run with the best validation
score; only that one is compared, beside EEGPT's and LaBraM's published rows,
which come from fold-averaged protocols scored on the folds they select on.
One subject-disjoint split here, with a test set nothing selects on.

The splits are built beforehand (`EEG/finetune_eeg_c1.sh` prints the command
when one is missing):

```bash
source $HOME/pwprep/bin/activate          # mne
python EEG/sleep_edf_finetune.py --out-dir $PW_DATA_EEG/sleep_edf --split holdout
python EEG/download_p300.py --dest $PW_DATA_EEG/erpbci       # login node
python EEG/physio_p300_finetune.py --edf-dir $PW_DATA_EEG/erpbci \
    --out-dir $PW_DATA_EEG/p300_f0 --fold 0
```

A Siena-continued encoder: `PRETRAINED=$PW_CKPT_ROOT/pretrain_eeg_c1_siena/latest.pth
TAG=siena` (the runs get `_siena` in their directory and their own rows).
