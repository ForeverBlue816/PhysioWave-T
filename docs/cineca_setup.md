# Running PhysioWave on CINECA Leonardo

Everything site-specific lives in [`scripts/cineca_env.sh`](../scripts/cineca_env.sh),
which every launcher sources. Off-cluster the same file falls back to
repository-relative paths, so the launch scripts work unchanged on a laptop.

## Storage layout

| What | Where | Why |
|---|---|---|
| Code | `$HOME/PhysioWave-T` | backed up; 50 GB quota |
| Training venv | `$HOME/pw` | built on top of the `cineca-ai` module |
| Data-prep venv | `$HOME/pwprep` | `mne` / `wfdb`, kept away from the training env |
| Checkpoints, logs | `$FAST/yanlchen/runs` | permanent, project-shared, low-latency I/O |
| SSL + preprocess caches | `$FAST/yanlchen/cache` | rebuilt rarely, read constantly |
| Manifests | `$FAST/yanlchen/manifests` | small, must outlive a job |
| Training corpora | `$SCRATCH/bio/{eeg,ecg,emg}` | large, rebuildable |

> `$SCRATCH` is **temporary**: files are purged 40 days after creation. Only the
> HDF5 corpora live there, because they can be rebuilt from the raw archives.
> Anything you cannot rebuild belongs in `$FAST`.

Inside each modality directory, one subdirectory per registry dataset id:

```
$SCRATCH/bio/eeg/tueg/*.h5      $SCRATCH/bio/ecg/mimic_iv_ecg/*.h5
$SCRATCH/bio/eeg/siena/*.h5     $SCRATCH/bio/ecg/ptbxl/{train,val,test}.h5
$SCRATCH/bio/eeg/tuab/tuab_{train,val,test}.h5
                                $SCRATCH/bio/emg/ninapro_db6/*.h5
                                $SCRATCH/bio/emg/epn612/{train,val,test}.h5
```

The ids are the keys of `REGISTRY` in
[`physiowave/data/registry.py`](../physiowave/data/registry.py).

## Which environments you need

Two. One is enough to *train*; the split exists because `mne` drags in a newer
`numpy`/`scipy`/`matplotlib` that would shadow the ones `cineca-ai` compiled for
Leonardo. Data preparation is a one-off CPU job that never runs alongside
training, so it gets its own env.

Actual third-party imports across the whole repository:

- **hard:** `torch numpy scipy h5py pandas PyWavelets scikit-learn pyyaml tqdm`
- **soft** (guarded by `try/except`): `tensorboard`, `omegaconf`
- **data prep only:** `mne` (EEG EDF), `wfdb` (ECG)
- **tests only:** `pytest`, `ruff`

`requirements.txt` additionally lists `transformers`, `accelerate`, `datasets`,
`evaluate`, `peft`, `matplotlib` and `seaborn`, which **nothing imports**. Do not
install them here: `accelerate` and `peft` depend on `torch`, pip cannot see the
module-provided torch, and it will pull a ~2.5 GB pip wheel that shadows the
Leonardo build and destroys multi-node NCCL performance.

## Install order

```bash
# 1. Log in (see the group's CINECA tutorial for the STEP client)
ssh <user>@login.leonardo.cineca.it

# 2. Code
mkdir -p $HOME && cd $HOME
git clone <repo> PhysioWave-T

# 3. Modules -- BEFORE creating the venv
module load profile/deeplrn
module av cineca-ai                 # confirm 4.3.0 exists
module load cineca-ai/4.3.0

# 4. Verify the module's interpreter before building anything on top of it
python -V                                                    # need >= 3.10
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; assert hasattr(torch.amp,'GradScaler')"
```

That last assertion is a hard gate:
[`pretrain_main.py`](../physiowave/train/pretrain_main.py) constructs
`torch.amp.GradScaler("cuda", ...)` unconditionally, which only exists from
torch 2.3. If it fails, pick a newer `cineca-ai` from `module av`.

```bash
# 5. Training env
cd $HOME && python -m venv pw --system-site-packages && source pw/bin/activate

# 6. Install only what the module does not already provide
for m in numpy scipy h5py pandas sklearn yaml tqdm pywt; do
  python -c "import $m" 2>/dev/null && echo "OK   $m" || echo "MISS $m"
done
pip install --no-cache-dir --no-deps PyWavelets     # usually the only gap
pip install --no-cache-dir nvitop                   # GPU monitoring

# 7. Confirm torch was NOT replaced -- must still point into .../cineca-ai/...
python -c "import torch; print(torch.__file__)"

# 8. Data-prep env (separate, does not use cineca-ai)
deactivate && module purge
module load python/3.11
cd $HOME && python -m venv pwprep && source pwprep/bin/activate
pip install --no-cache-dir mne wfdb h5py numpy scipy pandas tqdm
```

## Smoke test (login node, CPU, free)

```bash
module load profile/deeplrn cineca-ai/4.3.0 && source $HOME/pw/bin/activate
cd $HOME/PhysioWave-T
pip install --no-cache-dir --no-deps pytest && pytest tests/ -q
PYTHON=python bash scripts/run_tpami.sh smoke
PYTHON=python bash scripts/run_tpami.sh all --dry-run
```

## Interactive GPU debugging

```bash
srun --nodes=1 --gpus=4 --ntasks-per-node=4 --cpus-per-task=8 \
     -A iscrb_wearusfm -p boost_usr_prod --pty /bin/bash
source $HOME/pw/bin/activate && cd $HOME/PhysioWave-T
bash EEG/pretrain_eeg.sh > ~/eeg.log 2>&1 & nvitop
```

Billing is by CPU hours with one GPU counted as eight CPUs, so four GPUs with
eight CPUs each bills 32 CPU-hours per wall-clock hour.

## Batch jobs

```bash
# Pretraining, one modality per job
sbatch --export=ALL,MODALITY=eeg,EPOCHS=100  scripts/slurm/cineca_pretrain.sbatch
sbatch --export=ALL,MODALITY=ecg             scripts/slurm/cineca_pretrain.sbatch
sbatch --export=ALL,MODALITY=semg            scripts/slurm/cineca_pretrain.sbatch

# RALF fusion; encoders are discovered from $FAST/yanlchen/runs automatically
sbatch scripts/slurm/cineca_fusion.sbatch
```

Both templates refuse to start when the configured corpus has no `*.h5` files,
rather than falling back to the synthetic smoke dataset — a silent fallback would
spend a 24 h four-node reservation producing numbers that are not results.

Override anything through `--export`: `PW_CKPT_ROOT`, `PW_DATA_EEG`, `DATASETS`,
`PW_VENV`, `PROJECT_DIR`, `EPOCHS`, `BATCH_SIZE`, `PRECISION`, `EXTRA`.

## Monitoring

```bash
squeue --me
scancel <JOBID>
```
