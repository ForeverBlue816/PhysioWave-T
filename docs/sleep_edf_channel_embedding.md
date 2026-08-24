# Sleep-EDF channel embedding

Does telling the model *which derivation a row is* help, on a fixed two-channel
montage? Six variants, one changed thing each, measured against the current
strong baseline.

## What this can and cannot conclude

Sleep-EDF has exactly two bipolar derivations and never varies them. So:

* an ID embedding here is two learned row vectors, and the existing 2-D position
  embedding already distinguishes two rows. C1 is close to a null hypothesis by
  construction, which is why it is in the table.
* nothing here supports a claim about **cross-montage generalisation**. Every
  run sees the same two channels.
* nothing here is **reference-invariant**, and the signed encoding must not be
  described that way. A learned map of a signed derivation is not invariant to
  re-referencing; it stops being blind to it.
* nothing here says anything about 19/32/64/128-channel topology. kNN graphs,
  Laplacians and CSD all degenerate on two channels, so no topology term is
  included — a component that cannot be evaluated is a claim, not a result.

What a stable C4/C5 gain would license, exactly:

> Signed derivation-aware channel metadata improves a strong fixed two-channel
> sleep-staging model when it jointly conditions scale folding and Transformer
> tokens.

Heterogeneous montages and topology belong to a later multi-dataset study.

## The design

```
channel metadata ──> ChannelEncoder ──> e_c  [C, Dc]
                                         │
                                         ├──> W_f ──> tanh(g_f)·b  ──> dynamic ScaleFold logits
                                         │
folded EEG ──> PatchEmbed ──────────────>├──> W_t ──> tanh(g_t)·δ  ──> patch tokens
                                         ▼
                          existing 2-D PE ──> existing RoPE Transformer ──> head
```

**The code is never added to the waveform.** A per-channel vector on raw EEG is
a DC offset that every downstream filter then has to model around.

### Modes

| `--channel_encoding` | what it is |
|---|---|
| `none` | no code; the model is the one that existed before this feature |
| `id` | learned embedding of the channel name (EEGPT-style, their Eq. 11) |
| `signed` | the derivation's geometry, midpoint and direction kept apart |
| `hybrid` | `Norm(id + signed)` |

For a bipolar channel `A-B`, with `φ(x,y,z) = [x, y, z, xy, xz, yz, x²−y², 3z²−1]`
on the unit sphere:

```
m = (φ(r_A) + φ(r_B)) / 2         midpoint     — identical for A-B and B-A
d =  φ(r_A) − φ(r_B)              direction    — exactly negated for B-A
e = W_m·m + W_d·d + e_bipolar     then RMSNorm
```

The two branches have separate weights on purpose. Sharing them would make the
code a function of `φ(A)` and `φ(B)` separately, and the split into "where" and
"which way round" would carry no meaning.

| `--channel_injection` | where the code enters |
|---|---|
| `none` | nowhere |
| `token` | added to every patch token of its own channel |
| `fold` | biases the dynamic fold's scale logits (needs `--scale_fold dynamic`) |
| `dual` | both |

## Shapes, confirmed from the code

```
x                        [B, 2, 3000]
wavelet decomposition    [B, (3+1)·2, 3000] = [B, 8, 3000]     scale-major
ScaleFold(dynamic)       [B, 2, 3000]
  channel prior          [C, S] = [2, 4] -> broadcast [B, C, 1, S] into the logits
unsqueeze                [B, 1, 2, 3000]
PatchEmbed (1,50)        [B, D, 2, 60] -> flatten(2) -> [B, D, 120] -> [B, 120, D]
  semantic view          [B, 2, 60, D]
  token index            c·60 + p          CHANNEL-MAJOR, TIME-MINOR
position embedding       [B, 120, D]
```

The flatten order is `Conv2d` → `permute` → `flatten(2)` → `transpose` in
`transformer_modules.PatchEmbed.forward`, which walks the channel axis before
the time axis. A sentinel test marks one channel's code and asserts the other
channel's tokens are untouched, so this is pinned rather than assumed.

## Zero gates, non-zero projections

`g_f` and `g_t` start at exactly 0, so `tanh(g)·(…)` is exactly 0 and the
backbone's arithmetic at step 0 is the baseline's — verified to `|Δ| = 0`, not
to a tolerance.

The projections and the encoder start at **normal random values**, and that
asymmetry is required:

* `∂loss/∂g` carries the projection's output → non-zero at step 0
* `∂loss/∂W` carries `tanh(g)` → exactly zero until the gate moves

Zero-initialising both would put a zero on each side of the product and the
branch would never receive gradient at all. Measured:

```
step 1  gates: fold=7.2e-06 token=5.0e-04   proj: 0.0  0.0   enc: 0.0  0.0
step 2  gates: fold=1.2e-05 token=4.7e-04   proj: 2.0e-08  9.7e-04   enc: 2.2e-04  1.2e-04
```

Every channel module is constructed **after** the legacy modules and after
`apply(_init_weights)`. Constructing a module draws from the global RNG, so
building them earlier would shift every legacy draw and the variants would no
longer share a backbone. A test hashes the legacy parameters across all six
variants at one seed.

## Running it

### 1. Build the splits (the decode cache is reused)

```bash
python EEG/sleep_edf_finetune.py \
  --stage split --split eegpt-fold --fold 0 \
  --cache-dir "$PW_DATA_EEG/sleep_edf/cache" \
  --out-dir "$PW_DATA_EEG/sleep_edf_channel/fold0"
```

A new output directory, so the existing baseline files are untouched. The
metadata is written once per file, not once per 30 s window, and its hash goes
into `split.json` and into every HDF5's attributes.

### 2. One variant

```bash
CHANNEL_ENCODING=signed CHANNEL_INJECTION=dual \
DATA_DIR=$PW_DATA_EEG/sleep_edf_channel/fold0 \
OUTPUT_DIR=$PW_CKPT_ROOT/sleep_c4 bash EEG/finetune_sleep.sh
```

| variant | `CHANNEL_ENCODING` | `CHANNEL_INJECTION` |
|---|---|---|
| C0 | `none` | `none` |
| C1 | `id` | `token` |
| C2 | `signed` | `token` |
| C3 | `signed` | `fold` |
| C4 | `signed` | `dual` |
| C5 | `hybrid` | `dual` |

### 3. The whole matrix

```bash
# plumbing check: two variants, one seed, the four-GPU path the real runs use
VARIANTS=C0,C4 SEEDS=42 FOLDS=0 NUM_GPUS=4 EPOCHS=2 \
    bash EEG/run_sleep_channel_ablation.sh

# full: 6 variants x 10 folds x 3 seeds = 180 runs, order of 60 GPU-hours
VARIANTS=C0,C1,C2,C3,C4,C5 SEEDS=42,43,44 FOLDS=0,1,2,3,4,5,6,7,8,9 \
    NUM_GPUS=4 bash EEG/run_sleep_channel_ablation.sh
```

That is 2 runs, not 15. `VARIANTS=C0,C1,C2,C3,C4 SEEDS=42,43,44 NUM_GPUS=1`
is 15 runs on one GPU, which at the header's ~20 min per 20-epoch four-GPU fold
works out near two hours -- a sweep, not a smoke. Use `NUM_GPUS=4` for the
check, because one process never exercises the multi-rank path that the real
runs take.

`DRY_RUN=1` prints the commands without running them. A run counts as finished
only when it wrote `test_results.json`, so a job killed mid-epoch is redone
rather than silently dropped from the mean.

### 4. Collect

```bash
python scripts/collect_sleep_channel_ablation.py $PW_CKPT_ROOT/sleep_channel_ablation
```

Writes `channel_ablation.csv` and `.json`, and prints per-run metrics, mean ± sd,
per-class F1, confusion matrices, and the **paired** delta against C0 within
each (fold, seed) cell. Paired, because fold-to-fold and seed-to-seed variation
on Sleep-EDF is larger than the effect being looked for — an unpaired difference
of means is mostly a difference of folds. It exits non-zero when the matrix has
holes, since the means would then be over different subsets per variant.

## What the log says

```
[Train] Epoch 0: ... alpha=[0.250 0.250 0.250 0.251] sd_t=[0.000 ...] sd_c=[...]
        | alpha/chan=[0.250 0.250 0.250 0.251] [0.250 0.250 0.250 0.251]
        | g_f=+0.00007 g_t=-0.00012 tok_w=0.361
```

`alpha/chan` is per (channel, scale): a channel prior's purpose is to make the
two derivations want different bands, and the marginal over channels is exactly
the axis that would hide it. `g_f`/`g_t` are the gates; `tok_w` is the token
projection's norm, so a gate that moved and a branch whose output is still zero
are distinguishable.

## Compatibility

* **Old HDF5, `--channel_encoding none`**: runs unchanged. The metadata is
  optional and its absence is not an error in that mode.
* **Old HDF5, any other mode**: refused, with the `--stage split` command that
  fixes it. Reading a missing key as zero would train on a montage of origin
  points.
* **Old checkpoint**: loads. The channel keys are the only ones allowed to be
  missing, plus the task head; anything else missing raises, because
  `strict=False` on its own would leave those tensors random and the run would
  report a fine-tune of something it is not.
* **`train`/`val`/`test` must agree** on the metadata hash, and multi-file
  inputs must agree with each other. Concatenating two montages produces a
  corpus with no channel semantics and nothing downstream inspects it closely
  enough to notice.

## A note on `QK_NORM`

`EEG/finetune_sleep.sh` uses `${QK_NORM:+--qk_norm}`, which is a *presence*
test: `QK_NORM=0` is a non-empty string and turns the flag **on**. That
expansion is deliberately left as it was — changing it would silently alter what
an existing invocation resolves to, and an ablation needs everything except the
channel flags to stay still. The resolved value is now printed at launch and
stored in each run's `provenance`. With `QK_NORM` unset it is **False**, which
is what the recorded baseline used, and every variant must use the same value.
