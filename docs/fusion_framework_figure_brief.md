# Brief: the RALF multimodal framework figure

A specification for whoever draws the architecture figure of the multimodal
seizure/behaviour pipeline (three pretrained C1 encoders + RALF), at the
standard of a TPAMI Figure 1. Everything below is what the code does
(`physiowave/fusion/{encoders,model,train}.py`, `docs/c1_pretraining_architecture.md`);
draw it, do not improve it.

## Deliverable

- One figure, two-column width: **17.8 cm wide**, height 9 to 12 cm. Vector
  (SVG and PDF, text as text, fonts embedded), plus a 600 dpi PNG preview.
- Font: Times New Roman (or a Times-metric face), **bold** for every label,
  as the paper's other figures. Minimum 7 pt at print size; panel letters
  9 pt; nothing smaller than 6 pt anywhere, including tensor shapes.
- Palette: Okabe-Ito, fixed per modality and used nowhere else:
  EEG `#0072B2` (blue), ECG `#D55E00` (vermilion), EMG `#009E73` (green).
  Fusion / shared blocks: neutral greys `#4D4D4D` fill or outline on white.
  Reliability gate: `#CC79A7` (purple). Losses: `#E69F00` (orange) labels.
  No gradients, no drop shadows, no 3-D, no icons of brains or hearts, no
  clip art. White background. Thin (0.6 pt) black or grey outlines.
- Arrows: straight or orthogonal elbows, 0.8 pt, one arrowhead style. Data
  flow left to right. Loss terms enter from above/below with dashed arrows.
- Every block carries its tensor shape in small type under its name, in the
  notation `[B, tokens, D]`.
- The figure must be readable in greyscale: encode modality by hue AND by a
  small text tag (EEG / ECG / EMG) on every modality-specific block.
- Deliver also the source file (e.g. `.svg` made in Inkscape/Illustrator/Figma,
  or the drawing script if generated with code such as matplotlib/TikZ/D2).

## Layout: five regions, left to right (plus a strip for training signals)

```
 (a) inputs        (b) three C1 encoders (frozen)        (c) summary   (d) RALF core                       (e) outputs
 one 4 s window    per modality, identical structure     pooling       reliability-gated cross-attention   per-task heads
```

A dashed horizontal strip **under** regions (b)–(e), labelled
"(f) training signals", holds the corruption module and the four loss terms.
A small inset at the far right top, "(g) a missing modality", shows the mask
path. Panel letters (a)–(g) in bold at the top-left of each region.

### (a) Inputs: one multimodal window

Three stacked signal strips (short real-looking traces, 2–3 lines each, drawn
as thin polylines, not stock images), left edge:

| strip | text under it |
|---|---|
| EEG (blue) | 2 channels behind the ear, 4 s @ 256 Hz → `[B, 2, 1024]` |
| ECG (vermilion) | 1 channel, 4 s @ 250 Hz → `[B, 1, 1000]` |
| EMG (green) | 1 channel, 4 s @ 2000 Hz → `[B, 1, 8000]` |

Caption-style note in the margin (6.5 pt): "MPDB: 59 EEG, 1 ECG, 4 EMG
channels; the same pipeline". Each strip is z-scored per channel (one word:
"z-score" on the arrow out).

### (b) Three C1 encoders, one per modality, pretrained and frozen

Three horizontal lanes in the modality colour (light tint fill, 15% opacity,
solid colour outline), each lane containing the same five blocks in a row.
A padlock glyph (simple outline, no fill) at the lane's top-right with the
word "frozen"; a small unlocked padlock on the first two blocks' corner with
"trains" where noted below.

Blocks, left to right, in every lane:

1. **Channel adapter / channel identity** — a thin block: "channel-ID
   embedding, gate tanh(g)". Note "new rows for names outside the vocabulary
   (trains)". For SeizeIT2 EEG that is the two behind-the-ear names.
2. **Wavelet frontend** — "SoftGate wavelet decomposition, J = 3, 5 learnable
   16-tap wavelets, softmax selector" → "Spec(x) `[B, 4C, T]`" → **ScaleFold**
   "4 bands → 1 row per channel, `base + γ·(Σ α_s g_s * b_s − base)`" →
   `[B, C, T]`. Draw this as one block with two sub-cells (decomposition |
   fold). Tag: "fresh and trains when the montage sits on no pretraining
   route" (small, 6 pt).
3. **Patch embed** — "Conv (1 × patch)": EEG patch 128 → 8 patches/channel;
   ECG patch 125 → 8; EMG patch 250 → 32. Output `[B, C·P, D]`.
4. **+ position** — "2-D sinusoidal (channel, time) + RoPE".
5. **Shared Transformer** — the largest block in the lane: "pre-RMSNorm,
   QK-norm, SwiGLU". Per lane write the size under the block:
   EEG **512-d × 8 layers (30 M)**, ECG **512-d × 9 layers (30 M)**,
   EMG **384-d × 6 layers (12 M)**. Output tokens `[B, C·P, D_m]`.

Above the three lanes one line of text spanning them: "pretrained by masked
reconstruction (70 % frequency-guided masking; spec head MSE + raw head
SmoothL1) on 15.4 M EEG / 7.6 M ECG / 5.1 M sEMG windows".

### (c) Summary pooling

Per lane one block "SummaryPool: 4 learned queries cross-attend over the
encoder's tokens" → `[B, 4, D_m]`. Then a thin "Linear D_m → 256 + modality
token" block → `[B, 4, 256]`. The three lanes' outputs converge toward (d)
as three arrows in their colours.

### (d) RALF core (the figure's centre of gravity; give it the most space)

A grey-outlined rounded box titled **RALF: Reliability-Aware Late Fusion**.
Inside, from left to right:

1. **Reliability heads** (purple): one per modality, "LayerNorm → MLP →
   sigmoid → r_m ∈ [0, 1]" fed by the mean of that modality's 4 summary
   tokens. Draw r_EEG, r_ECG, r_EMG as three small dials or bars (purple).
   Each modality's tokens are **multiplied by r_m** before fusion: draw a ⊗
   on each coloured arrow with the purple r_m entering it.
2. **Fusion tokens**: "8 learned fusion tokens `[B, 8, 256]`" entering
   **two cross-attention layers** (8 heads) whose keys/values are the
   concatenated gated summary tokens `[B, 12, 256]`; each layer followed by
   an FFN (4×). Label the key-padding mask: "absent modality → masked out".
   Output: mean over the 8 tokens → "fused `[B, 256]`".
3. **Per-task heads**: a small fan of two heads from the fused vector,
   "SeizeIT2: 2 classes" and "MPDB: 5 classes" (the same encoders and core
   train both; draw the MPDB head in lighter grey with "per task").
4. **Unimodal heads**: from each modality's pooled token a thin linear
   "unimodal logits z_m" (modality colour).
5. **Residual combination** (right edge of the box): a ⊕ node with the
   equation set in math type:
   `logits = z_fusion + 0.5 · Σ_m r_m z_m / Σ_m r_m`
   (sum over present modalities).

### (e) Outputs

"P(seizure) per 4 s window" → a small strip of a continuous probability
trace over time with a threshold line at 0.5 and a bar "event-level
post-processing: merge < 2 s, drop < 8 s" (6.5 pt). This is the only place
with a time axis on the right.

### (f) Training signals (dashed strip under b–e)

Left: **SignalCorruptor** block: "p = 0.5 per modality: noise, drift,
saturation, bad channel, dropout; level ℓ_m ∈ [0, 1]" with a dashed arrow
into the inputs of (b) ("applied before the encoders"). Then four loss
labels, orange, each with a dashed arrow to where it attaches:

- `L_task = CE(logits, y)`, √-inverse-frequency class weights → at the ⊕ node
- `L_uni = 0.2 · Σ_m CE(z_m, y)` → at the unimodal heads
- `L_rel = Σ_m MSE(r_m, 1 − ℓ_m)` → at the reliability heads (arrow from the corruptor's ℓ_m)
- `L_cons = symKL(logits_subset ‖ logits_full)`, "a random modality subset
  (drop p = 0.3) must agree with the full prediction" → a small ghost copy of
  the core, drawn in 40 % grey, sharing weights (a "shared" tie line).

One line: "AdamW, lr 1e-4 (3e-4 on MPDB), cosine, 15–60 epochs, selection on
validation AUPRC; encoders frozen (or fine-tuned with layer-wise decay 0.75)".

### (g) Inset: a missing modality

A miniature of (d) with the ECG lane greyed out and crossed: "absent at
test time → key mask; r_ECG := 0; Σ renormalised over EEG, EMG". Four
words max per line.

## Checks before delivery

- Every modality-specific block has both its colour and its text tag.
- All shapes written: `[B, 2, 1024]`, `[B, 1, 1000]`, `[B, 1, 8000]`,
  `[B, C·P, D_m]`, `[B, 4, D_m]`, `[B, 4, 256]`, `[B, 12, 256]`, `[B, 8, 256]`,
  `[B, 256]`.
- The equation `logits = z_fusion + 0.5 · Σ r_m z_m / Σ r_m` appears once, at ⊕.
- No block in (b) implies the encoders are trained end-to-end: the padlocks
  are on the Transformer and frontend, the open padlock only on the channel
  rows (and on the frontend only when tagged "fresh").
- Print the figure at 17.8 cm and read every label at arm's length; shrink
  nothing below 6 pt to fit. If it does not fit, drop (g) first, then the
  signal thumbnails in (a).
- Export: `fig_framework.svg`, `fig_framework.pdf`, `fig_framework.png`
  (600 dpi), and the editable source.
