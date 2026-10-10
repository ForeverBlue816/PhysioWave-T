# ECG C1 downstream

## PTB-XL, 5 superclasses, single label (paper: accuracy)

| | Acc | BalAcc | F1-mac | AUROC |
|---|---|---|---|---|
| C1 fine-tuned | 0.7386 | 0.6468 | 0.6503 | 0.9104 |
| C1 linear probe | 0.6478 | 0.4395 | 0.4518 | 0.8364 |
| C1 from scratch | 0.7045 | 0.5665 | 0.5778 | 0.8853 |
| PhysioWave v1 (paper)* | 0.7310 |    -   |    -   |    -   |

## PTB-XL superdiagnostic, multi-label (literature: macro AUROC)

| | AUROC-mac | AUROC-mic | AUPRC | F1-mac |
|---|---|---|---|---|
| C1 fine-tuned | 0.9032 | 0.9194 | 0.7695 | 0.7089 |
| C1 linear probe | 0.8287 | 0.8574 | 0.6362 | 0.6063 |
| C1 from scratch | 0.8771 | 0.8970 | 0.7253 | 0.6686 |

\* PhysioWave v1 scored 4.1 s windows with a fixed threshold and its own splits; indicative, not like-for-like.
