# EEG C1 downstream

## Sleep-EDF (Sleep Cassette), 5 stages, Fpz-Cz + Pz-Oz, subject-disjoint holdout

| | lr | BalAcc | Kappa | F1-w | Acc | val | |
|---|---|---|---|---|---|---|---|
|   C1 fine-tuned | 1e-04 | 0.6942 | 0.7281 | 0.7985 | 0.8094 | 0.6800 | ep 6 |
| ✓ C1 fine-tuned | 3e-05 | 0.6918 | 0.7170 | 0.7871 | 0.8010 | 0.6820 | ep 6 |
| ✓ C1 linear probe | 1e-03 | 0.6654 | 0.7200 | 0.7897 | 0.8052 | 0.6817 | ep 8 |
| ✓ C1 from scratch | 4e-04 | 0.6924 | 0.7363 | 0.8043 | 0.8152 | 0.6906 | ep 8 |
| EEGPT (frozen + 4-layer head)* | | 0.6917 | 0.6857 |    -   |    -   | | |

pretrained - scratch (BalAcc, each chosen on validation): -0.0006

\* EEGPT's table: Sleep-EDF 10-fold, P300 leave-one-subject-out, each scored on the fold it also validates on. Here one subject-disjoint split with a held-out test set nothing selects on. Indicative only.
