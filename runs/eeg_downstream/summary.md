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

## PhysioNet P300 (erpbci), target vs non-target, subject-disjoint split

| | lr | BalAcc | Kappa | AUROC | F1-w | val | |
|---|---|---|---|---|---|---|---|
|   C1 fine-tuned [cw_els0.1] | 1e-03 | 0.6682 ± 0.0011 | 0.2619 ± 0.0347 | 0.7265 ± 0.0124 | 0.7576 ± 0.0330 | 0.7850 | ep 9/9/7  n=3 |
| ✓ C1 fine-tuned [route_cw_els0.1] | 1e-03 | 0.6567 ± 0.0199 | 0.2286 ± 0.0187 | 0.7194 ± 0.0194 | 0.7356 ± 0.0303 | 0.8214 | ep 6/6/8  n=3 |
|   C1 fine-tuned | 1e-04 | 0.6014 | 0.2465 | 0.7109 | 0.8069 | 0.7690 | ep 10 |
|   C1 fine-tuned | 3e-05 | 0.6179 | 0.2603 | 0.7151 | 0.8036 | 0.6733 | ep 19 |
| ✓ C1 linear probe | 1e-03 | 0.6217 | 0.2981 | 0.7576 | 0.8206 | 0.7718 | ep 9 |
|   C1 linear probe [cw] | 1e-03 | 0.6767 ± 0.0088 | 0.2906 ± 0.0127 | 0.7417 ± 0.0051 | 0.7781 ± 0.0177 | 0.7619 | ep 8/10/8  n=3 |
|   C1 linear probe [route_cw] | 1e-03 | 0.5075 ± 0.0091 | 0.0068 ± 0.0078 | 0.5277 ± 0.0156 | 0.4635 ± 0.1318 | 0.5105 | ep 0/2/0  n=3 |
|   C1 from scratch | 5e-04 | 0.5857 | 0.2381 | 0.7489 | 0.8116 | 0.7652 | ep 13 |
|   C1 from scratch [cw] | 5e-04 | 0.6586 ± 0.0198 | 0.2776 ± 0.0215 | 0.7362 ± 0.0049 | 0.7806 ± 0.0291 | 0.7641 | ep 10/13/14  n=3 |
| ✓ C1 from scratch [route_cw] | 5e-04 | 0.6512 ± 0.0027 | 0.2337 ± 0.0222 | 0.7194 ± 0.0108 | 0.7486 ± 0.0249 | 0.7985 | ep 5/9/6  n=3 |
| EEGPT (linear probe)* | | 0.6502 | 0.2999 | 0.7168 |    -   | | |
| LaBraM (linear probe)* | | 0.6477 | 0.2935 | 0.7068 |    -   | | |

pretrained - scratch (BalAcc, each chosen on validation): +0.0055

\* EEGPT's table: Sleep-EDF 10-fold, P300 leave-one-subject-out, each scored on the fold it also validates on. Here one subject-disjoint split with a held-out test set nothing selects on. Indicative only.
