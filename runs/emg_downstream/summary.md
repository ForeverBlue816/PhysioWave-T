# sEMG C1 downstream

## EMG-EPN-612, 6 gestures, user-dependent (the paper's protocol; paper: accuracy)

| | lr | Acc | BalAcc | F1-mac | Kappa | val | |
|---|---|---|---|---|---|---|---|
| ✓ C1 fine-tuned [e38ld] | 3e-04 | 0.9438 | 0.9438 | 0.9439 | 0.9325 | 0.9622 | ep 28 |
|   C1 fine-tuned [e38] | 1e-04 | 0.9384 | 0.9384 | 0.9385 | 0.9261 | 0.9530 | ep 29 |
|   C1 fine-tuned [e38ld] | 1e-04 | 0.9151 | 0.9151 | 0.9151 | 0.8981 | 0.9288 | ep 29 |
| ✓ C1 linear probe [e38] | 1e-03 | 0.9018 | 0.9018 | 0.9015 | 0.8821 | 0.9110 | ep 26 |
| ✓ C1 from scratch | 1e-04 | 0.9434 | 0.9434 | 0.9434 | 0.9320 | 0.9570 | ep 27 |
|   C1 from scratch [e38] | 1e-04 | 0.9420 | 0.9420 | 0.9420 | 0.9304 | 0.9563 | ep 27 |
| PhysioWave v1 (paper)* | | 0.9450 |    -   |    -   |    -   | | |

pretrained - scratch (Acc, each chosen on validation): +0.0004

## EMG-EPN-612, 6 gestures, user-independent (test users never trained on)

| | lr | Acc | BalAcc | F1-mac | Kappa | val | |
|---|---|---|---|---|---|---|---|
| ✓ C1 fine-tuned [e38ld] | 3e-04 | 0.9284 | 0.9284 | 0.9283 | 0.9141 | 0.9224 | ep 28 |
|   C1 fine-tuned [e38] | 1e-04 | 0.9201 | 0.9201 | 0.9201 | 0.9041 | 0.9200 | ep 16 |
|   C1 fine-tuned [e38ld] | 1e-04 | 0.9105 | 0.9105 | 0.9107 | 0.8927 | 0.9060 | ep 28 |
| ✓ C1 linear probe [e38] | 1e-03 | 0.8982 | 0.8982 | 0.8982 | 0.8778 | 0.9018 | ep 21 |
| ✓ C1 from scratch | 1e-04 | 0.9280 | 0.9280 | 0.9281 | 0.9136 | 0.9219 | ep 22 |
|   C1 from scratch [e38] | 1e-04 | 0.9288 | 0.9288 | 0.9289 | 0.9146 | 0.9198 | ep 29 |

pretrained - scratch (Acc, each chosen on validation): +0.0003

## GRABMyo, 17 classes, inter-session (sessions 1-2 train, session 3 tests)

| | lr | Acc | BalAcc | F1-mac | Kappa | val | |
|---|---|---|---|---|---|---|---|
| ✓ C1 fine-tuned [e38ld] | 3e-04 | 0.8967 | 0.8967 | 0.8965 | 0.8903 | 0.9454 | ep 29 |
|   C1 fine-tuned [e38] | 1e-04 | 0.8736 | 0.8736 | 0.8734 | 0.8657 | 0.9307 | ep 29 |
|   C1 fine-tuned [e38ld] | 1e-04 | 0.8546 | 0.8546 | 0.8546 | 0.8455 | 0.9166 | ep 27 |
| ✓ C1 linear probe [e38] | 1e-03 | 0.8670 | 0.8670 | 0.8670 | 0.8587 | 0.9166 | ep 27 |
| ✓ C1 from scratch | 1e-04 | 0.8516 | 0.8516 | 0.8513 | 0.8424 | 0.9294 | ep 28 |
|   C1 from scratch [e38] | 1e-04 | 0.8512 | 0.8512 | 0.8510 | 0.8419 | 0.9276 | ep 27 |

pretrained - scratch (Acc, each chosen on validation): +0.0451

## GRABMyo, 17 classes, subject-independent

| | lr | Acc | BalAcc | F1-mac | Kappa | val | |
|---|---|---|---|---|---|---|---|
| ✓ C1 fine-tuned [e38ld] | 3e-04 | 0.8196 | 0.8196 | 0.8183 | 0.8083 | 0.7513 | ep 6 |
|   C1 fine-tuned [e38] | 1e-04 | 0.7903 | 0.7903 | 0.7888 | 0.7772 | 0.7364 | ep 16 |
|   C1 fine-tuned [e38ld] | 1e-04 | 0.7929 | 0.7929 | 0.7919 | 0.7800 | 0.7002 | ep 27 |
| ✓ C1 linear probe [e38] | 1e-03 | 0.8321 | 0.8321 | 0.8321 | 0.8216 | 0.7598 | ep 14 |
| ✓ C1 from scratch [e38] | 1e-04 | 0.7837 | 0.7837 | 0.7805 | 0.7702 | 0.7021 | ep 23 |

pretrained - scratch (Acc, each chosen on validation): +0.0358

## NinaPro DB5, exercises A+B+C, 52 movements + rest, two Myo at 200 Hz (reps 1,3,4,6 / 2,5)

| | lr | Acc | BalAcc | F1-mac | Kappa | val | |
|---|---|---|---|---|---|---|---|
| ✓ C1 fine-tuned [e38ld] | 3e-04 | 0.5984 | 0.5886 | 0.5834 | 0.5899 | 0.5361 | ep 28 |
|   C1 fine-tuned [e38ld] | 1e-04 | 0.5085 | 0.4995 | 0.4899 | 0.4984 | 0.4396 | ep 28 |
| ✓ C1 linear probe [e38ld] | 1e-03 | 0.3428 | 0.3265 | 0.3216 | 0.3289 | 0.3147 | ep 20 |
| ✓ C1 from scratch [e38ld] | 1e-04 | 0.5034 | 0.4922 | 0.4869 | 0.4930 | 0.4436 | ep 28 |

pretrained - scratch (Acc, each chosen on validation): +0.0950

## NinaPro DB2, exercises B+C+D, 49 movements + rest (reps 1,3,4,6 / 2,5)

| | lr | Acc | BalAcc | F1-mac | Kappa | val | |
|---|---|---|---|---|---|---|---|
| ✓ C1 fine-tuned [e38] | 1e-04 | 0.3418 | 0.3416 | 0.3419 | 0.3264 | 0.3013 | ep 6 |
| ✓ C1 linear probe [e38] | 1e-03 | 0.1935 | 0.1803 | 0.1725 | 0.1725 | 0.1972 | ep 5 |
|   C1 from scratch | 1e-04 | 0.4017 | 0.4065 | 0.4077 | 0.3880 | 0.3373 | ep 28 |
| ✓ C1 from scratch [e38] | 1e-04 | 0.4019 | 0.4058 | 0.4075 | 0.3881 | 0.3410 | ep 29 |

pretrained - scratch (Acc, each chosen on validation): -0.0601

\* v1 used max-abs normalisation, 1024-sample inputs and its own training recipe on the same split. Indicative, not like-for-like.
