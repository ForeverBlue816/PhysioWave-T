# SeizeIT2, event-level, official split (test = sub-097..125)

RMS gate: cross. Threshold 0.5 unless stated. AUROC / AUPR (paper) are the dataset paper's epoch-based areas; the pooled per-second AUROC / AUPRC are ours and not comparable with them.

| run / condition | sens % | FA/h | score | sens @<=11 FA/h | AUROC (paper) | AUPR (paper) | pooled sens | FA/24h | sens @1 FA/h | sens @6.22 FA/h | per-second AUROC | per-second AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eeg_c1_finetune_sze / complete | 57.3 | 9.06 | 4.9 | 61.4 | 0.723 | 0.0107 | 0.403 | 302.9 | 0.122 | 0.282 | 0.703 | 0.0066 |
| eeg_c1_frozen_sze / complete | 38.7 | 3.40 | 22.2 | 54.5 | 0.715 | 0.0131 | 0.276 | 134.5 | 0.144 | 0.276 | 0.694 | 0.0056 |
| eeg_c1native_finetune_sze / complete | 35.6 | 3.08 | 20.4 | 48.8 | 0.698 | 0.0076 | 0.227 | 102.1 | 0.099 | 0.249 | 0.676 | 0.0027 |
| eeg_c1native_frozen_sze / complete | 33.7 | 2.03 | 23.1 | 45.7 | 0.706 | 0.0088 | 0.221 | 70.0 | 0.138 | 0.243 | 0.679 | 0.0027 |
| eeg_eegpt_finetune_sze / complete | 23.5 | 2.42 | 12.9 | 42.4 | 0.676 | 0.0059 | 0.166 | 55.5 | 0.110 | 0.282 | 0.683 | 0.0031 |
| eeg_eegpt_frozen_sze / complete | 42.3 | 4.95 | 16.8 | 53.3 | 0.710 | 0.0146 | 0.326 | 167.5 | 0.149 | 0.282 | 0.698 | 0.0053 |
| eeg_labram_finetune_sze / complete | 40.7 | 2.46 | 28.6 | 58.0 | 0.725 | 0.0161 | 0.282 | 99.1 | 0.144 | 0.309 | 0.706 | 0.0070 |
| eeg_labram_frozen_sze / complete | 28.0 | 2.04 | 18.9 | 49.3 | 0.714 | 0.0094 | 0.177 | 91.1 | 0.105 | 0.232 | 0.681 | 0.0041 |
| eeg_luna_huge_finetune_sze / complete | 57.2 | 6.24 | 19.9 | 63.4 | 0.723 | 0.0305 | 0.436 | 264.9 | 0.193 | 0.326 | 0.713 | 0.0060 |
| eeg_luna_huge_frozen_sze / complete | 39.6 | 2.90 | 23.6 | 58.2 | 0.725 | 0.0240 | 0.260 | 129.4 | 0.166 | 0.260 | 0.697 | 0.0113 |
| eeg_luna_large_finetune_sze / complete | 47.4 | 3.66 | 25.8 | 60.9 | 0.732 | 0.0206 | 0.431 | 164.6 | 0.177 | 0.414 | 0.712 | 0.0057 |
| eeg_luna_large_frozen_sze / complete | 25.9 | 3.60 | 3.8 | 54.6 | 0.710 | 0.0083 | 0.171 | 131.4 | 0.066 | 0.171 | 0.668 | 0.0030 |
| **published** | | | | | | | | | | | | |
| SVM, bte-EEG (Bhagubai et al., Sci. Data 2025) (same split and scoring) | 71.1 | 11.00 | | 71.1 | 0.700 | 0.0810 | | | | | | |
| ChronoNet, bte-EEG (Bhagubai et al., Sci. Data 2025) (same split and scoring) | 84.2 | 100.50 | |  | 0.792 | 0.1042 | | | | | | |
| bte-EEG + ECG algorithm (Swinnen et al., Epilepsia Open 2026) (per-patient mean, own cohort and protocol) | 78.0 | 6.22 | |  |   -   |   -   | | | | | | |
