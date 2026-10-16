# SeizeIT2 multimodal figures: ralf_frozen_szo

Drawn by `scripts/visualize_seizeit2.py` at commit `c98eb6f373c0858ea67ca090d3cd013dbd6e95f4` from `/leonardo_scratch/fast/IscrB_WearUsFM/yanlchen/runs/fusion/ralf_frozen_szo` (and the runs it is compared with). Vector figures with editable text; the arrays behind each are in `figure_data/` (not committed), what produced each in `figure_metadata/`.

## fig_seizure_example

ralf / frozen: sub-102 run-01, seizure 1 of 166 ranked candidates (49 s)
ralf / frozen: sub-102 run-01, seizure 1 of 166 ranked candidates (49 s)
grey band: annotated seizure; hatched: seconds the 13-150 µV RMS gate forces to 0; black bar: detection after post-processing at threshold 0.5

Recording `sub-102_run-01`, chosen as rank 1 of 166 candidates by probability contrast (0.67); say so in the caption.

![fig_seizure_example](figures/fig_seizure_example.pdf)

## fig_sensitivity_fa

SeizeIT2 test patients (sub-097..125), the dataset paper's scoring; dotted: 11 FA/h, the SVM's operating point
SeizeIT2 test patients (sub-097..125), the dataset paper's scoring; dotted: 11 FA/h, the SVM's operating point

![fig_sensitivity_fa](figures/fig_sensitivity_fa.pdf)

## fig_modality_subsets

Modality subsets: which signals carry the detection
Modality subsets: which signals carry the detection

![fig_modality_subsets](figures/fig_modality_subsets.pdf)

## fig_reliability

Corrupted inputs: the reliability head notices, and what each fusion keeps
Corrupted inputs: the reliability head notices, and what each fusion keeps

![fig_reliability](figures/fig_reliability.pdf)

## fig_per_patient


![fig_per_patient](figures/fig_per_patient.pdf)

## fig_learning_curves

Learning curves; the dot is the epoch the test figures come from
Learning curves; the dot is the epoch the test figures come from

![fig_learning_curves](figures/fig_learning_curves.pdf)

