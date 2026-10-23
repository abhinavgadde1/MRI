## Table 6 — Voxel classification metrics (held-out, LCC)

Primary model: **v3** (`brats_scale_full`). Ablation: v1 → v2 → v3. Post-process: **lcc**.

Predicted masks were not stored from the original held-out runs (`checkpoints/brats_eval_heldout*/lcc/` contains `metrics.csv` only). Metrics below were recomputed by re-running LCC inference on the 60 held-out cases.

F1 vs Dice sanity check: **pass**. Per-case-mean F1 matches binary Dice from the confusion matrix and matches table 2 held-out LCC Dice within 0.0001.

Accuracy is high even for poor segmentations because tumor voxels are a small fraction of the volume (see prevalence rows). Precision / recall / F1 are the informative metrics here.

### Pooled voxels (all 60 cases concatenated)

| region | n_cases | n_with_gt | metric | v1 | v2 | v3 |
| --- | --- | --- | --- | --- | --- | --- |
| ET | 60 | 58 | TP | 471521 | 957216 | 988498 |
| ET | 60 | 58 | FP | 352095090 | 3993554 | 738921 |
| ET | 60 | 58 | FN | 702872 | 217177 | 185895 |
| ET | 60 | 58 | TN | 182410517 | 530512053 | 533766686 |
| ET | 60 | 58 | Accuracy | 0.3414 | 0.9921 | 0.9983 |
| ET | 60 | 58 | Precision | 0.0013 | 0.1933 | 0.5722 |
| ET | 60 | 58 | Recall | 0.4015 | 0.8151 | 0.8417 |
| ET | 60 | 58 | Specificity | 0.3413 | 0.9925 | 0.9986 |
| ET | 60 | 58 | F1 (= Dice) | 0.0027 | 0.3126 | 0.6813 |
| ET | 60 | 58 | Tumor / brain voxels | 0.0025 | 0.0025 | 0.0025 |
| ET | 60 | 58 | Tumor / all voxels | 0.0022 | 0.0022 | 0.0022 |
| TC | 60 | 60 | TP | 2325392 | 1766623 | 1836024 |
| TC | 60 | 60 | FP | 533354608 | 2139358 | 829822 |
| TC | 60 | 60 | FN | 0 | 558769 | 489368 |
| TC | 60 | 60 | TN | 0 | 531215250 | 532524786 |
| TC | 60 | 60 | Accuracy | 0.0043 | 0.9950 | 0.9975 |
| TC | 60 | 60 | Precision | 0.0043 | 0.4523 | 0.6887 |
| TC | 60 | 60 | Recall | 1.0000 | 0.7597 | 0.7896 |
| TC | 60 | 60 | Specificity | 0.0000 | 0.9960 | 0.9984 |
| TC | 60 | 60 | F1 (= Dice) | 0.0086 | 0.5670 | 0.7357 |
| TC | 60 | 60 | Tumor / brain voxels | 0.0049 | 0.0049 | 0.0049 |
| TC | 60 | 60 | Tumor / all voxels | 0.0043 | 0.0043 | 0.0043 |
| WT | 60 | 60 | TP | 4323357 | 5369296 | 4975227 |
| WT | 60 | 60 | FP | 904317 | 6542363 | 609289 |
| WT | 60 | 60 | FN | 1417386 | 371447 | 765516 |
| WT | 60 | 60 | TN | 529034940 | 523396894 | 529329968 |
| WT | 60 | 60 | Accuracy | 0.9957 | 0.9871 | 0.9974 |
| WT | 60 | 60 | Precision | 0.8270 | 0.4508 | 0.8909 |
| WT | 60 | 60 | Recall | 0.7531 | 0.9353 | 0.8667 |
| WT | 60 | 60 | Specificity | 0.9983 | 0.9877 | 0.9989 |
| WT | 60 | 60 | F1 (= Dice) | 0.7883 | 0.6083 | 0.8786 |
| WT | 60 | 60 | Tumor / brain voxels | 0.0120 | 0.0120 | 0.0120 |
| WT | 60 | 60 | Tumor / all voxels | 0.0107 | 0.0107 | 0.0107 |

### Per-case mean (precision / recall / F1 skip empty-GT cases, same as table 2)

| region | n_cases | n_with_gt | metric | v1 | v2 | v3 |
| --- | --- | --- | --- | --- | --- | --- |
| ET | 60 | 58 | Accuracy | 0.3414 | 0.9921 | 0.9983 |
| ET | 60 | 58 | Precision | 0.0014 | 0.1897 | 0.5518 |
| ET | 60 | 58 | Recall | 0.3812 | 0.8299 | 0.8217 |
| ET | 60 | 58 | Specificity | 0.3412 | 0.9925 | 0.9986 |
| ET | 60 | 58 | F1 (= Dice) | 0.0027 | 0.2898 | 0.6327 |
| ET | 60 | 58 | Reported Dice (table 2) | 0.0027 | 0.2898 | 0.6327 |
| ET | 60 | 58 | F1 vs reported Dice | ok | ok | ok |
| ET | 60 | 58 | Tumor / brain voxels | 0.0025 | 0.0025 | 0.0025 |
| ET | 60 | 58 | Tumor / all voxels | 0.0022 | 0.0022 | 0.0022 |
| TC | 60 | 60 | Accuracy | 0.0043 | 0.9950 | 0.9975 |
| TC | 60 | 60 | Precision | 0.0043 | 0.4650 | 0.6791 |
| TC | 60 | 60 | Recall | 1.0000 | 0.8016 | 0.8233 |
| TC | 60 | 60 | Specificity | 0.0000 | 0.9960 | 0.9984 |
| TC | 60 | 60 | F1 (= Dice) | 0.0086 | 0.5454 | 0.7159 |
| TC | 60 | 60 | Reported Dice (table 2) | 0.0086 | 0.5454 | 0.7159 |
| TC | 60 | 60 | F1 vs reported Dice | ok | ok | ok |
| TC | 60 | 60 | Tumor / brain voxels | 0.0049 | 0.0049 | 0.0049 |
| TC | 60 | 60 | Tumor / all voxels | 0.0043 | 0.0043 | 0.0043 |
| WT | 60 | 60 | Accuracy | 0.9957 | 0.9871 | 0.9974 |
| WT | 60 | 60 | Precision | 0.8057 | 0.4634 | 0.8874 |
| WT | 60 | 60 | Recall | 0.7797 | 0.9321 | 0.8730 |
| WT | 60 | 60 | Specificity | 0.9983 | 0.9877 | 0.9988 |
| WT | 60 | 60 | F1 (= Dice) | 0.7777 | 0.5966 | 0.8713 |
| WT | 60 | 60 | Reported Dice (table 2) | 0.7777 | 0.5966 | 0.8713 |
| WT | 60 | 60 | F1 vs reported Dice | ok | ok | ok |
| WT | 60 | 60 | Tumor / brain voxels | 0.0121 | 0.0121 | 0.0121 |
| WT | 60 | 60 | Tumor / all voxels | 0.0107 | 0.0107 | 0.0107 |

F1 = 2·Precision·Recall / (Precision+Recall) = 2 TP / (2 TP + FP + FN) = binary Dice.

