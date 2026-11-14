# Batch2 weakly supervised CV summary

Negatives: tumor-free status is verbal and unverified (radiologist-reported / claimed_unverified).

Localization: These regions are unvalidated model proposals. No volume or outline accuracy can be claimed without reference masks.

## Overall (pooled held-out weaksup)

- ROC AUC: **0.777** 95% bootstrap CI [0.667, 0.870]
- Threshold (median of train-fold Youden): 0.9897
- Sensitivity: 0.579; Specificity: 0.823; FPR on held-out negatives: 0.177

## Shuffle control

- Shuffled-label weaksup AUC: **0.441** (must be near 0.5; high values imply bug or leakage).
- Real vs shuffled: 0.777 vs 0.441

## Stand-in positives only (n≈9)

- AUC: 0.748 [0.589, 0.877] (n_standin_pos=9)
- Sens=0.444 Spec=0.823 FPR_neg=0.177

## Per-fold metrics

| exp | fold | model | AUC | sens | spec | FPR_neg | thr |
|---|---:|---|---:|---:|---:|---:|---:|
| real | 0 | v3_baseline | 0.588 | 1.000 | 0.150 | 0.850 | 0.9953 |
| real | 0 | weaksup | 0.775 | 0.750 | 0.600 | 0.400 | 0.9906 |
| real | 1 | v3_baseline | 0.137 | 0.600 | 0.105 | 0.895 | 1.0000 |
| real | 1 | weaksup | 0.979 | 0.800 | 0.895 | 0.105 | 0.9230 |
| real | 2 | v3_baseline | 0.526 | 0.750 | 0.263 | 0.737 | 0.9998 |
| real | 2 | weaksup | 0.750 | 0.500 | 0.842 | 0.158 | 0.9897 |
| real | 3 | v3_baseline | 0.298 | 0.667 | 0.211 | 0.789 | 0.9998 |
| real | 3 | weaksup | 0.561 | 0.333 | 0.842 | 0.158 | 0.9863 |
| real | 4 | v3_baseline | 0.526 | 1.000 | 0.158 | 0.842 | 0.9998 |
| real | 4 | weaksup | 0.789 | 0.333 | 0.842 | 0.158 | 0.9934 |
| shuffle | 0 | v3_baseline | 0.294 | 0.250 | 0.750 | 0.250 | 1.0000 |
| shuffle | 0 | weaksup | 0.175 | 0.000 | 0.600 | 0.400 | 0.9950 |
| shuffle | 1 | v3_baseline | 0.276 | 0.250 | 0.895 | 0.105 | 1.0000 |
| shuffle | 1 | weaksup | 0.553 | 0.750 | 0.158 | 0.842 | 0.5876 |
| shuffle | 2 | v3_baseline | 0.368 | 1.000 | 0.000 | 1.000 | 0.5000 |
| shuffle | 2 | weaksup | 0.303 | 0.500 | 0.158 | 0.842 | 0.9342 |
| shuffle | 3 | v3_baseline | 0.439 | 1.000 | 0.000 | 1.000 | 0.5000 |
| shuffle | 3 | weaksup | 0.526 | 0.333 | 0.684 | 0.316 | 0.9991 |
| shuffle | 4 | v3_baseline | 0.362 | 0.250 | 0.421 | 0.579 | 1.0000 |
| shuffle | 4 | weaksup | 0.566 | 0.000 | 0.895 | 0.105 | 0.9987 |

## BraTS held-out forgetting (20-case subset)

- Flag if mean Dice drops by > 0.03: **FLAGGED**
- ET: baseline=0.569 weaksup=0.505 Δ=-0.064
- TC: baseline=0.650 weaksup=0.647 Δ=-0.003
- WT: baseline=0.841 weaksup=0.831 Δ=-0.010

## Timing

- Mean epoch: 136.4s
- Wall: 27322.2s
- Estimated full CV hours (from smoke): 5.683347951018044

Negatives reminder: tumor-free status is verbal and unverified (radiologist-reported / claimed_unverified).

