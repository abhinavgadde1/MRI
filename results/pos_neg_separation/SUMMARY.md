# Positive vs negative separation (v3+LCC)

Measurement only — no training. Checkpoint `brats_scale_full` read-only.

## Limits

- **19** batch2 positives only.
- Negatives: tumor-free status from `manifest.csv` (radiologist_report / radiologist_verbal / claimed_unverified).
- Radiologist-confirmed negatives available for analysis: **0** — **confirmed subset skipped; (b) all claimed only**.
- Scanner mix differs (batch2 mostly Philips Ingenia; negatives mostly Achieva).

## ROC AUC tables

| subset | score | n_pos | n_neg | AUC | 95% CI | best thr | sens | spec |
|---|---|---:|---:|---:|---|---:|---:|---:|
| all_claimed_negatives | WT_lcc_ml | 19 | 99 | 0.777 | [0.634, 0.908] | 14.2 | 0.684 | 0.823 |
| all_claimed_negatives | TC_lcc_ml | 19 | 99 | 0.815 | [0.685, 0.929] | 6.495 | 0.684 | 0.865 |
| all_claimed_negatives | ET_lcc_ml | 19 | 99 | 0.809 | [0.670, 0.927] | 1.61 | 0.684 | 0.927 |
| all_claimed_negatives | WT_raw_n_cc | 19 | 99 | 0.280 | [0.168, 0.404] | 5 | 1.000 | 0.021 |
| all_claimed_negatives | WT_max_prob_on_lcc | 19 | 99 | 0.426 | [0.309, 0.544] | 0.9998 | 0.947 | 0.188 |
| native_T1c_positives_vs_all_neg | WT_lcc_ml | 10 | 99 | 0.805 | [0.633, 0.960] | 14.2 | 0.700 | 0.823 |
| standin_T1c_positives_vs_all_neg | WT_lcc_ml | 9 | 99 | 0.745 | [0.517, 0.954] | 15.45 | 0.667 | 0.854 |
| native_T1c_positives_vs_all_neg | TC_lcc_ml | 10 | 99 | 0.873 | [0.758, 0.966] | 2.76 | 0.900 | 0.750 |
| standin_T1c_positives_vs_all_neg | TC_lcc_ml | 9 | 99 | 0.750 | [0.513, 0.967] | 9.08 | 0.667 | 0.906 |
| native_T1c_positives_vs_all_neg | ET_lcc_ml | 10 | 99 | 0.951 | [0.881, 0.994] | 1.721 | 0.900 | 0.938 |
| standin_T1c_positives_vs_all_neg | ET_lcc_ml | 9 | 99 | 0.650 | [0.445, 0.864] | 1.61 | 0.444 | 0.927 |
| native_T1c_positives_vs_all_neg | WT_raw_n_cc | 10 | 99 | 0.312 | [0.154, 0.485] | 5 | 1.000 | 0.021 |
| standin_T1c_positives_vs_all_neg | WT_raw_n_cc | 9 | 99 | 0.244 | [0.108, 0.402] | 8 | 1.000 | 0.042 |
| native_T1c_positives_vs_all_neg | WT_max_prob_on_lcc | 10 | 99 | 0.427 | [0.311, 0.546] | 1 | 0.900 | 0.344 |
| standin_T1c_positives_vs_all_neg | WT_max_prob_on_lcc | 9 | 99 | 0.425 | [0.253, 0.617] | 0.9953 | 1.000 | 0.094 |

## Shortcut check (non-tumor features)

Features: in-plane spacing, brain mask fraction, scanner model, T1c native/stand-in, mean brain intensity. Logistic regression, patient-grouped 5-fold CV.

- Mean CV AUC: **0.850** (folds=[0.9, 0.7263157894736842, 0.9750000000000001, 0.9666666666666667, 0.6833333333333333])

> **WARNING:** Non-tumor features separate positives from negatives well (CV AUC ≥ 0.75). Any classifier trained on these groups may be reading the scanner / acquisition protocol — not the tumor.

## Verdict

**Does v3 WT volume separate the groups: partly**

Primary WT AUC vs all claimed negatives = 0.777 [0.634, 0.908].

Plots: `roc_distributions_*.png` in this folder.
