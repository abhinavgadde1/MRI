# Audit after v3 channel-order fix

**Bug:** several scripts stacked modalities as `(T1c, T1, T2, FLAIR)` instead of BraTS/v3 `(FLAIR, T1, T1c, T2)`. That produced stale `outputs/batch2_labeling/*/v3_draft_label.nii.gz` (archived as `*.stale_wrong_channel_order.nii.gz`). Checkpoints were not modified.

## 1. Hardcoded 4-channel / modality-stack audit

| file:line | construct | verdict |
|---|---|---|
| `src/preprocessing/h5_to_nifti.py:37` | `MODALITY_NAMES = ("FLAIR", "T1", "T1c", "T2")` | **correct** (canonical) |
| `src/segmentation/model.py:14` | `INPUT_MODALITIES = MODALITY_NAMES` | **correct** |
| `src/segmentation/dataset.py:62` | image list via `MODALITY_NAMES` | **correct** |
| `src/segmentation/pseudo_label.py:99` | image list via `MODALITY_NAMES` | **correct** |
| `src/segmentation/finetune.py:111` | image list via `MODALITY_NAMES` | **correct** |
| `src/segmentation/batch2_dataset.py:266` | image list via `MODALITY_NAMES` | **correct** |
| `src/segmentation/batch2_weaksup.py:255,283` | image list via `MODALITY_NAMES` | **correct** |
| `src/segmentation/experimental_draft_ft.py:171,324` | image list via `MODALITY_NAMES` | **correct** |
| `src/segmentation/batch2_augment.py:27–29` | channel indices from `MODALITY_NAMES` | **correct** |
| `scripts/run_pos_neg_separation.py:74` | image list via `MODALITY_NAMES` | **correct** |
| `scripts/run_batch2_neg_baseline_v3.py:53` | image list via `MODALITY_NAMES` | **correct** |
| `scripts/run_batch2_confirmed_fp_baseline.py:167` | image list via `MODALITY_NAMES` | **correct** |
| `scripts/prepare_batch2_labeling.py` | now imports `MODALITY_NAMES` | **fixed** (was wrong) |
| `scripts/rerun_batch2_native_t1c.py` | now imports `MODALITY_NAMES` | **fixed** (was `("T1c","T1","T2","FLAIR")`) |
| `scripts/run_batch2_v3_preview.py` | now imports `MODALITY_NAMES` | **fixed** (was `("T1c","T1","T2","FLAIR")`) |
| `scripts/prepare_batch2_labeling.py` / ingest loops `("T1","T1c","T2","FLAIR")` | file copy / metadata only | **OK** (not model input stack) |
| `src/preprocessing/batch_pipeline.py:33` | `BRATS_H5_CHANNELS = ("FLAIR","T1","T1c","T2")` | **correct** |

Regression guard: `tests/test_modality_channel_order.py`.

## 2. Consumer impact & reruns

| artifact | used stale v3 / wrong stack? | action |
|---|---|---|
| `outputs/batch2_labeling/*/v3_draft_label.nii.gz` | **yes** (wrong stack) | regenerated earlier; stale archived |
| `outputs/batch2_draft_qa/` | **yes** (read stale files) | regenerated earlier |
| `outputs/batch2_intensity_draft/` | **yes** (Dice/agree3 vs v3 file) | **rerun** |
| `outputs/batch2_second_opinion/agreement.csv` + SUMMARY PART C | **yes** (compared to stale v3) | **rerun compare** (HD-GLIO segs kept) |
| `outputs/batch2_second_opinion/CONTROL_TEST.md` | **no** (BraTS held-out + stored v3 LCC metrics) | no rerun |
| `results/batch2_weaksup/` | **no** (live v3 via `MODALITY_NAMES`; WT mL already matches corrected) | no rerun |
| `results/pos_neg_separation/` | **no** (`live_v3` via `MODALITY_NAMES`; batch2_01 WT=71.39) | no rerun |
| `splits/batch2_pos_neg/` | **no** (ID lists only; no v3 volumes) | no rerun |
| `outputs/batch2_v3_preview/`, `outputs/batch2_native_t1c/*preview*` | **yes** (wrong-order scripts) | scripts fixed; preview NIfTIs remain historical/stale (labeling packages are source of truth) |

## 3. Before / after tables

### 3a. HD-GLIO vs v3 WT (agreement.csv)

| study | before Dice | before class | before v3 mL | after Dice | after class | after v3 mL |
|---|---:|---|---:|---:|---|---:|
| batch2_01 | 0.000 | disagree | 7.66 | 0.698 | partial | 71.39 |
| batch2_02 | 0.202 | disagree | 2.92 | 0.518 | partial | 14.20 |
| batch2_03 | 0.000 | disagree | 6.40 | 0.578 | partial | 121.70 |
| batch2_05 | 0.003 | disagree | 1.17 | 0.762 | both_agree | 68.04 |
| batch2_06 | 0.028 | disagree | 7.07 | 0.020 | disagree | 4.59 |
| batch2_07 | 0.198 | disagree | 10.22 | 0.594 | partial | 119.31 |
| batch2_08 | 0.000 | one_finds_no_tumor | 0.65 | 0.000 | one_finds_no_tumor | 1.42 |
| batch2_10 | 0.284 | disagree | 10.86 | 0.645 | partial | 48.92 |
| batch2_12 | 0.000 | disagree | 7.60 | 0.714 | both_agree | 19.51 |
| batch2_13 | 0.000 | disagree | 13.10 | 0.770 | both_agree | 4.04 |

Class counts: before **disagree 9 / one_finds_no_tumor 1 / both_agree 0 / partial 0** → after **both_agree 3 / partial 5 / disagree 1 / one_finds_no_tumor 1**.

### 3b. Intensity draft three-way ET centroid agree + Dice vs v3

| study | before agree3 | after agree3 | before Dice WT v3 | after Dice WT v3 | after v3 ET mL (in SUMMARY) |
|---|:---:|:---:|---:|---:|---:|
| batch2_01 | no | no | 0.000 | 0.000 | 7.60 |
| batch2_02 | no | no | 0.000 | 0.000 | 4.28 |
| batch2_03 | no | no | 0.000 | 0.000 | 7.78 |
| batch2_05 | no | no | 0.000 | ~0 | 3.06 |
| batch2_06 | no | no | 0.000 | 0.000 | 1.72 |
| batch2_07 | no | no | 0.000 | 0.000 | 4.80 |
| batch2_08 | no | no | 0.000 | 0.000 | 0.13 |
| batch2_10 | no | no | 0.000 | 0.000 | 15.01 |
| batch2_12 | no | no | 0.000 | 0.000 | 2.43 |
| batch2_13 | no | no | 0.000 | 0.000 | 2.47 |

agree3 count: **0 → 0** (unchanged). Intensity vs corrected v3 still essentially no WT/ET overlap; v3 volumes in the intensity SUMMARY did update.

### 3c. Draft QA (Claude vs v3) — already regenerated

| study | stale Dice WT v3 | corrected Dice WT v3 |
|---|---:|---:|
| batch2_01 | 0.000 | 0.594 |
| batch2_07 | 0.198 | 0.420 |
| batch2_10 | 0.126 | 0.702 |
| batch2_12 | 0.000 | 0.758 |

## 4. Conclusions that CHANGED

1. **Withdrawn:** “v3 and HD-GLIO disagree on location on essentially all native-T1c studies” (was driven by wrong-order v3 drafts with tiny/misplaced WT).
2. **Now:** v3 vs HD-GLIO WT is **both_agree on 3/10**, **partial on 5/10**, **disagree on 1/10**, **one-sided on 1/10** (`batch2_08`).
3. **Withdrawn (draft QA):** Claude vs v3 WT Dice = 0 on `batch2_01`/`12` as a property of the drafts — that was stale v3; corrected values are 0.594 / 0.758.
4. **Unchanged:** three-way intensity∩v3∩HD-GLIO ET centroid agreement remains **none**; intensity drafts still do not colocalize with model drafts.
5. **Unchanged:** no verified clinical masks; scanner-shortcut AUC ≈ 0.85; cohort results remain qualitative only.
6. **Unchanged / not rerun:** weaksup scores/montages and pos_neg_separation already used correct live channel order; CONTROL_TEST is BraTS-only.

## 5. LIMITATIONS

Updated `LIMITATIONS.md` and `results/paper/LIMITATIONS.md` §10 (and `scripts/make_paper_assets.py` generator) to match the corrected reruns above.
