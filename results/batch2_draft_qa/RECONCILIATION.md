# v3 draft vs LOSO Dice reconciliation

## Root cause

`scripts/prepare_batch2_labeling.py` previously stacked modalities as
`(T1c, T1, T2, FLAIR)` instead of the BraTS / v3 order `(FLAIR, T1, T1c, T2)`.
Same checkpoint (`brats_scale_full`), same registered 1 mm inputs, same
`get_patient_inference_transforms` + sliding-window + LCC + nesting —
only the channel order was wrong. Reproducing the wrong order yields
Dice WT = 1.0 vs the archived stale `v3_draft_label` files.

`experimental_draft_ft` LOSO used `preprocessing.h5_to_nifti.MODALITY_NAMES`
(correct order) on the labeling-package volumes → those Dice vs Claude
are the valid ones.

## T1c stand-in

These packages are `native_t1c: true` (real Gd T1c). Stand-in T1→T1c was
not used for these draft studies.

## Stale files

Yes — previous `v3_draft_label.nii.gz` are stale. Archived beside each
package as `v3_draft_label.stale_wrong_channel_order.nii.gz`; rewritten with correct channel order.
Checkpoints were not modified.

## Correct WT Dice vs Claude

| study | stale file vs Claude | correct v3 (regen = LOSO) |
|---|---:|---:|
| batch2_01 | 0.000 | **0.594** |
| batch2_07 | 0.198 | **0.420** |
| batch2_10 | 0.126 | **0.702** |
| batch2_12 | 0.000 | **0.758** |

Regenerated `outputs/batch2_draft_qa/per_case.csv` now matches these values.
`scripts/prepare_batch2_labeling.py` now imports `MODALITY_NAMES` from
`preprocessing.h5_to_nifti` so future packages keep the correct order.

