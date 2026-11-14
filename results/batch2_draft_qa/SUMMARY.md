# Batch2 draft QA (unverified drafts)

**All masks here are draft / unverified — not clinical reference.**

n_studies=4. Edge band = 5.0 mm inside brain mask; midline strip = central ~5 voxels; thin midline CC (<4.0 mm) → `possible vessel`.

| study | WT mL | TC mL | ET mL | n_CC | edge% | mid% | vessel? | Dice WT v3 | Dice WT HD | Δcent v3 mm | Δcent HD mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| batch2_01 | 30.35 | 0.00 | 0.00 | 2 | 0.1 | 5.4 | — | 0.594 | 0.854 | 10.6 | 5.3 |
| batch2_07 | 34.75 | 4.78 | 4.50 | 2 | 6.5 | 0.1 | — | 0.420 | 0.555 | 6.9 | 19.8 |
| batch2_10 | 38.27 | 5.94 | 5.94 | 1 | 4.0 | 4.6 | — | 0.702 | 0.549 | 7.3 | 7.9 |
| batch2_12 | 12.91 | 0.00 | 0.00 | 1 | 0.2 | 0.0 | — | 0.758 | 0.909 | 13.7 | 1.7 |

## Notes

- Overlap with v3 / HD-GLIO is descriptive only; none of the three are verified.
- High edge% or midline% suggests possible extracranial / sinus / falx leak.
- **v3 Dice here matches LOSO** after regenerating `v3_draft_label.nii.gz` with BraTS channel order `(FLAIR, T1, T1c, T2)`. Prior QA used stale files from a `(T1c, T1, T2, FLAIR)` stacking bug — see `RECONCILIATION.md`.

