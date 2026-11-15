# Registration QC diagnosis (BraTS positive control)

## Cause

Brain Dice used the **full patient voxel grid** as FOV support. For BraTS
(origin 0,0,0; 240×240×155 covering most of MNI after centering) that pulled
nearly the entire template brain (~1.89M voxels) into the denominator while
patient brain is ~1.4–1.5M → Dice capped ~0.75 even at a good centered init.
Batch2 HD-BET FOVs did not cover the template the same way, so they passed.
Tumor cost mask was **not** the primary failure mode (BraTS failed without it too).

## Fix (threshold unchanged at 0.85)

- Hole-filled LCC brain mask (same policy for BraTS + HD-BET).
- Dice FOV = dilated brain mask, not the full grid.
- Keep best of centered-init / affine / B-spline by Dice (optimizer can degrade BraTS).
- Cost mask still excludes WT dilated 5 mm.

## (a) ± tumor cost mask — brain Dice / MI (pre-fix register.py)

| Case | Cost mask | MI before→after | Brain Dice | Quality |
|---|---|---:|---:|---|
| BraTS20_Training_002 | no | -0.0085→-0.0489 | 0.575 | registration unreliable |
| BraTS20_Training_002 | yes | -0.0085→-0.0482 | 0.541 | registration unreliable |
| BraTS20_Training_013 | no | -0.0049→-0.0916 | 0.681 | registration unreliable |
| BraTS20_Training_013 | yes | -0.0049→-0.0809 | 0.659 | registration unreliable |
| BraTS20_Training_161 | no | -0.0023→-0.0903 | 0.713 | registration unreliable |
| BraTS20_Training_161 | yes | -0.0023→-0.0327 | 0.536 | registration unreliable |
| batch2_01 | no | -0.1067→-0.5652 | 0.941 | good |
| batch2_01 | yes | -0.1067→-0.5267 | 0.925 | good |
| batch2_05 | no | -0.1417→-0.5490 | 0.911 | good |
| batch2_05 | yes | -0.1417→-0.5585 | 0.907 | good |
| batch2_10 | no | -0.3247→-0.5529 | 0.935 | good |
| batch2_10 | yes | -0.3247→-0.5742 | 0.934 | good |

## Before → after (production path, with cost mask)

| Case | Before Dice | After Dice | After QC |
|---|---:|---:|---|
| BraTS20_Training_002 | 0.749 | 0.887 | good |
| BraTS20_Training_013 | 0.661 | 0.872 | good |
| BraTS20_Training_161 | 0.544 | 0.871 | good |

Overlays: `results/location_report/brats_overlays/*_template_outline.png`.
