## Table 5 — Segmentation vs ellipsoid-formula error (v3-lcc, WT, n=60)

| quantity | value_ml | mean_ml | n |
| --- | --- | --- | --- |
| median |pred voxel − GT voxel| (seg error) | 5.85 | 10.62 | 60 |
| median |GT ABC/2 − GT voxel| (formula error on GT) | 38.85 | 51.31 | 60 |
| median |pred ABC/2 − GT voxel| (pipeline ellipsoid error) | 21.17 | 36.02 | 60 |
| Spearman(|GT-formula error|, |pipeline ell error|) | 0.473 (p=1.33e-04) |  | 60 |
| Spearman(|seg error|, |pipeline ell error|) | 0.145 (p=2.70e-01) |  | 60 |

### Illustrative case BraTS20_Training_257

| subject_id | role | gt_voxel_ml | pred_voxel_ml | gt_ellipsoid_abc2_ml | pred_ellipsoid_abc2_ml | abs_seg_ml | abs_gt_formula_ml |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BraTS20_Training_257 | illustrative (near-perfect seg, residual formula gap) | 82.02 | 82.22 | 113.73 | 101.50 | 0.19 | 31.71 |
