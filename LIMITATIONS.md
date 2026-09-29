# Limitations

Canonical write-up for the paper submission lives at
[`results/paper/LIMITATIONS.md`](results/paper/LIMITATIONS.md).

Short summary for reuse:

1. **Primary model is v3** (`checkpoints/brats_scale_full`), trained on 289 BraTS
   cases. Earlier v1/v2 runs used a 40-case subset and are ablation-only
   (`experiments/ablation_v1_v2/`).
2. **No clinical ground-truth volumes** — ellipsoid formulas are compared to BraTS
   voxel volumes from GT masks.
3. **T1-as-T1c stand-in** on real-patient DICOMs when no native T1c series exists
   (see `outputs/real_patients_demo/`).
4. **LCC post-process** can drop the true tumor if a larger false-positive component
   exists (v3: 1/60 held-out, case 259).
5. **Ellipsoid-formula error dominates** once segmentation is strong (Table 5).

Read the full list before interpreting clinical or paper numbers.
