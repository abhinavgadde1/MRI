# Limitations

1. **Early models used a small BraTS subset.** v1/v2 were trained on a 40-case
   train / 10-case val split (seed 42) under Mac CPU/MPS and disk constraints.
   The **primary reported model (v3)** scales to 289 train / 20 val cases
   (excluding the 60 held-out) over 20 epochs.

2. **No manual mask correction.** Clinical fine-tuning / corrected masks were
   not used for the held-out comparison; results reflect BraTS-pretrained (or
   loss-fixed / scaled) weights only.

3. **T1-as-T1c stand-in.** Among successfully processed real-patient studies,
   none had a native contrast T1c series in the DICOM modality probe; T1 was
   copied to the T1c slot for model input (`prepare_correction_set.fill_missing_t1c_from_t1`).
   Three studies failed entirely (missing T1; FLAIR+T2 only).

4. **No clinical ground-truth volumes.** Ellipsoid formulas (ABC/2, π/6×ABC) were
   validated against BraTS **voxel** volumes from GT masks, not against
   radiologist-measured clinical volumes.

5. **WT/ET–TC trade-off was specific to the 40-case v2 run (ablation finding).**
   Best checkpoints are chosen by **mean** val Dice over ET/TC/WT. On the small
   v2 run, improving ET/TC came with a held-out WT regression vs v1 (paired
   ΔDice ≈ −0.18). Scaling to **v3 (289 train cases)** removed that trade-off:
   v3 beats both v1 and v2 on ET, TC, and WT simultaneously. This is documented
   as an ablation result, **not** a limitation of the final reported model.

6. **LCC largest-component failure.** Post-process `lcc` keeps the largest
   26-connected component. Failure rate on held-out: v1 2/60, v2 2/60, v3 1/60
   (case 259). Mechanism unchanged: a large false-positive blob is retained and
   the true tumor component is discarded (raw∩GT>0, LCC∩GT=0).

7. **v2 training instability (historical).** After switching to DiceCELoss over
   all channels, epoch-to-epoch val Dice was noisy under ReduceLROnPlateau
   inherited from v1. v3 replaced that schedule with linear warmup + cosine
   annealing and gradient clipping (1.0).

8. **Ellipsoid-formula error dominates once segmentation is strong.** For v3-lcc
   WT, predicted-vs-GT **voxel** ICC is ~0.93 (median |seg error| ~5.9 mL), but
   predicted radiologist ABC/2 vs GT voxel ICC remains ~0.55 — nearly identical
   to GT-only ABC/2 vs GT voxel (~0.58). Pipeline ellipsoid error correlates with
   GT-formula error (Spearman ~0.47) far more than with segmentation error
   (~0.15). Further improving the segmentation model would therefore **not**
   proportionally improve clinical volume estimates that use the ellipsoid
   method specifically (see Table 5; case 257).
