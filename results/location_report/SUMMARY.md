# Tumor location report

Optional feature (**OFF by default**). Checkpoints untouched.
Transforms / QC PNGs under `outputs/location_*/**/registration/` (gitignored).
No patient scans or atlas binaries committed.

BraTS overlays: `brats_overlays/`. QC thresholds justified below.

**JHU:** JHU atlas not present — CST / optic-radiation distances skipped.

Run `python scripts/setup_jhu_atlas.py` to copy from `$FSLDIR` if available.

## QC rule

`good` requires brain Dice ≥ 0.85 **and** ventricle Dice ≥ 0.60 (SynthSeg) or ≥ 0.30 (CSF heuristic) **and** surface p95 ≤ 8 mm **and** FOV coverage ≥ 0.90; else `caution` or `registration unreliable`. Transform chosen by Mattes MI (not Dice); stage reported (`centered_init` → summaries labeled `coarse (translation only)`).

## BraTS held-out style cases (GT masks)

| Case | QC | Stage | MI | Brain | Vent | Surf mean/p95 | Cov | Hemi | Lobe | % | Summary |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| BraTS20_Training_002 | caution | centered_init | -0.009→-0.167 | 0.887 | 0.262 | 1.7/4.6 | 0.988 | left | temporal | 50 | coarse (translation only): left temporal, 50%, abuts lateral ventricle |
| BraTS20_Training_013 | caution | centered_init | -0.005→-0.133 | 0.872 | 0.211 | 1.7/4.6 | 0.988 | right | frontal | 80 | coarse (translation only): right frontal, 80%, abuts lateral ventricle, near midline |
| BraTS20_Training_161 | caution | centered_init | -0.002→-0.097 | 0.871 | 0.173 | 1.7/4.6 | 0.988 | left | parietal | 63 | coarse (translation only): left parietal, 63%, abuts lateral ventricle, near midline |

### BraTS nearest-lobe breakdown (WT)

#### BraTS20_Training_002

| Lobe | % | mL |
|---|---:|---:|
| temporal | 50.0 | 33.49 |
| insula | 23.9 | 16.02 |
| frontal | 9.3 | 6.22 |
| occipital | 8.0 | 5.35 |
| parietal | 7.1 | 4.77 |
| deep/central | 1.7 | 1.16 |

#### BraTS20_Training_013

| Lobe | % | mL |
|---|---:|---:|
| frontal | 80.1 | 37.12 |
| deep/central | 17.4 | 8.08 |
| insula | 2.4 | 1.11 |
| parietal | 0.1 | 0.03 |

#### BraTS20_Training_161

| Lobe | % | mL |
|---|---:|---:|
| parietal | 62.6 | 33.66 |
| frontal | 29.0 | 15.60 |
| deep/central | 4.4 | 2.37 |
| insula | 2.2 | 1.21 |
| occipital | 1.4 | 0.73 |
| temporal | 0.4 | 0.21 |

## Batch2 studies (neutral IDs only)

Every row is **model-estimated, unverified**. QC overlays: `outputs/location_batch2/<id>/registration/registration_qc.png`.

| Study | QC | Stage | Brain | Vent | Surf p95 | Cov | Hemi | Lobe | % | Summary |
|---|---|---|---:|---:|---:|---:|---|---|---:|---|
| batch2_01 | good | bspline | 0.936 | 0.380 | 4.1 | 1.000 | bilateral | frontal | 83 | bilateral (L 70% / R 30%) frontal, 83%, abuts lateral ventricle, near midline |
| batch2_02 | gated | — | — | — | — | — | — | — | 0 | model volume 14.20 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_03 | good | bspline | 0.938 | 0.313 | 4.2 | 1.000 | right | parietal | 37 | right parietal, 37%, abuts lateral ventricle, near midline |
| batch2_04 | gated | — | — | — | — | — | — | — | 0 | model volume 0.70 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_05 | registration unreliable | bspline | 0.915 | 0.237 | 5.0 | 0.719 | left | frontal | 73 | Registration unreliable — anatomical location not claimed; see registration_qc.png. |
| batch2_06 | gated | — | — | — | — | — | — | — | 0 | model volume 4.60 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_07 | caution | centered_init | 0.816 | 0.215 | 4.9 | 0.999 | left | frontal | 71 | coarse (translation only): left frontal, 71%, abuts lateral ventricle, near midline |
| batch2_08 | gated | — | — | — | — | — | — | — | 0 | model volume 1.42 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_09 | caution | affine | 0.901 | 0.211 | 5.0 | 0.828 | left | frontal | 82 | left frontal, 82%, near ventricle (3 mm) |
| batch2_10 | good | bspline | 0.938 | 0.434 | 4.1 | 1.000 | bilateral | parietal | 62 | bilateral (L 12% / R 88%) parietal, 62%, abuts lateral ventricle, near midline |
| batch2_11 | gated | — | — | — | — | — | — | — | 0 | model volume 3.50 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_12 | gated | — | — | — | — | — | — | — | 0 | model volume 19.51 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_13 | gated | — | — | — | — | — | — | — | 0 | model volume 4.04 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_14 | registration unreliable | bspline | 0.816 | 0.048 | 4.5 | 0.582 | bilateral | deep/central | 57 | Registration unreliable — anatomical location not claimed; see registration_qc.png. |
| batch2_15 | caution | centered_init | 0.815 | 0.126 | 5.0 | 0.969 | left | frontal | 67 | coarse (translation only): left frontal, 67% |
| batch2_16 | caution | affine | 0.902 | 0.182 | 5.0 | 0.891 | left | parietal | 31 | left parietal, 31%, abuts lateral ventricle, near midline |
| batch2_17 | gated | — | — | — | — | — | — | — | 0 | model volume 1.74 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_18 | caution | centered_init | 0.859 | 0.115 | 5.0 | 0.998 | left | parietal | 50 | coarse (translation only): left parietal, 50%, abuts lateral ventricle, near midline |
| batch2_20 | gated | — | — | — | — | — | — | — | 0 | model volume 15.45 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |

## QC threshold justification

Gates: brain≥0.85; ventricle≥0.60 (SynthSeg) / ≥0.30 (CSF); surface p95≤8 mm; coverage≥0.90.

CSF-mode ventricle Dice on well-registered batch2 tops out ~0.4–0.5 (partial-volume T1); 0.30 is the CSF good gate. SynthSeg keeps 0.60.

| Case | Stage | Brain | Vent | Surf p95 | Cov | QC |
|---|---|---:|---:|---:|---:|---|
| BraTS20_Training_002 | centered_init | 0.887 | 0.262 | 4.6 | 0.988 | caution |
| BraTS20_Training_013 | centered_init | 0.872 | 0.211 | 4.6 | 0.988 | caution |
| BraTS20_Training_161 | centered_init | 0.871 | 0.173 | 4.6 | 0.988 | caution |
| batch2_01 | bspline | 0.936 | 0.380 | 4.1 | 1.000 | good |
| batch2_03 | bspline | 0.938 | 0.313 | 4.2 | 1.000 | good |
| batch2_05 | bspline | 0.915 | 0.237 | 5.0 | 0.719 | registration unreliable |
| batch2_07 | centered_init | 0.816 | 0.215 | 4.9 | 0.999 | caution |
| batch2_09 | affine | 0.901 | 0.211 | 5.0 | 0.828 | caution |
| batch2_10 | bspline | 0.938 | 0.434 | 4.1 | 1.000 | good |
| batch2_15 | centered_init | 0.815 | 0.126 | 5.0 | 0.969 | caution |
| batch2_16 | affine | 0.902 | 0.182 | 5.0 | 0.891 | caution |
| batch2_18 | centered_init | 0.859 | 0.115 | 5.0 | 0.998 | caution |

## Plausibility gate (batch2)

Location statements are withheld when WT volume is below the held-out GT floor (`pipeline.volume_plausibility`) or the largest component holds < 50% of WT. Wording: `model volume X mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified)`.

Gated studies: batch2_02, batch2_04, batch2_06, batch2_08, batch2_11, batch2_12, batch2_13, batch2_17, batch2_20.

| Study | WT mL | LCC fraction | Message |
|---|---:|---:|---|
| batch2_02 | 14.20 | 1.00 | model volume 14.20 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_04 | 0.70 | 1.00 | model volume 0.70 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_06 | 4.60 | 1.00 | model volume 4.60 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_08 | 1.42 | 1.00 | model volume 1.42 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_11 | 3.50 | 1.00 | model volume 3.50 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_12 | 19.51 | 1.00 | model volume 19.51 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_13 | 4.04 | 1.00 | model volume 4.04 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_17 | 1.74 | 1.00 | model volume 1.74 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |
| batch2_20 | 15.45 | 1.00 | model volume 15.45 mL is below the 20.9 mL BraTS size floor; location not reported (small lesion or false positive, unverified) |

## Atlas license notes

- **MNI152 / ICBM152:** ICBM 152 Nonlinear atlases (McGill / ICBM). Distributed via nilearn (fetch_icbm152_2009 / load_mni152_template). See ICBM / nilearn docs for redistribution terms; academic research use.
- **Harvard-Oxford:** Harvard-Oxford cortical and subcortical structural atlases as distributed with FSL and fetched by nilearn (fetch_atlas_harvard_oxford, maxprob thr25 1mm). FSL atlas license; cite Makris / Desikan / Frazier / Harvard Center for Morphometric Analysis.
- **JHU ICBM tracts:** JHU white-matter tractography atlas (Mori lab / Johns Hopkins) as distributed with FSL (JHU-ICBM-tracts-prob-2mm). Optional in this pipeline; used only when files are present under atlases/jhu/. Cite Hua / Mori / Wakana / FSL.

## Feature notes

- Nearest-lobe assignment: every tumor voxel → nearest HO cortical lobe within 15 mm, else `deep/central` (headline). Cortical-only % also reported.
- Transform selection: Mattes MI among centered_init / affine / bspline; QC metrics (brain Dice, ventricle Dice, surface distance, FOV coverage) are independent of selection.
- Cost mask excludes WT dilated 5 mm; hole-filled LCC brain mask; Dice FOV = dilated brain.
- Template outline QC: `registration/template_brain_outline_qc.png`.
- CLI: `python -m pipeline.cli run ... --location-report`
