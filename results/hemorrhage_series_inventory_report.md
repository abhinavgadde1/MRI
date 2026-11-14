# Hemorrhage-sensitive DICOM inventory — real-patient cohort

**Status:** diagnostic only. Ingestion, preprocessing, and the pipeline were not changed.
**Date:** 27 Oct 2026

## Question

Do the original real-patient DICOMs contain hemorrhage-sensitive sequences
(SWI / SWAN / GRE / T2*) that the current pipeline never ingested?
The processed path only converts **T1, T1c, T2, FLAIR**.

## Source

| item | value |
| --- | --- |
| Raw DICOM root | `MRI DATA` (`config.yaml` → `paths.raw_dicom`) |
| Study discovery | `preprocessing.batch_pipeline.discover_patient_studies` |
| Cohort | same 100 study folders as Table 1 (97 success + 3 failed) |
| What was read | every DICOM series header (`stop_before_pixels`), not `01_dicom_nifti` |
| Tags | `SeriesDescription`, `ProtocolName`, `SequenceName` |
| Ingestion filter | `run_patient_preprocessing` → `load_study_to_nifti(modalities=[T1,T1c,T2,FLAIR], include_other=False)` |

Search (case-insensitive substring) of the three tags above:
`SWI`, `SWAN`, `susceptibility`, `GRE`, `T2star`, `T2*`, `T2_star`, `gradient echo`.
`SeriesDescription` is reported raw (not normalized).

## Headline

| | n |
| --- | ---: |
| Studies scanned | 100 |
| **Has SWI/GRE in raw DICOM** | **96** |
| No hemorrhage-sensitive series | 4 |
| Matching series (all studies) | 202 |
| Studies with *any* extra series beyond T1/T1c/T2/FLAIR | **100** |
| Studies with *only* T1/T1c/T2/FLAIR series | **0** |

**96/100 studies already have Philips SWI in the source DICOMs.**
None of those series were converted, because ingestion drops `modality_label="other"`.

## Matching series (raw `SeriesDescription`)

Every one of the 202 hits matched the pattern **`SWI`** inside the description.
No series matched `SWAN`, `GRE`, `T2*`, `T2star`, `T2_star`, `susceptibility`, or `gradient echo`.
There were no GRE-substring false positives (e.g. nothing like “degree”).

| raw SeriesDescription | series | instances per series | what it is |
| --- | ---: | --- | --- |
| `VSWIp_fast MINI MIP` | 101 | 19–19 (median 19) | Philips SWIp |
| `SWIp_fast` | 101 | 260–290 (median 260) | Philips SWIp |

- 91 studies have one `SWIp_fast` + one `VSWIp_fast MINI MIP`.
- 5 studies have two of each (duplicate series UIDs / re-exports).
- `SWIp_fast` (~260–290 slices) is the source SWI volume.
- `VSWIp_fast MINI MIP` (always 19 slices) is a derived mini-MIP, not a second acquisition.

## Studies with no SWI/GRE (4)

These still have unused extra series (mostly DWI / Survey), just not hemorrhage-sensitive ones.

| study_id | n series | unused extra SeriesDescription (instance count) |
| --- | ---: | --- |
| `patient_A01` | 5 | `dReg - DWI` (22), `isoReg - DWI` (22), `Survey` (5) |
| `patient_A23` | 5 | `Reg - DWI` (115), `DWI` (115) |
| `patient_A50` | 6 | `dDWI` (23), `isoDWI` (23), `Survey` (5) |
| `patient_A62` | 12 | `BTFE_RT SENSE` (45), `isoDWI` (23), `STIR_longTE` (23), `SURVEY_bFFE` (17), `BTFE_RT SENSE` (50), `dDWI` (23), `BTFE_RT SENSE` (75) |

## Extra series the pipeline currently ignores

A series is “extra” if `identify_modality()` does **not** label it T1, T1c, T2, or FLAIR.
There are **706** such series across **100/100** studies.

Most frequent unused `SeriesDescription` values:

| count | SeriesDescription |
| ---: | --- |
| 110 | `Survey` |
| 101 | `VSWIp_fast MINI MIP` |
| 101 | `SWIp_fast` |
| 61 | `isoDWI` |
| 61 | `dDWI` |
| 39 | `dReg - DWI` |
| 39 | `isoReg - DWI` |
| 37 | `MIP - s3DI_MC_HR` |
| 37 | `s3DI_MC_HR` |
| 21 | `VEN_3D_PCA` |
| 21 | `MIP - VEN_3D_PCA` |
| 14 | `VVEN_3D_PCA` |
| 7 | `s3DI_MC_F` |
| 7 | `MIP - s3DI_MC_F` |
| 6 | `Vs3DI_MC_F` |
| 4 | `MobiView SURV_MT_COR` |
| 4 | `STIR_TSE_MT` |
| 3 | `MobiView SURV_MT_SAG` |
| 3 | `eMobiView STIR_TSE_MT` |
| 3 | `BTFE_RT SENSE` |

Rough groups (substring, a series can appear in more than one):

| group | extra series |
| --- | ---: |
| Survey | 232 |
| DWI / isoDWI | 202 |
| SWI (`SWIp` / `VSWIp`) | 202 |
| 3D angio-like (`s3DI` / MIP) | 96 |
| Venous PCA (`VEN_3D_PCA`) | 56 |

## Per-study inventory

Multiple matches in one study are joined with ` || `. Empty description/count means no hit.

| study_id | has_swi_or_gre | matching_series_description | series_instance_count |
| --- | --- | --- | --- |
| `patient_A01` | no | — | — |
| `patient_A02` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A03` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A04` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A05` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A06` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A07` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A08` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A09` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A10` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A11` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A12` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A13` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A14` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A15` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A16` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A17` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A18` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A19` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A20` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A21` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A22` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A23` | no | — | — |
| `patient_A24` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A25` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A26` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A27` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A28` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A29` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A30` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A31` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A32` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A33` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 280 || 280 |
| `patient_A34` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A35` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A36` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A37` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A38` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A39` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A40` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 260 || 280 |
| `patient_A41` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A42` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A43` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `patient_A44` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A45` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A46` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A47` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A48` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A49` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A50` | no | — | — |
| `patient_A51` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A52` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A53` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A54` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A55` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A56` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A57` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A58` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A59` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A60` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A61` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A62` | no | — | — |
| `patient_A63` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A64` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 290 || 19 |
| `patient_A65` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `patient_A66` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A67` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A68` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A69` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A70` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A71` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A72` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `patient_A73` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A74` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `patient_A75` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A76` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A77` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A78` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A79` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A80` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `patient_A81` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast | 280 || 19 || 19 || 280 |
| `patient_A82` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A83` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A84` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 280 || 280 |
| `patient_A85` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A86` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A87` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A88` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A89` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A90` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A91` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A92` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A93` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `patient_A94` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A95` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A96` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A97` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `patient_A98` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A99` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `patient_A100` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 280 || 280 |

## Files

Alongside this report:

- `study_hemorrhage_inventory.csv` — one row per study
- `matching_series.csv` — one row per matching series (UID, protocol, instance count)
- `extra_series_not_ingested.csv` — every non-T1/T1c/T2/FLAIR series
- `summary.json`

## What this does *not* do

- Does not convert SWI (or DWI, TOF, Survey) to NIfTI
- Does not change `identify_modality()` or `include_other=False`
- Does not re-run preprocessing or registration
- Does not claim the mini-MIP is usable as a 3D SWI volume; the candidate
  for any future ingest is `SWIp_fast`

