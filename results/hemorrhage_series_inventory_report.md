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
| `26070216161267453698_MR_Study1` | 5 | `dReg - DWI` (22), `isoReg - DWI` (22), `Survey` (5) |
| `26071304320513066722_MR_Study1` | 5 | `Reg - DWI` (115), `DWI` (115) |
| `26072506341762140975_MR_Study1` | 6 | `dDWI` (23), `isoDWI` (23), `Survey` (5) |
| `26073006234885442426_MR_Study1` | 12 | `BTFE_RT SENSE` (45), `isoDWI` (23), `STIR_longTE` (23), `SURVEY_bFFE` (17), `BTFE_RT SENSE` (50), `dDWI` (23), `BTFE_RT SENSE` (75) |

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
| `26070216161267453698_MR_Study1` | no | — | — |
| `26070305080270453758_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26070314071267604026_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26070408200268074217_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26070409265767144257_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26070409540769354286_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26070409540769354286_MR_Study1 2` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26070605394269334502_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26070613014390354814_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26070614152390454835_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26070707084390895010_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26070708365890165072_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26070709151891795096_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26070709594892225134_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26070905302392205679_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26070906421390285747_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26070911455390915925_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26070912134892975927_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26071007172890826108_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26071011152890086256_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26071012590390146272_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26071304012516286710_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26071304320513066722_MR_Study1` | no | — | — |
| `26071308571015536948_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26071313115513007068_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26071404271014157110_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26071406563514347220_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26071408185516237288_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26071409424013477341_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26071413483512457432_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26071413495515087434_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26071508002014517662_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26071508245013857670_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 280 || 280 |
| `26071511453513707764_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26071608095638498039_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26071609030141018101_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26071612044138988181_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072007242137269199_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26072008063136939243_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26072109394762409708_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 260 || 280 |
| `26072110054761049720_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26072110054761049720_MR_Study1 2` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26072113304759879786_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 280 || 19 |
| `26072114125261929790_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26072209432761090085_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26072306520261330311_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26072310183761270463_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072312074760640521_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072413462262120855_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072506341762140975_MR_Study1` | no | — | — |
| `26072510562762381109_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26072708432886901452_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072710225385761524_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072711473884981574_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072904510384911995_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26072906541386952125_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26072911164385742279_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072912020385312283_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26072912460887602295_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26072913580886662307_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26072914582884332315_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26073006234885442426_MR_Study1` | no | — | — |
| `26073007020887122462_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26073009564886362586_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 290 || 19 |
| `26073011501385172644_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `26073105544886072756_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26073107263508562864_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26073108355508012892_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26073109163009922908_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26073109560034012914_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26073110381008022940_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26080106174008113074_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `26080106414508393088_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26080109094509783162_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `26080109411703950057_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26080109440916670059_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26080304521011253338_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080309094008153581_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080312074008673751_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080409020110314029_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 290 |
| `26080411131609064091_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast | 280 || 19 || 19 || 280 |
| `26080508352607484385_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080607105609534682_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26080610235109424796_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 280 || 280 |
| `26080705433108294942_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080706533610904998_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26080707233109445020_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080708573609205118_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080709560611015148_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080711190110325174_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26080810164210735473_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26081004291713445679_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26081006095708455780_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 280 |
| `26081009054708325987_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26081009243210326011_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `26081011444209986065_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26081012130710966071_MR_Study1` | yes | SWIp_fast \|\| VSWIp_fast MINI MIP | 260 || 19 |
| `26081013263710686081_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `CHINNAMMAI.PRATHIPATI   50Y F_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| SWIp_fast | 19 || 260 |
| `MADHU U  29Y M_MR_Study1` | yes | VSWIp_fast MINI MIP \|\| VSWIp_fast MINI MIP \|\| SWIp_fast \|\| SWIp_fast | 19 || 19 || 280 || 280 |

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

