## Table 1 — Dataset & preprocessing

| item | value |
| --- | --- |
| BraTS train cases | 40 |
| BraTS val cases (train-time) | 10 |
| BraTS held-out cases | 60 |
| Split seed | 42 |
| Real patients processed (success) | 97 |
| Real patients failed | 3 (FileNotFoundError×3) |
| T1c native in DICOM (success cohort) | 0 |
| T1→T1c stand-in (success cohort) | 97 |
| Target spacing | 1.0×1.0×1.0 mm isotropic |
| Bias correction | N4 (SimpleITK) |
| Skull stripping | HD-BET if available, else SimpleITK Otsu fallback |
| Registration | Rigid to T1 (ITK), then 1 mm isotropic resample |
| Source log | data/processed/batch_summary_patients.csv |

### Real-patient MRI study IDs (n=100)

Full machine-readable list: `table1_real_patient_study_ids.csv`.

#### Successful (n=97)

| # | study_id | status |
| --- | --- | --- |
| 1 | 26070305080270453758_MR_Study1 | success |
| 2 | 26070314071267604026_MR_Study1 | success |
| 3 | 26070408200268074217_MR_Study1 | success |
| 4 | 26070409265767144257_MR_Study1 | success |
| 5 | 26070409540769354286_MR_Study1 | success |
| 6 | 26070409540769354286_MR_Study1 2 | success |
| 7 | 26070605394269334502_MR_Study1 | success |
| 8 | 26070613014390354814_MR_Study1 | success |
| 9 | 26070614152390454835_MR_Study1 | success |
| 10 | 26070707084390895010_MR_Study1 | success |
| 11 | 26070708365890165072_MR_Study1 | success |
| 12 | 26070709151891795096_MR_Study1 | success |
| 13 | 26070709594892225134_MR_Study1 | success |
| 14 | 26070905302392205679_MR_Study1 | success |
| 15 | 26070906421390285747_MR_Study1 | success |
| 16 | 26070911455390915925_MR_Study1 | success |
| 17 | 26070912134892975927_MR_Study1 | success |
| 18 | 26071007172890826108_MR_Study1 | success |
| 19 | 26071011152890086256_MR_Study1 | success |
| 20 | 26071012590390146272_MR_Study1 | success |
| 21 | 26071304012516286710_MR_Study1 | success |
| 22 | 26071304320513066722_MR_Study1 | success |
| 23 | 26071308571015536948_MR_Study1 | success |
| 24 | 26071313115513007068_MR_Study1 | success |
| 25 | 26071404271014157110_MR_Study1 | success |
| 26 | 26071406563514347220_MR_Study1 | success |
| 27 | 26071408185516237288_MR_Study1 | success |
| 28 | 26071409424013477341_MR_Study1 | success |
| 29 | 26071413483512457432_MR_Study1 | success |
| 30 | 26071413495515087434_MR_Study1 | success |
| 31 | 26071508002014517662_MR_Study1 | success |
| 32 | 26071508245013857670_MR_Study1 | success |
| 33 | 26071511453513707764_MR_Study1 | success |
| 34 | 26071608095638498039_MR_Study1 | success |
| 35 | 26071609030141018101_MR_Study1 | success |
| 36 | 26071612044138988181_MR_Study1 | success |
| 37 | 26072007242137269199_MR_Study1 | success |
| 38 | 26072008063136939243_MR_Study1 | success |
| 39 | 26072109394762409708_MR_Study1 | success |
| 40 | 26072110054761049720_MR_Study1 | success |
| 41 | 26072110054761049720_MR_Study1 2 | success |
| 42 | 26072113304759879786_MR_Study1 | success |
| 43 | 26072114125261929790_MR_Study1 | success |
| 44 | 26072209432761090085_MR_Study1 | success |
| 45 | 26072306520261330311_MR_Study1 | success |
| 46 | 26072310183761270463_MR_Study1 | success |
| 47 | 26072312074760640521_MR_Study1 | success |
| 48 | 26072413462262120855_MR_Study1 | success |
| 49 | 26072510562762381109_MR_Study1 | success |
| 50 | 26072708432886901452_MR_Study1 | success |
| 51 | 26072710225385761524_MR_Study1 | success |
| 52 | 26072711473884981574_MR_Study1 | success |
| 53 | 26072904510384911995_MR_Study1 | success |
| 54 | 26072906541386952125_MR_Study1 | success |
| 55 | 26072911164385742279_MR_Study1 | success |
| 56 | 26072912020385312283_MR_Study1 | success |
| 57 | 26072912460887602295_MR_Study1 | success |
| 58 | 26072913580886662307_MR_Study1 | success |
| 59 | 26072914582884332315_MR_Study1 | success |
| 60 | 26073007020887122462_MR_Study1 | success |
| 61 | 26073009564886362586_MR_Study1 | success |
| 62 | 26073011501385172644_MR_Study1 | success |
| 63 | 26073105544886072756_MR_Study1 | success |
| 64 | 26073107263508562864_MR_Study1 | success |
| 65 | 26073108355508012892_MR_Study1 | success |
| 66 | 26073109163009922908_MR_Study1 | success |
| 67 | 26073109560034012914_MR_Study1 | success |
| 68 | 26073110381008022940_MR_Study1 | success |
| 69 | 26080106174008113074_MR_Study1 | success |
| 70 | 26080106414508393088_MR_Study1 | success |
| 71 | 26080109094509783162_MR_Study1 | success |
| 72 | 26080109411703950057_MR_Study1 | success |
| 73 | 26080109440916670059_MR_Study1 | success |
| 74 | 26080304521011253338_MR_Study1 | success |
| 75 | 26080309094008153581_MR_Study1 | success |
| 76 | 26080312074008673751_MR_Study1 | success |
| 77 | 26080409020110314029_MR_Study1 | success |
| 78 | 26080411131609064091_MR_Study1 | success |
| 79 | 26080508352607484385_MR_Study1 | success |
| 80 | 26080607105609534682_MR_Study1 | success |
| 81 | 26080610235109424796_MR_Study1 | success |
| 82 | 26080705433108294942_MR_Study1 | success |
| 83 | 26080706533610904998_MR_Study1 | success |
| 84 | 26080707233109445020_MR_Study1 | success |
| 85 | 26080708573609205118_MR_Study1 | success |
| 86 | 26080709560611015148_MR_Study1 | success |
| 87 | 26080711190110325174_MR_Study1 | success |
| 88 | 26080810164210735473_MR_Study1 | success |
| 89 | 26081004291713445679_MR_Study1 | success |
| 90 | 26081006095708455780_MR_Study1 | success |
| 91 | 26081009054708325987_MR_Study1 | success |
| 92 | 26081009243210326011_MR_Study1 | success |
| 93 | 26081011444209986065_MR_Study1 | success |
| 94 | 26081012130710966071_MR_Study1 | success |
| 95 | 26081013263710686081_MR_Study1 | success |
| 96 | CHINNAMMAI.PRATHIPATI   50Y F_MR_Study1 | success |
| 97 | MADHU U  29Y M_MR_Study1 | success |

#### Failed (n=3)

| # | study_id | status |
| --- | --- | --- |
| 1 | 26070216161267453698_MR_Study1 | failed |
| 2 | 26072506341762140975_MR_Study1 | failed |
| 3 | 26073006234885442426_MR_Study1 | failed |
