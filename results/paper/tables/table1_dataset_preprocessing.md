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
| 1 | patient_A02 | success |
| 2 | patient_A03 | success |
| 3 | patient_A04 | success |
| 4 | patient_A05 | success |
| 5 | patient_A06 | success |
| 6 | patient_A07 | success |
| 7 | patient_A08 | success |
| 8 | patient_A09 | success |
| 9 | patient_A10 | success |
| 10 | patient_A11 | success |
| 11 | patient_A12 | success |
| 12 | patient_A13 | success |
| 13 | patient_A14 | success |
| 14 | patient_A15 | success |
| 15 | patient_A16 | success |
| 16 | patient_A17 | success |
| 17 | patient_A18 | success |
| 18 | patient_A19 | success |
| 19 | patient_A20 | success |
| 20 | patient_A21 | success |
| 21 | patient_A22 | success |
| 22 | patient_A23 | success |
| 23 | patient_A24 | success |
| 24 | patient_A25 | success |
| 25 | patient_A26 | success |
| 26 | patient_A27 | success |
| 27 | patient_A28 | success |
| 28 | patient_A29 | success |
| 29 | patient_A30 | success |
| 30 | patient_A31 | success |
| 31 | patient_A32 | success |
| 32 | patient_A33 | success |
| 33 | patient_A34 | success |
| 34 | patient_A35 | success |
| 35 | patient_A36 | success |
| 36 | patient_A37 | success |
| 37 | patient_A38 | success |
| 38 | patient_A39 | success |
| 39 | patient_A40 | success |
| 40 | patient_A41 | success |
| 41 | patient_A42 | success |
| 42 | patient_A43 | success |
| 43 | patient_A44 | success |
| 44 | patient_A45 | success |
| 45 | patient_A46 | success |
| 46 | patient_A47 | success |
| 47 | patient_A48 | success |
| 48 | patient_A49 | success |
| 49 | patient_A51 | success |
| 50 | patient_A52 | success |
| 51 | patient_A53 | success |
| 52 | patient_A54 | success |
| 53 | patient_A55 | success |
| 54 | patient_A56 | success |
| 55 | patient_A57 | success |
| 56 | patient_A58 | success |
| 57 | patient_A59 | success |
| 58 | patient_A60 | success |
| 59 | patient_A61 | success |
| 60 | patient_A63 | success |
| 61 | patient_A64 | success |
| 62 | patient_A65 | success |
| 63 | patient_A66 | success |
| 64 | patient_A67 | success |
| 65 | patient_A68 | success |
| 66 | patient_A69 | success |
| 67 | patient_A70 | success |
| 68 | patient_A71 | success |
| 69 | patient_A72 | success |
| 70 | patient_A73 | success |
| 71 | patient_A74 | success |
| 72 | patient_A75 | success |
| 73 | patient_A76 | success |
| 74 | patient_A77 | success |
| 75 | patient_A78 | success |
| 76 | patient_A79 | success |
| 77 | patient_A80 | success |
| 78 | patient_A81 | success |
| 79 | patient_A82 | success |
| 80 | patient_A83 | success |
| 81 | patient_A84 | success |
| 82 | patient_A85 | success |
| 83 | patient_A86 | success |
| 84 | patient_A87 | success |
| 85 | patient_A88 | success |
| 86 | patient_A89 | success |
| 87 | patient_A90 | success |
| 88 | patient_A91 | success |
| 89 | patient_A92 | success |
| 90 | patient_A93 | success |
| 91 | patient_A94 | success |
| 92 | patient_A95 | success |
| 93 | patient_A96 | success |
| 94 | patient_A97 | success |
| 95 | patient_A98 | success |
| 96 | patient_A99 | success |
| 97 | patient_A100 | success |

#### Failed (n=3)

| # | study_id | status |
| --- | --- | --- |
| 1 | patient_A01 | failed |
| 2 | patient_A50 | failed |
| 3 | patient_A62 | failed |
