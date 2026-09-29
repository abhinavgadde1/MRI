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
