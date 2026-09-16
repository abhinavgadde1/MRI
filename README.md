# Medical Imaging Pipeline

Python package for an end-to-end MRI analysis workflow: DICOM ingestion and preprocessing, MONAI-based tumor segmentation, 3D surface reconstruction, quantitative validation, white-matter tract risk assessment, and interactive Plotly visualization.

## Project layout

```text
MRI/
├── src/
│   ├── preprocessing/     # DICOM→NIfTI, bias correction, skull strip, registration
│   ├── segmentation/      # MONAI training and inference
│   ├── reconstruction/    # Marching cubes meshes and volume metrics
│   ├── validation/        # Ellipsoid comparison and Bland–Altman analysis
│   ├── tracts/            # TractSeg, distances, risk classification
│   └── visualization/     # Plotly 3D rendering
├── data/
│   ├── brats/             # BraTS (or similar) public training data
│   ├── real_patients/     # Anonymized clinical DICOM / NIfTI
│   └── processed/         # Intermediate and final pipeline outputs
├── notebooks/             # Exploratory analysis and demos
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Pipeline stages

### 1. Preprocessing (`src/preprocessing`)

Convert raw DICOM series to NIfTI, then normalize geometry and intensity for downstream models:

1. **DICOM → NIfTI** — series discovery and conversion (`dicom2nifti` / `pydicom` + `nibabel`).
2. **Bias field correction** — N4 (or equivalent) intensity nonuniformity correction via SimpleITK.
3. **Skull stripping** — brain mask extraction to remove non-brain tissue.
4. **Registration** — align modalities / sessions to a common reference space.

Outputs land under `data/processed/`.

### 2. Segmentation (`src/segmentation`)

Train and run MONAI segmentation models (e.g. U-Net / SegResNet) on BraTS-style or clinical volumes:

- **Training** — dataset loaders, transforms, loss, metrics, checkpointing.
- **Inference** — load a trained checkpoint and produce tumor / region masks as NIfTI.

### 3. Reconstruction (`src/reconstruction`)

Turn binary or label masks into measurable 3D geometry:

- **Marching cubes** — isosurface mesh extraction (`scikit-image` → `trimesh`).
- **Volume computation** — voxel-count and mesh-based tumor volume estimates.

### 4. Validation (`src/validation`)

Compare automated measurements against reference geometry or clinical estimates:

- **Ellipsoid comparison** — fit / compare axis-aligned or oriented ellipsoids to segmented lesions.
- **Bland–Altman** — agreement plots and statistics between paired volume (or diameter) measurements.

### 5. Tract analysis (`src/tracts`)

Assess spatial relationship between lesions and white-matter pathways:

- **TractSeg integration** — run TractSeg (with DIPY support) to obtain tract bundles.
- **Distance computation** — minimum / mean distances from tumor surface to tracts of interest.
- **Risk classification** — map distances (and optional overlap) to clinical risk tiers.

### 6. Visualization (`src/visualization`)

Interactive Plotly 3D scenes combining MRI context, tumor meshes, and tract bundles for review and reporting.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"            # preferred: install from pyproject.toml
# or
pip install -r requirements.txt
pip install -e .
```

Place BraTS (or equivalent) data in `data/brats/` and anonymized patient studies in `data/real_patients/`. Processed intermediates go in `data/processed/`.

## Usage (package imports)

After an editable install, modules import by package name:

```python
from preprocessing import dicom_to_nifti, bias_correction, skull_stripping, registration
from segmentation import train, inference
from reconstruction import marching_cubes, volume
from validation import ellipsoid, bland_altman
from tracts import tractseg_integration, distance, risk
from visualization import plotly_3d
```

## Notebooks

Use `notebooks/` for exploratory work (conversion QC, segmentation overlays, Bland–Altman drafts, 3D demos). Keep reusable logic in `src/` rather than in notebook cells.

## Notes

- Keep PHI out of the repo; only anonymized data under `data/real_patients/`.
- Large binaries (NIfTI, DICOM, checkpoints) are gitignored; use `.gitkeep` placeholders for empty data dirs.
- TractSeg and MONAI may need extra system / GPU setup depending on your environment.
