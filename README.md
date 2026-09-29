# MRI tumor segmentation & ellipsoid-volume pipeline

End-to-end BraTS / clinical-MRI pipeline: DICOM or NIfTI ingest → multilabel SegResNet
segmentation (primary model **v3**) → LCC post-process → 3D reconstruction →
ellipsoid validation → HTML report. Paper tables and qualitative real-patient demos
are regenerated from frozen held-out metrics under `results/`.

## Setup

```bash
./setup.sh                  # creates .venv, pip install -r requirements.txt, editable install
source .venv/bin/activate
export PYTHONPATH=src       # Makefile sets this for its targets
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

**Device notes (Mac / MPS):** Torch uses MPS when available, else CUDA, else CPU.
Training and sliding-window inference on MPS work; if you hit operator gaps, pass
`--device cpu`. Prevent sleep during long trainings — see `docs/SLEEP_PREVENTION.md`.

Place v3 weights at `checkpoints/brats_scale_full/best_model.pt` (~54 MB; gitignored).
`make demo` runs without weights (random init) for a synthetic smoke path.

## Reproduce paper tables 1–5

Tables and figures are assembled (without retraining) by:

```bash
make paper
# equivalent: PYTHONPATH=src python3 scripts/make_paper_assets.py
```

That script reads frozen CSVs under `results/` and `checkpoints/brats_eval_heldout*/`
and writes `results/paper/tables/table{1–5}_*.csv|.md` plus figures.

| Table | What | Upstream commands that produced the source metrics |
|-------|------|-----------------------------------------------------|
| **1** | Dataset / preprocessing counts | `scripts/make_paper_assets.py` ← `data/processed/batch_summary_patients.csv`, BraTS metadata |
| **2** | Segmentation Dice v1/v2/v3 (raw + LCC) | Held-out eval → `results/segmentation_table_v3.csv`; three-way: `PYTHONPATH=src python3 -m segmentation.compare_v3 --skip-inference-extras`; ablation v1↔v2: `PYTHONPATH=src python3 experiments/ablation_v1_v2/compare_v1_v2.py --skip-inference` |
| **3** | Ellipsoid GT radiologist + v3 pipeline agreement | `PYTHONPATH=src python3 -m validation.ellipsoid_batch` (outputs under `results/ellipsoid/…`); paper copies `results/ellipsoid/brats_scale_full/pipeline_level.csv` |
| **4** | LCC failure rates | Same held-out LCC CSVs + `make_paper_assets` Table 4 |
| **5** | Seg vs formula error (v3-lcc WT) | `results/ellipsoid/brats_scale_full/lcc/case_volumes.csv` via `make_paper_assets` |

Primary checkpoint: `checkpoints/brats_scale_full/best_model.pt`. Full paper bundle:
[`results/paper/`](results/paper/) (`MANIFEST.md`, `LIMITATIONS.md`, `environment.txt`).

## Inference on a new case

```bash
PYTHONPATH=src python3 -m pipeline.cli run \
  --input <dicom_or_nifti_dir> \
  --out outputs/<case>/
```

- Default checkpoint = v3 (`config.yaml` → `inference.checkpoint`, overridable with `--checkpoint`).
- Default post-process = `lcc` (`--postprocess raw|lcc|lcc_min`).
- Input may be a DICOM study directory or a folder of 4 registered 1 mm NIfTIs
  (or `…/04_registered_1mm/`).
- Writes `outputs/<case>/report.html` plus meshes / metrics.

Synthetic smoke (no BraTS / DICOM):

```bash
make demo
# → outputs/demo_synthetic/report.html
```

Real-patient qualitative demo (4 stand-in T1c cases + BraTS overlays):

```bash
make patients-demo
# → outputs/real_patients_demo/report.html
```

Open: [`outputs/real_patients_demo/report.html`](outputs/real_patients_demo/report.html)

## Layout (high level)

```
src/pipeline/          # CLI: ingest → segment → reconstruct → validate → report
src/segmentation/      # v3 train / eval / compare_v3 (main path)
src/preprocessing/ src/reconstruction/ src/validation/ …
experiments/ablation_v1_v2/   # historical v1→v2 / loss-bug scripts only
scripts/               # make_paper_assets.py, real_patients_demo.py
results/paper/         # submission tables & figures
checkpoints/           # local weights (*.pt gitignored)
```

## Known limitations

See [`LIMITATIONS.md`](LIMITATIONS.md) (points to the full paper list).

## Tests

```bash
make test
```
