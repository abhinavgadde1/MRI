"""Tests for batch2 corrected-label QC (synthetic NIfTIs)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from segmentation.batch2_label_qc import (
    check_all_studies,
    check_corrected_label,
    format_progress_table,
    write_progress_md,
)


def _save(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aff = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(data.astype(np.float32), aff), str(path))


def _make_package(
    root: Path,
    study_id: str,
    *,
    corrected: np.ndarray | None = None,
    draft: np.ndarray | None = None,
    write_corrected: bool = True,
    affine: np.ndarray | None = None,
    t1c_affine: np.ndarray | None = None,
    brain: np.ndarray | None = None,
) -> Path:
    study = root / study_id
    study.mkdir(parents=True, exist_ok=True)
    shape = (12, 12, 10)
    aff = np.eye(4) if affine is None else affine
    t1c_aff = aff if t1c_affine is None else t1c_affine
    _save(study / "T1c.nii.gz", np.zeros(shape, dtype=np.float32), t1c_aff)
    # Other modalities not required by QC, but keep packages realistic.
    for m in ("T1", "T2", "FLAIR"):
        _save(study / f"{m}.nii.gz", np.zeros(shape, dtype=np.float32), aff)

    if corrected is None:
        corrected = np.zeros(shape, dtype=np.float32)
        corrected[3:8, 3:8, 3:7] = 2
        corrected[4:7, 4:7, 4:6] = 1
        corrected[5:6, 5:6, 4:6] = 4
    if write_corrected:
        _save(study / "corrected_label.nii.gz", corrected, aff)
    if draft is None:
        draft = corrected.copy()
        draft[3, 3, 3] = 0  # differ slightly by default
    _save(study / "v3_draft_label.nii.gz", draft, aff)
    if brain is not None:
        _save(study / "T1c_brainmask.nii.gz", brain.astype(np.float32), aff)
    return study


def test_not_labeled(tmp_path: Path):
    _make_package(tmp_path, "s0", write_corrected=False)
    r = check_corrected_label(tmp_path / "s0")
    assert r.status == "not_labeled"


def test_valid_labeled(tmp_path: Path):
    study = _make_package(tmp_path, "ok")
    r = check_corrected_label(study)
    assert r.status == "labeled"
    assert r.wt_ml is not None and r.wt_ml > 0
    assert r.tc_ml is not None and r.et_ml is not None
    assert r.wt_n_components >= 1
    assert r.dice_vs_draft


def test_grid_mismatch(tmp_path: Path):
    bad_aff = np.eye(4)
    bad_aff[0, 0] = 2.0
    study = _make_package(tmp_path, "mis", t1c_affine=bad_aff)
    r = check_corrected_label(study)
    assert r.status == "invalid"
    assert any("mismatch" in x.lower() for x in r.reasons)


def test_disallowed_label_values(tmp_path: Path):
    shape = (12, 12, 10)
    lab = np.zeros(shape, dtype=np.float32)
    lab[4:6, 4:6, 4:6] = 3  # project enhancing — not allowed in corrected classic
    study = _make_package(tmp_path, "badval", corrected=lab)
    r = check_corrected_label(study)
    assert r.status == "invalid"
    assert any("disallowed" in x for x in r.reasons)


def test_empty_label(tmp_path: Path):
    lab = np.zeros((12, 12, 10), dtype=np.float32)
    study = _make_package(tmp_path, "empty", corrected=lab)
    r = check_corrected_label(study)
    assert r.status == "invalid"
    assert any("empty" in x.lower() for x in r.reasons)


def test_identical_to_draft_warns(tmp_path: Path):
    shape = (12, 12, 10)
    lab = np.zeros(shape, dtype=np.float32)
    lab[4:8, 4:8, 3:7] = 2
    lab[5:7, 5:7, 4:6] = 4
    study = _make_package(tmp_path, "same", corrected=lab, draft=lab.copy())
    r = check_corrected_label(study)
    assert r.status == "labeled"
    assert r.identical_to_draft is True
    assert any("identical" in w for w in r.warnings)


def test_border_and_tiny_island_flags(tmp_path: Path):
    shape = (12, 12, 10)
    lab = np.zeros(shape, dtype=np.float32)
    # Main tumor
    lab[4:8, 4:8, 3:7] = 2
    lab[5:7, 5:7, 4:6] = 4
    # Tiny disconnected island (1 voxel ≈ 0.001 mL at 1mm)
    lab[1, 1, 1] = 2
    # Touch border
    lab[0, 5, 5] = 2
    study = _make_package(tmp_path, "flags", corrected=lab)
    r = check_corrected_label(study)
    assert r.status == "labeled"
    joined = " ".join(r.flags).lower()
    assert "border" in joined
    assert "0.1" in joined or "connected component" in joined


def test_outside_brain_flag(tmp_path: Path):
    shape = (12, 12, 10)
    lab = np.zeros(shape, dtype=np.float32)
    lab[4:8, 4:8, 3:7] = 2
    lab[5:7, 5:7, 4:6] = 4
    brain = np.zeros(shape, dtype=np.float32)
    brain[5:7, 5:7, 4:6] = 1  # brain only covers ET core — WT spills outside
    study = _make_package(tmp_path, "oob", corrected=lab, brain=brain)
    r = check_corrected_label(study)
    assert any("outside brain" in f.lower() for f in r.flags)


def test_progress_table_and_write(tmp_path: Path):
    _make_package(tmp_path, "a")
    _make_package(tmp_path, "b", write_corrected=False)
    results = check_all_studies(tmp_path)
    assert len(results) == 2
    md = format_progress_table(results)
    assert "labeled" in md
    assert "not_labeled" in md
    out = write_progress_md(results, tmp_path / "LABELING_PROGRESS.md")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "Reference masks" in text
    # Re-run is safe / idempotent content-wise
    write_progress_md(results, out)
    assert out.read_text(encoding="utf-8") == text
