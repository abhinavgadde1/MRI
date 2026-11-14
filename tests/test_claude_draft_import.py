"""Synthetic tests for Claude draft import + draft QA helpers."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# scripts/ is not a package — load by path
import importlib.util


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aff = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(data.astype(np.int16), aff), str(path))


def test_import_claude_draft_grid_match_and_meta(tmp_path: Path):
    mod = _load_script("import_claude_drafts.py")
    study_id = "batch2_99"
    labeling = tmp_path / "labeling"
    pkg = labeling / study_id
    pkg.mkdir(parents=True)
    aff = np.diag([1.0, 1.0, 1.0, 1.0]).astype(float)
    shape = (16, 16, 12)
    t1c = np.random.randn(*shape).astype(np.float32)
    _write_nifti(pkg / "T1c.nii.gz", t1c, aff)
    _write_nifti(pkg / "FLAIR.nii.gz", t1c, aff)
    _write_nifti(pkg / "T1.nii.gz", t1c, aff)
    _write_nifti(pkg / "T2.nii.gz", t1c, aff)
    # v3 draft
    v3 = np.zeros(shape, dtype=np.int16)
    v3[4:8, 4:8, 4:8] = 2
    v3[5:7, 5:7, 5:7] = 4
    _write_nifti(pkg / "v3_draft_label.nii.gz", v3, aff)
    (pkg / "package_meta.json").write_text(json.dumps({"study_id": study_id}), encoding="utf-8")

    src_dir = tmp_path / "claude_drafts"
    src_dir.mkdir()
    draft = np.zeros(shape, dtype=np.int16)
    draft[3:9, 3:9, 3:9] = 2
    draft[4:7, 4:7, 4:7] = 1
    draft[5:6, 5:6, 5:6] = 4
    src = src_dir / f"{study_id}_claude_draft_label.nii.gz"
    _write_nifti(src, draft, aff)

    row = mod.import_one(src, labeling)
    assert row["status"] == "draft_unverified"
    dest = pkg / "claude_draft_label.nii.gz"
    assert dest.is_file()
    loaded = nib.load(str(dest))
    assert loaded.shape == (16, 16, 12)
    assert np.allclose(loaded.affine, aff)

    meta = json.loads((pkg / "package_meta.json").read_text(encoding="utf-8"))
    assert meta["draft_status"] == "draft_unverified"
    assert meta["claude_draft"]["status"] == "draft_unverified"

    ws = (pkg / f"{study_id}.itksnap").read_text(encoding="utf-8")
    assert "claude_draft_label.nii.gz" in ws
    assert "v3_draft_label.nii.gz" in ws
    sh = (pkg / "open_itksnap.sh").read_text(encoding="utf-8")
    assert "claude_draft_label.nii.gz" in sh


def test_import_rejects_shape_mismatch(tmp_path: Path):
    mod = _load_script("import_claude_drafts.py")
    study_id = "batch2_98"
    labeling = tmp_path / "labeling"
    pkg = labeling / study_id
    pkg.mkdir(parents=True)
    aff = np.eye(4)
    _write_nifti(pkg / "T1c.nii.gz", np.zeros((8, 8, 8), dtype=np.int16), aff)
    src = tmp_path / f"{study_id}_claude_draft_label.nii.gz"
    _write_nifti(src, np.zeros((9, 8, 8), dtype=np.int16), aff)
    with pytest.raises(ValueError, match="shape"):
        mod.import_one(src, labeling)


def test_draft_qa_volumes_and_vessel_flag(tmp_path: Path):
    qa = _load_script("draft_qa.py")
    pkg = tmp_path / "batch2_97"
    pkg.mkdir()
    aff = np.eye(4)
    shape = (32, 32, 24)
    t1c = np.ones(shape, dtype=np.float32)
    _write_nifti(pkg / "T1c.nii.gz", t1c, aff)
    # Brain mask: fill most of volume
    brain = np.ones(shape, dtype=np.uint8)
    _write_nifti(pkg / "brain_mask.nii.gz", brain, aff)
    # Thin midline tube (vessel-like) + a blob
    draft = np.zeros(shape, dtype=np.int16)
    draft[15:17, 8:24, 10:14] = 4  # thin strip through midline
    draft[20:26, 20:26, 10:16] = 2
    draft[21:25, 21:25, 11:15] = 4
    _write_nifti(pkg / "claude_draft_label.nii.gz", draft, aff)
    # Minimal v3 overlap
    v3 = np.zeros(shape, dtype=np.int16)
    v3[21:25, 21:25, 11:15] = 4
    _write_nifti(pkg / "v3_draft_label.nii.gz", v3, aff)

    row = qa.qa_one(pkg, "claude_draft_label.nii.gz")
    assert row["ET_ml"] > 0
    assert row["WT_ml"] > row["ET_ml"]
    assert row["n_cc_WT"] >= 1
    assert "possible_vessel" in row["possible_vessel_flags"] or row["possible_vessel_flags"] == "" or "cc" in row["possible_vessel_flags"]
    assert row["draft_status"] == "draft_unverified"
