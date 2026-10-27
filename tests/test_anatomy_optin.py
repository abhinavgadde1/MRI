"""Opt-in SynthSeg anatomy overlay: isolated venv, never fail the job."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go

from pipeline import anatomy_optin
from visualization.plotly_3d import MESH_COLORS


def _tiny_out(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    vol = np.zeros((8, 8, 8), dtype=np.uint8)
    vol[2:5, 2:5, 2:5] = 2
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(out / "pred_label_lcc.nii.gz"))
    (out / "mesh_3d.html").write_text("<html>tumor-only</html>", encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps({"case_id": "synth", "registered_dir": str(tmp_path / "missing")}),
        encoding="utf-8",
    )
    return out


def test_try_add_anatomy_missing_venv_is_unavailable(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(anatomy_optin, "SYNTHSEG_PYTHON", tmp_path / "no-such-python")
    out = _tiny_out(tmp_path)
    original = (out / "mesh_3d.html").read_text(encoding="utf-8")
    status, reason = anatomy_optin.try_add_anatomy_mesh(out)
    assert status == "unavailable"
    assert "venv-synthseg" in reason.lower() or "not installed" in reason.lower()
    assert (out / "mesh_3d.html").read_text(encoding="utf-8") == original


def test_try_add_anatomy_never_raises(tmp_path: Path, monkeypatch):
    def boom(_out):
        raise RuntimeError("synthseg exploded")

    monkeypatch.setattr(anatomy_optin, "_try_add_anatomy_mesh", boom)
    status, reason = anatomy_optin.try_add_anatomy_mesh(_tiny_out(tmp_path))
    assert status == "unavailable"
    assert "RuntimeError" in reason


def test_timeout_allows_native_resolution_patches():
    assert anatomy_optin.ANATOMY_TIMEOUT_S >= 180
    assert anatomy_optin.MIN_STRUCTURE_VOXELS >= 200


def test_anatomy_palette_distinct_from_tumor_and_each_other():
    tumor = {c.lower() for c in MESH_COLORS.values()}
    colors = [spec[2].lower() for spec in anatomy_optin.ANAT_SPEC]
    assert len(set(colors)) == len(colors)
    for color in colors:
        assert color not in tumor
    opacities = [spec[3] for spec in anatomy_optin.ANAT_SPEC]
    assert max(opacities) <= 0.30
    # Pink/purple pair from the previous palette should be gone.
    assert "#e89ab8" not in colors
    assert "#9b59b6" not in colors


def test_plan_anatomy_structures_drops_below_threshold():
    labels = np.zeros((20, 20, 20), dtype=np.int16)
    labels[0:10, 0:8, 0:8] = 3  # left cortex = 640 voxels, above 400
    labels[0, 10, 10] = 16  # brainstem sliver
    wt = np.zeros_like(labels, dtype=bool)
    rows = anatomy_optin.plan_anatomy_structures(labels, wt, min_voxels=400)
    by_key = {r["key"]: r for r in rows}
    assert by_key["left_cortex"]["rendered"] is True
    assert by_key["brainstem"]["rendered"] is False
    assert by_key["brainstem"]["reason"] == "below threshold, not rendered"
    assert by_key["brainstem"]["voxels"] == 1
    assert by_key["right_cortex"]["reason"] == "empty after WT mask, not rendered"


def test_laterality_masks_are_spatially_separated_not_a_label_bug():
    labels = np.zeros((30, 20, 20), dtype=np.int16)
    labels[2:8, 4:12, 4:12] = 3
    labels[20:28, 4:12, 4:12] = 42
    labels[1:7, 12:16, 4:10] = 2
    labels[21:27, 12:16, 4:10] = 41
    report = anatomy_optin.laterality_report(labels)
    assert report["cortex_voxel_overlap"] == 0
    assert report["wm_voxel_overlap"] == 0
    assert report["cortex_masks_spatially_separated"] is True
    assert report["wm_masks_spatially_separated"] is True
    assert report["cortex_left_centroid_x_lt_right"] is True


def test_tumor_traces_forced_opaque_and_last():
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=[0], y=[0], z=[0], name="Brain surface", opacity=0.28))
    fig.add_trace(go.Mesh3d(x=[0], y=[0], z=[0], name="Whole Tumor (WT)", opacity=0.5))
    fig.add_trace(go.Mesh3d(x=[0], y=[0], z=[0], name="Left cerebral cortex", opacity=0.14))
    fig.add_trace(go.Mesh3d(x=[0], y=[0], z=[0], name="Tumor Core (TC)", opacity=0.5))
    anatomy_optin._tumor_traces_on_top(fig)
    names = [tr.name for tr in fig.data]
    assert names[0] == "Brain surface"
    assert names[1] == "Left cerebral cortex"
    assert names[-2:] == ["Whole Tumor (WT)", "Tumor Core (TC)"]
    assert all(tr.opacity == 1.0 for tr in fig.data if tr.name in anatomy_optin.TUMOR_TRACE_NAMES)
