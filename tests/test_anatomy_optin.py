"""Opt-in SynthSeg anatomy overlay: isolated venv, never fail the job."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from pipeline import anatomy_optin


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
