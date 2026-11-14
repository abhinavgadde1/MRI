"""Regression: T1→T1c stand-in and API zip staging for real-patient layouts."""

from __future__ import annotations

import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from preprocessing.registration import REGISTERED_NAMES
from pipeline.run_case import T1C_STANDIN_NOTE, ensure_t1c_standin, t1c_is_standin


def _write_tiny_nifti(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))


def test_ensure_t1c_standin_copies_t1_when_t1c_missing(tmp_path: Path):
    reg = tmp_path / "04_registered_1mm"
    for key, fname in REGISTERED_NAMES.items():
        if key == "T1c":
            continue
        _write_tiny_nifti(reg / fname)
    assert not (reg / REGISTERED_NAMES["T1c"]).is_file()
    notes = ensure_t1c_standin(reg)
    assert (reg / REGISTERED_NAMES["T1c"]).is_file()
    assert any("stand-in" in n.lower() or "t1c" in n.lower() for n in notes)


def test_t1c_is_standin_when_files_are_byte_identical(tmp_path: Path):
    reg = tmp_path / "04_registered_1mm"
    for _key, fname in REGISTERED_NAMES.items():
        _write_tiny_nifti(reg / fname)
    # Overwrite T1c with T1 bytes (clinical fill already done).
    t1 = (reg / REGISTERED_NAMES["T1"]).read_bytes()
    (reg / REGISTERED_NAMES["T1c"]).write_bytes(t1)
    assert t1c_is_standin(reg)
    notes = ensure_t1c_standin(reg)
    assert any("stand-in" in n.lower() for n in notes)
    assert T1C_STANDIN_NOTE in notes


def test_t1c_is_not_standin_when_t1c_differs(tmp_path: Path):
    reg = tmp_path / "04_registered_1mm"
    for key, fname in REGISTERED_NAMES.items():
        _write_tiny_nifti(reg / fname)
        if key == "T1c":
            vol = np.ones((8, 8, 8), dtype=np.float32)
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(reg / fname))
    assert not t1c_is_standin(reg)
    notes = ensure_t1c_standin(reg)
    assert T1C_STANDIN_NOTE not in notes


def test_report_html_banners_t1c_standin(tmp_path: Path):
    import pandas as pd

    from pipeline.run_case import _write_report

    metrics = pd.DataFrame(
        [
            {"region": "ET", "volume_ml": 1.2, "sphericity": 0.5, "mesh_faces": 10, "trusted": False},
            {"region": "WT", "volume_ml": 12.0, "sphericity": 0.4, "mesh_faces": 20, "trusted": True},
        ]
    )
    path = _write_report(
        tmp_path,
        case_id="clinical_demo",
        checkpoint=Path("best_model.pt"),
        postprocess="lcc",
        metrics=metrics,
        ellipsoid={"n_wt_voxels": 1},
        notes=[T1C_STANDIN_NOTE],
        t1c_is_standin=True,
    )
    html = path.read_text(encoding="utf-8")
    assert "t1c-standin-banner" in html
    assert "ET hidden" in html or "stand-in" in html.lower()
    assert "untrusted" in html


def test_ensure_t1c_standin_raises_clear_error_if_t1_missing(tmp_path: Path):
    reg = tmp_path / "04_registered_1mm"
    _write_tiny_nifti(reg / REGISTERED_NAMES["FLAIR"])
    with pytest.raises(FileNotFoundError, match="missing required modalities"):
        ensure_t1c_standin(reg)


def test_api_find_pipeline_input_ignores_macosx_and_unwraps_study(tmp_path: Path):
    from api.app import _find_pipeline_input

    extract = tmp_path / "extracted"
    study = extract / "patient_A02"
    series = study / "1.2.3.series"
    series.mkdir(parents=True)
    (extract / "__MACOSX").mkdir()
    # Fake DICOM files (extension only — finder is path-based)
    for i in range(12):
        (series / f"slice_{i}.dcm").write_bytes(b"DICM")

    chosen = _find_pipeline_input(extract)
    assert chosen == study or chosen == series or study in chosen.parents or chosen == study
    assert "__MACOSX" not in chosen.parts


def test_api_error_unwraps_monai_transform_message():
    from api.app import _format_job_error

    root = FileNotFoundError(
        "No such file or no access: '/tmp/04_registered_1mm/T1c_registered_1mm.nii.gz'"
    )
    wrapped = RuntimeError(
        "applying transform <monai.transforms.io.dictionary.LoadImaged object at 0x11b>"
    )
    wrapped.__cause__ = root
    msg, tb = _format_job_error(wrapped)
    assert "T1c_registered_1mm" in msg
    assert "LoadImaged object at" not in msg or "underlying cause" in msg
    assert "FileNotFoundError" in msg
