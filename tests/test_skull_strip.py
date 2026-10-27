"""Tests for skull-strip fallback behaviour."""

import json
import logging
from pathlib import Path

import SimpleITK as sitk

from preprocessing.skull_strip import (
    hd_bet_available,
    read_strip_method,
    skull_strip,
    write_study_strip_summary,
)


def _write_blob(path: Path) -> None:
    img = sitk.Image(24, 24, 16, sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0, 2.0))
    arr = sitk.GetArrayFromImage(img)
    arr[4:12, 6:18, 6:18] = 200.0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path))


def test_simpleitk_fallback_when_hd_bet_disabled(tmp_path: Path, caplog):
    src = tmp_path / "brain.nii.gz"
    dst = tmp_path / "brain_stripped.nii.gz"
    _write_blob(src)

    with caplog.at_level(logging.INFO):
        result = skull_strip(src, dst, prefer_hd_bet=False)

    assert result.method == "simpleitk_otsu"
    assert result.stripped_path.is_file()
    assert result.mask_path.is_file()
    assert "prefer_hd_bet=False" in result.reason
    assert any("SimpleITK Otsu" in r.message for r in caplog.records)

    sidecar = tmp_path / "brain_stripped_skullstrip_method.json"
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["method"] == "simpleitk_otsu"
    assert "prefer_hd_bet=False" in payload["reason"]
    assert payload["prefer_hd_bet"] is False
    assert read_strip_method(result.mask_path) == "simpleitk_otsu"
    assert read_strip_method(tmp_path) == "simpleitk_otsu"


def test_study_strip_summary_json(tmp_path: Path):
    path = write_study_strip_summary(
        tmp_path,
        study_name="studyA",
        methods={"T1": "hd-bet", "T2": "simpleitk_otsu"},
        reasons={
            "T1": "HD-BET inference succeeded (backend=cli, device=mps, mode=fast)",
            "T2": "HD-BET failed (RuntimeError: boom); falling back to SimpleITK Otsu",
        },
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["study_name"] == "studyA"
    assert data["methods"]["T1"] == "hd-bet"
    assert "falling back" in data["reasons"]["T2"]


def test_hd_bet_availability_is_bool():
    assert isinstance(hd_bet_available(), bool)
