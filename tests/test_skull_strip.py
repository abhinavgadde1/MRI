"""Tests for skull-strip fallback behaviour."""

import logging
from pathlib import Path

import SimpleITK as sitk

from preprocessing.skull_strip import hd_bet_available, skull_strip


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
    assert any("SimpleITK Otsu" in r.message for r in caplog.records)


def test_hd_bet_availability_is_bool():
    assert isinstance(hd_bet_available(), bool)
