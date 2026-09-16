"""Tests for rigid registration and isotropic resampling."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from preprocessing.registration import (
    register_modalities_to_reference,
    resample_isotropic,
    rigid_register,
)


def _blob(path: Path, *, spacing=(1.5, 1.5, 3.0), shift=(0.0, 0.0, 0.0)) -> Path:
    size = (32, 32, 20)
    img = sitk.Image(*size, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    img.SetOrigin(shift)
    arr = sitk.GetArrayFromImage(img)
    arr[5:15, 8:24, 8:24] = 150.0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(path))
    return path


def test_resample_isotropic_spacing(tmp_path: Path):
    src = _blob(tmp_path / "t1.nii.gz", spacing=(1.5, 1.5, 3.0))
    dst = tmp_path / "t1_1mm.nii.gz"
    resample_isotropic(src, dst, spacing_mm=1.0)
    out = sitk.ReadImage(str(dst))
    assert np.allclose(out.GetSpacing(), (1.0, 1.0, 1.0))


def test_rigid_register_and_modality_bundle(tmp_path: Path):
    t1 = _blob(tmp_path / "T1.nii.gz")
    t2 = _blob(tmp_path / "T2.nii.gz", shift=(2.0, 1.0, 0.0))
    flair = _blob(tmp_path / "FLAIR.nii.gz", shift=(-1.0, 0.5, 1.0))

    out = tmp_path / "reg"
    results = register_modalities_to_reference(
        {"T1": t1, "T2": t2, "FLAIR": flair},
        out,
        spacing_mm=1.0,
    )
    assert "T1" in results and "T2" in results and "FLAIR" in results
    for path in results.values():
        img = sitk.ReadImage(str(path))
        assert np.allclose(img.GetSpacing(), (1.0, 1.0, 1.0))
        assert path.is_file()

    # Direct rigid API still works
    registered, _tfm = rigid_register(t2, t1, tmp_path / "t2_on_t1.nii.gz")
    assert registered.is_file()
