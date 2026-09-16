"""Tests for clinical ellipsoid volume (A × B × C / 2)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from validation.ellipsoid import (
    clinical_ellipsoid_from_diameters,
    clinical_ellipsoid_volume_cm3,
    compare_to_ellipsoid,
    diameters_from_mask_principal_axes,
    fit_ellipsoid,
    principal_axis_bounding_diameters,
)


def test_clinical_formula_abc_over_2():
    # A,B,C in mm → V = ABC/2 / 1000 cm³
    assert clinical_ellipsoid_volume_cm3((40.0, 30.0, 20.0)) == pytest.approx(
        40 * 30 * 20 / 2 / 1000
    )


def test_manual_diameters_sorted():
    ell = clinical_ellipsoid_from_diameters((20.0, 40.0, 30.0))
    assert ell.diameters_mm == (40.0, 30.0, 20.0)
    assert ell.source == "manual"
    assert ell.volume_cm3 == pytest.approx(12.0)


def test_principal_axis_extents_of_box():
    # Axis-aligned box 10×6×4 mm (voxel centers on a grid)
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 10, 11),
        np.linspace(0, 6, 7),
        np.linspace(0, 4, 5),
        indexing="ij",
    )
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    diameters, center, _ = principal_axis_bounding_diameters(pts)
    assert diameters[0] == pytest.approx(10.0, abs=0.01)
    assert diameters[1] == pytest.approx(6.0, abs=0.01)
    assert diameters[2] == pytest.approx(4.0, abs=0.01)
    assert center[0] == pytest.approx(5.0, abs=0.01)


def test_fit_from_mask_nifti(tmp_path: Path):
    data = np.zeros((30, 20, 16), dtype=np.uint8)
    # Cuboid 20×10×8 voxels at 1 mm → diameters ~19, 9, 7 (voxel-index extents)
    data[5:25, 5:15, 4:12] = 1
    path = tmp_path / "mask.nii.gz"
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(img, str(path))

    ell = fit_ellipsoid(path, label=1)
    assert ell.source == "principal_axes"
    assert ell.diameters_mm[0] >= ell.diameters_mm[1] >= ell.diameters_mm[2]
    assert ell.volume_cm3 == pytest.approx(
        clinical_ellipsoid_volume_cm3(ell.diameters_mm)
    )

    comparison = compare_to_ellipsoid(path, label=1)
    assert comparison.ellipsoid_volume_ml == pytest.approx(ell.volume_ml)
    assert comparison.mask_volume_ml > 0


def test_fit_manual_overrides_mask():
    ell = fit_ellipsoid(diameters_mm=(50.0, 40.0, 30.0))
    assert ell.source == "manual"
    assert ell.volume_cm3 == pytest.approx(30.0)


def test_rejects_nonpositive_diameters():
    with pytest.raises(ValueError):
        clinical_ellipsoid_volume_cm3((10.0, 0.0, 5.0))
