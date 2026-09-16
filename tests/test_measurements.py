"""Tests for tumor morphometric measurements."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from reconstruction.measurements import (
    measure_from_mask,
    measure_from_mesh,
    measure_tumor,
    principal_axes_pca,
    sphericity_index,
    volume_cm3_from_mask,
)


def _sphere_mask(radius_vox: int = 8, size: int = 32) -> np.ndarray:
    zz, yy, xx = np.ogrid[:size, :size, :size]
    center = size // 2
    return ((xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2) <= radius_vox**2


def test_volume_cm3_from_mask_spacing():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True  # 27 voxels
    vol = volume_cm3_from_mask(mask, (2.0, 1.0, 1.0))
    assert vol == pytest.approx(27 * 2.0 / 1000.0)


def test_sphericity_of_sphere_near_one():
    # Analytic sphere: V = 4/3 π r³, A = 4 π r² → Ψ = 1
    r = 10.0
    v_cm3 = (4.0 / 3.0) * np.pi * r**3 / 1000.0
    a_mm2 = 4.0 * np.pi * r**2
    assert sphericity_index(v_cm3, a_mm2) == pytest.approx(1.0, rel=1e-6)


def test_pca_axis_order_and_anisotropy():
    # Elongated cloud along x
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [
            rng.normal(0, 10, 500),
            rng.normal(0, 2, 500),
            rng.normal(0, 1, 500),
        ]
    )
    lengths, center, components = principal_axes_pca(pts)
    assert lengths[0] >= lengths[1] >= lengths[2]
    assert abs(center[0]) < 1.0
    assert components.shape == (3, 3)


def test_measure_from_mask_sphere(tmp_path: Path):
    mask = _sphere_mask(radius_vox=8, size=40).astype(np.uint8)
    path = tmp_path / "sphere.nii.gz"
    spacing = (1.0, 1.0, 1.0)
    img = nib.Nifti1Image(mask, np.eye(4))
    img.header.set_zooms(spacing)
    nib.save(img, str(path))

    m = measure_from_mask(path, label=1)
    assert m.volume_cm3 > 0
    assert m.surface_area_mm2 > 0
    assert 0.7 < m.sphericity <= 1.05  # discrete sphere ≈ 1
    assert m.principal_axis_lengths_mm[0] >= m.principal_axis_lengths_mm[2]
    assert m.source == "mask"


def test_measure_from_mesh_and_combined(tmp_path: Path):
    # Unit-ish box mesh in mm
    mesh = trimesh.creation.box(extents=[10.0, 8.0, 6.0])
    mesh_path = tmp_path / "box.stl"
    mesh.export(str(mesh_path))

    from_mesh = measure_from_mesh(mesh_path)
    assert from_mesh.volume_cm3 == pytest.approx(10 * 8 * 6 / 1000.0, rel=0.05)
    assert from_mesh.surface_area_mm2 == pytest.approx(mesh.area, rel=1e-5)
    assert from_mesh.source == "mesh"

    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:15, 6:14, 7:13] = 1
    combined = measure_tumor(mask=mask, mesh=mesh, spacing_mm=(1.0, 1.0, 1.0), label=1)
    assert combined.source == "mask+mesh"
    assert combined.principal_axis_lengths_mm[0] > 0
