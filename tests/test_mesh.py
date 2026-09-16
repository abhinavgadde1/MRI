"""Tests for marching-cubes mesh extraction with physical spacing."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from reconstruction.mesh import mask_to_mesh


def _write_mask(path: Path, data: np.ndarray, zooms: tuple[float, float, float]) -> Path:
    affine = np.diag([*zooms, 1.0])
    img = nib.Nifti1Image(data.astype(np.uint8), affine)
    img.header.set_zooms(zooms)
    nib.save(img, str(path))
    return path


def test_mask_to_mesh_uses_spacing_and_writes_obj(tmp_path: Path):
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[5:15, 5:15, 5:15] = 1
    zooms = (1.5, 1.0, 2.0)
    mask_path = _write_mask(tmp_path / "mask.nii.gz", data, zooms)
    out = mask_to_mesh(mask_path, tmp_path / "tumor.obj", label=1)

    assert out.is_file()
    mesh = trimesh.load(str(out), force="mesh")
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    # Bounding box extent should reflect anisotropic spacing (cube 10 voxels).
    extents = mesh.bounding_box.extents
    assert extents[0] == pytest.approx(10 * 1.5, rel=0.15)
    assert extents[1] == pytest.approx(10 * 1.0, rel=0.15)
    assert extents[2] == pytest.approx(10 * 2.0, rel=0.15)


def test_mask_to_mesh_writes_stl(tmp_path: Path):
    data = np.zeros((12, 12, 12), dtype=np.uint8)
    data[3:9, 3:9, 3:9] = 2
    mask_path = _write_mask(tmp_path / "mask.nii.gz", data, (1.0, 1.0, 1.0))
    out = mask_to_mesh(mask_path, tmp_path / "tumor.stl", label=2)
    assert out.suffix == ".stl"
    assert out.is_file()


def test_rejects_unsupported_format(tmp_path: Path):
    data = np.ones((8, 8, 8), dtype=np.uint8)
    mask_path = _write_mask(tmp_path / "mask.nii.gz", data, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="Unsupported mesh format"):
        mask_to_mesh(mask_path, tmp_path / "tumor.ply", label=1)


def test_empty_label_raises(tmp_path: Path):
    data = np.zeros((8, 8, 8), dtype=np.uint8)
    data[2:4, 2:4, 2:4] = 1
    mask_path = _write_mask(tmp_path / "mask.nii.gz", data, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="No foreground"):
        mask_to_mesh(mask_path, tmp_path / "tumor.obj", label=9)
