"""Tumor / region volume computation from masks and meshes."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh


def compute_mask_volume_ml(
    mask_path: str | Path,
    *,
    label: int | None = 1,
) -> float:
    """Compute volume in milliliters from a labeled NIfTI mask.

    ``label=None`` counts any non-zero voxel (whole-tumor style).
    """
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    voxel_ml = float(np.prod(spacing) / 1000.0)
    if label is None:
        return float(np.count_nonzero(data) * voxel_ml)
    return float(np.count_nonzero(data == label) * voxel_ml)


def compute_mesh_volume_ml(mesh_path: str | Path) -> float:
    """Compute watertight mesh volume in milliliters via trimesh."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {mesh_path}")
    # trimesh volumes are in cubed vertex units (mm^3 if vertices are mm).
    return float(abs(mesh.volume) / 1000.0)
