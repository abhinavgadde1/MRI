"""Distance metrics between tumor surfaces and tract bundles."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree


def _mask_surface_points(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Approximate surface voxels as on-mask voxels with an off-mask neighbor."""
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    surface = mask & ~eroded
    ijk = np.column_stack(np.nonzero(surface))
    if ijk.size == 0:
        ijk = np.column_stack(np.nonzero(mask))
    homo = np.hstack([ijk.astype(float), np.ones((ijk.shape[0], 1))])
    return (affine @ homo.T).T[:, :3]


def minimum_surface_distance_mm(
    tumor_mask_path: str | Path,
    tract_mask_path: str | Path,
    *,
    tumor_label: int = 1,
    tract_label: int = 1,
) -> float:
    """Minimum Euclidean distance (mm) between tumor and tract surfaces.

    Returns ``0.0`` when masks overlap.
    """
    tumor_img = nib.load(str(tumor_mask_path))
    tract_img = nib.load(str(tract_mask_path))
    tumor = np.asanyarray(tumor_img.dataobj) == tumor_label
    tract = np.asanyarray(tract_img.dataobj) == tract_label

    if not np.any(tumor):
        raise ValueError(f"No tumor label {tumor_label} in {tumor_mask_path}")
    if not np.any(tract):
        raise ValueError(f"No tract label {tract_label} in {tract_mask_path}")

    # Overlap in a common voxel grid when shapes match; otherwise use world points only.
    if tumor.shape == tract.shape and np.any(tumor & tract):
        return 0.0

    tumor_pts = _mask_surface_points(tumor, tumor_img.affine)
    tract_pts = _mask_surface_points(tract, tract_img.affine)
    tree = cKDTree(tract_pts)
    distances, _ = tree.query(tumor_pts, k=1)
    return float(np.min(distances))
