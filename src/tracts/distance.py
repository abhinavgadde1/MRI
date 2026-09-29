"""Distance metrics between lesion masks and white-matter tract masks."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree


def _foreground_points_mm(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    """Map foreground voxel indices to physical coordinates (mm) via spacing."""
    ijk = np.column_stack(np.nonzero(mask)).astype(float)
    if ijk.size == 0:
        raise ValueError("Mask has no foreground voxels")
    return ijk * np.asarray(spacing_mm, dtype=float)


def minimum_distance_mm(
    lesion_mask: str | Path | np.ndarray,
    tract_mask: str | Path | np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] | None = None,
    lesion_label: int = 1,
    tract_label: int = 1,
) -> float:
    """Minimum Euclidean distance (mm) from lesion mask to tract mask.

    Coordinates are ``ijk * spacing`` from the NIfTI header (or the provided
    ``spacing_mm`` when arrays are passed). Returns ``0.0`` on overlap.
    """
    if isinstance(lesion_mask, (str, Path)):
        lesion_img = nib.load(str(lesion_mask))
        lesion = np.asanyarray(lesion_img.dataobj) == lesion_label
        zooms = lesion_img.header.get_zooms()[:3]
        lesion_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    else:
        if spacing_mm is None:
            raise ValueError("spacing_mm is required when lesion_mask is an array")
        lesion = np.asarray(lesion_mask) == lesion_label
        lesion_spacing = spacing_mm

    if isinstance(tract_mask, (str, Path)):
        tract_img = nib.load(str(tract_mask))
        tract = np.asanyarray(tract_img.dataobj) == tract_label
        zooms = tract_img.header.get_zooms()[:3]
        tract_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    else:
        if spacing_mm is None:
            raise ValueError("spacing_mm is required when tract_mask is an array")
        tract = np.asarray(tract_mask) == tract_label
        tract_spacing = spacing_mm

    if not np.any(lesion):
        raise ValueError(f"No lesion label {lesion_label}")
    if not np.any(tract):
        raise ValueError(f"No tract label {tract_label}")

    if lesion.shape == tract.shape and np.any(lesion & tract):
        return 0.0

    lesion_pts = _foreground_points_mm(lesion, lesion_spacing)
    tract_pts = _foreground_points_mm(tract, tract_spacing)
    tree = cKDTree(tract_pts)
    distances, _ = tree.query(lesion_pts, k=1)
    return float(np.min(distances))


# Alias matching earlier surface-distance naming.
minimum_surface_distance_mm = minimum_distance_mm
