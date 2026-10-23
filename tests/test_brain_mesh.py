"""Brain-surface mesh cleaning: single CC, tumor containment, shared affine."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from visualization.plotly_3d import (
    binary_mask_to_trimesh,
    brain_mask_from_volume,
    brain_surface_mesh,
    clean_brain_mask,
    mesh_index_affine,
    tumor_outside_brain_fraction,
    validate_tumor_inside_brain,
)


def _noisy_brain_volume(shape=(48, 48, 40), seed: int = 0) -> np.ndarray:
    """Ellipsoid brain + scattered high-intensity speckles outside it."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.array(shape) / 2.0
    radii = np.array(shape) * 0.35
    brain = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0
    vol = np.zeros(shape, dtype=np.float32)
    vol[brain] = 1.0 + 0.1 * rng.random(int(brain.sum()))
    # Sprinkle islands
    for _ in range(80):
        z = int(rng.integers(0, shape[0]))
        y = int(rng.integers(0, shape[1]))
        x = int(rng.integers(0, shape[2]))
        if not brain[z, y, x]:
            vol[z, y, x] = 2.0
    return vol


def _edge_tumor_case(shape=(40, 40, 32)):
    """Brain ellipsoid + WT blob that touches the FOV z=0 face (protrusion case)."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.array(shape) / 2.0
    radii = np.array([shape[0] * 0.35, shape[1] * 0.35, shape[2] * 0.4])
    brain = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0
    vol = np.zeros(shape, dtype=np.float32)
    vol[brain] = 1.0
    # Tumor near inferior edge, touching z=0
    wt = np.zeros(shape, dtype=bool)
    wt[18:24, 18:26, 0:8] = True
    wt &= brain | (zz < 2)  # allow a few voxels at the FOV edge
    return vol, wt


def test_clean_brain_mask_single_component():
    vol = _noisy_brain_volume()
    thr = float(np.percentile(vol[vol > 0], 5))
    raw = vol > thr
    _, n_raw = ndi.label(raw)
    assert n_raw > 1

    cleaned = clean_brain_mask(raw)
    _, n_clean = ndi.label(cleaned)
    assert n_clean == 1
    assert cleaned.sum() > 0


def test_brain_mask_from_volume_is_single_component():
    vol = _noisy_brain_volume()
    mask = brain_mask_from_volume(vol)
    _, n = ndi.label(mask)
    assert n == 1


def test_brain_surface_mesh_builds_and_is_reasonable():
    """``target_faces`` is a contract, not a leftover production budget.

    The gyri/sulci brain shell defaults to 55_000 faces so folds survive.
    This tiny ellipsoid (~10k raw marching-cubes triangles) asks for 5_000
    to check that quadric decimation actually ran. A +50 slack covers
    backends that overshoot by a handful of faces; 10k+ means collapse
    was skipped (typically missing ``fast_simplification`` under trimesh 5).
    """
    vol = _noisy_brain_volume()
    requested = 5_000
    mesh = brain_surface_mesh(
        vol, spacing=(1.0, 1.0, 1.0), target_faces=requested
    )
    assert mesh is not None
    assert len(mesh.faces) > 0
    assert len(mesh.faces) <= requested + 50, (
        f"decimation did not honor target_faces={requested}; "
        f"got {len(mesh.faces)} (raw marching-cubes is ~10k on this fixture)"
    )
    parts = mesh.split(only_watertight=False)
    assert len(parts) == 1


def test_mesh_index_affine_shared():
    aff = mesh_index_affine((1.0, 1.0, 1.0))
    assert aff.shape == (4, 4)
    assert np.allclose(aff, np.eye(4))
    aff2 = mesh_index_affine((1.0, 1.0, 1.0))
    assert np.allclose(aff, aff2)


def test_tumor_union_keeps_edge_tumor_inside_brain_mask():
    vol, wt = _edge_tumor_case()
    mask_plain = brain_mask_from_volume(vol, include_masks=None, dilate_voxels=0)
    # Without union, FOV-edge tumor voxels can fall outside intensity brain mask
    # (zeros / low intensity at border). With union they must be inside.
    mask_union = brain_mask_from_volume(
        vol, include_masks={"whole_tumor": wt}, dilate_voxels=1
    )
    _, n = ndi.label(mask_union)
    assert n == 1
    frac = tumor_outside_brain_fraction(mask_union, wt)
    assert frac == 0.0
    # Sanity: validation helper agrees
    fracs = validate_tumor_inside_brain(mask_union, {"whole_tumor": wt})
    assert fracs["whole_tumor"] == 0.0


def test_brain_and_tumor_meshes_share_coordinate_frame():
    vol, wt = _edge_tumor_case()
    spacing = (1.0, 1.0, 1.0)
    brain = brain_surface_mesh(
        vol,
        spacing,
        include_masks={"whole_tumor": wt},
        peel_voxels=0,
        target_faces=8_000,
    )
    tumor = binary_mask_to_trimesh(wt, spacing)
    assert brain is not None and tumor is not None
    # Same origin/spacing frame: tumor bbox should sit inside (or on) brain bbox
    # with a small tolerance for surface sampling.
    bb, tb = brain.bounds, tumor.bounds
    assert np.all(tb[0] >= bb[0] - 1.5)
    assert np.all(tb[1] <= bb[1] + 1.5)
    assert len(brain.split(only_watertight=False)) == 1
