"""Binary mask post-processing: connectivity cleanup for multi-label predictions."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import ndimage as ndi

PostprocessMode = Literal["raw", "lcc", "lcc_min"]
POSTPROCESS_MODES: tuple[PostprocessMode, ...] = ("raw", "lcc", "lcc_min")

# 26-connectivity in 3D (face + edge + corner neighbors).
CONN_26 = np.ones((3, 3, 3), dtype=np.int8)

# Defaults for ``lcc_min``: keep CC if volume >= max(5% of largest, 1 mL).
DEFAULT_LCC_MIN_FRAC = 0.05
DEFAULT_LCC_MIN_ML = 1.0


def keep_largest_cc(binary: np.ndarray, *, connectivity: int = 26) -> np.ndarray:
    """Keep only the largest connected component of a binary mask."""
    mask = np.asarray(binary, dtype=bool)
    if not np.any(mask):
        return mask.copy()
    structure = CONN_26 if connectivity == 26 else None
    labeled, n_labels = ndi.label(mask, structure=structure)
    if n_labels <= 1:
        return mask.copy()
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    largest = int(np.argmax(counts))
    return labeled == largest


def keep_components_min_size(
    binary: np.ndarray,
    *,
    min_frac_of_largest: float = DEFAULT_LCC_MIN_FRAC,
    min_volume_ml: float = DEFAULT_LCC_MIN_ML,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    connectivity: int = 26,
) -> np.ndarray:
    """Keep the largest CC plus any other CC above a volume threshold.

    A non-largest component is retained when its volume is at least
    ``max(min_frac_of_largest * vol_largest, min_volume_ml)``.
    """
    mask = np.asarray(binary, dtype=bool)
    if not np.any(mask):
        return mask.copy()
    structure = CONN_26 if connectivity == 26 else None
    labeled, n_labels = ndi.label(mask, structure=structure)
    if n_labels <= 1:
        return mask.copy()

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    largest_vox = int(counts[largest_label])
    voxel_ml = float(np.prod(spacing_mm) / 1000.0)
    largest_ml = largest_vox * voxel_ml
    threshold_ml = max(float(min_frac_of_largest) * largest_ml, float(min_volume_ml))
    threshold_vox = threshold_ml / voxel_ml if voxel_ml > 0 else float("inf")

    keep = np.zeros_like(mask)
    for lab in range(1, n_labels + 1):
        size = int(counts[lab])
        if lab == largest_label or size >= threshold_vox:
            keep |= labeled == lab
    return keep


def fill_holes_binary(binary: np.ndarray) -> np.ndarray:
    """Fill interior holes in a 3D binary mask."""
    mask = np.asarray(binary, dtype=bool)
    if not np.any(mask):
        return mask.copy()
    return ndi.binary_fill_holes(mask)


def lcc_then_fill(binary: np.ndarray, *, connectivity: int = 26) -> np.ndarray:
    """Largest connected component, then fill holes."""
    return fill_holes_binary(keep_largest_cc(binary, connectivity=connectivity))


def lcc_min_then_fill(
    binary: np.ndarray,
    *,
    min_frac_of_largest: float = DEFAULT_LCC_MIN_FRAC,
    min_volume_ml: float = DEFAULT_LCC_MIN_ML,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    connectivity: int = 26,
) -> np.ndarray:
    """Largest + sizable secondary CCs, then fill holes."""
    kept = keep_components_min_size(
        binary,
        min_frac_of_largest=min_frac_of_largest,
        min_volume_ml=min_volume_ml,
        spacing_mm=spacing_mm,
        connectivity=connectivity,
    )
    return fill_holes_binary(kept)


def apply_binary_postprocess(
    binary: np.ndarray,
    mode: PostprocessMode = "raw",
    *,
    connectivity: int = 26,
    min_frac_of_largest: float = DEFAULT_LCC_MIN_FRAC,
    min_volume_ml: float = DEFAULT_LCC_MIN_ML,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Apply ``raw``, ``lcc``, or ``lcc_min`` post-processing."""
    mask = np.asarray(binary, dtype=bool)
    if mode == "raw":
        return mask.copy()
    if mode == "lcc":
        return lcc_then_fill(mask, connectivity=connectivity)
    if mode == "lcc_min":
        return lcc_min_then_fill(
            mask,
            min_frac_of_largest=min_frac_of_largest,
            min_volume_ml=min_volume_ml,
            spacing_mm=spacing_mm,
            connectivity=connectivity,
        )
    raise ValueError(f"Unknown postprocess mode {mode!r}; expected raw|lcc|lcc_min")


def apply_channel_postprocess(
    pred: np.ndarray,
    mode: PostprocessMode = "raw",
    *,
    connectivity: int = 26,
    min_frac_of_largest: float = DEFAULT_LCC_MIN_FRAC,
    min_volume_ml: float = DEFAULT_LCC_MIN_ML,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Post-process each channel of a (C, *spatial) or (1, C, *spatial) mask."""
    arr = np.asarray(pred)
    squeeze_batch = False
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
        squeeze_batch = True
    if arr.ndim != 4:
        raise ValueError(f"Expected (C, H, W, D) prediction, got shape {arr.shape}")

    out = np.empty_like(arr, dtype=arr.dtype)
    for c in range(arr.shape[0]):
        cleaned = apply_binary_postprocess(
            arr[c] > 0.5 if arr.dtype != bool else arr[c],
            mode,
            connectivity=connectivity,
            min_frac_of_largest=min_frac_of_largest,
            min_volume_ml=min_volume_ml,
            spacing_mm=spacing_mm,
        )
        out[c] = cleaned.astype(arr.dtype)
    if squeeze_batch:
        return out[np.newaxis, ...]
    return out


def component_distances_to_largest_mm(
    labeled: np.ndarray,
    largest_label: int,
    spacing_mm: tuple[float, float, float],
) -> list[float]:
    """Min Euclidean distance (mm) from each smaller CC to the largest CC."""
    lcc = labeled == largest_label
    if not np.any(lcc):
        return []
    # Distance from every voxel to the nearest LCC voxel.
    dt = ndi.distance_transform_edt(~lcc, sampling=spacing_mm)
    distances: list[float] = []
    for lab in range(1, int(labeled.max()) + 1):
        if lab == largest_label:
            continue
        sel = labeled == lab
        if not np.any(sel):
            continue
        distances.append(float(dt[sel].min()))
    return distances


def fragmentation_stats(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    connectivity: int = 26,
) -> dict[str, Any]:
    """Connected-component fragmentation summary for a binary WT mask."""
    mask = np.asarray(binary, dtype=bool)
    voxel_ml = float(np.prod(spacing_mm) / 1000.0)
    total_vox = int(np.count_nonzero(mask))
    if total_vox == 0:
        return {
            "n_components": 0,
            "frac_largest": float("nan"),
            "frac_outside_largest": float("nan"),
            "vol_largest_ml": 0.0,
            "vol_total_ml": 0.0,
            "n_small_components": 0,
            "max_dist_to_lcc_mm": float("nan"),
            "mean_dist_to_lcc_mm": float("nan"),
            "distances_to_lcc_mm": "",
        }

    structure = CONN_26 if connectivity == 26 else None
    labeled, n_labels = ndi.label(mask, structure=structure)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    largest_vox = int(counts[largest_label])
    distances = component_distances_to_largest_mm(labeled, largest_label, spacing_mm)
    frac_largest = float(largest_vox / total_vox)

    return {
        "n_components": int(n_labels),
        "frac_largest": frac_largest,
        "frac_outside_largest": float(1.0 - frac_largest),
        "vol_largest_ml": float(largest_vox * voxel_ml),
        "vol_total_ml": float(total_vox * voxel_ml),
        "n_small_components": int(n_labels - 1),
        "max_dist_to_lcc_mm": float(np.max(distances)) if distances else 0.0,
        "mean_dist_to_lcc_mm": float(np.mean(distances)) if distances else 0.0,
        "distances_to_lcc_mm": ";".join(f"{d:.2f}" for d in distances),
    }
