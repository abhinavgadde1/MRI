"""Hemisphere, lobe, structure overlap, and distance metrics for tumor masks."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy import ndimage as ndi

from location.atlases import cortical_label_to_lobe_map

# Coarse lobes used for nearest-lobe assignment / headlines.
NEAREST_LOBE_NAMES: tuple[str, ...] = (
    "frontal",
    "parietal",
    "temporal",
    "occipital",
    "insula",
)
DEEP_LABEL = "deep/central"
NEAREST_MAX_MM = 15.0


def _voxel_ml(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_mm) / 1000.0)


def hemisphere_fractions(
    mask: np.ndarray,
    affine: np.ndarray,
    *,
    bilateral_min_pct: float = 10.0,
) -> dict[str, Any]:
    """Left / right / bilateral from RAS x (affine); percent of tumor voxels."""
    mask_b = np.asarray(mask, dtype=bool)
    if not mask_b.any():
        return {
            "label": "none",
            "left_pct": 0.0,
            "right_pct": 0.0,
            "midline_x_mm": float("nan"),
        }

    shape = mask_b.shape
    center_ijk = np.array(
        [[shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0, 1.0]], dtype=np.float64
    )
    center_xyz = (np.asarray(affine, dtype=np.float64) @ center_ijk.T).T[0, :3]
    mid_x = float(center_xyz[0])

    ijk = np.argwhere(mask_b)
    ones = np.ones((len(ijk), 1), dtype=np.float64)
    xyz = (np.asarray(affine, dtype=np.float64) @ np.hstack([ijk, ones]).T).T[:, :3]

    left = xyz[:, 0] < mid_x
    right = ~left
    n = len(xyz)
    left_pct = 100.0 * float(left.sum()) / n
    right_pct = 100.0 * float(right.sum()) / n
    if left_pct >= bilateral_min_pct and right_pct >= bilateral_min_pct:
        label = "bilateral"
    elif left_pct >= right_pct:
        label = "left"
    else:
        label = "right"
    return {
        "label": label,
        "left_pct": round(left_pct, 1),
        "right_pct": round(right_pct, 1),
        "midline_x_mm": mid_x,
    }


def _counts_to_lobe_table(
    counts: Mapping[str, int],
    total: int,
    spacing_mm: tuple[float, float, float],
) -> dict[str, dict[str, float | int]]:
    vox_ml = _voxel_ml(spacing_mm)
    denom = max(int(total), 1)
    return {
        name: {
            "pct": round(100.0 * c / denom, 1),
            "ml": round(c * vox_ml, 3),
            "voxels": int(c),
        }
        for name, c in sorted(counts.items(), key=lambda kv: -kv[1])
        if c > 0
    }


def lobe_masks_from_atlas(
    cortical_atlas: np.ndarray,
    cortical_labels: Mapping[int, str],
) -> dict[str, np.ndarray]:
    """Boolean masks for each coarse cortical lobe (patient-space HO atlas)."""
    atlas = np.asarray(cortical_atlas)
    lobe_of = cortical_label_to_lobe_map(dict(cortical_labels))
    out: dict[str, np.ndarray] = {
        name: np.zeros(atlas.shape, dtype=bool) for name in NEAREST_LOBE_NAMES
    }
    for idx, lobe in lobe_of.items():
        if lobe in out:
            out[lobe] |= atlas == int(idx)
    return out


def lobe_occupancy(
    mask: np.ndarray,
    cortical_atlas: np.ndarray,
    cortical_labels: Mapping[int, str],
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    nearest_max_mm: float = NEAREST_MAX_MM,
) -> dict[str, Any]:
    """Cortical-only and nearest-lobe occupancy for a tumor mask.

    * ``lobes_cortical_only`` — distribution over tumor voxels that land on a
      Harvard-Oxford cortical lobe label (renormalized; no ``other``).
    * ``lobes_nearest`` — every tumor voxel assigned to the nearest cortical
      lobe within ``nearest_max_mm``, else ``deep/central``. Headline uses this.
    """
    mask_b = np.asarray(mask, dtype=bool)
    atlas = np.asarray(cortical_atlas)
    lobe_of = cortical_label_to_lobe_map(dict(cortical_labels))
    total = int(mask_b.sum())
    empty = {
        "lobes": {},
        "lobes_cortical_only": {},
        "lobes_nearest": {},
        "dominant_lobe": None,
        "dominant_pct": 0.0,
        "dominant_cortical_lobe": None,
        "dominant_cortical_pct": 0.0,
        "unlabeled_pct": 0.0,
        "nearest_max_mm": float(nearest_max_mm),
    }
    if total == 0:
        return empty

    # --- Direct cortical overlap ---
    cortical_counts: dict[str, int] = {n: 0 for n in NEAREST_LOBE_NAMES}
    labels_in = atlas[mask_b]
    labeled = 0
    for lab in labels_in:
        lab_i = int(lab)
        if lab_i == 0:
            continue
        lobe = lobe_of.get(lab_i)
        if lobe in cortical_counts:
            cortical_counts[lobe] += 1
            labeled += 1
    unlabeled = total - labeled
    cortical_only = _counts_to_lobe_table(
        {k: v for k, v in cortical_counts.items() if v > 0},
        labeled if labeled > 0 else 1,
        spacing_mm,
    )
    # Dominant among direct cortical hits (may be None if tumor is entirely deep).
    if labeled > 0:
        dom_c = max(cortical_counts, key=lambda k: cortical_counts[k])
        dom_c_pct = 100.0 * cortical_counts[dom_c] / labeled
    else:
        dom_c, dom_c_pct = None, 0.0

    # --- Nearest-lobe assignment for every tumor voxel ---
    lobe_masks = lobe_masks_from_atlas(atlas, cortical_labels)
    # Precompute distance-to-lobe for the whole volume (cheap at BraTS sizes).
    dist_maps: dict[str, np.ndarray] = {}
    for name, lm in lobe_masks.items():
        if not np.any(lm):
            dist_maps[name] = np.full(atlas.shape, np.inf, dtype=np.float32)
        else:
            dist_maps[name] = ndi.distance_transform_edt(
                ~lm, sampling=spacing_mm
            ).astype(np.float32)

    # Vectorized assignment on tumor voxels only.
    tumor_idx = np.where(mask_b)
    n_t = int(mask_b.sum())
    # Stack distances (n_lobes, n_tumor)
    dist_stack = np.stack(
        [dist_maps[name][tumor_idx] for name in NEAREST_LOBE_NAMES], axis=0
    )
    best_i = np.argmin(dist_stack, axis=0)
    best_d = dist_stack[best_i, np.arange(n_t)]
    assigned = np.array(NEAREST_LOBE_NAMES, dtype=object)[best_i]
    assigned = assigned.copy()
    assigned[best_d > float(nearest_max_mm)] = DEEP_LABEL

    # Direct HO cortical labels override distance.
    labs = atlas[tumor_idx].astype(np.int32)
    for lobe_name in NEAREST_LOBE_NAMES:
        # Label ids that map to this lobe
        ids = [i for i, L in lobe_of.items() if L == lobe_name]
        if not ids:
            continue
        hit = np.isin(labs, ids)
        assigned[hit] = lobe_name

    nearest_counts: dict[str, int] = {n: 0 for n in NEAREST_LOBE_NAMES}
    nearest_counts[DEEP_LABEL] = 0
    for name, cnt in zip(*np.unique(assigned, return_counts=True)):
        nearest_counts[str(name)] = int(cnt)

    nearest_table = _counts_to_lobe_table(nearest_counts, total, spacing_mm)
    # Headline dominant: never "other"; deep/central allowed.
    dom_n = max(nearest_counts, key=lambda k: nearest_counts[k])
    dom_n_pct = 100.0 * nearest_counts[dom_n] / total

    return {
        # Back-compat: ``lobes`` == nearest-lobe table (headline source).
        "lobes": nearest_table,
        "lobes_cortical_only": cortical_only,
        "lobes_nearest": nearest_table,
        "dominant_lobe": dom_n,
        "dominant_pct": round(dom_n_pct, 1),
        "dominant_cortical_lobe": dom_c,
        "dominant_cortical_pct": round(float(dom_c_pct), 1),
        "unlabeled_pct": round(100.0 * unlabeled / total, 1),
        "nearest_max_mm": float(nearest_max_mm),
    }


def top_structures(
    mask: np.ndarray,
    atlas: np.ndarray,
    labels: Mapping[int, str],
    *,
    top_k: int = 5,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    skip_background: bool = True,
) -> list[dict[str, Any]]:
    """Top-k atlas structures by overlap volume with the tumor mask."""
    mask_b = np.asarray(mask, dtype=bool)
    atlas = np.asarray(atlas)
    total = int(mask_b.sum())
    if total == 0:
        return []
    vals, counts = np.unique(atlas[mask_b], return_counts=True)
    vox_ml = _voxel_ml(spacing_mm)
    rows: list[dict[str, Any]] = []
    for val, cnt in zip(vals.tolist(), counts.tolist()):
        idx = int(val)
        if skip_background and idx == 0:
            continue
        name = labels.get(idx, f"label_{idx}")
        rows.append(
            {
                "id": idx,
                "name": name,
                "ml": round(int(cnt) * vox_ml, 3),
                "pct": round(100.0 * int(cnt) / total, 1),
                "voxels": int(cnt),
            }
        )
    rows.sort(key=lambda r: -r["voxels"])
    return rows[: int(top_k)]


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    mask_b = np.asarray(mask, dtype=bool)
    if not mask_b.any():
        return mask_b
    eroded = ndi.binary_erosion(mask_b)
    surface = mask_b & ~eroded
    return surface if surface.any() else mask_b


def min_distance_mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Minimum Euclidean surface distance (mm). 0 if masks overlap."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if not a.any() or not b.any():
        return float("nan")
    if np.any(a & b):
        return 0.0
    dt = ndi.distance_transform_edt(~b, sampling=spacing_mm)
    surface = _surface_voxels(a)
    return float(dt[surface].min())


def midline_distance_mm(
    mask: np.ndarray,
    affine: np.ndarray,
) -> float:
    """Min |x − midline| in mm for tumor voxels (RAS x from affine)."""
    mask_b = np.asarray(mask, dtype=bool)
    if not mask_b.any():
        return float("nan")
    shape = mask_b.shape
    center_ijk = np.array(
        [[shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0, 1.0]], dtype=np.float64
    )
    mid_x = float((np.asarray(affine, dtype=np.float64) @ center_ijk.T).T[0, 0])
    ijk = np.argwhere(mask_b)
    ones = np.ones((len(ijk), 1), dtype=np.float64)
    xyz = (np.asarray(affine, dtype=np.float64) @ np.hstack([ijk, ones]).T).T[:, :3]
    return float(np.min(np.abs(xyz[:, 0] - mid_x)))


def plain_language_summary(
    *,
    hemisphere: Mapping[str, Any],
    lobes: Mapping[str, Any],
    distances: Mapping[str, float],
    registration_quality: str,
) -> str:
    """One-line summary from nearest-lobe headline (never 'other')."""
    if registration_quality in {"poor", "registration unreliable"}:
        return (
            "Registration unreliable — anatomical location not claimed; "
            "see registration_qc.png."
        )
    if registration_quality == "caution":
        # Still emit a descriptive line; report HTML already warns.
        pass

    hemi = str(hemisphere.get("label", "unknown"))
    if hemi == "bilateral":
        hemi_txt = (
            f"bilateral (L {hemisphere.get('left_pct', 0):.0f}% / "
            f"R {hemisphere.get('right_pct', 0):.0f}%)"
        )
    elif hemi == "none":
        return "No tumor voxels in this mask."
    else:
        hemi_txt = hemi

    dom = lobes.get("dominant_lobe")
    pct = float(lobes.get("dominant_pct") or 0.0)
    if dom == DEEP_LABEL:
        lobe_txt = f"deep/central, {pct:.0f}%"
    elif dom:
        lobe_txt = f"{dom}, {pct:.0f}%"
    else:
        lobe_txt = "lobe uncertain"

    bits: list[str] = []
    vent = distances.get("ventricles_mm")
    if vent == vent:
        if vent <= 2.0:
            bits.append("abuts lateral ventricle")
        elif vent <= 8.0:
            bits.append(f"near ventricle ({vent:.0f} mm)")
    cst = distances.get("corticospinal_tract_mm")
    if cst == cst and cst <= 5.0:
        bits.append("near corticospinal tract" if cst > 0 else "involves corticospinal tract")
    optic = distances.get("optic_radiation_mm")
    if optic == optic and optic <= 5.0:
        bits.append("near optic radiation" if optic > 0 else "involves optic radiation")
    mid = distances.get("midline_mm")
    if mid == mid and mid <= 3.0:
        bits.append("near midline")

    suffix = f", {', '.join(bits)}" if bits else ""
    return f"{hemi_txt} {lobe_txt}{suffix}"


def region_location_block(
    mask: np.ndarray,
    *,
    affine: np.ndarray,
    cortical: np.ndarray,
    cortical_labels: Mapping[int, str],
    subcortical: np.ndarray,
    subcortical_labels: Mapping[int, str],
    ventricle_mask: np.ndarray,
    cst_mask: np.ndarray | None,
    optic_mask: np.ndarray | None,
    spacing_mm: tuple[float, float, float],
    registration_quality: str,
) -> dict[str, Any]:
    """Full location dict for one tumor region (WT / TC / ET)."""
    hemi = hemisphere_fractions(mask, affine)
    lobes = lobe_occupancy(mask, cortical, cortical_labels, spacing_mm=spacing_mm)
    top_c = top_structures(mask, cortical, cortical_labels, spacing_mm=spacing_mm)
    top_s = top_structures(mask, subcortical, subcortical_labels, spacing_mm=spacing_mm)
    merged = sorted(top_c + top_s, key=lambda r: -r["voxels"])[:5]

    distances = {
        "ventricles_mm": min_distance_mm(mask, ventricle_mask, spacing_mm),
        "corticospinal_tract_mm": (
            min_distance_mm(mask, cst_mask, spacing_mm)
            if cst_mask is not None and np.any(cst_mask)
            else float("nan")
        ),
        "optic_radiation_mm": (
            min_distance_mm(mask, optic_mask, spacing_mm)
            if optic_mask is not None and np.any(optic_mask)
            else float("nan")
        ),
        "midline_mm": midline_distance_mm(mask, affine),
    }
    summary = plain_language_summary(
        hemisphere=hemi,
        lobes=lobes,
        distances=distances,
        registration_quality=registration_quality,
    )
    return {
        "hemisphere": hemi,
        "lobes": lobes["lobes"],
        "lobes_cortical_only": lobes["lobes_cortical_only"],
        "lobes_nearest": lobes["lobes_nearest"],
        "dominant_lobe": lobes["dominant_lobe"],
        "dominant_pct": lobes["dominant_pct"],
        "dominant_cortical_lobe": lobes.get("dominant_cortical_lobe"),
        "dominant_cortical_pct": lobes.get("dominant_cortical_pct"),
        "unlabeled_pct": lobes.get("unlabeled_pct"),
        "top_structures": merged,
        "distances_mm": distances,
        "summary": summary,
        "volume_ml": round(
            int(np.asarray(mask).astype(bool).sum()) * _voxel_ml(spacing_mm), 3
        ),
    }
