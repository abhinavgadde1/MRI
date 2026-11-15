"""Independent registration QC metrics (separate from transform selection)."""

from __future__ import annotations

from typing import Any

import numpy as np

# "good" requires all four. Justified on BraTS positive controls + batch2 goods
# (see results/location_report/SUMMARY.md QC justification table).
BRAIN_DICE_GOOD = 0.85
# SynthSeg ventricle labels support a strict 0.60 Dice gate. CSF-heuristic
# mode tops out ~0.4–0.5 even on well-registered batch2 (see SUMMARY), so the
# CSF-mode good gate is 0.30 (still well above chance / empty overlap).
VENTRICLE_DICE_GOOD = 0.60
VENTRICLE_DICE_GOOD_CSF = 0.30
SURFACE_P95_GOOD_MM = 8.0
COVERAGE_GOOD = 0.90

# Below these → registration unreliable (vs caution).
BRAIN_DICE_UNRELIABLE = 0.75
VENTRICLE_DICE_UNRELIABLE = 0.10
VENTRICLE_DICE_UNRELIABLE_CSF = 0.05
SURFACE_P95_UNRELIABLE_MM = 15.0
COVERAGE_UNRELIABLE = 0.75

# FreeSurfer / SynthSeg lateral ventricle labels.
SYNTHSEG_LATERAL_VENTRICLE = (4, 43, 5, 44)


def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a_b = np.asarray(a, dtype=bool)
    b_b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a_b, b_b).sum())
    denom = int(a_b.sum()) + int(b_b.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * inter / denom)


def _csf_dark_mask(
    t1: np.ndarray,
    brain_b: np.ndarray,
    t2: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Dark-on-T1 (optionally T2-bright) CSF candidate voxels inside brain."""
    from scipy import ndimage as ndi

    t1_np = np.asarray(t1, dtype=np.float32)
    t1_brain = t1_np[brain_b]
    if t1_brain.size == 0:
        return np.zeros(brain_b.shape, dtype=bool), "empty"

    core = ndi.binary_erosion(brain_b, iterations=5)
    if int(core.sum()) < 1000:
        core = brain_b

    # Soft CSF: darkest ~40% of brain T1 (ventricles are darker than cortex/WM
    # but partial-volumed; a 15–20% cut was too strict on clinical / BraTS T1).
    t1_p40 = float(np.percentile(t1_brain, 40))
    dark = core & (t1_np <= t1_p40)
    source = "t1_csf"
    if t2 is not None and np.asarray(t2).shape == t1_np.shape:
        t2_np = np.asarray(t2, dtype=np.float32)
        t2_p60 = float(np.percentile(t2_np[brain_b], 60))
        dark = dark & (t2_np >= t2_p60)
        source = "t1_t2_csf"
    return ndi.binary_opening(dark, iterations=1), source


def estimate_patient_ventricle_mask(
    t1: np.ndarray,
    brain: np.ndarray,
    *,
    t2: np.ndarray | None = None,
    synthseg: np.ndarray | None = None,
    atlas_ventricle: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Patient lateral-ventricle mask for QC Dice.

    Prefers SynthSeg lateral-ventricle labels when present. Otherwise uses a
    CSF heuristic (dark T1, optionally T2-bright). When a warped atlas
    ventricle is provided, CSF is restricted to a neighbourhood of that atlas
    label — this checks whether the registered ventricle lands on CSF without
    requiring a perfect free-standing ventricle segmentation.
    """
    from scipy import ndimage as ndi

    brain_b = np.asarray(brain, dtype=bool)
    if synthseg is not None:
        lab = np.asarray(synthseg)
        if lab.shape == brain_b.shape:
            vent = np.isin(lab, SYNTHSEG_LATERAL_VENTRICLE) & brain_b
            if int(vent.sum()) >= 50:
                return vent, "synthseg"

    dark, source = _csf_dark_mask(t1, brain_b, t2)
    if not np.any(dark):
        return np.zeros(brain_b.shape, dtype=bool), source

    if atlas_ventricle is not None and np.any(atlas_ventricle):
        near = ndi.binary_dilation(
            np.asarray(atlas_ventricle, dtype=bool), iterations=6
        )
        cand = dark & near
        if int(cand.sum()) >= 100:
            return cand, source + "_near_atlas"

    # Independent fallback: largest central CSF components.
    lab, n = ndi.label(dark)
    if n == 0:
        return np.zeros(brain_b.shape, dtype=bool), source
    coords = np.array(np.where(brain_b))
    brain_center = coords.mean(axis=1)
    extent = coords.max(axis=1) - coords.min(axis=1) + 1e-6
    sizes = ndi.sum(dark, lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = np.zeros(brain_b.shape, dtype=bool)
    kept = 0
    for idx in order:
        comp = lab == (int(idx) + 1)
        if float(sizes[idx]) < 500:
            break
        centroid = np.array(np.where(comp)).mean(axis=1)
        if float(np.max(np.abs(centroid - brain_center) / extent)) > 0.40:
            continue
        keep |= comp
        kept += 1
        if kept >= 2:
            break
    if not np.any(keep):
        keep = lab == (int(order[0]) + 1)
    return keep, source


def surface_distance_mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[float, float]:
    """Boundary distance (mean, p95) in mm between two binary masks.

    Uses the symmetric average of directed surface distances, but only over
    surface voxels that lie inside a shared ROI (dilated intersection) so
    unmatched template cortex outside the patient brain does not inflate p95.
    """
    from scipy import ndimage as ndi

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if not np.any(a) or not np.any(b):
        return float("inf"), float("inf")

    # Shared neighborhood around both masks (~5 mm at 1 mm spacing).
    roi = ndi.binary_dilation(a, iterations=5) & ndi.binary_dilation(b, iterations=5)
    a_roi = a & roi
    b_roi = b & roi
    if not np.any(a_roi) or not np.any(b_roi):
        a_roi, b_roi = a, b

    a_surf = a_roi & ~ndi.binary_erosion(a_roi)
    b_surf = b_roi & ~ndi.binary_erosion(b_roi)
    if not np.any(a_surf):
        a_surf = a_roi
    if not np.any(b_surf):
        b_surf = b_roi

    dt_b = ndi.distance_transform_edt(~b_roi, sampling=spacing_mm)
    dt_a = ndi.distance_transform_edt(~a_roi, sampling=spacing_mm)
    d_ab = dt_b[a_surf]
    d_ba = dt_a[b_surf]
    all_d = np.concatenate([d_ab.ravel(), d_ba.ravel()])
    return float(np.mean(all_d)), float(np.percentile(all_d, 95))


def fov_coverage_fraction(
    template_brain: np.ndarray,
    warped_patient_fov: np.ndarray,
) -> float:
    """Fraction of template brain voxels covered by the patient FOV in MNI."""
    tb = np.asarray(template_brain, dtype=bool)
    fov = np.asarray(warped_patient_fov, dtype=bool)
    denom = int(tb.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(tb, fov).sum() / denom)


def ventricle_gates(ventricle_source: str | None) -> tuple[float, float]:
    """Return (good_min, unreliable_min) for ventricle Dice given mask source."""
    src = (ventricle_source or "").lower()
    if src == "synthseg":
        return VENTRICLE_DICE_GOOD, VENTRICLE_DICE_UNRELIABLE
    return VENTRICLE_DICE_GOOD_CSF, VENTRICLE_DICE_UNRELIABLE_CSF


def flag_registration_quality(
    *,
    brain_dice: float,
    ventricle_dice: float | None,
    surface_p95_mm: float,
    coverage: float,
    stage: str,
    ventricle_source: str | None = None,
) -> tuple[str, str]:
    """Map independent QC metrics → good | caution | registration unreliable."""
    vent = float(ventricle_dice) if ventricle_dice is not None else 0.0
    vent_missing = ventricle_dice is None
    vent_good, vent_bad = ventricle_gates(ventricle_source)

    unreliable_reasons: list[str] = []
    if float(brain_dice) < BRAIN_DICE_UNRELIABLE:
        unreliable_reasons.append(
            f"brain Dice {brain_dice:.3f} < {BRAIN_DICE_UNRELIABLE}"
        )
    if float(coverage) < COVERAGE_UNRELIABLE:
        unreliable_reasons.append(
            f"FOV coverage {coverage:.3f} < {COVERAGE_UNRELIABLE}"
        )
    if float(surface_p95_mm) > SURFACE_P95_UNRELIABLE_MM:
        unreliable_reasons.append(
            f"surface p95 {surface_p95_mm:.1f} mm > {SURFACE_P95_UNRELIABLE_MM}"
        )
    if not vent_missing and vent < vent_bad:
        unreliable_reasons.append(
            f"ventricle Dice {vent:.3f} < {vent_bad} ({ventricle_source or 'csf'})"
        )
    if unreliable_reasons:
        return (
            "registration unreliable",
            "Unreliable QC: " + "; ".join(unreliable_reasons) + ".",
        )

    good = (
        float(brain_dice) >= BRAIN_DICE_GOOD
        and (not vent_missing and vent >= vent_good)
        and float(surface_p95_mm) <= SURFACE_P95_GOOD_MM
        and float(coverage) >= COVERAGE_GOOD
    )
    metrics = (
        f"brain Dice={brain_dice:.3f}, ventricle Dice={vent:.3f} "
        f"[{ventricle_source or 'n/a'}; good≥{vent_good}], "
        f"surface p95={surface_p95_mm:.1f} mm, coverage={coverage:.3f}, "
        f"stage={stage}"
    )
    if good:
        return "good", f"All QC gates passed ({metrics})."

    caution_bits: list[str] = []
    if float(brain_dice) < BRAIN_DICE_GOOD:
        caution_bits.append(f"brain Dice {brain_dice:.3f} < {BRAIN_DICE_GOOD}")
    if vent_missing or vent < vent_good:
        caution_bits.append(
            f"ventricle Dice {vent:.3f} < {vent_good}"
            if not vent_missing
            else "ventricle Dice unavailable"
        )
    if float(surface_p95_mm) > SURFACE_P95_GOOD_MM:
        caution_bits.append(
            f"surface p95 {surface_p95_mm:.1f} mm > {SURFACE_P95_GOOD_MM}"
        )
    if float(coverage) < COVERAGE_GOOD:
        caution_bits.append(f"coverage {coverage:.3f} < {COVERAGE_GOOD} (partial FOV)")
    return (
        "caution",
        "Caution — " + "; ".join(caution_bits) + f". ({metrics})",
    )


def gate_message(wt_ml: float, floor_ml: float) -> str:
    """Plausibility-gate wording (never 'no tumor detected')."""
    return (
        f"model volume {wt_ml:.2f} mL is below the {floor_ml:.1f} mL BraTS size floor; "
        "location not reported (small lesion or false positive, unverified)"
    )


def qc_as_dict(
    *,
    brain_dice: float,
    ventricle_dice: float | None,
    surface_mean_mm: float,
    surface_p95_mm: float,
    coverage: float,
    coverage_partial: bool,
    ventricle_source: str | None,
    stage: str,
    quality: str,
    quality_note: str,
) -> dict[str, Any]:
    return {
        "brain_dice": float(brain_dice),
        "ventricle_dice": None if ventricle_dice is None else float(ventricle_dice),
        "surface_mean_mm": float(surface_mean_mm),
        "surface_p95_mm": float(surface_p95_mm),
        "coverage": float(coverage),
        "coverage_partial": bool(coverage_partial),
        "ventricle_source": ventricle_source,
        "stage": stage,
        "quality": quality,
        "quality_note": quality_note,
        "thresholds": {
            "brain_dice_good": BRAIN_DICE_GOOD,
            "ventricle_dice_good": VENTRICLE_DICE_GOOD,
            "surface_p95_good_mm": SURFACE_P95_GOOD_MM,
            "coverage_good": COVERAGE_GOOD,
        },
    }
