"""Classical intensity-based enhancement + FLAIR drafts for native-T1c batch2.

Rule-based drafts for **manual correction only** — not verified masks.
Vessels, flow artifacts and misregistration can create false enhancement.

Writes exclusively under ``outputs/batch2_intensity_draft/<study_id>/``.
Does not modify checkpoints, heldout splits, ``data/processed/real_patients/``,
or any existing draft / corrected / v3 / HD-GLIO output.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import frangi

from config import PROJECT_ROOT
from segmentation.batch2_label_qc import find_brain_mask
from segmentation.batch2_scan_qc import normalized_cross_correlation
from segmentation.hdglio_labels import brats_regions_from_classic

logger = logging.getLogger(__name__)

NATIVE_T1C_STUDIES = (
    "batch2_01",
    "batch2_02",
    "batch2_03",
    "batch2_05",
    "batch2_06",
    "batch2_07",
    "batch2_08",
    "batch2_10",
    "batch2_12",
    "batch2_13",
)

LABELING_ROOT = PROJECT_ROOT / "outputs" / "batch2_labeling"
HDGLIO_ROOT = PROJECT_ROOT / "outputs" / "batch2_second_opinion"
OUT_ROOT = PROJECT_ROOT / "outputs" / "batch2_intensity_draft"

DRAFT_DISCLAIMER = (
    "Rule-based intensity drafts for manual correction only — not verified masks. "
    "Vessels, flow artifacts and misregistration can create false enhancement."
)

# Enhancement threshold: top 0.5% of enhancement z-scores inside the brain.
ENHANCE_PERCENTILE = 99.5
FLAIR_Z_THRESH = 1.5
MIN_ENHANCE_ML = 0.3
MIN_FLAIR_ML = 1.0
# T1 vs T1c are different contrasts — absolute NCC is often moderate.
# Flag when refine *hurts* NCC or leaves it very low.
NCC_POOR_ABS = 0.40
NCC_DROP_FLAG = 0.03
CENTROID_AGREE_MM = 15.0
GAUSS_SIGMA_MM = 1.0
TOP_K_ENHANCE = 3


@dataclass
class ComponentInfo:
    label: int
    volume_ml: float
    centroid_mm: tuple[float, float, float]
    mean_enhancement: float


@dataclass
class StudyIntensityResult:
    study_id: str
    nawm_estimation: str
    nawm_t1_median: float
    nawm_t1_mad: float
    ncc_before: float
    ncc_after: float
    alignment_flag: str
    enhance_percentile: float
    enhance_threshold: float
    flair_z_threshold: float
    csf_rule: str
    tc_core_rule: str
    candidates: list[ComponentInfo] = field(default_factory=list)
    dice_vs_v3: dict[str, float] = field(default_factory=dict)
    dice_vs_hdglio: dict[str, float] = field(default_factory=dict)
    vol_intensity_ml: dict[str, float] = field(default_factory=dict)
    vol_v3_ml: dict[str, float] = field(default_factory=dict)
    vol_hdglio_ml: dict[str, float] = field(default_factory=dict)
    main_et_centroid_mm: tuple[float, float, float] | None = None
    v3_et_centroid_mm: tuple[float, float, float] | None = None
    hdglio_et_centroid_mm: tuple[float, float, float] | None = None
    et_location_agree_all_three: bool = False
    grid_match_ok: bool = False
    notes: list[str] = field(default_factory=list)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, np.asarray(img.affine), zooms


def _voxel_ml(spacing: tuple[float, float, float]) -> float:
    return float(np.prod(spacing) / 1000.0)


def nawm_proxy_mask(t1: np.ndarray, brain: np.ndarray) -> tuple[np.ndarray, str]:
    """Boolean NAWM proxy from T1 brain histogram mode ± 0.25·IQR."""
    vals = t1[brain]
    vals = vals[np.isfinite(vals)]
    mask = np.zeros_like(brain, dtype=bool)
    if vals.size < 100:
        mask[brain] = True
        return mask, "fallback_full_brain (too few voxels for mode)"
    hist, edges = np.histogram(vals, bins=256)
    mode_idx = int(np.argmax(hist))
    mode = 0.5 * (edges[mode_idx] + edges[mode_idx + 1])
    q25, q75 = np.percentile(vals, [25, 75])
    iqr = float(max(q75 - q25, 1e-6))
    half = 0.25 * iqr
    proxy = brain & (t1 >= mode - half) & (t1 <= mode + half)
    if int(proxy.sum()) < 50:
        proxy = brain.copy()
        note = f"T1 mode={mode:.4g}; IQR window empty → full brain proxy"
    else:
        note = (
            f"T1 histogram mode={mode:.4g}; NAWM proxy = voxels within "
            f"±0.25·IQR ({half:.4g}) of mode (n={int(proxy.sum())})"
        )
    return proxy, note


def estimate_nawm_stats(
    vol: np.ndarray,
    proxy: np.ndarray,
) -> tuple[float, float]:
    """Median / MAD on a NAWM proxy mask; MAD floored to avoid explode-y z-scores."""
    vals = vol[proxy]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    floor = max(0.01 * abs(med), 1.0)
    mad = max(mad, floor)
    return med, mad


def robust_zscore(
    vol: np.ndarray,
    brain: np.ndarray,
    *,
    median: float,
    mad: float,
) -> np.ndarray:
    """(x - median) / (1.4826 · MAD); zeros outside brain."""
    scale = 1.4826 * max(float(mad), 1e-6)
    out = np.zeros_like(vol, dtype=np.float32)
    out[brain] = ((vol[brain] - float(median)) / scale).astype(np.float32)
    return out


def rigid_register_masked(
    moving: np.ndarray,
    fixed: np.ndarray,
    brain: np.ndarray,
    affine: np.ndarray,
    *,
    number_of_iterations: int = 100,
    moving_path: Path | None = None,
    fixed_path: Path | None = None,
) -> tuple[np.ndarray, float, float]:
    """Rigid refine moving→fixed using MI, metric masked to brain.

    Prefer reading NIfTI paths via SimpleITK when provided (correct geometry).
    Returns ``(resampled_moving, ncc_before, ncc_after)``.
    """
    ncc_before = normalized_cross_correlation(moving, fixed, brain)

    if moving_path is not None and fixed_path is not None:
        fixed_s = sitk.Cast(sitk.ReadImage(str(fixed_path)), sitk.sitkFloat32)
        moving_s = sitk.Cast(sitk.ReadImage(str(moving_path)), sitk.sitkFloat32)
        mask_img = sitk.GetImageFromArray(
            np.swapaxes(brain.astype(np.uint8), 0, 2)
        )
        mask_img.CopyInformation(fixed_s)
        mask_s = sitk.Cast(mask_img, sitk.sitkUInt8)
    else:
        # Synthetic / in-memory: axis-aligned spacing from affine norms
        spacing = tuple(
            float(np.linalg.norm(affine[:3, i])) for i in range(3)
        )
        fixed_s = sitk.GetImageFromArray(np.swapaxes(fixed, 0, 2).astype(np.float32))
        moving_s = sitk.GetImageFromArray(np.swapaxes(moving, 0, 2).astype(np.float32))
        fixed_s.SetSpacing(spacing)
        moving_s.SetSpacing(spacing)
        mask_s = sitk.GetImageFromArray(np.swapaxes(brain.astype(np.uint8), 0, 2))
        mask_s.CopyInformation(fixed_s)

    initial = sitk.CenteredTransformInitializer(
        fixed_s,
        moving_s,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricFixedMask(mask_s)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=number_of_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        transform = registration.Execute(fixed_s, moving_s)
        resampled_s = sitk.Resample(
            moving_s,
            fixed_s,
            transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        resampled = np.swapaxes(sitk.GetArrayFromImage(resampled_s), 0, 2).astype(
            np.float32
        )
        if resampled.shape != moving.shape:
            logger.warning(
                "Resample shape %s != moving %s; keeping original",
                resampled.shape,
                moving.shape,
            )
            resampled = moving.copy()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Masked rigid registration failed (%s); keeping original", exc)
        resampled = moving.copy()

    ncc_after = normalized_cross_correlation(resampled, fixed, brain)
    # Packages are already co-registered; keep original if refine hurts NCC.
    if np.isfinite(ncc_before) and np.isfinite(ncc_after) and ncc_after < ncc_before - 1e-4:
        logger.info(
            "Rigid refine lowered NCC (%.3f→%.3f); keeping original moving volume",
            ncc_before,
            ncc_after,
        )
        return moving.copy(), float(ncc_before), float(ncc_before)
    return resampled, float(ncc_before), float(ncc_after)


def gaussian_smooth_mm(
    vol: np.ndarray,
    spacing: tuple[float, float, float],
    sigma_mm: float = GAUSS_SIGMA_MM,
) -> np.ndarray:
    sig = tuple(max(sigma_mm / max(s, 1e-6), 1e-6) for s in spacing)
    return ndi.gaussian_filter(vol.astype(np.float32), sigma=sig)


def enhancement_candidates(
    enhance: np.ndarray,
    brain: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    percentile: float = ENHANCE_PERCENTILE,
    min_ml: float = MIN_ENHANCE_ML,
    top_k: int = TOP_K_ENHANCE,
) -> tuple[np.ndarray, float, list[ComponentInfo], dict[str, Any]]:
    """Threshold enhancement, drop vessels/border/small, keep top-k components."""
    brain_vals = enhance[brain]
    thr = float(np.percentile(brain_vals, percentile)) if brain_vals.size else 0.0
    raw = (enhance >= thr) & brain

    # Frangi vesselness on |enhancement| (tubular bright structures → vessels)
    # Restrict to brain bbox for speed.
    vessel = np.zeros_like(enhance, dtype=np.float32)
    if np.any(brain):
        zs = np.where(brain)
        z0, z1 = int(zs[0].min()), int(zs[0].max()) + 1
        y0, y1 = int(zs[1].min()), int(zs[1].max()) + 1
        x0, x1 = int(zs[2].min()), int(zs[2].max()) + 1
        crop = np.abs(enhance[z0:z1, y0:y1, x0:x1])
        # Frangi sigmas in voxels ≈ 0.5–1.5 mm vessel calibre.
        mean_sp = float(np.mean(spacing))
        sigma_list = [0.5 / mean_sp, 1.0 / mean_sp, 1.5 / mean_sp]
        try:
            v_crop = frangi(crop, sigmas=sigma_list, black_ridges=False)
            vessel[z0:z1, y0:y1, x0:x1] = v_crop.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Frangi failed (%s); skipping vessel filter", exc)

    # Only suppress voxels with clearly elevated vesselness (avoid wiping
    # everything when Frangi is ~0 everywhere, where p97 would be 0).
    v_brain = vessel[brain]
    vmax = float(v_brain.max()) if v_brain.size else 0.0
    if vmax > 1e-6:
        v_thr = float(np.percentile(v_brain[v_brain > 0], 90.0)) if np.any(
            v_brain > 0
        ) else vmax
        v_thr = max(v_thr, 0.1 * vmax)
        vessel_mask = vessel >= v_thr
    else:
        vessel_mask = np.zeros_like(brain, dtype=bool)

    # Border-touching: erode brain and drop anything outside eroded mask
    eroded = ndi.binary_erosion(brain, iterations=2)
    border_touch = brain & ~eroded

    # Drop border voxels before CC; score tubularity per component (Frangi +
    # elongation) so compact enhancing blobs are not erased by vesselness.
    cleaned = raw & ~border_touch
    vox_ml = _voxel_ml(spacing)
    min_vox = max(1, int(np.ceil(min_ml / vox_ml)))

    labeled, nlab = ndi.label(cleaned)
    comps: list[ComponentInfo] = []
    keep = np.zeros_like(cleaned, dtype=bool)
    sizes = ndi.sum(cleaned, labeled, index=list(range(1, nlab + 1))) if nlab else []
    order = np.argsort(sizes)[::-1] if nlab else []
    kept = 0
    n_vessel_dropped = 0
    for idx in order:
        lab = int(idx) + 1
        size = int(sizes[idx])
        if size < min_vox:
            continue
        mask = labeled == lab
        if float(mask[border_touch].sum()) / max(size, 1) > 0.25:
            continue
        # Tubular / vessel heuristic: high mean Frangi AND elongated PCA axes
        mean_v = float(vessel[mask].mean()) if vmax > 1e-6 else 0.0
        coords = np.column_stack(np.where(mask)).astype(np.float64)
        elong = 1.0
        if coords.shape[0] >= 10:
            cov = np.cov(coords.T)
            evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            if evals[0] > 1e-6:
                elong = float(evals[0] / max(evals[-1], 1e-6))
        if mean_v > 0.15 * vmax and elong >= 4.0:
            n_vessel_dropped += 1
            continue
        # Also drop if mostly covered by the hard vessel mask and small
        if float(vessel_mask[mask].mean()) > 0.7 and size < 3 * min_vox and elong >= 3.0:
            n_vessel_dropped += 1
            continue

        cents = ndi.center_of_mass(mask)
        centroid_mm = tuple(float(cents[i] * spacing[i]) for i in range(3))
        mean_e = float(enhance[mask].mean())
        comps.append(
            ComponentInfo(
                label=kept + 1,
                volume_ml=float(size * vox_ml),
                centroid_mm=centroid_mm,  # type: ignore[arg-type]
                mean_enhancement=mean_e,
            )
        )
        keep |= mask
        kept += 1
        if kept >= top_k:
            break

    meta = {
        "percentile_rule": f"top {100 - percentile:.2f}% of enhancement z inside brain "
        f"(percentile={percentile})",
        "threshold": thr,
        "vessel_filter": "Frangi + PCA elongation (drop if high vesselness and elongated)",
        "min_ml": min_ml,
        "n_raw_components": int(nlab),
        "n_vessel_dropped": n_vessel_dropped,
        "n_kept": kept,
    }
    return keep, thr, comps, meta


def flair_candidates(
    flair_z: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    brain: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    z_thresh: float = FLAIR_Z_THRESH,
    min_ml: float = MIN_FLAIR_ML,
) -> tuple[np.ndarray, str]:
    """FLAIR hyperintensity candidates with CSF / ventricular-rim removal."""
    raw = (flair_z >= z_thresh) & brain

    # CSF rule: low T1 (below brain median) AND T2-bright (above 75th pct of brain T2)
    t1_brain = t1[brain]
    t2_brain = t2[brain]
    t1_med = float(np.median(t1_brain)) if t1_brain.size else 0.0
    t2_p75 = float(np.percentile(t2_brain, 75)) if t2_brain.size else 0.0
    csf = brain & (t1 < t1_med) & (t2 > t2_p75)
    csf_rule = (
        f"CSF = brain voxels with T1 < brain-median ({t1_med:.4g}) AND "
        f"T2 > brain 75th percentile ({t2_p75:.4g}); removed from FLAIR map"
    )

    # Thin rim along ventricles: dilate a CSF core that is also very dark on T1
    t1_p20 = float(np.percentile(t1_brain, 20)) if t1_brain.size else 0.0
    vent_core = brain & (t1 < t1_p20) & (t2 > t2_p75)
    vent_rim = ndi.binary_dilation(vent_core, iterations=2) & ~vent_core

    cleaned = raw & ~csf & ~vent_rim
    vox_ml = _voxel_ml(spacing)
    min_vox = max(1, int(np.ceil(min_ml / vox_ml)))
    labeled, nlab = ndi.label(cleaned)
    keep = np.zeros_like(cleaned, dtype=bool)
    if nlab:
        for lab in range(1, nlab + 1):
            m = labeled == lab
            if int(m.sum()) >= min_vox:
                keep |= m
    return keep, csf_rule


def build_nested_brats(
    et_mask: np.ndarray,
    flair_mask: np.ndarray,
    t1_z: np.ndarray,
    t2: np.ndarray,
    brain: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, str]:
    """ET / TC / WT nested classic BraTS labels {1,2,4}.

    TC core rule: T1-dark (z < −0.5) OR T2-bright (T2 > brain 70th pct) voxels
    that are adjacent (26-connected dilation of ET by 2 voxels) to ET, then
    union with ET. Necrotic/cystic core sits next to enhancing tissue.
    """
    t2_brain = t2[brain]
    t2_p70 = float(np.percentile(t2_brain, 70)) if t2_brain.size else 0.0
    near_et = ndi.binary_dilation(et_mask, iterations=2)
    core = near_et & brain & ((t1_z < -0.5) | (t2 > t2_p70))
    tc = et_mask | core
    tc_rule = (
        "TC = ET ∪ voxels within 2-voxel dilation of ET that are "
        f"T1-z < -0.5 OR T2 > brain 70th pct ({t2_p70:.4g})"
    )

    # WT: FLAIR components that contain or touch TC, plus TC
    labeled, nlab = ndi.label(flair_mask)
    wt = tc.copy()
    if nlab:
        for lab in range(1, nlab + 1):
            m = labeled == lab
            if np.any(m & tc) or np.any(ndi.binary_dilation(m, iterations=1) & tc):
                wt |= m
    wt |= tc

    # Classic: 4=ET, 1=TC\\ET (necrosis), 2=WT\\TC (edema)
    out = np.zeros(et_mask.shape, dtype=np.int16)
    out[wt & ~tc] = 2
    out[tc & ~et_mask] = 1
    out[et_mask] = 4
    return out, tc_rule


def region_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = float(np.count_nonzero(a & b))
    denom = float(np.count_nonzero(a) + np.count_nonzero(b))
    if denom == 0:
        return float("nan")
    return 2.0 * inter / denom


def largest_et_centroid_mm(
    classic: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    et = classic == 4
    if not np.any(et):
        return None
    labeled, n = ndi.label(et)
    if n == 0:
        return None
    sizes = ndi.sum(et, labeled, index=list(range(1, n + 1)))
    lab = int(np.argmax(sizes)) + 1
    c = ndi.center_of_mass(labeled == lab)
    return tuple(float(c[i] * spacing[i]) for i in range(3))  # type: ignore[return-value]


def centroid_dist_mm(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> float:
    if a is None or b is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _norm_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(float)
    p2, p98 = np.percentile(sl, (2, 98))
    if p98 <= p2:
        return np.zeros_like(sl)
    return np.clip((sl - p2) / (p98 - p2), 0, 1)


def _contour(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    return m & ~ndi.binary_erosion(m, iterations=1)


def _pick_slices(mask: np.ndarray, n: int = 6) -> list[int]:
    m = np.asarray(mask, dtype=bool)
    z = m.shape[2]
    if not np.any(m):
        return [int(round(i * (z - 1) / max(n - 1, 1))) for i in range(n)]
    zs = np.where(m.any(axis=(0, 1)))[0]
    if len(zs) <= n:
        mid = list(map(int, zs))
        while len(mid) < n:
            mid.append(int(zs[len(zs) // 2]))
        return sorted(mid)[:n]
    idx = np.linspace(0, len(zs) - 1, n).astype(int)
    return [int(zs[i]) for i in idx]


def save_montage(
    out_png: Path,
    *,
    study_id: str,
    t1c: np.ndarray,
    enhance: np.ndarray,
    flair: np.ndarray,
    intensity_label: np.ndarray,
    v3_label: np.ndarray | None,
) -> None:
    wt = intensity_label > 0
    slices = _pick_slices(wt, 6)
    v3_wt = (v3_label > 0) if v3_label is not None else np.zeros_like(wt)
    fig, axs = plt.subplots(3, 6, figsize=(14, 7))
    row_titles = ["T1c + intensity (red) / v3 (green)", "Enhancement map", "FLAIR + contours"]
    for col, z in enumerate(slices):
        e_i = _contour(wt[:, :, z])
        e_v = _contour(v3_wt[:, :, z])

        base = _norm_slice(t1c[:, :, z])
        rgb = np.stack([base, base, base], axis=-1)
        rgb[e_v] = (0.15, 0.85, 0.25)
        rgb[e_i] = (0.95, 0.2, 0.15)
        axs[0, col].imshow(np.rot90(rgb), origin="lower")

        enh = enhance[:, :, z]
        e2, e98 = np.percentile(enh, (2, 98))
        axs[1, col].imshow(
            np.rot90(np.clip((enh - e2) / max(e98 - e2, 1e-6), 0, 1)),
            origin="lower",
            cmap="hot",
        )

        base_f = _norm_slice(flair[:, :, z])
        rgb_f = np.stack([base_f, base_f, base_f], axis=-1)
        rgb_f[e_v] = (0.15, 0.85, 0.25)
        rgb_f[e_i] = (0.95, 0.2, 0.15)
        axs[2, col].imshow(np.rot90(rgb_f), origin="lower")

        for r in range(3):
            axs[r, col].axis("off")
            if col == 0:
                axs[r, col].set_ylabel(row_titles[r], fontsize=8)
    fig.suptitle(f"{study_id} intensity draft\n{DRAFT_DISCLAIMER}", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_open_itksnap(out_dir: Path, study_id: str) -> Path:
    t1c = out_dir / "T1c.nii.gz"
    flair = out_dir / "FLAIR.nii.gz"
    enh = out_dir / "enhancement_map.nii.gz"
    draft = out_dir / "intensity_draft_label.nii.gz"
    cmd = (
        f'itksnap -g "{t1c}" '
        f'-s "{draft}" '
        f'-o "{flair}" "{enh}"'
    )
    path = out_dir / "open_itksnap.sh"
    path.write_text(
        "#!/bin/zsh\n"
        "# Open this intensity draft in ITK-SNAP (requires `itksnap` on PATH).\n"
        f"# {DRAFT_DISCLAIMER}\n"
        f'cd "{out_dir}"\n'
        f"{cmd}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    (out_dir / "OPEN.txt").write_text(
        f"{DRAFT_DISCLAIMER}\n\nITK-SNAP open command:\n\n{cmd}\n",
        encoding="utf-8",
    )
    return path


def process_study(study_id: str, out_root: Path | None = None) -> StudyIntensityResult:
    out_root = Path(out_root or OUT_ROOT)
    out_dir = out_root / study_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pkg = LABELING_ROOT / study_id
    t1c, aff, sp = _load(pkg / "T1c.nii.gz")
    t1, _, _ = _load(pkg / "T1.nii.gz")
    flair, _, _ = _load(pkg / "FLAIR.nii.gz")
    t2, _, _ = _load(pkg / "T2.nii.gz")

    brain, brain_note = find_brain_mask(pkg, target_shape=t1c.shape, target_affine=aff)
    if brain is None:
        # Fallback: T1 > 0
        brain = t1 > 0
        brain_note = "fallback T1>0 (no HD-BET mask found)"
    notes = [f"brain_mask: {brain_note}"]

    # 1. NAWM proxy from T1 mode; per-modality median/MAD on that mask;
    #    refine T1c→T1 rigid (brain-masked MI).
    proxy, nawm_note = nawm_proxy_mask(t1, brain)
    med_t1, mad_t1 = estimate_nawm_stats(t1, proxy)
    t1c_ref, ncc_b, ncc_a = rigid_register_masked(
        t1c,
        t1,
        brain,
        aff,
        moving_path=pkg / "T1c.nii.gz",
        fixed_path=pkg / "T1.nii.gz",
    )
    med_t1c, mad_t1c = estimate_nawm_stats(t1c_ref, proxy)
    nawm_note = (
        f"{nawm_note}; T1 median/MAD={med_t1:.4g}/{mad_t1:.4g}; "
        f"T1c median/MAD on same proxy={med_t1c:.4g}/{mad_t1c:.4g} "
        f"(MAD floored to max(1%|median|, 1))"
    )
    align_flag = "ok"
    if not np.isfinite(ncc_a) or ncc_a < NCC_POOR_ABS or (ncc_b - ncc_a) > NCC_DROP_FLAG:
        align_flag = (
            f"POOR_ALIGNMENT ncc={ncc_b:.3f}→{ncc_a:.3f} "
            f"(flag if after<{NCC_POOR_ABS} or drop>{NCC_DROP_FLAG})"
        )
        notes.append(align_flag)

    t1_z = robust_zscore(t1, brain, median=med_t1, mad=mad_t1)
    t1c_z = robust_zscore(t1c_ref, brain, median=med_t1c, mad=mad_t1c)

    # 2–3. Enhancement map + candidates
    enhance = gaussian_smooth_mm(t1c_z - t1_z, sp, GAUSS_SIGMA_MM)
    enhance = enhance.astype(np.float32)
    enhance[~brain] = 0.0
    et_mask, thr, comps, enh_meta = enhancement_candidates(
        enhance, brain, sp, percentile=ENHANCE_PERCENTILE
    )

    # 4. FLAIR
    flair_med = float(np.median(flair[brain])) if np.any(brain) else 0.0
    flair_mad = float(np.median(np.abs(flair[brain] - flair_med))) if np.any(brain) else 1.0
    flair_z = robust_zscore(flair, brain, median=flair_med, mad=max(flair_mad, 1e-6))
    flair_mask, csf_rule = flair_candidates(flair_z, t1, t2, brain, sp, z_thresh=FLAIR_Z_THRESH)

    # 5. Nested BraTS draft
    classic, tc_rule = build_nested_brats(et_mask, flair_mask, t1_z, t2, brain, sp)

    # Save on exact T1c grid
    t1c_img = nib.load(str(pkg / "T1c.nii.gz"))
    assert classic.shape == t1c_img.shape
    grid_ok = bool(
        classic.shape == t1c_img.shape
        and np.allclose(aff, t1c_img.affine, atol=1e-5)
    )
    draft_path = out_dir / "intensity_draft_label.nii.gz"
    nib.save(
        nib.Nifti1Image(classic.astype(np.int16), t1c_img.affine, t1c_img.header),
        str(draft_path),
    )
    # Verify reload
    re = nib.load(str(draft_path))
    grid_ok = grid_ok and re.shape == t1c_img.shape and np.allclose(
        re.affine, t1c_img.affine, atol=1e-5
    )

    # Copies / derived maps for ITK-SNAP (symlinks would be fragile across OS)
    for name, arr in (
        ("T1c.nii.gz", t1c),
        ("T1.nii.gz", t1),
        ("FLAIR.nii.gz", flair),
        ("T2.nii.gz", t2),
        ("enhancement_map.nii.gz", enhance),
        ("t1c_rigid_to_t1.nii.gz", t1c_ref),
        ("brain_mask.nii.gz", brain.astype(np.uint8)),
    ):
        nib.save(
            nib.Nifti1Image(arr.astype(np.float32 if "mask" not in name else np.uint8),
                            t1c_img.affine, t1c_img.header),
            str(out_dir / name),
        )

    write_open_itksnap(out_dir, study_id)

    # 6. Compare with v3 + HD-GLIO
    v3_path = pkg / "v3_draft_label.nii.gz"
    hd_path = HDGLIO_ROOT / study_id / "hdglio_draft_label.nii.gz"
    v3_lab = (
        np.rint(np.asanyarray(nib.load(str(v3_path)).dataobj)).astype(np.int16)
        if v3_path.is_file()
        else None
    )
    hd_lab = (
        np.rint(np.asanyarray(nib.load(str(hd_path)).dataobj)).astype(np.int16)
        if hd_path.is_file()
        else None
    )
    if v3_lab is not None and v3_lab.shape != classic.shape:
        notes.append(f"v3 draft shape {v3_lab.shape} ≠ intensity {classic.shape}")
        v3_lab = None
    if hd_lab is not None and hd_lab.shape != classic.shape:
        notes.append(f"HD-GLIO draft shape {hd_lab.shape} ≠ intensity {classic.shape}")
        hd_lab = None

    regs_i = brats_regions_from_classic(classic)
    dice_v3: dict[str, float] = {}
    dice_hd: dict[str, float] = {}
    vol_i: dict[str, float] = {}
    vol_v3: dict[str, float] = {}
    vol_hd: dict[str, float] = {}
    vox_ml = _voxel_ml(sp)
    for rname, key in (("ET", "ET"), ("TC", "TC"), ("WT", "WT")):
        vol_i[rname] = float(regs_i[key].sum() * vox_ml)
    if v3_lab is not None:
        regs_v = brats_regions_from_classic(v3_lab)
        for rname, key in (("ET", "ET"), ("TC", "TC"), ("WT", "WT")):
            dice_v3[rname] = region_dice(regs_i[key], regs_v[key])
            vol_v3[rname] = float(regs_v[key].sum() * vox_ml)
    if hd_lab is not None:
        regs_h = brats_regions_from_classic(hd_lab)
        for rname, key in (("ET", "ET"), ("TC", "TC"), ("WT", "WT")):
            dice_hd[rname] = region_dice(regs_i[key], regs_h[key])
            vol_hd[rname] = float(regs_h[key].sum() * vox_ml)

    c_i = largest_et_centroid_mm(classic, sp)
    c_v = largest_et_centroid_mm(v3_lab, sp) if v3_lab is not None else None
    c_h = largest_et_centroid_mm(hd_lab, sp) if hd_lab is not None else None
    agree = (
        centroid_dist_mm(c_i, c_v) <= CENTROID_AGREE_MM
        and centroid_dist_mm(c_i, c_h) <= CENTROID_AGREE_MM
        and centroid_dist_mm(c_v, c_h) <= CENTROID_AGREE_MM
    )

    save_montage(
        out_dir / "review_montage.png",
        study_id=study_id,
        t1c=t1c,
        enhance=enhance,
        flair=flair,
        intensity_label=classic,
        v3_label=v3_lab,
    )

    result = StudyIntensityResult(
        study_id=study_id,
        nawm_estimation=nawm_note,
        nawm_t1_median=med_t1,
        nawm_t1_mad=mad_t1,
        ncc_before=ncc_b,
        ncc_after=ncc_a,
        alignment_flag=align_flag,
        enhance_percentile=ENHANCE_PERCENTILE,
        enhance_threshold=thr,
        flair_z_threshold=FLAIR_Z_THRESH,
        csf_rule=csf_rule,
        tc_core_rule=tc_rule,
        candidates=comps,
        dice_vs_v3=dice_v3,
        dice_vs_hdglio=dice_hd,
        vol_intensity_ml=vol_i,
        vol_v3_ml=vol_v3,
        vol_hdglio_ml=vol_hd,
        main_et_centroid_mm=c_i,
        v3_et_centroid_mm=c_v,
        hdglio_et_centroid_mm=c_h,
        et_location_agree_all_three=agree,
        grid_match_ok=grid_ok,
        notes=notes,
    )

    meta = {
        "disclaimer": DRAFT_DISCLAIMER,
        "nawm_estimation": nawm_note,
        "nawm_t1_median": med_t1,
        "nawm_t1_mad": mad_t1,
        "nawm_t1c_median": med_t1c,
        "nawm_t1c_mad": mad_t1c,
        "ncc_before": ncc_b,
        "ncc_after": ncc_a,
        "alignment_flag": align_flag,
        "enhancement": enh_meta,
        "flair_z_threshold": FLAIR_Z_THRESH,
        "csf_rule": csf_rule,
        "tc_core_rule": tc_rule,
        "candidates": [asdict(c) for c in comps],
        "dice_vs_v3": dice_v3,
        "dice_vs_hdglio": dice_hd,
        "volumes_ml": {"intensity": vol_i, "v3": vol_v3, "hdglio": vol_hd},
        "et_centroids_mm": {
            "intensity": c_i,
            "v3": c_v,
            "hdglio": c_h,
        },
        "et_location_agree_all_three": agree,
        "grid_match_ok": grid_ok,
        "notes": notes,
    }
    (out_dir / "study_meta.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        f"# {study_id} intensity draft\n\n{DRAFT_DISCLAIMER}\n\n"
        f"Open with `./open_itksnap.sh`.\n",
        encoding="utf-8",
    )
    logger.info(
        "%s | NCC %.3f→%.3f %s | ET cand=%d | agree3=%s | grid=%s",
        study_id,
        ncc_b,
        ncc_a,
        align_flag,
        len(comps),
        agree,
        grid_ok,
    )
    return result


def write_summary(results: list[StudyIntensityResult], out_root: Path) -> Path:
    lines = [
        "# Batch2 intensity-based drafts (native T1c)",
        "",
        f"**{DRAFT_DISCLAIMER}**",
        "",
        "## Methods (fixed rules)",
        "",
        "- **NAWM z-score**: histogram mode of T1 inside HD-BET brain mask; "
        "NAWM proxy = voxels within ±0.25·IQR of that mode; per-modality median/MAD "
        "on that proxy (MAD floored to max(1%·|median|, 1)); robust z = "
        "(x − median) / (1.4826·MAD).",
        "- **T1c→T1 rigid refine**: Mattes MI, metric masked to brain; NCC before/after "
        f"(different contrasts — absolute NCC often moderate); flag if NCC after "
        f"< {NCC_POOR_ABS} or drops by > {NCC_DROP_FLAG}.",
        f"- **Enhancement**: z(T1c)−z(T1), Gaussian σ≤{GAUSS_SIGMA_MM} mm; "
        f"threshold = brain **percentile {ENHANCE_PERCENTILE}** "
        f"(top {100 - ENHANCE_PERCENTILE:.2f}%); Frangi vesselness + border + "
        f"<{MIN_ENHANCE_ML} mL removed; keep top {TOP_K_ENHANCE} components.",
        f"- **FLAIR**: brain z > **{FLAIR_Z_THRESH}**; CSF = low T1 & T2-bright; "
        f"ventricular rim dilated from dark-T1 CSF; keep components > {MIN_FLAIR_ML} mL.",
        "- **Nested BraTS draft**: ET = enhancing candidates; TC = ET ∪ adjacent "
        "T1-dark / T2-bright core; WT = FLAIR touching/containing TC ∪ TC "
        "(labels 1 / 2 / 4 on the study T1c grid).",
        "",
        "## Per-study alignment & enhancement",
        "",
        "| study | NCC before | NCC after | flag | enh thr (p99.5) | n cand | agree ET loc |",
        "|---|---:|---:|---|---:|---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.study_id} | {r.ncc_before:.3f} | {r.ncc_after:.3f} | "
            f"{r.alignment_flag} | {r.enhance_threshold:.3f} | {len(r.candidates)} | "
            f"{'yes' if r.et_location_agree_all_three else 'no'} |"
        )

    lines += [
        "",
        "## Dice vs v3 / HD-GLIO drafts (not reference truth)",
        "",
        "| study | Dice ET v3 | TC v3 | WT v3 | Dice ET HD | TC HD | WT HD | "
        "vol ET int/v3/hd (mL) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.study_id} | "
            f"{r.dice_vs_v3.get('ET', float('nan')):.3f} | "
            f"{r.dice_vs_v3.get('TC', float('nan')):.3f} | "
            f"{r.dice_vs_v3.get('WT', float('nan')):.3f} | "
            f"{r.dice_vs_hdglio.get('ET', float('nan')):.3f} | "
            f"{r.dice_vs_hdglio.get('TC', float('nan')):.3f} | "
            f"{r.dice_vs_hdglio.get('WT', float('nan')):.3f} | "
            f"{r.vol_intensity_ml.get('ET', float('nan')):.2f} / "
            f"{r.vol_v3_ml.get('ET', float('nan')):.2f} / "
            f"{r.vol_hdglio_ml.get('ET', float('nan')):.2f} |"
        )

    agree = [r.study_id for r in results if r.et_location_agree_all_three]
    lines += [
        "",
        f"## Main enhancing-component location agreement (centroids within {CENTROID_AGREE_MM:.0f} mm)",
        "",
        f"Studies where intensity, v3 and HD-GLIO main ET centroids all agree: "
        f"**{', '.join(agree) if agree else '(none)'}**",
        "",
        "## Candidate components (intensity draft)",
        "",
    ]
    for r in results:
        lines.append(f"### {r.study_id}")
        lines.append("")
        lines.append(f"- NAWM: {r.nawm_estimation}")
        lines.append(f"- CSF rule: {r.csf_rule}")
        lines.append(f"- TC rule: {r.tc_core_rule}")
        if not r.candidates:
            lines.append("- No enhancement candidates retained.")
        else:
            lines.append("| # | vol mL | centroid mm | mean enh z |")
            lines.append("|---:|---:|---|---:|")
            for c in r.candidates:
                lines.append(
                    f"| {c.label} | {c.volume_ml:.2f} | "
                    f"({c.centroid_mm[0]:.1f}, {c.centroid_mm[1]:.1f}, {c.centroid_mm[2]:.1f}) | "
                    f"{c.mean_enhancement:.3f} |"
                )
        lines.append("")

    lines += [
        "## Outputs",
        "",
        "Per study under `outputs/batch2_intensity_draft/<study_id>/`:",
        "- `intensity_draft_label.nii.gz` (BraTS 1/2/4 on T1c grid)",
        "- `enhancement_map.nii.gz`, `review_montage.png`, `study_meta.json`",
        "- `open_itksnap.sh` — loads T1c, FLAIR, enhancement map, and this draft",
        "",
        f"**{DRAFT_DISCLAIMER}**",
        "",
    ]
    path = out_root / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for r in results:
        rows.append(
            {
                "study_id": r.study_id,
                "ncc_before": r.ncc_before,
                "ncc_after": r.ncc_after,
                "alignment_flag": r.alignment_flag,
                "enhance_threshold": r.enhance_threshold,
                "n_candidates": len(r.candidates),
                "et_agree_all_three": r.et_location_agree_all_three,
                "dice_ET_v3": r.dice_vs_v3.get("ET"),
                "dice_WT_v3": r.dice_vs_v3.get("WT"),
                "dice_ET_hdglio": r.dice_vs_hdglio.get("ET"),
                "dice_WT_hdglio": r.dice_vs_hdglio.get("WT"),
                "grid_match_ok": r.grid_match_ok,
            }
        )
    pd.DataFrame(rows).to_csv(out_root / "summary_table.csv", index=False)
    return path


def run_all(
    study_ids: list[str] | None = None,
    out_root: Path | None = None,
) -> list[StudyIntensityResult]:
    out_root = Path(out_root or OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    ids = list(study_ids or NATIVE_T1C_STUDIES)
    results = []
    for sid in ids:
        logger.info("=== intensity draft %s ===", sid)
        results.append(process_study(sid, out_root=out_root))
    write_summary(results, out_root)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classical intensity enhancement/FLAIR drafts for native-T1c batch2."
    )
    parser.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help="Subset of study ids (default: all 10 native-T1c)",
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_all(study_ids=args.studies, out_root=args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
