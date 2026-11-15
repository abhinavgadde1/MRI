"""Register patient T1 → MNI152 (affine + mild B-spline) and warp atlases back."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from location.qc import (
    BRAIN_DICE_GOOD,
    COVERAGE_GOOD,
    dice_bool,
    flag_registration_quality,
    fov_coverage_fraction,
    surface_distance_mm,
)

logger = logging.getLogger(__name__)

# Back-compat alias used by tests / callers.
BRAIN_DICE_MIN = BRAIN_DICE_GOOD

TUMOR_EXCLUDE_DILATE_MM = 5.0
# Dilate patient brain this many voxels when building the FOV support for Dice
# so BraTS-sized crops are not scored against uncovered MNI template corners.
BRAIN_FOV_DILATE_VOX = 5


@dataclass
class RegistrationResult:
    """Outputs of patient→MNI registration."""

    reg_dir: Path
    affine_tfm: Path
    bspline_tfm: Path | None
    composite_tfm: Path  # manifest JSON listing component transforms
    transform: sitk.Transform  # in-memory fixed(MNI)→moving(patient)
    t1_in_mni: Path
    mi_before: float
    mi_after: float
    brain_dice: float
    quality: str  # good | caution | registration unreliable (finalized after ventricle QC)
    quality_note: str
    cost_mask_path: Path | None = None
    stage: str = "affine"
    surface_mean_mm: float = float("nan")
    surface_p95_mm: float = float("nan")
    coverage: float = float("nan")
    coverage_partial: bool = False
    ventricle_dice: float | None = None
    ventricle_source: str | None = None
    patient_brain_xyz: np.ndarray | None = None
    template_brain_xyz: np.ndarray | None = None
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reg_dir": str(self.reg_dir),
            "affine_tfm": str(self.affine_tfm),
            "bspline_tfm": str(self.bspline_tfm) if self.bspline_tfm else None,
            "composite_tfm": str(self.composite_tfm),
            "t1_in_mni": str(self.t1_in_mni),
            "mi_before": float(self.mi_before),
            "mi_after": float(self.mi_after),
            "brain_dice": float(self.brain_dice),
            "ventricle_dice": self.ventricle_dice,
            "surface_mean_mm": float(self.surface_mean_mm),
            "surface_p95_mm": float(self.surface_p95_mm),
            "coverage": float(self.coverage),
            "coverage_partial": bool(self.coverage_partial),
            "ventricle_source": self.ventricle_source,
            "stage": self.stage,
            "quality": self.quality,
            "quality_note": self.quality_note,
            "cost_mask": str(self.cost_mask_path) if self.cost_mask_path else None,
        }


def estimate_brain_mask(t1: np.ndarray | sitk.Image) -> np.ndarray:
    """Robust brain mask for registration cost / Dice (HD-BET and BraTS).

    - Zero-background skull-strips (HD-BET): nonzero voxels.
    - Dense / z-scored FOVs (BraTS): positive tissue (or Otsu fallback).
    Always hole-fills and keeps the largest connected component so noise
    speckles and ventricles-as-holes do not distort Dice or the cost mask.
    Returns a boolean array in (X, Y, Z) order matching ``_sitk_to_numpy_xyz``.
    """
    from scipy import ndimage as ndi

    if isinstance(t1, sitk.Image):
        t1_np = _sitk_to_numpy_xyz(t1).astype(np.float32)
        sitk_img: sitk.Image | None = t1
    else:
        t1_np = np.asarray(t1, dtype=np.float32)
        sitk_img = None

    nz = t1_np != 0
    if float(np.mean(nz)) < 0.5:
        # Classic skull-stripped (HD-BET): zeros outside brain.
        brain = nz
    else:
        # Dense FOV (z-scored BraTS, synthetic filled volumes): positive tissue.
        pos = t1_np > 0
        if float(pos.mean()) >= 0.05:
            brain = pos
        elif sitk_img is not None:
            otsu = sitk.OtsuThreshold(sitk_img, 0, 1)
            brain = _sitk_to_numpy_xyz(otsu) > 0
            if float(brain.mean()) > 0.7:
                brain = ~brain
        else:
            thr = float(np.percentile(t1_np, 75))
            brain = t1_np > thr

    brain = ndi.binary_fill_holes(np.asarray(brain, dtype=bool))
    lab, n = ndi.label(brain)
    if n:
        sizes = ndi.sum(brain, lab, index=np.arange(1, n + 1))
        brain = lab == (int(np.argmax(sizes)) + 1)
    brain = ndi.binary_fill_holes(ndi.binary_closing(brain, iterations=1))
    return brain


def build_registration_cost_mask(
    t1: np.ndarray | sitk.Image,
    tumor_mask: np.ndarray | None,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    dilate_mm: float = TUMOR_EXCLUDE_DILATE_MM,
) -> np.ndarray:
    """Brain tissue minus dilated whole-tumor — used as registration metric mask.

    Tumor voxels (and a ``dilate_mm`` neighbourhood) are excluded so large
    lesions do not pull the template. Returns a boolean array in patient space.
    """
    if isinstance(t1, sitk.Image):
        sp = tuple(float(s) for s in t1.GetSpacing())
        spacing_mm = (sp[0], sp[1], sp[2])

    brain = estimate_brain_mask(t1)
    cost = brain.copy()
    if tumor_mask is not None and np.any(tumor_mask):
        tumor = np.asarray(tumor_mask, dtype=bool)
        if tumor.shape != cost.shape:
            raise ValueError(
                f"Tumor mask shape {tumor.shape} != T1 shape {cost.shape}"
            )
        # Dilate by ~dilate_mm using EDT (spacing-aware).
        dist = ndi_distance_to_tumor(tumor, spacing_mm)
        exclude = dist <= float(dilate_mm)
        cost = cost & ~exclude
    return cost


def ndi_distance_to_tumor(
    tumor: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.distance_transform_edt(~np.asarray(tumor, dtype=bool), sampling=spacing_mm)


def _brain_fov_mask(brain: np.ndarray, dilate_vox: int = BRAIN_FOV_DILATE_VOX) -> np.ndarray:
    """FOV support for Dice: dilated brain, not the full rectangular grid."""
    from scipy import ndimage as ndi

    if dilate_vox <= 0:
        return np.asarray(brain, dtype=bool)
    return ndi.binary_dilation(np.asarray(brain, dtype=bool), iterations=int(dilate_vox))


def _dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    return dice_bool(a, b)


def _numpy_to_sitk_mask(mask_xyz: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Write an (X,Y,Z) boolean mask as a SimpleITK image matching ``reference``."""
    # SimpleITK arrays are (Z,Y,X).
    arr_zyx = np.transpose(np.asarray(mask_xyz, dtype=np.uint8), (2, 1, 0))
    img = sitk.GetImageFromArray(arr_zyx)
    img.CopyInformation(reference)
    return sitk.Cast(img, sitk.sitkUInt8)


def _sitk_to_numpy_xyz(img: sitk.Image) -> np.ndarray:
    return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))


def load_t1_to_mni_transform(reg_dir: str | Path) -> sitk.Transform:
    """Rebuild patient→MNI from saved affine (+ optional B-spline) files."""
    reg_dir = Path(reg_dir)
    manifest = reg_dir / "t1_to_mni_composite.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        affine = sitk.ReadTransform(str(reg_dir / meta["affine"]))
        bspline_name = meta.get("bspline")
        if bspline_name:
            bspline = sitk.ReadTransform(str(reg_dir / bspline_name))
            composite = sitk.CompositeTransform(3)
            composite.AddTransform(affine)
            composite.AddTransform(bspline)
            return composite
        return affine
    affine_path = reg_dir / "t1_to_mni_affine.tfm"
    return sitk.ReadTransform(str(affine_path))


def _mattes_mi(fixed: sitk.Image, moving: sitk.Image, transform: sitk.Transform) -> float:
    """Mattes mutual information (higher / less-negative is better alignment)."""
    metric = sitk.ImageRegistrationMethod()
    metric.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    metric.SetMetricSamplingStrategy(metric.RANDOM)
    metric.SetMetricSamplingPercentage(0.2, seed=42)
    metric.SetInterpolator(sitk.sitkLinear)
    metric.SetInitialTransform(transform, inPlace=False)
    # GetMetricValue requires Execute setup; use MetricEvaluate instead.
    try:
        return float(metric.MetricEvaluate(fixed, moving))
    except Exception:  # noqa: BLE001
        # Fallback: resample then histogram MI approximation via sitk filter.
        resampled = sitk.Resample(
            moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.2, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetInitialTransform(sitk.Transform(), inPlace=False)
        return float(reg.MetricEvaluate(fixed, resampled))


def register_t1_to_mni(
    t1_path: str | Path,
    mni_path: str | Path,
    out_dir: str | Path,
    *,
    tumor_mask: np.ndarray | None = None,
    affine_iterations: int = 200,
    bspline_iterations: int = 50,
    bspline_mesh: int = 6,
    skip_bspline: bool = False,
    dice_min: float = BRAIN_DICE_MIN,
) -> RegistrationResult:
    """Affine then mild B-spline: moving patient T1 → fixed MNI152.

    When ``tumor_mask`` is provided, a cost-function mask excludes the tumor
    dilated by 5 mm so large lesions do not dominate MI. Saves transforms under
    ``out_dir`` (typically ``outputs/<case>/registration/``).
    """
    t1_path = Path(t1_path)
    mni_path = Path(mni_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_raw = sitk.Cast(sitk.ReadImage(str(mni_path)), sitk.sitkFloat32)
    moving_raw = sitk.Cast(sitk.ReadImage(str(t1_path)), sitk.sitkFloat32)

    fixed = sitk.Normalize(fixed_raw)
    moving = sitk.Normalize(moving_raw)

    # Cost-function mask on the moving (patient) image.
    sp = tuple(float(s) for s in moving_raw.GetSpacing())
    spacing_mm = (sp[0], sp[1], sp[2])
    cost_np = build_registration_cost_mask(
        moving_raw, tumor_mask, spacing_mm=spacing_mm
    )
    # If tumor exclusion wiped the mask (tiny FOV / huge lesion), fall back to brain.
    if int(cost_np.sum()) < 1000:
        logger.warning(
            "Cost mask too small (%d voxels); using full brain mask.",
            int(cost_np.sum()),
        )
        cost_np = build_registration_cost_mask(
            moving_raw, tumor_mask=None, spacing_mm=spacing_mm
        )
    cost_mask_path = out_dir / "registration_cost_mask.nii.gz"
    cost_sitk = _numpy_to_sitk_mask(cost_np, moving_raw)
    sitk.WriteImage(cost_sitk, str(cost_mask_path))

    # Template + patient brain masks (hole-filled LCC) for metric + Dice.
    fixed_brain = estimate_brain_mask(fixed_raw)
    fixed_brain_sitk = _numpy_to_sitk_mask(fixed_brain, fixed_raw)
    patient_brain = estimate_brain_mask(moving_raw)
    patient_fov = _brain_fov_mask(patient_brain)
    patient_brain_sitk = _numpy_to_sitk_mask(patient_brain, moving_raw)
    patient_fov_sitk = _numpy_to_sitk_mask(patient_fov, moving_raw)
    sitk.WriteImage(patient_brain_sitk, str(out_dir / "patient_brain_mask.nii.gz"))

    identity = sitk.Transform()
    mi_before = _mattes_mi(fixed, moving, identity)

    def _configure(reg: sitk.ImageRegistrationMethod, *, sampling: float) -> None:
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(sampling, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)
        # Moving mask: brain minus dilated tumor. Fixed mask: template brain.
        reg.SetMetricMovingMask(cost_sitk)
        reg.SetMetricFixedMask(fixed_brain_sitk)

    # Full rectangular FOV of the patient scan (for coverage), plus brain FOV for Dice.
    patient_scan_fov = np.ones(patient_brain.shape, dtype=bool)
    patient_scan_fov_sitk = _numpy_to_sitk_mask(patient_scan_fov, moving_raw)

    def _warp_mask(mask_sitk: sitk.Image, tfm: sitk.Transform) -> np.ndarray:
        out = sitk.Resample(
            mask_sitk,
            fixed_raw,
            tfm,
            sitk.sitkNearestNeighbor,
            0.0,
            sitk.sitkUInt8,
        )
        return _sitk_to_numpy_xyz(out) > 0

    def _brain_dice_for(tfm: sitk.Transform) -> float:
        """FOV-aware brain Dice (reported for QC; NOT used to pick the transform)."""
        warped_brain = _warp_mask(patient_brain_sitk, tfm)
        warped_fov = _warp_mask(patient_fov_sitk, tfm)
        template_in_fov = fixed_brain & warped_fov
        return _dice_bool(warped_brain & warped_fov, template_in_fov)

    def _mi_for(tfm: sitk.Transform) -> float:
        return _mattes_mi(fixed, moving, tfm)

    # --- Affine ---
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    _configure(reg, sampling=0.25)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=int(affine_iterations),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    affine_path = out_dir / "t1_to_mni_affine.tfm"
    bspline_path: Path | None = None
    try:
        affine = reg.Execute(fixed, moving)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Affine registration failed (%s); using centered init.", exc)
        affine = initial
        skip_bspline = True

    sitk.WriteTransform(affine, str(affine_path))

    # Candidates selected by MI only (Mattes: more negative = better).
    candidates: list[tuple[str, sitk.Transform]] = [
        ("centered_init", initial),
        ("affine", affine),
    ]
    bspline_out: sitk.Transform | None = None
    if not skip_bspline:
        mesh = [int(bspline_mesh)] * 3
        bspline = sitk.BSplineTransformInitializer(fixed, mesh)
        reg2 = sitk.ImageRegistrationMethod()
        _configure(reg2, sampling=0.2)
        reg2.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-4,
            numberOfIterations=int(bspline_iterations),
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=200,
        )
        reg2.SetMovingInitialTransform(affine)
        reg2.SetInitialTransform(bspline, inPlace=False)
        reg2.SetShrinkFactorsPerLevel([2, 1])
        reg2.SetSmoothingSigmasPerLevel([1, 0])
        reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        try:
            bspline_out = reg2.Execute(fixed, moving)
            composite_bs = sitk.CompositeTransform(3)
            composite_bs.AddTransform(affine)
            composite_bs.AddTransform(bspline_out)
            candidates.append(("bspline", composite_bs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("B-spline stage failed (%s); skipping.", exc)
            bspline_out = None

    scored: list[tuple[str, sitk.Transform, float]] = []
    for name, tfm in candidates:
        try:
            scored.append((name, tfm, _mi_for(tfm)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MI eval failed for %s (%s); skipping candidate.", name, exc)
    if not scored:
        scored = [("centered_init", initial, mi_before)]
    # Best = lowest Mattes MI value (most negative).
    scored.sort(key=lambda t: t[2])
    used_stage, composite, mi_after = scored[0]
    mi_table = {name: mi for name, _, mi in scored}

    if used_stage == "bspline" and bspline_out is not None:
        bspline_path = out_dir / "t1_to_mni_bspline.tfm"
        sitk.WriteTransform(bspline_out, str(bspline_path))
        # Persist the affine that B-spline was composed with.
        sitk.WriteTransform(affine, str(affine_path))
    elif used_stage == "centered_init":
        sitk.WriteTransform(initial, str(affine_path))
    else:
        sitk.WriteTransform(affine, str(affine_path))

    composite_path = out_dir / "t1_to_mni_composite.json"
    composite_path.write_text(
        json.dumps(
            {
                "affine": affine_path.name,
                "bspline": bspline_path.name if bspline_path is not None else None,
                "stage": used_stage,
                "mi_by_stage": mi_table,
                "selection": "mattes_mi",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    t1_in_mni = out_dir / "t1_in_mni.nii.gz"
    resampled = sitk.Resample(
        moving_raw, fixed_raw, composite, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    sitk.WriteImage(resampled, str(t1_in_mni))

    # --- Independent QC (not used for selection) ---
    warped_brain = _warp_mask(patient_brain_sitk, composite)
    warped_brain_fov = _warp_mask(patient_fov_sitk, composite)
    warped_scan_fov = _warp_mask(patient_scan_fov_sitk, composite)
    template_in_fov = fixed_brain & warped_brain_fov
    brain_dice = _dice_bool(warped_brain & warped_brain_fov, template_in_fov)
    coverage = fov_coverage_fraction(fixed_brain, warped_scan_fov)
    coverage_partial = coverage < COVERAGE_GOOD
    # Template spacing is 1 mm isotropic for MNI152 1mm.
    mni_sp = tuple(float(s) for s in fixed_raw.GetSpacing())
    # Surface distance inside the patient FOV so missing cerebellum does not
    # inflate p95; FOV coverage is reported separately.
    template_in_scan = fixed_brain & warped_scan_fov
    surf_mean, surf_p95 = surface_distance_mm(
        warped_brain, template_in_scan, (mni_sp[0], mni_sp[1], mni_sp[2])
    )

    # Provisional quality (ventricle Dice filled in by report after HO warp).
    quality, note = flag_registration_quality(
        brain_dice=brain_dice,
        ventricle_dice=None,
        surface_p95_mm=surf_p95,
        coverage=coverage,
        stage=used_stage,
    )
    # Without ventricle yet, do not call "good" — report will upgrade/downgrade.
    if quality == "good":
        quality, note = (
            "caution",
            f"Pending ventricle QC. {note}",
        )

    _save_template_outline_overlay(
        resampled,
        fixed_brain,
        out_dir / "template_brain_outline_qc.png",
        title=(
            f"Template brain outline · Dice={brain_dice:.3f} · "
            f"cov={coverage:.2f} · {used_stage}"
        ),
    )

    meta = {
        "mi_before": mi_before,
        "mi_after": mi_after,
        "mi_by_stage": mi_table,
        "brain_dice": brain_dice,
        "brain_dice_min": float(dice_min),
        "surface_mean_mm": surf_mean,
        "surface_p95_mm": surf_p95,
        "coverage": coverage,
        "coverage_partial": coverage_partial,
        "quality": quality,
        "quality_note": note,
        "affine_iterations": affine_iterations,
        "bspline": bspline_path is not None,
        "stage": used_stage,
        "selection": "mattes_mi",
        "fov_dice": "dilated_brain",
        "tumor_excluded_dilate_mm": TUMOR_EXCLUDE_DILATE_MM,
        "composite_manifest": composite_path.name,
    }
    (out_dir / "registration_qc.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return RegistrationResult(
        reg_dir=out_dir,
        affine_tfm=affine_path,
        bspline_tfm=bspline_path,
        composite_tfm=composite_path,
        transform=composite,
        t1_in_mni=t1_in_mni,
        mi_before=mi_before,
        mi_after=mi_after,
        brain_dice=brain_dice,
        quality=quality,
        quality_note=note,
        cost_mask_path=cost_mask_path,
        stage=used_stage,
        surface_mean_mm=surf_mean,
        surface_p95_mm=surf_p95,
        coverage=coverage,
        coverage_partial=coverage_partial,
        patient_brain_xyz=patient_brain,
        template_brain_xyz=fixed_brain,
        spacing_mm=spacing_mm,
    )


def _save_template_outline_overlay(
    t1_mni: sitk.Image,
    template_brain_xyz: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    """Side-by-side axial/coronal/sagittal: template brain outline on T1-in-MNI."""
    import matplotlib.pyplot as plt

    t1 = _sitk_to_numpy_xyz(t1_mni).astype(np.float32)
    brain = np.asarray(template_brain_xyz, dtype=bool)
    if np.any(t1 != 0):
        com = np.array(np.where(t1 != 0)).mean(axis=1).astype(int)
    else:
        com = np.array(t1.shape) // 2
    planes = [
        ("axial", 2, int(np.clip(com[2], 0, t1.shape[2] - 1))),
        ("coronal", 1, int(np.clip(com[1], 0, t1.shape[1] - 1))),
        ("sagittal", 0, int(np.clip(com[0], 0, t1.shape[0] - 1))),
    ]

    def _plane(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
        if axis == 0:
            return vol[index]
        if axis == 1:
            return vol[:, index]
        return vol[:, :, index]

    def _norm(sl: np.ndarray) -> np.ndarray:
        pos = sl[sl > 0]
        if pos.size == 0:
            return np.zeros_like(sl)
        p2, p98 = np.percentile(pos, (2, 98))
        if p98 <= p2:
            return np.zeros_like(sl)
        return np.clip((sl - p2) / (p98 - p2), 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, axis, idx) in zip(axs, planes):
        img = _norm(_plane(t1, axis, idx))
        lab = _plane(brain, axis, idx)
        ax.imshow(img.T, cmap="gray", origin="lower")
        if np.any(lab):
            ax.contour(lab.T.astype(float), levels=[0.5], colors="#00e5ff", linewidths=0.8)
        ax.set_title(f"{name} {idx}")
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _invert_transform(transform: sitk.Transform) -> sitk.Transform:
    """Invert a registration transform when an analytic inverse exists."""
    try:
        return transform.GetInverse()
    except Exception:
        pass
    if isinstance(transform, sitk.CompositeTransform):
        inverse = sitk.CompositeTransform(3)
        for i in range(transform.GetNumberOfTransforms() - 1, -1, -1):
            inverse.AddTransform(transform.GetNthTransform(i).GetInverse())
        return inverse
    clone = sitk.Transform(transform)
    if clone.SetInverse():
        return clone
    raise RuntimeError("Cannot invert registration transform")


def patient_to_mni_for_atlas_warp(
    reg_dir: str | Path | None = None,
    *,
    affine_tfm: str | Path | sitk.Transform | None = None,
    full_transform: sitk.Transform | None = None,
) -> sitk.Transform:
    """Patient→MNI transform for NN atlas warps.

    Prefers an analytic inverse of the full transform; B-spline composites are
    not analytically invertible in SimpleITK, so we fall back to the **affine
    inverse** (labels still use the robust linear mapping; the mild B-spline
    remains available for intensity QC via ``t1_in_mni.nii.gz``).
    """
    if full_transform is not None:
        try:
            return _invert_transform(full_transform)
        except Exception:
            logger.info(
                "Full transform not invertible (likely B-spline); "
                "atlas warp uses affine inverse."
            )
    if affine_tfm is None and reg_dir is not None:
        affine_tfm = Path(reg_dir) / "t1_to_mni_affine.tfm"
    if affine_tfm is None:
        raise ValueError("Need affine_tfm or reg_dir for atlas warp inverse")
    if isinstance(affine_tfm, sitk.Transform):
        return _invert_transform(affine_tfm)
    return _invert_transform(sitk.ReadTransform(str(affine_tfm)))


def warp_label_to_patient(
    atlas_path: str | Path,
    patient_t1_path: str | Path,
    t1_to_mni_tfm: str | Path | sitk.Transform,
    output_path: str | Path,
    *,
    interpolator: int = sitk.sitkNearestNeighbor,
    affine_tfm: str | Path | sitk.Transform | None = None,
) -> Path:
    """Bring an MNI-space label map onto the patient T1 grid (NN default).

    SimpleITK registration optimizes a transform that maps **fixed (MNI) →
    moving (patient)**. ``Resample`` maps reference coords → moving coords, so
    atlas→patient uses the **inverse** (patient → MNI).
    """
    atlas_path = Path(atlas_path)
    patient_t1_path = Path(patient_t1_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference = sitk.ReadImage(str(patient_t1_path))
    moving = sitk.ReadImage(str(atlas_path))
    if isinstance(t1_to_mni_tfm, (str, Path)):
        path = Path(t1_to_mni_tfm)
        if path.suffix == ".json":
            forward = load_t1_to_mni_transform(path.parent)
            if affine_tfm is None:
                affine_tfm = path.parent / "t1_to_mni_affine.tfm"
        else:
            forward = sitk.ReadTransform(str(path))
    else:
        forward = t1_to_mni_tfm

    patient_to_mni = patient_to_mni_for_atlas_warp(
        affine_tfm=affine_tfm, full_transform=forward
    )

    warped = sitk.Resample(
        moving,
        reference,
        patient_to_mni,
        interpolator,
        0.0,
        moving.GetPixelID(),
    )
    sitk.WriteImage(warped, str(output_path))
    return output_path


def warp_array_to_patient(
    array_mni: np.ndarray,
    mni_affine: np.ndarray,
    patient_t1_path: str | Path,
    t1_to_mni_tfm: str | Path | sitk.Transform,
    *,
    nearest: bool = True,
    affine_tfm: str | Path | sitk.Transform | None = None,
) -> np.ndarray:
    """Warp a numpy label/mask defined on the MNI grid into patient space."""
    tmp_dir = Path(patient_t1_path).parent
    tmp_atlas = tmp_dir / "_tmp_mni_label.nii.gz"
    tmp_out = tmp_dir / "_tmp_patient_label.nii.gz"
    img = nib.Nifti1Image(np.asarray(array_mni), np.asarray(mni_affine))
    nib.save(img, str(tmp_atlas))
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    warp_label_to_patient(
        tmp_atlas,
        patient_t1_path,
        t1_to_mni_tfm,
        tmp_out,
        interpolator=interp,
        affine_tfm=affine_tfm,
    )
    out = np.asanyarray(nib.load(str(tmp_out)).dataobj)
    tmp_atlas.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return out
