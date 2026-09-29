"""Rigid registration, 1 mm isotropic resampling, and full preprocessing chain."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import SimpleITK as sitk

from preprocessing.bias_correction import n4_bias_correct
from preprocessing.dicom_loader import ConvertedSeries, load_study_to_nifti
from preprocessing.skull_strip import skull_strip

logger = logging.getLogger(__name__)

TARGET_MODALITIES = ("T1", "T1c", "T2", "FLAIR")
MovingModality = Literal["T1c", "T2", "FLAIR"]

REGISTERED_NAMES: dict[str, str] = {
    "FLAIR": "FLAIR_registered_1mm.nii.gz",
    "T1": "T1_1mm.nii.gz",
    "T1c": "T1c_registered_1mm.nii.gz",
    "T2": "T2_registered_1mm.nii.gz",
}


@dataclass
class RegistrationResult:
    """Paths produced by registering one moving volume to a fixed reference."""

    modality: str
    registered_path: Path
    transform_path: Path | None = None


@dataclass
class PatientPreprocessResult:
    """Artifact paths from the full per-patient preprocessing chain."""

    study_name: str
    output_dir: Path
    dicom_nifti: dict[str, Path] = field(default_factory=dict)
    bias_corrected: dict[str, Path] = field(default_factory=dict)
    skull_stripped: dict[str, Path] = field(default_factory=dict)
    registered_1mm: dict[str, Path] = field(default_factory=dict)
    transforms: dict[str, Path] = field(default_factory=dict)
    reference: str | None = None
    manifest_path: Path | None = None


def register_to_reference(
    moving_path: str | Path,
    fixed_path: str | Path,
    output_path: str | Path,
    *,
    transform_path: str | Path | None = None,
    use_affine: bool = False,
) -> Path:
    """Register ``moving`` to ``fixed`` (rigid by default) and write the result."""
    registered, _transform = rigid_register(
        moving_path,
        fixed_path,
        output_path,
        transform_path=transform_path,
        use_affine=use_affine,
    )
    return registered


def rigid_register(
    moving_path: str | Path,
    fixed_path: str | Path,
    output_path: str | Path,
    *,
    transform_path: str | Path | None = None,
    use_affine: bool = False,
    number_of_iterations: int = 200,
) -> tuple[Path, sitk.Transform]:
    """Rigid (or optional affine) registration with Mattes mutual information."""
    moving_path = Path(moving_path)
    fixed_path = Path(fixed_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixed = sitk.Cast(sitk.ReadImage(str(fixed_path)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(moving_path)), sitk.sitkFloat32)

    transform_type: sitk.Transform = (
        sitk.AffineTransform(3) if use_affine else sitk.Euler3DTransform()
    )
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform_type,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.25, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=number_of_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    transform = registration.Execute(fixed, moving)
    logger.info(
        "Registered %s -> %s | metric=%.6f | stop=%s",
        moving_path.name,
        fixed_path.name,
        registration.GetMetricValue(),
        registration.GetOptimizerStopConditionDescription(),
    )

    resampled = sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    sitk.WriteImage(resampled, str(output_path))

    if transform_path is not None:
        transform_path = Path(transform_path)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, str(transform_path))

    return output_path, transform


def resample_isotropic(
    input_path: str | Path,
    output_path: str | Path,
    *,
    spacing_mm: float = 1.0,
    reference_path: str | Path | None = None,
    default_value: float = 0.0,
    interpolator: int = sitk.sitkLinear,
) -> Path:
    """Resample a volume to isotropic spacing (default 1 mm)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(input_path))
    reference = (
        sitk.ReadImage(str(reference_path)) if reference_path is not None else image
    )

    target_spacing = (float(spacing_mm),) * reference.GetDimension()
    iso_reference = _isotropic_grid(reference, target_spacing)
    resampled = sitk.Resample(
        image,
        iso_reference,
        sitk.Transform(),
        interpolator,
        default_value,
        sitk.sitkFloat32,
    )
    sitk.WriteImage(resampled, str(output_path))
    logger.info(
        "Resampled %s -> %s mm isotropic (%s)",
        input_path.name,
        spacing_mm,
        tuple(round(s, 3) for s in resampled.GetSpacing()),
    )
    return output_path


def _isotropic_grid(reference: sitk.Image, spacing: Sequence[float]) -> sitk.Image:
    """Build an empty image covering ``reference``'s FOV at ``spacing``."""
    old_spacing = reference.GetSpacing()
    old_size = reference.GetSize()
    new_size = [
        max(1, int(round(old_size[i] * old_spacing[i] / spacing[i])))
        for i in range(reference.GetDimension())
    ]
    grid = sitk.Image(new_size, sitk.sitkFloat32)
    grid.SetSpacing(tuple(spacing))
    grid.SetOrigin(reference.GetOrigin())
    grid.SetDirection(reference.GetDirection())
    return grid


def register_modalities_to_reference(
    modality_paths: dict[str, Path],
    output_dir: str | Path,
    *,
    atlas_path: str | Path | None = None,
    spacing_mm: float = 1.0,
) -> dict[str, Path]:
    """Rigid-register T1c/T2/FLAIR to T1 (or all modalities to an atlas).

    Writes 1 mm isotropic volumes using :data:`REGISTERED_NAMES`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir = output_dir / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    if atlas_path is not None:
        fixed_src = Path(atlas_path)
        reference_name = "atlas"
        if not fixed_src.is_file():
            raise FileNotFoundError(f"Atlas not found: {fixed_src}")
        fixed_1mm = output_dir / f"{reference_name}_1mm.nii.gz"
    else:
        if "T1" not in modality_paths:
            raise FileNotFoundError(
                "T1 volume is required for registration when no atlas is provided. "
                f"Available: {sorted(modality_paths)}"
            )
        fixed_src = Path(modality_paths["T1"])
        reference_name = "T1"
        fixed_1mm = output_dir / REGISTERED_NAMES["T1"]

    resample_isotropic(fixed_src, fixed_1mm, spacing_mm=spacing_mm)

    results: dict[str, Path] = {}
    if atlas_path is None:
        results["T1"] = fixed_1mm
    elif "T1" in modality_paths:
        t1_reg = output_dir / REGISTERED_NAMES["T1"]
        t1_tfm = transforms_dir / "T1_to_atlas.tfm"
        rigid_register(modality_paths["T1"], fixed_1mm, t1_reg, transform_path=t1_tfm)
        results["T1"] = t1_reg

    for modality in ("T1c", "T2", "FLAIR"):
        if modality not in modality_paths:
            logger.warning("Skipping missing modality: %s", modality)
            continue
        out = output_dir / REGISTERED_NAMES[modality]
        tfm = transforms_dir / f"{modality}_to_{reference_name}.tfm"
        rigid_register(modality_paths[modality], fixed_1mm, out, transform_path=tfm)
        results[modality] = out

    return results


def run_patient_preprocessing(
    study_dicom_dir: str | Path,
    output_study_dir: str | Path,
    *,
    study_name: str | None = None,
    atlas_path: str | Path | None = None,
    spacing_mm: float = 1.0,
    use_hd_bet: bool = True,
) -> PatientPreprocessResult:
    """Full chain: DICOM → bias → skull → register → 1 mm.

    Intermediates under ``output_study_dir``::

        01_dicom_nifti/
        02_bias/
        03_skull/
        04_registered_1mm/
    """
    study_dicom_dir = Path(study_dicom_dir)
    output_study_dir = Path(output_study_dir)
    study_name = study_name or study_dicom_dir.name
    output_study_dir.mkdir(parents=True, exist_ok=True)

    result = PatientPreprocessResult(
        study_name=study_name,
        output_dir=output_study_dir,
        reference="atlas" if atlas_path else "T1",
    )

    dicom_out = output_study_dir / "01_dicom_nifti"
    converted = load_study_to_nifti(
        study_dicom_dir,
        dicom_out,
        study_name=study_name,
        modalities=list(TARGET_MODALITIES),
        include_other=False,
    )
    selected = _select_primary_series(converted)
    result.dicom_nifti = dict(selected)
    logger.info("DICOM conversion selected modalities: %s", sorted(selected))

    if not selected:
        raise RuntimeError(f"No T1/T1c/T2/FLAIR series found in {study_dicom_dir}")

    return run_nifti_preprocessing(
        selected,
        output_study_dir,
        study_name=study_name,
        atlas_path=atlas_path,
        spacing_mm=spacing_mm,
        use_hd_bet=use_hd_bet,
        existing=result,
    )


def run_nifti_preprocessing(
    modality_paths: dict[str, Path],
    output_dir: str | Path,
    *,
    study_name: str | None = None,
    atlas_path: str | Path | None = None,
    spacing_mm: float = 1.0,
    use_hd_bet: bool = True,
    existing: PatientPreprocessResult | None = None,
) -> PatientPreprocessResult:
    """Bias-correct → skull-strip → rigid-register → 1 mm resample from NIfTIs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = study_name or output_dir.name

    result = existing or PatientPreprocessResult(
        study_name=study_name,
        output_dir=output_dir,
        reference="atlas" if atlas_path else "T1",
    )
    result.study_name = study_name
    result.output_dir = output_dir
    result.reference = "atlas" if atlas_path else "T1"
    if not result.dicom_nifti:
        result.dicom_nifti = {m: Path(p) for m, p in modality_paths.items()}

    if not modality_paths:
        raise RuntimeError("No modality NIfTI paths provided for preprocessing")

    bias_dir = output_dir / "02_bias"
    bias_dir.mkdir(parents=True, exist_ok=True)
    for modality, nifti in modality_paths.items():
        out = bias_dir / f"{modality}.nii.gz"
        n4_bias_correct(nifti, out)
        result.bias_corrected[modality] = out

    strip_dir = output_dir / "03_skull"
    strip_dir.mkdir(parents=True, exist_ok=True)
    for modality, nifti in result.bias_corrected.items():
        out = strip_dir / f"{modality}.nii.gz"
        mask = strip_dir / f"{modality}_brainmask.nii.gz"
        strip_result = skull_strip(
            nifti, out, mask_path=mask, prefer_hd_bet=use_hd_bet
        )
        result.skull_stripped[modality] = strip_result.stripped_path
        logger.info("Skull strip %s via %s", modality, strip_result.method)

    reg_dir = output_dir / "04_registered_1mm"
    registered = register_modalities_to_reference(
        result.skull_stripped,
        reg_dir,
        atlas_path=atlas_path,
        spacing_mm=spacing_mm,
    )
    result.registered_1mm = registered
    transforms_dir = reg_dir / "transforms"
    if transforms_dir.is_dir():
        result.transforms = {p.stem: p for p in transforms_dir.glob("*.tfm")}

    manifest_path = output_dir / "pipeline_manifest.json"
    _write_manifest(result, manifest_path)
    result.manifest_path = manifest_path
    logger.info("Patient preprocessing complete: %s", manifest_path)
    return result


def _select_primary_series(converted: Sequence[ConvertedSeries]) -> dict[str, Path]:
    """Pick one series per modality (most instances wins)."""
    best: dict[str, ConvertedSeries] = {}
    for item in converted:
        label = item.meta.modality_label
        if label not in TARGET_MODALITIES:
            continue
        current = best.get(label)
        if (
            current is None
            or item.meta.number_of_instances > current.meta.number_of_instances
        ):
            best[label] = item
    return {label: item.nifti_path for label, item in best.items()}


def _write_manifest(result: PatientPreprocessResult, path: Path) -> None:
    def _stringify(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _stringify(v) for k, v in obj.items()}
        return obj

    payload = _stringify(asdict(result))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
