"""N4 bias field correction via SimpleITK."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk


def n4_bias_correct(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mask_path: str | Path | None = None,
    shrink_factor: int = 2,
    num_iterations: tuple[int, ...] = (50, 50, 30, 20),
    convergence_threshold: float = 1e-6,
) -> Path:
    """Apply N4 bias-field correction to a NIfTI volume.

    Uses ``sitk.N4BiasFieldCorrectionImageFilter``. The bias field may be
    estimated on a shrinked image for speed, then applied at full resolution.
    The output preserves the input spacing, origin, and direction cosines.

    Parameters
    ----------
    input_path:
        Input NIfTI path (``.nii`` / ``.nii.gz``).
    output_path:
        Corrected NIfTI destination.
    mask_path:
        Optional binary mask NIfTI. When omitted, an Otsu mask is derived.
    shrink_factor:
        Integer downsample factor (≥1) used only for bias-field estimation.
    num_iterations:
        Maximum iterations at each N4 resolution level.
    convergence_threshold:
        N4 convergence threshold.

    Returns
    -------
    Path
        Path to the corrected volume.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {input_path}")
    if shrink_factor < 1:
        raise ValueError("shrink_factor must be >= 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = sitk.ReadImage(str(input_path))
    image = sitk.Cast(original, sitk.sitkFloat32)
    image.CopyInformation(original)

    if mask_path is not None:
        mask = sitk.ReadImage(str(mask_path))
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        mask.CopyInformation(original)
    else:
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        mask.CopyInformation(original)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(num_iterations))
    corrector.SetConvergenceThreshold(convergence_threshold)

    if shrink_factor == 1:
        corrected = corrector.Execute(image, mask)
    else:
        shrink_factors = [int(shrink_factor)] * image.GetDimension()
        image_shrunk = sitk.Shrink(image, shrink_factors)
        mask_shrunk = sitk.Shrink(mask, shrink_factors)
        corrector.Execute(image_shrunk, mask_shrunk)
        log_bias = corrector.GetLogBiasFieldAsImage(image)
        corrected = image / sitk.Exp(log_bias)

    corrected.CopyInformation(original)
    sitk.WriteImage(corrected, str(output_path))
    return output_path


# Backwards-compatible alias.
correct_bias_field = n4_bias_correct
