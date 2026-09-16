"""DICOM series to NIfTI conversion."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib


def convert_dicom_series(
    dicom_dir: str | Path,
    output_path: str | Path,
    *,
    reorient: bool = True,
) -> Path:
    """Convert a DICOM series directory to a NIfTI file.

    Parameters
    ----------
    dicom_dir:
        Directory containing a single DICOM series.
    output_path:
        Destination ``.nii`` or ``.nii.gz`` path.
    reorient:
        If True, reorient the volume to RAS after conversion when possible.

    Returns
    -------
    Path
        Path to the written NIfTI file.
    """
    import dicom2nifti

    dicom_dir = Path(dicom_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compress = output_path.name.endswith(".nii.gz")
    staging_dir = output_path.parent / f".{output_path.stem}_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        dicom2nifti.convert_directory(
            str(dicom_dir),
            str(staging_dir),
            compression=compress,
            reorient=reorient,
        )
        nifti_files = sorted(staging_dir.glob("*.nii*"))
        if not nifti_files:
            raise FileNotFoundError(f"No NIfTI produced from {dicom_dir}")
        img = nib.load(str(nifti_files[0]))
        nib.save(img, str(output_path))
    finally:
        for leftover in staging_dir.glob("*"):
            leftover.unlink()
        if staging_dir.exists():
            staging_dir.rmdir()

    return output_path
