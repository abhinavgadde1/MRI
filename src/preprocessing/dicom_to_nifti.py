"""Thin helper: convert a single DICOM series directory to NIfTI."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def convert_dicom_series(
    dicom_dir: str | Path,
    output_path: str | Path,
    *,
    reorient_ras: bool = False,
    backend: str = "auto",
) -> Path:
    """Convert a DICOM series directory to a NIfTI file.

    Tries SimpleITK first (when ``backend`` is ``\"auto\"``), then falls back to
    ``dicom2nifti`` if installed. Optionally reorients the result to RAS.

    Parameters
    ----------
    dicom_dir:
        Directory containing one DICOM series.
    output_path:
        Destination ``.nii`` or ``.nii.gz`` path.
    reorient_ras:
        If True, reorient the written volume to RAS+ after conversion.
    backend:
        ``\"auto\"``, ``\"simpleitk\"``, or ``\"dicom2nifti\"``.

    Returns
    -------
    Path
        Path to the written NIfTI file.
    """
    dicom_dir = Path(dicom_dir)
    output_path = Path(output_path)
    if not dicom_dir.is_dir():
        raise NotADirectoryError(f"DICOM directory not found: {dicom_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    backends = (
        ["simpleitk", "dicom2nifti"]
        if backend == "auto"
        else [backend]
    )

    for name in backends:
        try:
            if name == "simpleitk":
                _convert_simpleitk(dicom_dir, output_path)
            elif name == "dicom2nifti":
                _convert_dicom2nifti(dicom_dir, output_path)
            else:
                raise ValueError(f"Unknown backend: {name}")
            if reorient_ras:
                _reorient_to_ras(output_path)
            logger.info("Converted %s -> %s via %s", dicom_dir, output_path, name)
            return output_path
        except Exception as exc:  # noqa: BLE001 — try next backend
            errors.append(f"{name}: {exc}")
            logger.warning("DICOM→NIfTI via %s failed: %s", name, exc)

    raise RuntimeError(
        f"Failed to convert DICOM series at {dicom_dir}: " + " | ".join(errors)
    )


def _convert_simpleitk(dicom_dir: Path, output_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    if not file_names:
        raise FileNotFoundError(f"No DICOM files for series in {dicom_dir}")
    reader.SetFileNames(file_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path))


def _convert_dicom2nifti(dicom_dir: Path, output_path: Path) -> None:
    try:
        import dicom2nifti
    except ImportError as exc:
        raise ImportError(
            "dicom2nifti is not installed; pip install dicom2nifti"
        ) from exc

    compress = output_path.name.endswith(".nii.gz")
    with tempfile.TemporaryDirectory(prefix="dicom2nifti_") as tmp:
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        dicom2nifti.convert_directory(
            str(dicom_dir),
            str(out_dir),
            compression=compress,
            reorient=True,
        )
        produced = sorted(out_dir.glob("*.nii*"))
        if not produced:
            raise FileNotFoundError("dicom2nifti produced no NIfTI output")
        shutil.copy2(produced[0], output_path)


def _reorient_to_ras(nifti_path: Path) -> None:
    """Reorient a NIfTI on disk to RAS+ using nibabel when available."""
    try:
        import nibabel as nib
        from nibabel.orientations import axcodes2ornt, ornt_transform
    except ImportError:
        logger.warning("nibabel unavailable; skipping RAS reorient for %s", nifti_path)
        return

    img = nib.load(str(nifti_path))
    current = nib.orientations.io_orientation(img.affine)
    ras = axcodes2ornt("RAS")
    transform = ornt_transform(current, ras)
    reoriented = img.as_reoriented(transform)
    nib.save(reoriented, str(nifti_path))
