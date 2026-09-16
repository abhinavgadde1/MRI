"""Load multi-series DICOM studies, convert to NIfTI, and write metadata sidecars."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import FileDataset

logger = logging.getLogger(__name__)

ModalityLabel = Literal["T1", "T1c", "T2", "FLAIR", "other"]

# Ordered rules: first match wins. Patterns applied to SeriesDescription + ProtocolName.
_MODALITY_RULES: tuple[tuple[ModalityLabel, re.Pattern[str]], ...] = (
    ("FLAIR", re.compile(r"flair", re.I)),
    # Contrast / post-Gd T1 before generic T1
    (
        "T1c",
        re.compile(
            r"(t1\s*[c+]|\+?\s*c\b|gad|gado|gd[\s_-]?dtpa|post[\s_-]?contrast|"
            r"contrast|ce[\s_-]?t1|t1[\s_-]?ce|t1w?[\s_-]?post|mprage.*post)",
            re.I,
        ),
    ),
    ("T2", re.compile(r"\bt2(?!\s*flair)|t2w|t2[\s_-]?tse|t2[\s_-]?fse", re.I)),
    ("T1", re.compile(r"\bt1\b|t1w|t1[\s_-]?se|t1[\s_-]?ffe|mprage|spgr|bravo", re.I)),
)


@dataclass(frozen=True)
class SeriesMeta:
    """Geometry and identification metadata for one DICOM series."""

    series_instance_uid: str
    series_number: int | None
    series_description: str | None
    protocol_name: str | None
    modality_dicom: str | None
    modality_label: ModalityLabel
    pixel_spacing: list[float] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    image_orientation_patient: list[float] | None
    image_position_patient: list[float] | None
    rows: int | None
    columns: int | None
    number_of_instances: int
    patient_id: str | None = None
    study_instance_uid: str | None = None
    conversion_backend: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConvertedSeries:
    """Result of converting one series to NIfTI + JSON sidecar."""

    nifti_path: Path
    sidecar_path: Path
    meta: SeriesMeta


def discover_dicom_files(root: str | Path) -> list[Path]:
    """Recursively find readable DICOM files under ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in {".json", ".nii", ".gz", ".txt", ".csv", ".html"}:
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "SOPClassUID", None) is None and getattr(ds, "SeriesInstanceUID", None) is None:
            continue
        found.append(path)
    return sorted(found)


def group_by_series(dicom_files: Sequence[Path]) -> dict[str, list[Path]]:
    """Group DICOM paths by ``SeriesInstanceUID``."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception as exc:
            logger.warning("Skipping unreadable DICOM %s: %s", path, exc)
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if not uid:
            logger.warning("Skipping %s: missing SeriesInstanceUID", path)
            continue
        groups[uid].append(path)
    return dict(groups)


def identify_modality(
    series_description: str | None,
    protocol_name: str | None = None,
    *,
    sequence_name: str | None = None,
) -> ModalityLabel:
    """Map series/protocol text to T1 / T1c / T2 / FLAIR / other."""
    blob = " | ".join(
        part for part in (series_description, protocol_name, sequence_name) if part
    )
    if not blob.strip():
        return "other"
    for label, pattern in _MODALITY_RULES:
        if pattern.search(blob):
            return label
    return "other"


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except TypeError:
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_series_metadata(
    files: Sequence[Path],
    *,
    modality_label: ModalityLabel | None = None,
) -> SeriesMeta:
    """Read geometry tags from the first readable instance in the series."""
    if not files:
        raise ValueError("Cannot extract metadata from an empty series")

    ds: FileDataset | None = None
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Unable to read any DICOM header in series")

    series_description = _tag_str(ds, "SeriesDescription")
    protocol_name = _tag_str(ds, "ProtocolName")
    sequence_name = _tag_str(ds, "SequenceName")
    label = modality_label or identify_modality(
        series_description, protocol_name, sequence_name=sequence_name
    )

    series_number = getattr(ds, "SeriesNumber", None)
    try:
        series_number = int(series_number) if series_number is not None else None
    except (TypeError, ValueError):
        series_number = None

    return SeriesMeta(
        series_instance_uid=str(ds.SeriesInstanceUID),
        series_number=series_number,
        series_description=series_description,
        protocol_name=protocol_name,
        modality_dicom=_tag_str(ds, "Modality"),
        modality_label=label,
        pixel_spacing=_as_float_list(getattr(ds, "PixelSpacing", None)),
        slice_thickness=_as_float(getattr(ds, "SliceThickness", None)),
        spacing_between_slices=_as_float(getattr(ds, "SpacingBetweenSlices", None)),
        image_orientation_patient=_as_float_list(
            getattr(ds, "ImageOrientationPatient", None)
        ),
        image_position_patient=_as_float_list(getattr(ds, "ImagePositionPatient", None)),
        rows=int(ds.Rows) if getattr(ds, "Rows", None) is not None else None,
        columns=int(ds.Columns) if getattr(ds, "Columns", None) is not None else None,
        number_of_instances=len(files),
        patient_id=_tag_str(ds, "PatientID"),
        study_instance_uid=_tag_str(ds, "StudyInstanceUID"),
    )


def _tag_str(ds: FileDataset, name: str) -> str | None:
    value = getattr(ds, name, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_stem(text: str | None, fallback: str) -> str:
    raw = (text or fallback).strip() or fallback
    cleaned = re.sub(r"[^\w.\-]+", "_", raw)
    return cleaned.strip("_")[:80] or fallback


def _series_output_name(meta: SeriesMeta) -> str:
    parts = [
        f"ser{meta.series_number:03d}" if meta.series_number is not None else "ser",
        _safe_stem(meta.series_description, meta.series_instance_uid[-8:]),
    ]
    return "_".join(parts)


def convert_series_to_nifti(
    files: Sequence[Path],
    output_nifti: str | Path,
    *,
    backend: Literal["auto", "simpleitk", "dicom2nifti"] = "auto",
) -> tuple[Path, str]:
    """Convert one series (file list) to NIfTI.

    Returns ``(nifti_path, backend_used)``.
    """
    output_nifti = Path(output_nifti)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(p) for p in files)
    if len(files) < 1:
        raise ValueError("No DICOM files to convert")

    errors: list[str] = []
    backends: list[str]
    if backend == "auto":
        backends = ["simpleitk", "dicom2nifti"]
    else:
        backends = [backend]

    for name in backends:
        try:
            if name == "simpleitk":
                _convert_with_simpleitk(files, output_nifti)
            else:
                _convert_with_dicom2nifti(files, output_nifti)
            return output_nifti, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("Conversion via %s failed for %s: %s", name, output_nifti.name, exc)

    raise RuntimeError(
        "All conversion backends failed for series "
        f"({len(files)} files): " + " | ".join(errors)
    )


def _convert_with_simpleitk(files: Sequence[Path], output_nifti: Path) -> None:
    reader = sitk.ImageSeriesReader()
    # Prefer ITK's own series file sorting when a common directory exists.
    dirs = {p.parent for p in files}
    file_names: list[str]
    if len(dirs) == 1:
        series_dir = str(next(iter(dirs)))
        uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(series_dir)
        matched: list[str] | None = None
        wanted = {str(p.resolve()) for p in files}
        for uid in uids:
            candidate = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, uid)
            if {str(Path(c).resolve()) for c in candidate} == wanted:
                matched = list(candidate)
                break
        file_names = matched if matched is not None else [str(p) for p in files]
    else:
        file_names = [str(p) for p in files]

    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    sitk.WriteImage(image, str(output_nifti))


def _convert_with_dicom2nifti(files: Sequence[Path], output_nifti: Path) -> None:
    import dicom2nifti

    compress = output_nifti.name.endswith(".nii.gz")
    with tempfile.TemporaryDirectory(prefix="dicom2nifti_") as tmp:
        tmp_dir = Path(tmp)
        # dicom2nifti expects a directory of instances belonging to one series.
        for idx, src in enumerate(files):
            dest = tmp_dir / f"{idx:05d}_{src.name}"
            try:
                dest.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dest)

        out_dir = tmp_dir / "out"
        out_dir.mkdir()
        dicom2nifti.convert_directory(
            str(tmp_dir),
            str(out_dir),
            compression=compress,
            reorient=True,
        )
        produced = sorted(out_dir.glob("*.nii*"))
        if not produced:
            # Fallback: convert unsorted list API if available
            if hasattr(dicom2nifti, "dicom_series_to_nifti"):
                dicom2nifti.dicom_series_to_nifti(
                    str(tmp_dir),
                    str(output_nifti),
                    reorient_nifti=True,
                )
                if output_nifti.is_file():
                    return
            raise FileNotFoundError("dicom2nifti produced no NIfTI output")
        img = nib.load(str(produced[0]))
        nib.save(img, str(output_nifti))


def _sidecar_path_for(nifti_path: Path) -> Path:
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return nifti_path.with_name(name[: -len(".nii.gz")] + ".json")
    if name.endswith(".nii"):
        return nifti_path.with_name(name[: -len(".nii")] + ".json")
    return nifti_path.with_suffix(".json")


def write_sidecar(meta: SeriesMeta, sidecar_path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
    """Write series metadata JSON next to the NIfTI output."""
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(meta)
    if extra:
        payload.update(extra)
    # Ensure JSON-serializable paths / numpy types are plain Python.
    text = json.dumps(payload, indent=2, default=_json_default)
    sidecar_path.write_text(text + "\n", encoding="utf-8")
    logger.info(
        "Sidecar %s | modality=%s | PixelSpacing=%s | SliceThickness=%s | IOP=%s",
        sidecar_path.name,
        meta.modality_label,
        meta.pixel_spacing,
        meta.slice_thickness,
        meta.image_orientation_patient,
    )
    return sidecar_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_study_to_nifti(
    dicom_root: str | Path,
    output_root: str | Path,
    *,
    study_name: str | None = None,
    modalities: Iterable[ModalityLabel] | None = None,
    backend: Literal["auto", "simpleitk", "dicom2nifti"] = "auto",
    compress: bool = True,
    include_other: bool = True,
) -> list[ConvertedSeries]:
    """Convert all series under a DICOM study directory to modality-organized NIfTI.

    Parameters
    ----------
    dicom_root:
        Study (or multi-series) directory tree containing DICOM files.
    output_root:
        Destination root. Outputs are written to
        ``{output_root}/{study_name}/{modality}/{series}.nii.gz`` plus ``.json``.
    study_name:
        Folder name for this study; defaults to ``dicom_root.name``.
    modalities:
        Optional whitelist of modality labels to convert. When omitted, all
        identified modalities are kept (``other`` controlled by ``include_other``).
    backend:
        Conversion engine preference.
    compress:
        Write ``.nii.gz`` when True, else ``.nii``.
    include_other:
        Keep series that do not match T1 / T1c / T2 / FLAIR.

    Returns
    -------
    list[ConvertedSeries]
        Successfully converted series (failed series are logged and skipped).
    """
    dicom_root = Path(dicom_root)
    output_root = Path(output_root)
    study_name = study_name or dicom_root.name
    allowed = set(modalities) if modalities is not None else None

    files = discover_dicom_files(dicom_root)
    if not files:
        raise FileNotFoundError(f"No DICOM files found under {dicom_root}")

    groups = group_by_series(files)
    logger.info(
        "Found %d DICOM files in %d series under %s",
        len(files),
        len(groups),
        dicom_root,
    )

    results: list[ConvertedSeries] = []
    for uid, series_files in sorted(groups.items(), key=lambda item: item[0]):
        try:
            meta = extract_series_metadata(series_files)
        except Exception as exc:
            logger.error("Metadata extraction failed for series %s: %s", uid, exc)
            continue

        if meta.modality_label == "other" and not include_other:
            logger.info(
                "Skipping non-target series %s (%s)",
                uid,
                meta.series_description,
            )
            continue
        if allowed is not None and meta.modality_label not in allowed:
            logger.info(
                "Skipping series %s: modality %s not in %s",
                uid,
                meta.modality_label,
                sorted(allowed),
            )
            continue

        ext = ".nii.gz" if compress else ".nii"
        out_dir = output_root / study_name / meta.modality_label
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _series_output_name(meta)
        nifti_path = out_dir / f"{stem}{ext}"
        # Avoid collisions when descriptions repeat.
        if nifti_path.exists():
            nifti_path = out_dir / f"{stem}_{uid[-8:]}{ext}"
        sidecar_path = _sidecar_path_for(nifti_path)

        try:
            nifti_path, used_backend = convert_series_to_nifti(
                series_files, nifti_path, backend=backend
            )
        except Exception as exc:
            logger.error(
                "Failed to convert series %s (%s): %s",
                uid,
                meta.series_description,
                exc,
            )
            continue

        meta = SeriesMeta(**{**asdict(meta), "conversion_backend": used_backend})
        write_sidecar(
            meta,
            sidecar_path,
            extra={
                "nifti_path": str(nifti_path),
                "source_dicom_root": str(dicom_root),
                "source_files": [str(p) for p in series_files],
            },
        )
        results.append(
            ConvertedSeries(nifti_path=nifti_path, sidecar_path=sidecar_path, meta=meta)
        )
        logger.info(
            "Wrote %s (%s, %d instances, backend=%s)",
            nifti_path,
            meta.modality_label,
            meta.number_of_instances,
            used_backend,
        )

    return results
