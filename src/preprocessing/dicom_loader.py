"""Load multi-series DICOM studies, convert to NIfTI, and write JSON sidecars."""

from __future__ import annotations

import argparse
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

# Ordered rules: first match wins (FLAIR / T1c before generic T1 / T2).
_MODALITY_RULES: tuple[tuple[ModalityLabel, re.Pattern[str]], ...] = (
    ("FLAIR", re.compile(r"flair", re.I)),
    (
        "T1c",
        re.compile(
            r"(t1\s*[c+]|\+?\s*c\b|gad|gado|gd[\s_-]?dtpa|post[\s_-]?contrast|"
            r"contrast|ce[\s_-]?t1|t1[\s_-]?ce|t1w?[\s_-]?post|mprage.*post|"
            r"\bgd\b)",
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
        if getattr(ds, "SOPClassUID", None) is None and getattr(
            ds, "SeriesInstanceUID", None
        ) is None:
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
    """Map SeriesDescription / ProtocolName text to T1 / T1c / T2 / FLAIR / other."""
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


def _tag_str(ds: FileDataset, name: str) -> str | None:
    value = getattr(ds, name, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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

    rows = getattr(ds, "Rows", None)
    columns = getattr(ds, "Columns", None)
    try:
        rows = int(rows) if rows is not None else None
    except (TypeError, ValueError):
        rows = None
    try:
        columns = int(columns) if columns is not None else None
    except (TypeError, ValueError):
        columns = None

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
        image_position_patient=_as_float_list(
            getattr(ds, "ImagePositionPatient", None)
        ),
        rows=rows,
        columns=columns,
        number_of_instances=len(files),
        patient_id=_tag_str(ds, "PatientID"),
        study_instance_uid=_tag_str(ds, "StudyInstanceUID"),
    )


def _safe_stem(text: str | None, fallback: str) -> str:
    raw = (text or fallback).strip() or fallback
    cleaned = re.sub(r"[^\w.\-]+", "_", raw)
    return cleaned.strip("_")[:80] or fallback


def _series_output_name(meta: SeriesMeta) -> str:
    ser = f"ser{meta.series_number:03d}" if meta.series_number is not None else "ser"
    safe = _safe_stem(meta.series_description, meta.series_instance_uid[-8:])
    return f"{ser}_{safe}"


def _sidecar_path_for(nifti_path: Path) -> Path:
    """Derive the JSON sidecar path for a NIfTI file."""
    name = nifti_path.name
    if name.endswith(".nii.gz"):
        return nifti_path.with_name(name[: -len(".nii.gz")] + ".json")
    if name.endswith(".nii"):
        return nifti_path.with_name(name[: -len(".nii")] + ".json")
    return nifti_path.with_suffix(".json")


def convert_series_to_nifti(
    files: Sequence[Path],
    output_nifti: str | Path,
    *,
    backend: Literal["auto", "simpleitk", "dicom2nifti"] = "auto",
) -> tuple[Path, str]:
    """Convert one series (file list) to NIfTI. Returns ``(path, backend_used)``."""
    output_nifti = Path(output_nifti)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(p) for p in files)
    if not files:
        raise ValueError("No DICOM files to convert")

    errors: list[str] = []
    backends: list[str] = (
        ["simpleitk", "dicom2nifti"] if backend == "auto" else [backend]
    )

    for name in backends:
        try:
            if name == "simpleitk":
                _convert_with_simpleitk(files, output_nifti)
            else:
                _convert_with_dicom2nifti(files, output_nifti)
            return output_nifti, name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            logger.warning(
                "Conversion via %s failed for %s: %s", name, output_nifti.name, exc
            )

    raise RuntimeError(
        f"All conversion backends failed ({len(files)} files): " + " | ".join(errors)
    )


def _convert_with_simpleitk(files: Sequence[Path], output_nifti: Path) -> None:
    reader = sitk.ImageSeriesReader()
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


def write_sidecar(
    meta: SeriesMeta,
    sidecar_path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write series metadata JSON (PixelSpacing, SliceThickness, IOP, …)."""
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical keys requested by the pipeline, plus full SeriesMeta.
    payload: dict[str, Any] = {
        "PixelSpacing": meta.pixel_spacing,
        "SliceThickness": meta.slice_thickness,
        "ImageOrientationPatient": meta.image_orientation_patient,
        "SeriesDescription": meta.series_description,
        "modality": meta.modality_label,
        "SeriesInstanceUID": meta.series_instance_uid,
        **asdict(meta),
    }
    if extra:
        payload.update(extra)

    sidecar_path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Sidecar %s | modality=%s | PixelSpacing=%s | SliceThickness=%s",
        sidecar_path.name,
        meta.modality_label,
        meta.pixel_spacing,
        meta.slice_thickness,
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
    """Convert all series under a DICOM study to modality-organized NIfTI + JSON.

    Outputs::

        {output_root}/{study_id}/{modality}/ser{N}_{safe_name}.nii.gz
        {output_root}/{study_id}/{modality}/ser{N}_{safe_name}.json
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
            },
        )
        results.append(
            ConvertedSeries(
                nifti_path=nifti_path, sidecar_path=sidecar_path, meta=meta
            )
        )
        logger.info(
            "Wrote %s (%s, %d instances, backend=%s)",
            nifti_path,
            meta.modality_label,
            meta.number_of_instances,
            used_backend,
        )

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a DICOM study directory to modality-organized NIfTI."
    )
    parser.add_argument("dicom_dir", type=Path, help="Study DICOM root")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output root ({out}/{study_id}/{modality}/…)",
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Also convert series that are not T1/T1c/T2/FLAIR",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    results = load_study_to_nifti(
        args.dicom_dir,
        args.output,
        include_other=args.include_other,
    )
    logger.info("Converted %d series", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
