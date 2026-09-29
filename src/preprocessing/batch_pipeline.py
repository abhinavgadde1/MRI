"""Batch preprocessing over real-patient DICOM studies and BraTS volumes."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from config import load_config
from preprocessing.dicom_loader import (
    discover_dicom_files,
    group_by_series,
    identify_modality,
)
from preprocessing.registration import (
    PatientPreprocessResult,
    run_nifti_preprocessing,
    run_patient_preprocessing,
)

logger = logging.getLogger(__name__)

DatasetName = Literal["real_patients", "brats"]

BRATS_H5_CHANNELS: tuple[str, ...] = ("FLAIR", "T1", "T1c", "T2")


@dataclass
class StudySummary:
    """One row in the batch summary CSV."""

    dataset: DatasetName
    study_id: str
    status: Literal["success", "failed", "skipped"]
    modalities_found: str = ""
    modalities_processed: str = ""
    output_dir: str = ""
    duration_sec: float = 0.0
    error: str = ""


@dataclass
class BatchResult:
    """Aggregate outcome for a batch run."""

    dataset: DatasetName
    summaries: list[StudySummary] = field(default_factory=list)
    summary_csv: Path | None = None

    @property
    def n_success(self) -> int:
        return sum(1 for s in self.summaries if s.status == "success")

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.summaries if s.status == "failed")


def discover_patient_studies(raw_dicom_root: str | Path) -> list[Path]:
    """List patient study directories under ``MRI DATA`` (or equivalent).

    A directory looks like a patient study when it is a non-hidden folder
    that either contains DICOM files or nested series folders.
    """
    root = Path(raw_dicom_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Patient DICOM root not found: {root}")

    studies: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if _looks_like_patient_study(path):
            studies.append(path)

    logger.info("Discovered %d patient studies under %s", len(studies), root)
    return studies


def _looks_like_patient_study(path: Path) -> bool:
    """Heuristic: has .dcm files, DICOM-looking files, or UID-like subdirs."""
    for child in path.iterdir():
        if child.is_file():
            suffix = child.suffix.lower()
            if suffix in {".dcm", ".dicom", ".ima"} or suffix == "":
                return True
        elif child.is_dir() and not child.name.startswith("."):
            # Series folders are often UIDs or numeric
            if "." in child.name or child.name.isdigit() or child.name.startswith("1."):
                return True
            # Any nested file that looks like DICOM
            for nested in child.rglob("*"):
                if nested.is_file() and nested.suffix.lower() in {
                    ".dcm",
                    ".dicom",
                    ".ima",
                    "",
                }:
                    return True
                break
    return False


def discover_brats_subjects(brats_root: str | Path) -> list[dict[str, object]]:
    """Discover BraTS subjects as H5 volumes and/or classic NIfTI folders.

    Returns dicts with keys ``study_id``, ``kind`` (``h5`` / ``nifti_dir``),
    and ``path`` (volume index or directory).
    """
    root = Path(brats_root)
    if not root.is_dir():
        raise NotADirectoryError(f"BraTS root not found: {root}")

    subjects: list[dict[str, object]] = []

    nifti_dirs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and not p.name.startswith(".")
        and _looks_like_brats_nifti_subject(p)
    )
    for path in nifti_dirs:
        subjects.append({"study_id": path.name, "kind": "nifti_dir", "path": path})

    slice0 = sorted(root.glob("volume_*_slice_0.h5"))
    if slice0:
        mapping = _load_brats_name_mapping(root)
        for h5 in slice0:
            vol_id = int(h5.name.split("_")[1])
            study_id = mapping.get(vol_id, f"BraTS_volume_{vol_id:03d}")
            subjects.append({"study_id": study_id, "kind": "h5", "path": vol_id})

    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for item in subjects:
        sid = str(item["study_id"])
        if sid in seen:
            continue
        seen.add(sid)
        unique.append(item)

    logger.info(
        "Discovered %d BraTS subjects under %s (%d H5, %d NIfTI folders)",
        len(unique),
        root,
        sum(1 for s in unique if s["kind"] == "h5"),
        sum(1 for s in unique if s["kind"] == "nifti_dir"),
    )
    return unique


def _looks_like_brats_nifti_subject(path: Path) -> bool:
    names = [p.name.lower() for p in path.glob("*.nii*")]
    if not names:
        return False
    joined = " ".join(names)
    return any(tag in joined for tag in ("flair", "t1ce", "t1", "t2"))


def _load_brats_name_mapping(brats_root: Path) -> dict[int, str]:
    csv_path = brats_root / "name_mapping.csv"
    if not csv_path.is_file():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", csv_path, exc)
        return {}
    if "BraTS_2020_subject_ID" not in df.columns:
        return {}
    return {
        i + 1: str(row)
        for i, row in enumerate(df["BraTS_2020_subject_ID"].tolist())
    }


def probe_patient_modalities(study_dir: Path) -> list[str]:
    """Identify T1/T1c/T2/FLAIR series present in a DICOM study (header-only)."""
    try:
        files = discover_dicom_files(study_dir)
        groups = group_by_series(files)
    except Exception as exc:
        logger.warning("Modality probe failed for %s: %s", study_dir, exc)
        return []

    found: set[str] = set()
    for paths in groups.values():
        try:
            import pydicom

            ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
            label = identify_modality(
                getattr(ds, "SeriesDescription", None),
                getattr(ds, "ProtocolName", None),
                sequence_name=getattr(ds, "SequenceName", None),
            )
            if label in {"T1", "T1c", "T2", "FLAIR"}:
                found.add(label)
        except Exception:
            continue
    return sorted(found)


def convert_brats_h5_volume(
    brats_root: str | Path,
    volume_id: int,
    output_dir: str | Path,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Path]:
    """Stack BraTS H5 slices into per-modality NIfTI volumes."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for BraTS H5 conversion. Install with: pip install h5py"
        ) from exc
    import SimpleITK as sitk

    brats_root = Path(brats_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_paths = sorted(
        brats_root.glob(f"volume_{volume_id}_slice_*.h5"),
        key=lambda p: int(p.stem.split("_slice_")[1]),
    )
    if not slice_paths:
        raise FileNotFoundError(
            f"No H5 slices for volume_{volume_id} under {brats_root}"
        )

    channels: list[list[np.ndarray]] = [[] for _ in BRATS_H5_CHANNELS]
    for path in slice_paths:
        with h5py.File(path, "r") as handle:
            image = np.asarray(handle["image"])
        if image.ndim != 3 or image.shape[-1] < len(BRATS_H5_CHANNELS):
            raise ValueError(f"Unexpected image shape {image.shape} in {path}")
        for idx in range(len(BRATS_H5_CHANNELS)):
            channels[idx].append(np.asarray(image[:, :, idx], dtype=np.float32))

    modality_paths: dict[str, Path] = {}
    for modality, slices in zip(BRATS_H5_CHANNELS, channels):
        volume = np.stack(slices, axis=-1)
        sitk_img = sitk.GetImageFromArray(np.transpose(volume, (2, 0, 1)))
        sitk_img.SetSpacing(spacing_mm)
        out = output_dir / f"{modality}.nii.gz"
        sitk.WriteImage(sitk_img, str(out))
        modality_paths[modality] = out

    return modality_paths


def load_brats_nifti_subject(subject_dir: str | Path) -> dict[str, Path]:
    """Map classic BraTS NIfTI filenames to modality labels."""
    subject_dir = Path(subject_dir)
    mapping_rules: list[tuple[str, tuple[str, ...]]] = [
        ("FLAIR", ("flair",)),
        ("T1c", ("t1ce", "t1gd", "t1c")),
        ("T1", ("t1",)),
        ("T2", ("t2",)),
    ]
    files = list(subject_dir.glob("*.nii*"))
    found: dict[str, Path] = {}
    for modality, tokens in mapping_rules:
        for path in files:
            name = path.name.lower()
            if modality == "T1" and any(t in name for t in ("t1ce", "t1gd", "t1c")):
                continue
            if any(token in name for token in tokens):
                found[modality] = path
                break
    if not found:
        raise FileNotFoundError(f"No BraTS modality NIfTIs in {subject_dir}")
    return found


def write_summary_csv(summaries: Sequence[StudySummary], csv_path: str | Path) -> Path:
    """Write / overwrite a batch summary CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(s) for s in summaries])
    column_order = [
        "dataset",
        "study_id",
        "status",
        "modalities_found",
        "modalities_processed",
        "duration_sec",
        "output_dir",
        "error",
    ]
    df = df.reindex(columns=column_order)
    df.to_csv(csv_path, index=False)
    logger.info(
        "Wrote summary CSV %s (success=%d failed=%d skipped=%d)",
        csv_path,
        int((df["status"] == "success").sum()) if len(df) else 0,
        int((df["status"] == "failed").sum()) if len(df) else 0,
        int((df["status"] == "skipped").sum()) if len(df) else 0,
    )
    return csv_path


def run_batch_patients(
    raw_dicom_root: str | Path,
    output_root: str | Path,
    *,
    atlas_path: str | Path | None = None,
    spacing_mm: float = 1.0,
    use_hd_bet: bool = True,
    max_studies: int | None = None,
    summary_csv: str | Path | None = None,
    study_filter: Iterable[str] | None = None,
) -> BatchResult:
    """Run the full preprocessing chain on real-patient DICOM studies."""
    studies = discover_patient_studies(raw_dicom_root)
    if study_filter is not None:
        allow = set(study_filter)
        studies = [s for s in studies if s.name in allow]
    if max_studies is not None:
        studies = studies[: max(0, max_studies)]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch = BatchResult(dataset="real_patients")
    total = len(studies)
    logger.info("Starting real-patient batch: %d studies -> %s", total, output_root)

    for index, study_dir in enumerate(studies, start=1):
        study_id = study_dir.name
        out_dir = output_root / study_id
        modalities_found = probe_patient_modalities(study_dir)
        logger.info(
            "[%d/%d] Patient %s | modalities_found=%s",
            index,
            total,
            study_id,
            modalities_found or "none",
        )
        started = time.perf_counter()
        try:
            result = run_patient_preprocessing(
                study_dir,
                out_dir,
                study_name=study_id,
                atlas_path=atlas_path,
                spacing_mm=spacing_mm,
                use_hd_bet=use_hd_bet,
            )
            processed = sorted(result.registered_1mm.keys())
            batch.summaries.append(
                StudySummary(
                    dataset="real_patients",
                    study_id=study_id,
                    status="success",
                    modalities_found=",".join(modalities_found),
                    modalities_processed=",".join(processed),
                    output_dir=str(out_dir),
                    duration_sec=round(time.perf_counter() - started, 2),
                )
            )
            logger.info("[%d/%d] SUCCESS %s (%s)", index, total, study_id, processed)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[%d/%d] FAILED %s: %s\n%s",
                index,
                total,
                study_id,
                err,
                traceback.format_exc(),
            )
            batch.summaries.append(
                StudySummary(
                    dataset="real_patients",
                    study_id=study_id,
                    status="failed",
                    modalities_found=",".join(modalities_found),
                    modalities_processed="",
                    output_dir=str(out_dir),
                    duration_sec=round(time.perf_counter() - started, 2),
                    error=err,
                )
            )

    csv_path = (
        Path(summary_csv) if summary_csv else output_root / "batch_summary_patients.csv"
    )
    batch.summary_csv = write_summary_csv(batch.summaries, csv_path)
    logger.info(
        "Patient batch done: %d success, %d failed / %d total",
        batch.n_success,
        batch.n_failed,
        total,
    )
    return batch


def run_batch_brats(
    brats_root: str | Path,
    output_root: str | Path,
    *,
    atlas_path: str | Path | None = None,
    spacing_mm: float = 1.0,
    use_hd_bet: bool = True,
    max_studies: int | None = None,
    summary_csv: str | Path | None = None,
    study_filter: Iterable[str] | None = None,
) -> BatchResult:
    """Run preprocessing over BraTS H5 volumes and/or classic subject folders."""
    subjects = discover_brats_subjects(brats_root)
    if study_filter is not None:
        allow = set(study_filter)
        subjects = [s for s in subjects if str(s["study_id"]) in allow]
    if max_studies is not None:
        subjects = subjects[: max(0, max_studies)]

    brats_root = Path(brats_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    batch = BatchResult(dataset="brats")
    total = len(subjects)
    logger.info("Starting BraTS batch: %d subjects -> %s", total, output_root)

    for index, subject in enumerate(subjects, start=1):
        study_id = str(subject["study_id"])
        kind = str(subject["kind"])
        out_dir = output_root / study_id
        logger.info("[%d/%d] BraTS %s (kind=%s)", index, total, study_id, kind)
        started = time.perf_counter()
        modalities_found: list[str] = []
        try:
            nifti_stage = out_dir / "01_brats_nifti"
            if kind == "h5":
                volume_id = int(subject["path"])  # type: ignore[arg-type]
                modality_paths = convert_brats_h5_volume(
                    brats_root, volume_id, nifti_stage
                )
            elif kind == "nifti_dir":
                modality_paths = load_brats_nifti_subject(
                    Path(subject["path"])  # type: ignore[arg-type]
                )
                nifti_stage.mkdir(parents=True, exist_ok=True)
                staged: dict[str, Path] = {}
                for mod, src in modality_paths.items():
                    dest = nifti_stage / f"{mod}.nii.gz"
                    if not dest.exists():
                        shutil.copy2(src, dest)
                    staged[mod] = dest
                modality_paths = staged
            else:
                raise ValueError(f"Unknown BraTS subject kind: {kind}")

            modalities_found = sorted(modality_paths)
            result: PatientPreprocessResult = run_nifti_preprocessing(
                modality_paths,
                out_dir,
                study_name=study_id,
                atlas_path=atlas_path,
                spacing_mm=spacing_mm,
                use_hd_bet=use_hd_bet,
            )
            processed = sorted(result.registered_1mm.keys())
            batch.summaries.append(
                StudySummary(
                    dataset="brats",
                    study_id=study_id,
                    status="success",
                    modalities_found=",".join(modalities_found),
                    modalities_processed=",".join(processed),
                    output_dir=str(out_dir),
                    duration_sec=round(time.perf_counter() - started, 2),
                )
            )
            logger.info("[%d/%d] SUCCESS %s (%s)", index, total, study_id, processed)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[%d/%d] FAILED %s: %s\n%s",
                index,
                total,
                study_id,
                err,
                traceback.format_exc(),
            )
            batch.summaries.append(
                StudySummary(
                    dataset="brats",
                    study_id=study_id,
                    status="failed",
                    modalities_found=",".join(modalities_found),
                    modalities_processed="",
                    output_dir=str(out_dir),
                    duration_sec=round(time.perf_counter() - started, 2),
                    error=err,
                )
            )

    csv_path = (
        Path(summary_csv) if summary_csv else output_root / "batch_summary_brats.csv"
    )
    batch.summary_csv = write_summary_csv(batch.summaries, csv_path)
    logger.info(
        "BraTS batch done: %d success, %d failed / %d total",
        batch.n_success,
        batch.n_failed,
        total,
    )
    return batch


def run_all_batches(
    *,
    raw_dicom_root: str | Path | None = None,
    brats_root: str | Path | None = None,
    processed_root: str | Path | None = None,
    atlas_path: str | Path | None = None,
    use_hd_bet: bool = True,
    max_studies: int | None = None,
    config_path: str | Path | None = None,
) -> tuple[BatchResult, BatchResult]:
    """Run patient and BraTS batches using ``config.yaml`` defaults when omitted."""
    cfg = load_config(config_path)
    raw_dicom_root = Path(raw_dicom_root) if raw_dicom_root else cfg.paths.raw_dicom
    brats_root = Path(brats_root) if brats_root else cfg.paths.brats
    processed_root = Path(processed_root) if processed_root else cfg.paths.processed

    patients = run_batch_patients(
        raw_dicom_root,
        processed_root / "real_patients",
        atlas_path=atlas_path,
        use_hd_bet=use_hd_bet,
        max_studies=max_studies,
        summary_csv=processed_root / "batch_summary_patients.csv",
    )
    brats = run_batch_brats(
        brats_root,
        processed_root / "brats",
        atlas_path=atlas_path,
        use_hd_bet=use_hd_bet,
        max_studies=max_studies,
        summary_csv=processed_root / "batch_summary_brats.csv",
    )

    combined = patients.summaries + brats.summaries
    write_summary_csv(combined, processed_root / "batch_summary_all.csv")
    return patients, brats


def _configure_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch MRI preprocessing for real patients and BraTS."
    )
    parser.add_argument(
        "--dataset",
        choices=("patients", "brats", "all"),
        default="all",
        help="Which dataset batch to run",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--max-studies", type=int, default=None, help="Limit studies (debug)"
    )
    parser.add_argument("--atlas", type=Path, default=None, help="Optional atlas NIfTI")
    parser.add_argument(
        "--no-hd-bet",
        action="store_true",
        help="Force SimpleITK skull-strip fallback",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(verbose=not args.quiet)

    cfg = load_config(args.config)
    use_hd_bet = not args.no_hd_bet

    if args.dataset in {"patients", "all"}:
        run_batch_patients(
            cfg.paths.raw_dicom,
            cfg.paths.processed / "real_patients",
            atlas_path=args.atlas,
            use_hd_bet=use_hd_bet,
            max_studies=args.max_studies,
            summary_csv=cfg.paths.processed / "batch_summary_patients.csv",
        )
    if args.dataset in {"brats", "all"}:
        run_batch_brats(
            cfg.paths.brats,
            cfg.paths.processed / "brats",
            atlas_path=args.atlas,
            use_hd_bet=use_hd_bet,
            max_studies=args.max_studies,
            summary_csv=cfg.paths.processed / "batch_summary_brats.csv",
        )
    if args.dataset == "all":
        rows: list[StudySummary] = []
        for name in ("batch_summary_patients.csv", "batch_summary_brats.csv"):
            path = cfg.paths.processed / name
            if path.is_file():
                for record in pd.read_csv(path).to_dict(orient="records"):
                    rows.append(StudySummary(**record))
        if rows:
            write_summary_csv(rows, cfg.paths.processed / "batch_summary_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
