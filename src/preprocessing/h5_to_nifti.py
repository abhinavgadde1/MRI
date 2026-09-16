"""Convert BraTS per-slice HDF5 files into cached 3D NIfTI volumes.

Filenames in this project follow::

    volume_{patient_id}_slice_{slice_index}.h5

Confirmed by scanning ``archive/BraTS2020_training_data/content/data``
(e.g. ``volume_100_slice_0.h5``, ``volume_9_slice_99.h5``). Each file holds:

- ``image`` — shape ``(240, 240, 4)`` — FLAIR, T1, T1ce, T2
- ``mask``  — shape ``(240, 240, 3)`` — tumor sub-region channels (binary)

No slice thickness / spacing attributes were found on the H5 datasets or in
``meta_data.csv`` / ``name_mapping.csv`` / ``BraTS20 Training Metadata.csv``.
NIfTIs therefore use an identity affine (1 mm isotropic in world mm units),
which matches the nominal BraTS resolution.

Outputs are written once under ``data/processed/brats_nifti/`` and skipped on
subsequent runs unless ``force=True`` (do not re-convert every training epoch).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Confirmed filename pattern for this dataset dump.
SLICE_NAME_RE = re.compile(r"^volume_(?P<volume_id>\d+)_slice_(?P<slice_index>\d+)\.h5$")

# Channel order used by the Kaggle BraTS2020 H5 dump.
MODALITY_NAMES: tuple[str, ...] = ("FLAIR", "T1", "T1c", "T2")
MASK_CHANNEL_NAMES: tuple[str, ...] = ("mask_ch0", "mask_ch1", "mask_ch2")

EXPECTED_IMAGE_SHAPE_2D = (240, 240, 4)
EXPECTED_MASK_SHAPE_2D = (240, 240, 3)


@dataclass(frozen=True)
class SliceRef:
    path: Path
    volume_id: int
    slice_index: int


@dataclass
class PatientH5Group:
    volume_id: int
    study_id: str
    slices: list[SliceRef] = field(default_factory=list)

    @property
    def slice_indices(self) -> list[int]:
        return [s.slice_index for s in self.slices]


@dataclass
class ConversionRecord:
    volume_id: int
    study_id: str
    status: str
    n_slices: int = 0
    expected_slices: int | None = None
    missing_slices: list[int] = field(default_factory=list)
    duplicate_slices: list[int] = field(default_factory=list)
    out_of_order: bool = False
    output_dir: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    affine: str = "identity"
    spacing_source: str = "none"
    error: str = ""


def parse_slice_filename(path: Path) -> SliceRef | None:
    """Parse ``volume_{id}_slice_{index}.h5``; return None if unmatched."""
    match = SLICE_NAME_RE.match(path.name)
    if not match:
        return None
    return SliceRef(
        path=path,
        volume_id=int(match.group("volume_id")),
        slice_index=int(match.group("slice_index")),
    )


def inspect_naming_pattern(h5_dir: str | Path, *, sample_size: int = 12) -> list[str]:
    """Return sample filenames and log the inferred naming pattern."""
    h5_dir = Path(h5_dir)
    names = sorted(p.name for p in h5_dir.glob("*.h5"))[:sample_size]
    logger.info("Sample BraTS H5 filenames (%d): %s", len(names), names)
    matched = sum(1 for n in names if SLICE_NAME_RE.match(n))
    logger.info(
        "Naming pattern volume_{id}_slice_{index}.h5 matched %d/%d samples",
        matched,
        len(names),
    )
    return names


def load_study_id_mapping(h5_dir: str | Path) -> dict[int, str]:
    """Map volume index → BraTS_2020_subject_ID via ``name_mapping.csv`` when present."""
    csv_path = Path(h5_dir) / "name_mapping.csv"
    if not csv_path.is_file():
        logger.warning("No name_mapping.csv in %s; using volume_* study IDs", h5_dir)
        return {}
    df = pd.read_csv(csv_path)
    if "BraTS_2020_subject_ID" not in df.columns:
        return {}
    # Rows are ordered as volume 1..N in this dump.
    return {
        i + 1: str(name)
        for i, name in enumerate(df["BraTS_2020_subject_ID"].tolist())
    }


def probe_spacing_metadata(h5_dir: str | Path) -> tuple[np.ndarray, str]:
    """Return (affine, source_description).

    Inspects H5 attributes and companion CSVs. When nothing specifies spacing,
    returns an identity affine (nominal 1 mm isotropic) and logs that fact.
    """
    h5_dir = Path(h5_dir)
    # Probe a few H5 files for geometry attributes.
    for path in sorted(h5_dir.glob("volume_*_slice_0.h5"))[:5]:
        with h5py.File(path, "r") as handle:
            file_attrs = dict(handle.attrs)
            image_attrs = dict(handle["image"].attrs) if "image" in handle else {}
            for key in (
                "spacing",
                "pixdim",
                "zooms",
                "slice_thickness",
                "SliceThickness",
                "PixelSpacing",
            ):
                if key in file_attrs or key in image_attrs:
                    value = file_attrs.get(key, image_attrs.get(key))
                    logger.info("Found spacing-like attr %s=%s in %s", key, value, path.name)
                    spacing = np.array(value, dtype=float).ravel()
                    if spacing.size == 1:
                        spacing = np.array([spacing[0], spacing[0], spacing[0]])
                    if spacing.size >= 3:
                        affine = np.eye(4, dtype=float)
                        affine[0, 0], affine[1, 1], affine[2, 2] = spacing[:3]
                        return affine, f"h5_attr:{key}"

    # Companion CSVs in this dump do not include spacing columns (verified).
    for csv_name in ("meta_data.csv", "name_mapping.csv", "survival_info.csv"):
        csv_path = h5_dir / csv_name
        if not csv_path.is_file():
            continue
        columns = list(pd.read_csv(csv_path, nrows=0).columns)
        spacing_cols = [
            c
            for c in columns
            if any(tok in c.lower() for tok in ("spacing", "thickness", "pixdim", "zoom"))
        ]
        if spacing_cols:
            logger.info("Spacing columns in %s: %s", csv_name, spacing_cols)

    logger.warning(
        "No slice spacing / thickness metadata found in H5 attrs or CSVs under %s; "
        "using identity affine (nominal BraTS 1 mm isotropic)",
        h5_dir,
    )
    return np.eye(4, dtype=float), "identity_nominal_1mm"


def group_slices_by_patient(h5_dir: str | Path) -> dict[int, PatientH5Group]:
    """Scan the H5 directory and group slice files by volume / patient ID."""
    h5_dir = Path(h5_dir)
    if not h5_dir.is_dir():
        raise NotADirectoryError(f"BraTS H5 directory not found: {h5_dir}")

    inspect_naming_pattern(h5_dir)
    id_map = load_study_id_mapping(h5_dir)
    groups: dict[int, PatientH5Group] = {}
    unmatched = 0

    for path in h5_dir.glob("*.h5"):
        parsed = parse_slice_filename(path)
        if parsed is None:
            unmatched += 1
            logger.debug("Skipping unmatched H5 name: %s", path.name)
            continue
        group = groups.get(parsed.volume_id)
        if group is None:
            study_id = id_map.get(parsed.volume_id, f"volume_{parsed.volume_id:03d}")
            group = PatientH5Group(
                volume_id=parsed.volume_id, study_id=study_id, slices=[]
            )
            groups[parsed.volume_id] = group
        group.slices.append(parsed)

    for group in groups.values():
        group.slices.sort(key=lambda s: s.slice_index)

    logger.info(
        "Grouped %d patients from H5 slices (%d unmatched filenames)",
        len(groups),
        unmatched,
    )
    return groups


def validate_slice_sequence(group: PatientH5Group) -> ConversionRecord:
    """Check for missing, duplicate, or out-of-order slices."""
    indices = group.slice_indices
    sorted_indices = sorted(indices)
    out_of_order = indices != sorted_indices

    counts: dict[int, int] = {}
    for idx in indices:
        counts[idx] = counts.get(idx, 0) + 1
    duplicates = sorted(i for i, n in counts.items() if n > 1)

    unique_sorted = sorted(counts)
    missing: list[int] = []
    expected_n: int | None = None
    if unique_sorted:
        expected = list(range(unique_sorted[0], unique_sorted[-1] + 1))
        expected_n = len(expected)
        missing = [i for i in expected if i not in counts]
        # Prefer canonical 0..N-1 coverage when the series starts at 0.
        if unique_sorted[0] == 0 and unique_sorted[-1] >= 0:
            expected_n = unique_sorted[-1] + 1
            missing = [i for i in range(expected_n) if i not in counts]

    status = "ok"
    if missing or duplicates or out_of_order:
        status = "invalid_slices"
        logger.error(
            "Patient volume_%s (%s): missing=%s duplicates=%s out_of_order=%s "
            "(n_files=%d)",
            group.volume_id,
            group.study_id,
            missing,
            duplicates,
            out_of_order,
            len(indices),
        )

    return ConversionRecord(
        volume_id=group.volume_id,
        study_id=group.study_id,
        status=status,
        n_slices=len(indices),
        expected_slices=expected_n,
        missing_slices=missing,
        duplicate_slices=duplicates,
        out_of_order=out_of_order,
    )


def _patient_output_complete(output_dir: Path) -> bool:
    required = [f"{name}.nii.gz" for name in MODALITY_NAMES] + ["mask.nii.gz"]
    return output_dir.is_dir() and all((output_dir / name).is_file() for name in required)


def stack_patient_volume(
    group: PatientH5Group,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load sorted slices and stack into modality volumes + label mask.

    Returns
    -------
    modalities:
        Mapping name → ``float32`` array shaped ``(240, 240, Z)``.
    mask:
        ``uint8`` label volume ``(240, 240, Z)`` with values 0–3 from the
        three mutually exclusive H5 mask channels.
    """
    n = len(group.slices)
    modalities = {
        name: np.empty((240, 240, n), dtype=np.float32) for name in MODALITY_NAMES
    }
    mask = np.zeros((240, 240, n), dtype=np.uint8)

    for z, slice_ref in enumerate(group.slices):
        with h5py.File(slice_ref.path, "r") as handle:
            if "image" not in handle or "mask" not in handle:
                raise KeyError(f"Missing image/mask datasets in {slice_ref.path}")
            image = np.asarray(handle["image"])
            mask_slice = np.asarray(handle["mask"])

        if image.shape != EXPECTED_IMAGE_SHAPE_2D:
            raise ValueError(
                f"Unexpected image shape {image.shape} in {slice_ref.path}; "
                f"expected {EXPECTED_IMAGE_SHAPE_2D}"
            )
        if mask_slice.shape != EXPECTED_MASK_SHAPE_2D:
            raise ValueError(
                f"Unexpected mask shape {mask_slice.shape} in {slice_ref.path}; "
                f"expected {EXPECTED_MASK_SHAPE_2D}"
            )

        for channel, name in enumerate(MODALITY_NAMES):
            modalities[name][:, :, z] = image[:, :, channel].astype(np.float32)

        # Channels are binary and non-overlapping in this dump; encode 1..3.
        label = np.zeros((240, 240), dtype=np.uint8)
        for channel in range(3):
            label[mask_slice[:, :, channel] > 0] = np.uint8(channel + 1)
        mask[:, :, z] = label

    return modalities, mask


def save_patient_niftis(
    group: PatientH5Group,
    output_dir: str | Path,
    affine: np.ndarray,
) -> dict[str, Path]:
    """Stack slices and write modality + mask NIfTIs with ``nibabel``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modalities, mask = stack_patient_volume(group)

    outputs: dict[str, Path] = {}
    for name, data in modalities.items():
        path = output_dir / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        outputs[name] = path

    mask_path = output_dir / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))
    outputs["mask"] = mask_path

    meta = {
        "volume_id": group.volume_id,
        "study_id": group.study_id,
        "n_slices": len(group.slices),
        "modalities": list(MODALITY_NAMES),
        "mask_encoding": {
            "0": "background",
            "1": MASK_CHANNEL_NAMES[0],
            "2": MASK_CHANNEL_NAMES[1],
            "3": MASK_CHANNEL_NAMES[2],
        },
        "affine": affine.tolist(),
        "shape_xyz": list(modalities["T1"].shape),
    }
    (output_dir / "conversion_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def convert_patient(
    group: PatientH5Group,
    output_root: str | Path,
    affine: np.ndarray,
    *,
    spacing_source: str,
    force: bool = False,
) -> ConversionRecord:
    """Validate and convert one patient; skip if cached unless ``force``."""
    output_dir = Path(output_root) / group.study_id
    record = validate_slice_sequence(group)
    record.affine = "identity" if np.allclose(affine, np.eye(4)) else "custom"
    record.spacing_source = spacing_source
    record.output_dir = str(output_dir)

    if record.status == "invalid_slices":
        # Do not write a malformed volume.
        return record

    if _patient_output_complete(output_dir) and not force:
        record.status = "cached"
        record.outputs = {
            name: str(output_dir / f"{name}.nii.gz") for name in MODALITY_NAMES
        }
        record.outputs["mask"] = str(output_dir / "mask.nii.gz")
        logger.info("Cached NIfTIs for %s — skipping conversion", group.study_id)
        return record

    try:
        outputs = save_patient_niftis(group, output_dir, affine)
        record.status = "converted"
        record.outputs = {k: str(v) for k, v in outputs.items()}
        logger.info(
            "Converted volume_%s (%s): %d slices -> %s",
            group.volume_id,
            group.study_id,
            len(group.slices),
            output_dir,
        )
    except Exception as exc:
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed converting %s: %s", group.study_id, exc)
    return record


def convert_brats_h5_directory(
    h5_dir: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    volume_ids: Iterable[int] | None = None,
    summary_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Convert all (or selected) BraTS H5 patients to cached NIfTI volumes.

    Safe to call at dataset build time; subsequent calls hit the cache unless
    ``force=True``. Do not invoke inside the training epoch loop.
    """
    h5_dir = Path(h5_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    affine, spacing_source = probe_spacing_metadata(h5_dir)
    groups = group_slices_by_patient(h5_dir)
    if volume_ids is not None:
        allow = set(volume_ids)
        groups = {vid: grp for vid, grp in groups.items() if vid in allow}

    records: list[ConversionRecord] = []
    total = len(groups)
    for index, volume_id in enumerate(sorted(groups), start=1):
        group = groups[volume_id]
        logger.info("[%d/%d] %s (volume_%s)", index, total, group.study_id, volume_id)
        records.append(
            convert_patient(
                group,
                output_root,
                affine,
                spacing_source=spacing_source,
                force=force,
            )
        )

    df = pd.DataFrame([asdict(r) for r in records])
    # Flatten list columns for CSV readability.
    for col in ("missing_slices", "duplicate_slices"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ",".join(map(str, x)) if x else "")
    if "outputs" in df.columns:
        df["outputs"] = df["outputs"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

    csv_path = Path(summary_csv) if summary_csv else output_root / "h5_to_nifti_summary.csv"
    df.to_csv(csv_path, index=False)

    n_ok = int(df["status"].isin(["converted", "cached"]).sum()) if len(df) else 0
    n_bad = int((df["status"] == "invalid_slices").sum()) if len(df) else 0
    n_fail = int((df["status"] == "failed").sum()) if len(df) else 0
    logger.info(
        "H5→NIfTI done: ok/cached=%d invalid_slices=%d failed=%d | summary=%s",
        n_ok,
        n_bad,
        n_fail,
        csv_path,
    )

    manifest = {
        "h5_dir": str(h5_dir),
        "output_root": str(output_root),
        "affine": affine.tolist(),
        "spacing_source": spacing_source,
        "modality_channel_order": list(MODALITY_NAMES),
        "n_patients": total,
        "summary_csv": str(csv_path),
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return df


def main(argv: Sequence[str] | None = None) -> int:
    from config import load_config

    parser = argparse.ArgumentParser(description="Convert BraTS H5 slices to NIfTI (cached).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--h5-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Reconvert even if cached")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config)
    h5_dir = args.h5_dir or cfg.paths.brats
    output_dir = args.output_dir or (cfg.paths.processed / "brats_nifti")

    volume_ids = None
    if args.max_patients is not None:
        groups = group_slices_by_patient(h5_dir)
        volume_ids = sorted(groups)[: args.max_patients]

    convert_brats_h5_directory(
        h5_dir,
        output_dir,
        force=args.force,
        volume_ids=volume_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
