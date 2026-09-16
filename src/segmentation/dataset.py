"""MONAI Dataset and transforms for cached BraTS NIfTI volumes."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Sequence

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    Spacingd,
)

from config import load_config
from preprocessing.h5_to_nifti import MODALITY_NAMES

logger = logging.getLogger(__name__)

IMAGE_KEY = "image"
LABEL_KEY = "label"
DEFAULT_KEYS = (IMAGE_KEY, LABEL_KEY)

# BraTS label semantics from h5_to_nifti mask encoding.
BRATS_LABELS: dict[int, str] = {
    0: "background",
    1: "necrotic_core",
    2: "edema",
    3: "enhancing_tumor",
}


def discover_brats_subjects(brats_nifti_root: str | Path) -> list[Path]:
    """Return subject directories containing all modality + mask NIfTIs."""
    root = Path(brats_nifti_root)
    if not root.is_dir():
        raise NotADirectoryError(f"BraTS NIfTI root not found: {root}")

    required = [f"{modality}.nii.gz" for modality in MODALITY_NAMES] + ["mask.nii.gz"]
    subjects: list[Path] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((subject_dir / name).is_file() for name in required):
            subjects.append(subject_dir)
        else:
            missing = [name for name in required if not (subject_dir / name).is_file()]
            logger.warning("Skipping incomplete subject %s (missing %s)", subject_dir.name, missing)
    logger.info("Discovered %d BraTS subjects under %s", len(subjects), root)
    return subjects


def subject_to_datadict(subject_dir: str | Path) -> dict[str, Any]:
    """Build a MONAI-style record with four modalities stacked as channels."""
    subject_dir = Path(subject_dir)
    return {
        "subject_id": subject_dir.name,
        IMAGE_KEY: [str(subject_dir / f"{modality}.nii.gz") for modality in MODALITY_NAMES],
        LABEL_KEY: str(subject_dir / "mask.nii.gz"),
    }


def build_brats_file_list(brats_nifti_root: str | Path) -> list[dict[str, Any]]:
    """List all BraTS cases as MONAI data dicts."""
    return [subject_to_datadict(path) for path in discover_brats_subjects(brats_nifti_root)]


def split_train_val(
    records: Sequence[dict[str, Any]],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic train/validation split by subject."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_fraction)))
    val = items[:n_val]
    train = items[n_val:]
    if not train:
        raise ValueError("Train split is empty; reduce val_fraction or add more subjects")
    return train, val


def get_train_transforms(
    *,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    crop_size: tuple[int, int, int] = (96, 96, 96),
    pos: int = 1,
    neg: int = 1,
    num_samples: int = 2,
) -> Compose:
    """Training transforms with random positive/negative cropping."""
    return Compose(
        [
            LoadImaged(keys=list(DEFAULT_KEYS)),
            EnsureChannelFirstd(keys=list(DEFAULT_KEYS)),
            Spacingd(
                keys=list(DEFAULT_KEYS),
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=[IMAGE_KEY], nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=list(DEFAULT_KEYS),
                label_key=LABEL_KEY,
                spatial_size=crop_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
            ),
            EnsureTyped(keys=list(DEFAULT_KEYS)),
        ]
    )


def get_val_transforms(
    *,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Deterministic validation transforms (no random cropping)."""
    return Compose(
        [
            LoadImaged(keys=list(DEFAULT_KEYS)),
            EnsureChannelFirstd(keys=list(DEFAULT_KEYS)),
            Spacingd(
                keys=list(DEFAULT_KEYS),
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=[IMAGE_KEY], nonzero=True, channel_wise=True),
            EnsureTyped(keys=list(DEFAULT_KEYS)),
        ]
    )


def create_brats_datasets(
    brats_nifti_root: str | Path | None = None,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    crop_size: tuple[int, int, int] = (96, 96, 96),
    num_train_samples: int = 2,
) -> tuple[Dataset, Dataset]:
    """Create MONAI train and validation ``Dataset`` objects."""
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti

    records = build_brats_file_list(brats_nifti_root)
    train_files, val_files = split_train_val(records, val_fraction=val_fraction, seed=seed)

    train_ds = Dataset(
        data=train_files,
        transform=get_train_transforms(pixdim=pixdim, crop_size=crop_size, num_samples=num_train_samples),
    )
    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms(pixdim=pixdim),
    )
    logger.info(
        "BraTS datasets: train=%d val=%d (from %d subjects)",
        len(train_ds),
        len(val_ds),
        len(records),
    )
    return train_ds, val_ds


def create_brats_dataloaders(
    brats_nifti_root: str | Path | None = None,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    val_fraction: float = 0.2,
    seed: int = 42,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    crop_size: tuple[int, int, int] = (96, 96, 96),
    num_train_samples: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation ``DataLoader`` instances for BraTS."""
    train_ds, val_ds = create_brats_datasets(
        brats_nifti_root,
        val_fraction=val_fraction,
        seed=seed,
        pixdim=pixdim,
        crop_size=crop_size,
        num_train_samples=num_train_samples,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
