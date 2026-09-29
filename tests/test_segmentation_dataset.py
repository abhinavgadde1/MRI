"""Tests for BraTS MONAI dataset construction (synthetic fixtures only)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from preprocessing.h5_to_nifti import MODALITY_NAMES
from segmentation.dataset import (
    build_brats_file_list,
    discover_brats_subjects,
    get_train_transforms,
    get_val_transforms,
    split_train_val,
    subject_to_datadict,
)


def _write_subject(root: Path, name: str) -> Path:
    subject = root / name
    subject.mkdir(parents=True)
    affine = np.eye(4)
    for modality in MODALITY_NAMES:
        data = np.random.randn(16, 16, 12).astype(np.float32)
        nib.save(nib.Nifti1Image(data, affine), str(subject / f"{modality}.nii.gz"))
    mask = np.zeros((16, 16, 12), dtype=np.uint8)
    mask[4:10, 4:10, 3:8] = 2
    mask[6:9, 6:9, 4:7] = 1
    mask[7:8, 7:8, 5:6] = 3
    nib.save(nib.Nifti1Image(mask, affine), str(subject / "mask.nii.gz"))
    return subject


def test_discover_brats_subjects(tmp_path: Path):
    _write_subject(tmp_path, "case_001")
    _write_subject(tmp_path, "case_002")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "FLAIR.nii.gz").write_bytes(b"")

    subjects = discover_brats_subjects(tmp_path)
    assert len(subjects) == 2
    assert {p.name for p in subjects} == {"case_001", "case_002"}


def test_subject_datadict_stacks_modalities(tmp_path: Path):
    subject = _write_subject(tmp_path, "BraTS20_Training_001")
    record = subject_to_datadict(subject)
    assert record["subject_id"] == "BraTS20_Training_001"
    assert len(record["image"]) == 4
    assert record["image"][0].endswith("FLAIR.nii.gz")
    assert record["label"].endswith("mask.nii.gz")


def test_split_train_val(tmp_path: Path):
    for i in range(10):
        _write_subject(tmp_path, f"case_{i:03d}")
    records = build_brats_file_list(tmp_path)
    train, val = split_train_val(records, val_fraction=0.2, seed=42)
    assert len(train) + len(val) == 10
    assert len(val) == 2
    assert {r["subject_id"] for r in train}.isdisjoint({r["subject_id"] for r in val})


def test_transforms_compose():
    train_t = get_train_transforms()
    val_t = get_val_transforms()
    assert len(train_t.transforms) == 6
    assert len(val_t.transforms) == 5
    assert train_t.transforms[0].__class__.__name__ == "LoadImaged"
    assert val_t.transforms[-1].__class__.__name__ == "EnsureTyped"


def test_discover_missing_root():
    with pytest.raises(NotADirectoryError):
        discover_brats_subjects("/tmp/does_not_exist_brats_nifti")
