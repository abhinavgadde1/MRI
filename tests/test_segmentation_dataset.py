"""Tests for BraTS MONAI dataset construction."""

from pathlib import Path

from segmentation.dataset import (
    build_brats_file_list,
    discover_brats_subjects,
    get_train_transforms,
    get_val_transforms,
    split_train_val,
    subject_to_datadict,
)


def test_discover_brats_subjects():
    root = Path("data/processed/brats_nifti")
    subjects = discover_brats_subjects(root)
    assert len(subjects) == 369


def test_subject_datadict_stacks_modalities():
    subject = Path("data/processed/brats_nifti/BraTS20_Training_001")
    record = subject_to_datadict(subject)
    assert record["subject_id"] == "BraTS20_Training_001"
    assert len(record["image"]) == 4
    assert record["image"][0].endswith("FLAIR.nii.gz")
    assert record["label"].endswith("mask.nii.gz")


def test_split_train_val():
    records = build_brats_file_list("data/processed/brats_nifti")
    train, val = split_train_val(records, val_fraction=0.2, seed=42)
    assert len(train) + len(val) == len(records)
    assert len(val) == 74  # 20% of 369 rounded


def test_transforms_compose():
    train_t = get_train_transforms()
    val_t = get_val_transforms()
    assert len(train_t.transforms) == 6
    assert len(val_t.transforms) == 5
    assert train_t.transforms[0].__class__.__name__ == "LoadImaged"
    assert val_t.transforms[-1].__class__.__name__ == "EnsureTyped"
