"""Tests for BraTS H5 → NIfTI conversion and caching."""

from pathlib import Path

import nibabel as nib
import numpy as np

from preprocessing.h5_to_nifti import (
    convert_brats_h5_directory,
    group_slices_by_patient,
    inspect_naming_pattern,
    parse_slice_filename,
    validate_slice_sequence,
)


def test_parse_confirmed_filename_pattern():
    ref = parse_slice_filename(Path("volume_100_slice_10.h5"))
    assert ref is not None
    assert ref.volume_id == 100
    assert ref.slice_index == 10
    assert parse_slice_filename(Path("not_a_match.h5")) is None


def test_inspect_and_group_real_h5():
    h5_dir = Path("archive/BraTS2020_training_data/content/data")
    names = inspect_naming_pattern(h5_dir)
    assert names
    assert any(n.startswith("volume_") for n in names)
    groups = group_slices_by_patient(h5_dir)
    assert len(groups) == 369
    sample = groups[1]
    record = validate_slice_sequence(sample)
    assert record.status == "ok"
    assert len(sample.slices) == 155


def test_convert_one_patient_and_cache(tmp_path: Path):
    h5_dir = Path("archive/BraTS2020_training_data/content/data")
    out = tmp_path / "brats_nifti"
    df1 = convert_brats_h5_directory(h5_dir, out, volume_ids=[1])
    assert df1.iloc[0]["status"] == "converted"
    study = df1.iloc[0]["study_id"]
    t1 = out / study / "T1.nii.gz"
    mask = out / study / "mask.nii.gz"
    assert t1.is_file() and mask.is_file()
    img = nib.load(str(t1))
    assert img.shape == (240, 240, 155)
    assert np.allclose(img.affine, np.eye(4))

    df2 = convert_brats_h5_directory(h5_dir, out, volume_ids=[1])
    assert df2.iloc[0]["status"] == "cached"
