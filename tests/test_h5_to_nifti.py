"""Tests for BraTS H5 → NIfTI helpers (parse + lightweight mocks)."""

from pathlib import Path

import h5py
import numpy as np

from preprocessing.h5_to_nifti import (
    MODALITY_NAMES,
    PatientH5Group,
    SliceRef,
    multi_channel_mask_to_labels,
    parse_slice_filename,
    save_patient_niftis,
    validate_slice_sequence,
)


def test_parse_slice_filename():
    ref = parse_slice_filename(Path("volume_100_slice_10.h5"))
    assert ref is not None
    assert ref.volume_id == 100
    assert ref.slice_index == 10
    assert parse_slice_filename(Path("not_a_match.h5")) is None


def test_modality_names_order():
    assert MODALITY_NAMES == ("FLAIR", "T1", "T1c", "T2")


def test_multi_channel_mask_to_labels():
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    labels = multi_channel_mask_to_labels(mask)
    assert labels[0, 0] == 1
    assert labels[1, 1] == 2
    assert labels[2, 2] == 3
    assert labels[3, 3] == 0


def test_validate_slice_sequence_ok_and_gap():
    ok = PatientH5Group(
        volume_id=1,
        study_id="s1",
        slices=[
            SliceRef(Path("volume_1_slice_0.h5"), 1, 0),
            SliceRef(Path("volume_1_slice_1.h5"), 1, 1),
        ],
    )
    assert validate_slice_sequence(ok).status == "ok"

    gap = PatientH5Group(
        volume_id=1,
        study_id="s1",
        slices=[
            SliceRef(Path("volume_1_slice_0.h5"), 1, 0),
            SliceRef(Path("volume_1_slice_2.h5"), 1, 2),
        ],
    )
    record = validate_slice_sequence(gap)
    assert record.status == "invalid_slices"
    assert 1 in record.missing_slices


def test_save_patient_niftis_from_tiny_h5(tmp_path: Path):
    """Build a 2-slice toy H5 series and convert without touching real BraTS data."""
    slices: list[SliceRef] = []
    for z in range(2):
        path = tmp_path / f"volume_9_slice_{z}.h5"
        with h5py.File(path, "w") as handle:
            image = np.zeros((8, 8, 4), dtype=np.float32)
            image[..., 0] = z + 1
            mask = np.zeros((8, 8, 3), dtype=np.uint8)
            if z == 1:
                mask[2:4, 2:4, 1] = 1
            handle.create_dataset("image", data=image)
            handle.create_dataset("mask", data=mask)
        slices.append(SliceRef(path=path, volume_id=9, slice_index=z))

    group = PatientH5Group(volume_id=9, study_id="toy_patient", slices=slices)
    out = tmp_path / "nifti"
    outputs = save_patient_niftis(group, out, np.eye(4))
    assert set(outputs) >= set(MODALITY_NAMES) | {"mask"}
    assert all(p.is_file() for p in outputs.values())
