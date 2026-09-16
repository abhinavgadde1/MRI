"""Tests for pseudo-label generation helpers."""

from pathlib import Path

import numpy as np
import pytest

from segmentation.pseudo_label import (
    REGISTERED_NAMES,
    regions_to_label_map,
    registered_dir_to_datadict,
)


def test_regions_to_label_map():
    shape = (10, 10, 10)
    et = np.zeros(shape, dtype=bool)
    tc = np.zeros(shape, dtype=bool)
    wt = np.zeros(shape, dtype=bool)
    wt[2:8, 2:8, 2:8] = True
    tc[3:7, 3:7, 3:7] = True
    et[4:6, 4:6, 4:6] = True

    label = regions_to_label_map(et, tc, wt)
    assert label[2, 2, 2] == 2  # edema only
    assert label[3, 3, 3] == 1  # necrotic core
    assert label[4, 4, 4] == 3  # enhancing
    assert label[0, 0, 0] == 0


def test_registered_dir_to_datadict(tmp_path: Path):
    reg = tmp_path / "04_registered_1mm"
    reg.mkdir(parents=True)
    for name in REGISTERED_NAMES.values():
        (reg / name).write_bytes(b"")
    record = registered_dir_to_datadict(reg, "study_a")
    assert record["study_id"] == "study_a"
    assert len(record["image"]) == 4
    assert record["reference"].endswith("T1_1mm.nii.gz")
