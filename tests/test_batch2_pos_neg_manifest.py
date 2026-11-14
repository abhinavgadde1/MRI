"""Tests for batch2 pos/neg matching helpers and balanced folds."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from segmentation.batch2_dataset import (
    make_patient_folds,
    write_empty_label_like,
)
from segmentation.batch2_pos_neg_manifest import (
    StudyMeta,
    select_matched_negatives,
)


def _neg(i: int, *, standin: bool = True, thick: float = 5.0) -> StudyMeta:
    return StudyMeta(
        study_id=f"neg_{i:03d}",
        cohort="negative",
        patient_id=f"pid_{i}",
        study_uid=f"uid_{i}",
        manufacturer="Philips Medical Systems",
        model="Achieva",
        field_strength_t=1.5,
        slice_thickness_T1_mm=thick,
        slice_thickness_T1c_mm=thick,
        slice_thickness_T2_mm=thick,
        slice_thickness_FLAIR_mm=thick,
        t1c_status="standin" if standin else "native",
        has_all_four_registered=True,
        brain_mask_fraction=0.2,
        tumor_free_claim="claim",
    )


def test_select_matched_negatives_seed_deterministic():
    positives = [
        StudyMeta(
            study_id=f"pos_{i}",
            cohort="positive",
            patient_id=f"pp_{i}",
            study_uid=f"pu_{i}",
            manufacturer="Philips",
            model="Ingenia",
            field_strength_t=1.5,
            slice_thickness_T1_mm=5.0,
            slice_thickness_T1c_mm=1.2,
            slice_thickness_T2_mm=5.0,
            slice_thickness_FLAIR_mm=5.0,
            t1c_status="native" if i < 3 else "standin",
            has_all_four_registered=True,
        )
        for i in range(6)
    ]
    negatives = [_neg(i) for i in range(40)]
    # no native negatives → deviation flag
    a, unused, dev = select_matched_negatives(positives, negatives, n=10, seed=42)
    b, _, _ = select_matched_negatives(positives, negatives, n=10, seed=42)
    assert [x.study_id for x in a] == [x.study_id for x in b]
    assert len(a) == 10
    assert len(unused) == 30
    assert dev["t1c_match_impossible"] is True


def test_write_empty_label_like(tmp_path: Path):
    aff = np.eye(4)
    ref = tmp_path / "t1c.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), aff), str(ref))
    out = tmp_path / "empty.nii.gz"
    write_empty_label_like(ref, out)
    d = np.asanyarray(nib.load(str(out)).dataobj)
    assert d.shape == (4, 4, 4)
    assert int(d.sum()) == 0


def test_balanced_neg_folds():
    records = []
    for i in range(10):
        records.append(
            {
                "subject_id": f"p{i}",
                "patient_id": f"p{i}",
                "is_negative": False,
                "image": [],
                "label": "",
            }
        )
    for i in range(10):
        records.append(
            {
                "subject_id": f"n{i}",
                "patient_id": f"n{i}",
                "is_negative": True,
                "image": [],
                "label": "",
            }
        )
    folds = make_patient_folds(records, seed=42, max_lopo_patients=2, n_folds=5, balance_negatives=True)
    assert len(folds) == 5
    neg_counts = [f["n_negatives_held"] for f in folds]
    assert min(neg_counts) >= 1
    assert max(neg_counts) - min(neg_counts) <= 1
