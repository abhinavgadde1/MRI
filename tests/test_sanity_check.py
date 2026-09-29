"""Tests for segmentation sanity-check helpers."""

import numpy as np
import pytest

from segmentation.sanity_check import (
    assert_dice_agreement,
    assert_matching_geometry,
    dice_monai,
    dice_numpy,
    select_axial_slices,
)


def test_dice_manual_matches_monai():
    pred = np.zeros((32, 32, 32), dtype=bool)
    gt = np.zeros((32, 32, 32), dtype=bool)
    pred[10:20, 10:20, 10:20] = True
    gt[12:22, 12:22, 12:22] = True
    manual, monai = assert_dice_agreement(pred, gt, tolerance=1e-4)
    assert manual == pytest.approx(monai, abs=1e-4)


def test_geometry_mismatch_raises():
    pred = np.zeros((10, 10, 10), dtype=bool)
    gt = np.zeros((11, 10, 10), dtype=bool)
    affine = np.eye(4)
    with pytest.raises(ValueError, match="shape"):
        assert_matching_geometry(pred, gt, affine, affine)


def test_affine_mismatch_raises():
    pred = np.zeros((10, 10, 10), dtype=bool)
    gt = np.zeros((10, 10, 10), dtype=bool)
    with pytest.raises(ValueError, match="affine"):
        assert_matching_geometry(pred, gt, np.eye(4), np.eye(4) * 2)


def test_select_axial_slices_evenly_spaced():
    mask = np.zeros((20, 20, 40), dtype=bool)
    mask[5:15, 5:15, 10:30] = True
    slices = select_axial_slices(mask, n_slices=4)
    assert len(slices) == 4
    assert slices == sorted(slices)
    assert all(10 <= z < 30 for z in slices)


def test_dice_empty_union():
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert dice_numpy(empty, empty) == 1.0
    manual, monai = assert_dice_agreement(empty, empty)
    assert manual == 1.0 and monai == 1.0


def test_dice_monai_nonzero():
    pred = np.zeros((8, 8, 8), dtype=bool)
    gt = np.zeros((8, 8, 8), dtype=bool)
    pred[2:6, 2:6, 2:6] = True
    gt[2:6, 2:6, 2:6] = True
    assert dice_monai(pred, gt) == pytest.approx(1.0)
    assert dice_numpy(pred, gt) == pytest.approx(1.0)
