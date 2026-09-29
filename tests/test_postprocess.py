"""Tests for binary mask post-processing (LCC + hole fill)."""

import numpy as np
import pytest

from segmentation.postprocess import (
    apply_binary_postprocess,
    apply_channel_postprocess,
    fill_holes_binary,
    fragmentation_stats,
    keep_components_min_size,
    keep_largest_cc,
    lcc_then_fill,
)


def test_lcc_removes_small_disconnected_blob():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:15, 5:15, 5:15] = True  # large cube
    mask[0:2, 0:2, 0:2] = True  # small disconnected blob
    cleaned = keep_largest_cc(mask)
    assert cleaned[5:15, 5:15, 5:15].all()
    assert not cleaned[0:2, 0:2, 0:2].any()
    assert cleaned.sum() == 10 * 10 * 10


def test_hole_filling_works():
    mask = np.ones((12, 12, 12), dtype=bool)
    mask[4:8, 4:8, 4:8] = False  # interior cavity
    filled = fill_holes_binary(mask)
    assert filled[4:8, 4:8, 4:8].all()
    assert filled.all()


def test_lcc_then_fill_combo():
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[3:13, 3:13, 3:13] = True
    mask[5:9, 5:9, 5:9] = False  # hole in large component
    mask[0, 0, 0] = True  # stray voxel
    out = lcc_then_fill(mask)
    assert not out[0, 0, 0]
    assert out[5:9, 5:9, 5:9].all()


def test_raw_mode_unchanged():
    rng = np.random.default_rng(0)
    mask = rng.random((8, 8, 8)) > 0.7
    out = apply_binary_postprocess(mask, mode="raw")
    assert np.array_equal(out, mask)
    assert out is not mask  # copy


def test_channel_postprocess_raw_unchanged():
    pred = np.zeros((3, 10, 10, 10), dtype=np.float32)
    pred[2, 2:8, 2:8, 2:8] = 1.0
    pred[2, 0, 0, 0] = 1.0
    raw = apply_channel_postprocess(pred, mode="raw")
    assert np.array_equal(raw, pred)
    lcc = apply_channel_postprocess(pred, mode="lcc")
    assert lcc[2, 0, 0, 0] == 0.0
    assert lcc[2, 4, 4, 4] == 1.0


def test_lcc_min_keeps_large_secondary_drops_tiny():
    """Secondary CC ≥ max(5% of largest, 1 mL) kept; tiny blob dropped."""
    mask = np.zeros((40, 40, 40), dtype=bool)
    # Largest ~ 10×10×10 = 1000 vox = 1.0 mL at 1 mm
    mask[5:15, 5:15, 5:15] = True
    # Large secondary ~ 6×6×6 = 216 vox = 0.216 mL — below 1 mL and below 5% of 1 mL?
    # 5% of 1.0 mL = 0.05; threshold = max(0.05, 1.0) = 1.0 mL
    # So need secondary >= 1 mL to keep. Make secondary 10×10×10 = 1.0 mL elsewhere.
    mask[25:35, 25:35, 25:35] = True  # 1000 vox = 1.0 mL → kept
    mask[0:2, 0:2, 0:2] = True  # 8 vox tiny → dropped

    kept = keep_components_min_size(
        mask,
        min_frac_of_largest=0.05,
        min_volume_ml=1.0,
        spacing_mm=(1.0, 1.0, 1.0),
    )
    assert kept[5:15, 5:15, 5:15].all()
    assert kept[25:35, 25:35, 25:35].all()
    assert not kept[0:2, 0:2, 0:2].any()


def test_lcc_min_relative_threshold_when_tumor_large():
    """With a large primary, 5% floor exceeds 1 mL and drops a mid-size blob."""
    mask = np.zeros((60, 60, 60), dtype=bool)
    # Largest 30×30×30 = 27000 vox = 27 mL → 5% = 1.35 mL > 1 mL
    mask[5:35, 5:35, 5:35] = True
    # Secondary 10×10×10 = 1.0 mL < 1.35 → dropped
    mask[45:55, 45:55, 45:55] = True
    kept = keep_components_min_size(
        mask,
        min_frac_of_largest=0.05,
        min_volume_ml=1.0,
        spacing_mm=(1.0, 1.0, 1.0),
    )
    assert kept[5:35, 5:35, 5:35].all()
    assert not kept[45:55, 45:55, 45:55].any()


def test_lcc_min_mode_via_apply():
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[2:12, 2:12, 2:12] = True
    mask[18:28, 18:28, 18:28] = True
    mask[0, 0, 0] = True
    out = apply_binary_postprocess(
        mask,
        mode="lcc_min",
        min_frac_of_largest=0.05,
        min_volume_ml=1.0,
        spacing_mm=(1.0, 1.0, 1.0),
    )
    assert out[2:12, 2:12, 2:12].all()
    assert out[18:28, 18:28, 18:28].all()
    assert not out[0, 0, 0]


def test_fragmentation_stats_counts_components():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[1:5, 1:5, 1:5] = True
    mask[8:11, 8:11, 8:11] = True
    stats = fragmentation_stats(mask, (1.0, 1.0, 1.0))
    assert stats["n_components"] == 2
    assert stats["frac_largest"] == pytest.approx(64 / (64 + 27))
    assert stats["n_small_components"] == 1
    assert stats["max_dist_to_lcc_mm"] > 0
    assert stats["frac_outside_largest"] == pytest.approx(27 / (64 + 27))
