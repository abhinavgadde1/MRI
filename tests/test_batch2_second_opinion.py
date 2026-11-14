"""Tests for batch2 scan QC and HD-GLIO label conversion."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from segmentation.batch2_scan_qc import (
    high_frequency_energy_score,
    normalized_cross_correlation,
    top_percentile_mean,
)
from segmentation.hdglio_labels import (
    brats_regions_from_classic,
    classify_agreement,
    compare_v3_hdglio,
    hdglio_to_brats_classic,
    resample_seg_to_grid,
    save_hdglio_draft_on_t1c_grid,
)


def test_ncc_identical_is_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8, 8))
    assert normalized_cross_correlation(a, a) == pytest.approx(1.0, abs=1e-6)


def test_ncc_anticorrelated():
    a = np.arange(27, dtype=float).reshape(3, 3, 3)
    assert normalized_cross_correlation(a, -a) == pytest.approx(-1.0, abs=1e-6)


def test_hf_energy_smooth_lower_than_noisy():
    rng = np.random.default_rng(1)
    smooth = np.ones((32, 32, 8))
    smooth[:, :, :] = ndi_gaussian(smooth)
    noisy = rng.normal(size=(32, 32, 8))
    assert high_frequency_energy_score(noisy) > high_frequency_energy_score(smooth)


def ndi_gaussian(vol):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(vol.astype(float), sigma=2.0)


def test_top_percentile_mean():
    v = np.zeros((4, 4, 4))
    v[0, 0, 0] = 100
    v[1, 1, 1] = 50
    mask = np.ones_like(v, dtype=bool)
    assert top_percentile_mean(v, mask, 99.0) >= 50


def test_hdglio_to_brats_classic():
    seg = np.zeros((4, 4, 4), dtype=np.int16)
    seg[1, 1, 1] = 1  # CE → 4
    seg[2, 2, 2] = 2  # NE → 2
    out = hdglio_to_brats_classic(seg)
    assert out[1, 1, 1] == 4
    assert out[2, 2, 2] == 2
    regs = brats_regions_from_classic(out)
    assert regs["ET"][1, 1, 1]
    assert regs["WT"][2, 2, 2]
    assert not regs["ET"][2, 2, 2]


def test_classify_agreement():
    assert classify_agreement(0.8, v3_empty=False, hd_empty=False) == "both_agree"
    assert classify_agreement(0.5, v3_empty=False, hd_empty=False) == "partial"
    assert classify_agreement(0.1, v3_empty=False, hd_empty=False) == "disagree"
    assert classify_agreement(0.0, v3_empty=True, hd_empty=False) == "one_finds_no_tumor"


def test_compare_and_resample(tmp_path: Path):
    aff = np.eye(4)
    # HD-GLIO raw on same grid
    raw = np.zeros((6, 6, 6), dtype=np.int16)
    raw[2:4, 2:4, 2:4] = 1
    raw[1:5, 1:5, 1:3] = 2
    raw_path = tmp_path / "raw.nii.gz"
    t1c_path = tmp_path / "t1c.nii.gz"
    nib.save(nib.Nifti1Image(raw.astype(np.float32), aff), str(raw_path))
    nib.save(nib.Nifti1Image(np.zeros((6, 6, 6), dtype=np.float32), aff), str(t1c_path))
    out = tmp_path / "hdglio_draft_label.nii.gz"
    meta = save_hdglio_draft_on_t1c_grid(raw_path, t1c_path, out)
    assert out.is_file()
    assert meta["resampled_nearest"] is False
    assert 4 in meta["labels"]

    # Different grid → nearest resample
    raw2 = np.zeros((3, 3, 3), dtype=np.float32)
    raw2[1, 1, 1] = 1
    aff2 = np.diag([2.0, 2.0, 2.0, 1.0])
    raw2_path = tmp_path / "raw2.nii.gz"
    nib.save(nib.Nifti1Image(raw2, aff2), str(raw2_path))
    out2 = tmp_path / "draft2.nii.gz"
    meta2 = save_hdglio_draft_on_t1c_grid(raw2_path, t1c_path, out2)
    assert meta2["resampled_nearest"] is True
    # helper unit
    seg_img = nib.load(str(raw2_path))
    tgt = nib.load(str(t1c_path))
    r = resample_seg_to_grid(seg_img, tgt)
    assert r.shape == (6, 6, 6)

    v3 = hdglio_to_brats_classic(raw)
    hd = np.asarray(nib.load(str(out)).dataobj)
    rows = compare_v3_hdglio("s", v3, hd)
    assert any(r.region == "WT" and r.dice == 1.0 for r in rows)
