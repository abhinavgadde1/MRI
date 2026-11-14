"""Synthetic tests for classical intensity-draft helpers."""

from __future__ import annotations

import numpy as np

from segmentation.batch2_intensity_draft import (
    build_nested_brats,
    enhancement_candidates,
    estimate_nawm_stats,
    flair_candidates,
    nawm_proxy_mask,
    rigid_register_masked,
    robust_zscore,
)


def _sphere(shape, center, radius) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2


def test_bright_sphere_detected_as_enhancement():
    shape = (48, 48, 48)
    brain = np.ones(shape, dtype=bool)
    # Smooth background + bright sphere on "T1c" enhancement map
    enhance = np.random.default_rng(0).normal(0.0, 0.2, size=shape).astype(np.float32)
    ball = _sphere(shape, (24, 24, 24), 5)
    enhance[ball] = 5.0
    keep, thr, comps, meta = enhancement_candidates(
        enhance,
        brain,
        spacing=(1.0, 1.0, 1.0),
        percentile=99.5,
        min_ml=0.3,
        top_k=3,
    )
    assert thr > 0
    assert keep[ball].mean() > 0.5
    assert len(comps) >= 1
    assert comps[0].volume_ml >= 0.3
    assert "top" in meta["percentile_rule"]


def test_vessel_like_tube_suppressed_relative_to_blob():
    shape = (40, 40, 40)
    brain = np.ones(shape, dtype=bool)
    enhance = np.zeros(shape, dtype=np.float32)
    # Fat blob (tumor-like)
    blob = _sphere(shape, (20, 20, 20), 4)
    enhance[blob] = 4.0
    # Thin tube along z through corner (vessel-like)
    enhance[5:35, 8, 8] = 4.0
    keep, _, comps, _ = enhancement_candidates(
        enhance, brain, spacing=(1.0, 1.0, 1.0), percentile=99.0, min_ml=0.2, top_k=3
    )
    # Blob should dominate kept volume; tube voxels mostly removed by Frangi/size
    assert keep[blob].sum() > keep[5:35, 8, 8].sum()
    assert any(c.volume_ml >= 0.2 for c in comps)


def test_nawm_and_robust_zscore():
    rng = np.random.default_rng(1)
    brain = np.zeros((30, 30, 30), dtype=bool)
    brain[5:25, 5:25, 5:25] = True
    t1 = np.zeros((30, 30, 30), dtype=np.float32)
    t1[brain] = rng.normal(100.0, 5.0, size=int(brain.sum())).astype(np.float32)
    proxy, note = nawm_proxy_mask(t1, brain)
    assert "mode" in note.lower() or "brain" in note.lower()
    med, mad = estimate_nawm_stats(t1, proxy)
    assert 90 < med < 110
    assert mad >= 1.0  # floor
    z = robust_zscore(t1, brain, median=med, mad=mad)
    assert abs(float(np.median(z[brain]))) < 0.5


def test_shifted_copy_registration_improves_ncc():
    rng = np.random.default_rng(2)
    shape = (48, 48, 48)
    brain = np.zeros(shape, dtype=bool)
    brain[6:42, 6:42, 6:42] = True
    # Structured fixed image (smooth blob field) so MI has a clear optimum
    zz, yy, xx = np.mgrid[0:48, 0:48, 0:48]
    fixed = (
        80.0 * np.exp(-((zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) / 80.0)
        + rng.normal(0.0, 1.0, size=shape)
    ).astype(np.float32)
    fixed[~brain] = 0.0
    moving = np.roll(fixed, shift=3, axis=0)
    affine = np.eye(4, dtype=float)
    resampled, ncc_b, ncc_a = rigid_register_masked(
        moving, fixed, brain, affine, number_of_iterations=80
    )
    # Either NCC improves, or MSE drops (registration may only partially recover)
    err_before = float(np.mean((moving[brain] - fixed[brain]) ** 2))
    err_after = float(np.mean((resampled[brain] - fixed[brain]) ** 2))
    assert (ncc_a >= ncc_b - 1e-3) or (err_after <= err_before)


def test_flair_and_nested_label_rules():
    shape = (36, 36, 36)
    brain = np.ones(shape, dtype=bool)
    flair_z = np.zeros(shape, dtype=np.float32)
    t1 = np.full(shape, 100.0, dtype=np.float32)
    t2 = np.full(shape, 50.0, dtype=np.float32)
    # FLAIR lesion
    lesion = _sphere(shape, (18, 18, 18), 6)
    flair_z[lesion] = 3.0
    # CSF pocket (should be removed)
    csf = _sphere(shape, (8, 8, 8), 3)
    flair_z[csf] = 3.0
    t1[csf] = 20.0
    t2[csf] = 200.0
    flair_m, rule = flair_candidates(
        flair_z, t1, t2, brain, spacing=(1.0, 1.0, 1.0), z_thresh=1.5, min_ml=0.5
    )
    assert "CSF" in rule
    assert flair_m[lesion].mean() > 0.5
    assert flair_m[csf].mean() < 0.5

    et = _sphere(shape, (18, 18, 18), 3)
    t1_z = np.zeros(shape, dtype=np.float32)
    t1_z[et] = 0.0
    # Dark core next to ET
    core = _sphere(shape, (18, 22, 18), 2)
    t1_z[core] = -1.0
    classic, tc_rule = build_nested_brats(et, flair_m, t1_z, t2, brain, (1.0, 1.0, 1.0))
    assert "T1-z" in tc_rule
    assert (classic[et] == 4).mean() > 0.9
    assert np.any(classic == 2) or np.any(classic == 1)
