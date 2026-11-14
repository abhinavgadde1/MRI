"""Tests for pos/neg separation metrics helpers."""

from __future__ import annotations

import numpy as np

from segmentation.pos_neg_separation import (
    best_threshold_sens_spec,
    bootstrap_auc_ci,
    verdict_from_wt_auc,
)


def test_bootstrap_auc_perfect_separation():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=200, seed=0)
    assert auc == 1.0
    assert 0.0 <= lo <= auc <= hi <= 1.0


def test_best_threshold():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    thr, sens, spec = best_threshold_sens_spec(y, s)
    assert sens == 1.0
    assert spec == 1.0
    assert thr > 0.2


def test_verdict():
    assert verdict_from_wt_auc(0.9) == "yes"
    assert verdict_from_wt_auc(0.75) == "partly"
    assert verdict_from_wt_auc(0.5) == "no"
