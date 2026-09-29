"""Tests for tract distance and risk classification."""

import numpy as np
import pytest

from tracts.distance import minimum_distance_mm
from tracts.risk import RiskTier, classify_tract_risk
from tracts.tractseg_integration import run_tractseg


def test_minimum_distance_uses_spacing():
    lesion = np.zeros((20, 20, 20), dtype=np.uint8)
    tract = np.zeros((20, 20, 20), dtype=np.uint8)
    lesion[5:8, 5:8, 5:8] = 1
    tract[5:8, 5:8, 14:17] = 1  # nearest voxels z=7 → z=14 (Δ=7)
    # spacing z=2 mm → expected min = 7 * 2 = 14 mm
    d = minimum_distance_mm(lesion, tract, spacing_mm=(1.0, 1.0, 2.0))
    assert d == pytest.approx(14.0, abs=0.01)


def test_overlap_distance_is_zero():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[3:6, 3:6, 3:6] = 1
    assert minimum_distance_mm(mask, mask, spacing_mm=(1.0, 1.0, 1.0)) == 0.0


def test_risk_thresholds():
    assert classify_tract_risk(0.0) == RiskTier.HIGH
    assert classify_tract_risk(1.9) == RiskTier.HIGH
    assert classify_tract_risk(2.0) == RiskTier.MODERATE
    assert classify_tract_risk(4.9) == RiskTier.MODERATE
    assert classify_tract_risk(5.0) == RiskTier.LOW


def test_tractseg_missing_raises(tmp_path):
    with pytest.raises(RuntimeError, match="TractSeg"):
        run_tractseg(tmp_path, tmp_path / "out", python_executable="TractSeg_not_installed_xyz")
