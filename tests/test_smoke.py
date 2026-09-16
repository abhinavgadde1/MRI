"""Smoke tests for package imports and pure-Python helpers."""

from validation.bland_altman import bland_altman_stats
from tracts.risk import RiskTier, classify_tract_risk


def test_bland_altman_basic():
    a = [10.0, 12.0, 11.0, 13.0]
    b = [9.5, 12.5, 10.5, 13.5]
    result = bland_altman_stats(a, b)
    assert result.n == 4
    assert result.loa_lower < result.mean_diff < result.loa_upper


def test_risk_tiers():
    assert classify_tract_risk(0.0) is RiskTier.HIGH
    assert classify_tract_risk(3.0) is RiskTier.MODERATE
    assert classify_tract_risk(8.0) is RiskTier.LOW
