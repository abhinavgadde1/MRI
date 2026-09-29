"""Risk classification from tumor–tract distances."""

from __future__ import annotations

from enum import Enum


class RiskTier(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


def classify_tract_risk(
    distance_mm: float,
    *,
    high_threshold_mm: float = 2.0,
    moderate_threshold_mm: float = 5.0,
) -> RiskTier:
    """Map minimum lesion–tract distance to a discrete risk tier.

    Default thresholds:
    - high: distance < 2 mm (includes overlap / abutment)
    - moderate: 2 mm ≤ distance < 5 mm
    - low: distance ≥ 5 mm
    """
    if distance_mm < 0:
        raise ValueError("distance_mm must be non-negative")
    if distance_mm < high_threshold_mm:
        return RiskTier.HIGH
    if distance_mm < moderate_threshold_mm:
        return RiskTier.MODERATE
    return RiskTier.LOW
