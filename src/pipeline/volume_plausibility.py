"""WT volume plausibility vs labeled BraTS held-out ground truth.

Messaging only — does not change segmentation or block a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROJECT_ROOT

# Held-out GT WT voxel volumes (n=60), written by ellipsoid_gt_regions.
# Fallback numbers match that table / the real-patient FP audit
# (outputs/real_patient_fp_audit/summary.json).
_HELDOUT_GT_CSV = (
    PROJECT_ROOT / "results" / "ellipsoid" / "gt_only" / "case_volumes.csv"
)
_FALLBACK_MIN_ML = 20.935
_FALLBACK_MAX_ML = 222.999


def load_heldout_wt_bounds(csv_path: Path | None = None) -> tuple[float, float]:
    """Return (min, max) whole-tumor GT voxel volume in mL from held-out stats."""
    path = Path(csv_path) if csv_path is not None else _HELDOUT_GT_CSV
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError):
        return _FALLBACK_MIN_ML, _FALLBACK_MAX_ML
    if "region" in df.columns:
        wt = df.loc[df["region"].astype(str).eq("whole_tumor"), "voxel_volume_ml"]
    elif "source" in df.columns:
        wt = df.loc[df["source"].astype(str).eq("gt"), "voxel_volume_ml"]
    else:
        return _FALLBACK_MIN_ML, _FALLBACK_MAX_ML
    if wt.empty:
        return _FALLBACK_MIN_ML, _FALLBACK_MAX_ML
    return float(wt.min()), float(wt.max())


# Sourced from held-out GT WT voxel volumes, not a round number.
MIN_PLAUSIBLE_WT_ML, MAX_PLAUSIBLE_WT_ML = load_heldout_wt_bounds()


@dataclass(frozen=True)
class VolumePlausibility:
    status: str | None
    message: str | None
    wt_ml: float | None
    floor_ml: float
    max_ml: float

    def as_meta(self) -> dict[str, Any]:
        return {
            "volume_plausibility": self.status,
            "volume_plausibility_message": self.message,
            "volume_plausibility_floor_ml": round(self.floor_ml, 3),
            "volume_plausibility_max_ml": round(self.max_ml, 3),
        }


def assess_wt_plausibility(
    wt_ml: float | None,
    *,
    floor_ml: float | None = None,
    max_ml: float | None = None,
) -> VolumePlausibility:
    """Classify predicted WT volume against held-out GT min/max.

    ``status`` is ``low``, ``high``, ``plausible``, or ``None`` if volume is missing.
    """
    floor = float(MIN_PLAUSIBLE_WT_ML if floor_ml is None else floor_ml)
    hi = float(MAX_PLAUSIBLE_WT_ML if max_ml is None else max_ml)
    if wt_ml is None:
        return VolumePlausibility(None, None, None, floor, hi)
    try:
        vol = float(wt_ml)
    except (TypeError, ValueError):
        return VolumePlausibility(None, None, None, floor, hi)
    if vol != vol:  # NaN
        return VolumePlausibility(None, None, None, floor, hi)
    if vol < floor:
        msg = (
            f"Predicted whole-tumor volume ({vol:.2f} mL) is below the smallest "
            f"confirmed tumor volume observed in our labeled evaluation set "
            f"({floor:.1f} mL). This result is more consistent with a segmentation "
            f"artifact than a confirmed tumor detection. Clinical correlation is required."
        )
        return VolumePlausibility("low", msg, vol, floor, hi)
    if vol > hi:
        msg = (
            f"Predicted volume ({vol:.2f} mL) exceeds the largest confirmed "
            f"tumor volume in our evaluation set ({hi:.1f} mL); this may reflect "
            f"over-segmentation of normal tissue rather than tumor extent."
        )
        return VolumePlausibility("high", msg, vol, floor, hi)
    return VolumePlausibility("plausible", None, vol, floor, hi)


def wt_ml_from_metrics(metrics: pd.DataFrame | None) -> float | None:
    if metrics is None or getattr(metrics, "empty", True):
        return None
    if "region" not in metrics.columns or "volume_ml" not in metrics.columns:
        return None
    names = metrics["region"].astype(str).str.upper()
    hit = metrics.loc[names.isin(("WT", "WHOLE_TUMOR"))]
    if hit.empty:
        return None
    return float(hit.iloc[0]["volume_ml"])
