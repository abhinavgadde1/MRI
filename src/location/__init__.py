"""Optional tumor location reporting (OFF by default).

Registers skull-stripped T1 to MNI152, warps Harvard-Oxford (and optional JHU)
atlases into patient space, and summarizes hemisphere / lobes / distances.
Does not change checkpoints or default pipeline outputs.
"""

from __future__ import annotations

from location.report import LOCATION_REPORT_ENABLED_DEFAULT, run_location_report

__all__ = [
    "LOCATION_REPORT_ENABLED_DEFAULT",
    "run_location_report",
]
