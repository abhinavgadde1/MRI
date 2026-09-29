"""Quantitative validation: ellipsoid fits and Bland–Altman agreement."""

from validation.bland_altman import bland_altman_plot_frame, bland_altman_stats
from validation.ellipsoid import (
    ClinicalEllipsoid,
    EllipsoidComparison,
    EllipsoidFit,
    clinical_ellipsoid_from_diameters,
    clinical_ellipsoid_volume_cm3,
    compare_to_ellipsoid,
    diameters_from_binary,
    fit_ellipsoid,
    geometric_ellipsoid_volume_cm3,
)
from validation.ellipsoid_batch import checkpoint_tag, run_ellipsoid_heldout

__all__ = [
    "ClinicalEllipsoid",
    "EllipsoidFit",
    "EllipsoidComparison",
    "clinical_ellipsoid_volume_cm3",
    "geometric_ellipsoid_volume_cm3",
    "clinical_ellipsoid_from_diameters",
    "diameters_from_binary",
    "fit_ellipsoid",
    "compare_to_ellipsoid",
    "bland_altman_stats",
    "bland_altman_plot_frame",
    "checkpoint_tag",
    "run_ellipsoid_heldout",
]
