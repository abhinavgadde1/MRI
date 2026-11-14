"""Tests for radiologist tumor-free status mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segmentation.batch2_radiologist_status import (
    STATUS_REPORT,
    STATUS_UNVERIFIED,
    STATUS_VERBAL,
    apply_radiologist_status_to_manifest,
    bootstrap_fraction_ci,
    status_from_row,
)


def test_status_from_empty_is_unverified():
    assert status_from_row({"radiologist_read": "", "source": ""}) == STATUS_UNVERIFIED


def test_status_from_explicit_and_source():
    assert status_from_row({"radiologist_read": STATUS_VERBAL, "source": ""}) == STATUS_VERBAL
    assert (
        status_from_row({"radiologist_read": "tumor free", "source": "verbal"})
        == STATUS_VERBAL
    )
    assert (
        status_from_row({"radiologist_read": "tumor free", "source": "report"})
        == STATUS_REPORT
    )


def test_apply_manifest_defaults_unverified(tmp_path: Path):
    manif = tmp_path / "manifest.csv"
    rad = tmp_path / "reads.csv"
    pd.DataFrame(
        [
            {
                "study_id": "n1",
                "cohort": "negative",
                "selected": True,
                "unused_negative": False,
                "tumor_free_claim": "old",
                "has_all_four_registered": True,
                "t1c_status": "standin",
                "manufacturer": "Philips",
            },
            {
                "study_id": "n2",
                "cohort": "negative",
                "selected": False,
                "unused_negative": True,
                "tumor_free_claim": "old",
                "has_all_four_registered": True,
                "t1c_status": "standin",
                "manufacturer": "Philips",
            },
            {
                "study_id": "p1",
                "cohort": "positive",
                "selected": True,
                "unused_negative": False,
                "tumor_free_claim": "",
                "has_all_four_registered": True,
                "t1c_status": "native",
                "manufacturer": "Philips",
            },
        ]
    ).to_csv(manif, index=False)
    # Only n2 confirmed — do not invent for n1
    pd.DataFrame(
        [
            {
                "study_id": "n2",
                "radiologist_read": "no tumor",
                "other_findings": "sinusitis",
                "source": "report",
                "date": "2026-01-01",
            }
        ]
    ).to_csv(rad, index=False)
    out = apply_radiologist_status_to_manifest(manif, rad)
    assert out.loc[out.study_id == "n1", "tumor_free_status"].iloc[0] == STATUS_UNVERIFIED
    assert out.loc[out.study_id == "n2", "tumor_free_status"].iloc[0] == STATUS_REPORT
    assert out.loc[out.study_id == "n2", "other_findings"].iloc[0] == "sinusitis"
    assert out.loc[out.study_id == "n2", "read_date"].iloc[0] == "2026-01-01"
    assert out.loc[out.study_id == "p1", "tumor_free_status"].iloc[0] == ""


def test_bootstrap_ci_bounds():
    flags = np.array([True, True, False, False, True])
    frac, lo, hi = bootstrap_fraction_ci(flags, n_boot=500, seed=0)
    assert 0.0 <= lo <= frac <= hi <= 1.0
