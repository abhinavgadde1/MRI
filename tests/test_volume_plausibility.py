"""WT volume-floor plausibility: messaging only, no segmentation change."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.volume_plausibility import (
    MAX_PLAUSIBLE_WT_ML,
    MIN_PLAUSIBLE_WT_ML,
    assess_wt_plausibility,
    load_heldout_wt_bounds,
    wt_ml_from_metrics,
)


def test_bounds_come_from_heldout_gt_table():
    lo, hi = load_heldout_wt_bounds()
    assert lo == pytest.approx(20.935, abs=0.01)
    assert hi == pytest.approx(222.999, abs=0.01)
    assert MIN_PLAUSIBLE_WT_ML == pytest.approx(lo)
    assert MAX_PLAUSIBLE_WT_ML == pytest.approx(hi)


def test_assess_low_high_plausible_and_missing():
    low = assess_wt_plausibility(3.25)
    assert low.status == "low"
    assert "3.25 mL" in low.message
    assert "20.9 mL" in low.message
    assert "segmentation artifact" in low.message

    high = assess_wt_plausibility(250.0)
    assert high.status == "high"
    assert "250.00 mL" in high.message
    assert "223.0 mL" in high.message
    assert "over-segmentation" in high.message

    ok = assess_wt_plausibility(59.0)
    assert ok.status == "plausible"
    assert ok.message is None

    missing = assess_wt_plausibility(None)
    assert missing.status is None
    assert missing.message is None


def test_wt_ml_from_metrics():
    df = pd.DataFrame(
        [
            {"region": "ET", "volume_ml": 0.1},
            {"region": "WT", "volume_ml": 12.0},
        ]
    )
    assert wt_ml_from_metrics(df) == pytest.approx(12.0)
    assert wt_ml_from_metrics(pd.DataFrame()) is None


def _patch_tiny_or_plausible(monkeypatch, *, wt_voxels: int) -> None:
    import pipeline.run_case as rc

    monkeypatch.setattr(rc, "_load_or_init_model", lambda *a, **k: object())

    def fake_seg(_model, _device, registered_dir, _case_id, postprocess="lcc"):
        t1 = registered_dir / "T1_1mm.nii.gz"
        import nibabel as nib

        shape = nib.load(str(t1)).shape
        wt = np.zeros(shape, dtype=bool)
        wt.ravel()[: int(wt_voxels)] = True
        empty = np.zeros(shape, dtype=bool)
        display = np.zeros(shape, dtype=np.float32)
        return (
            {
                "enhancing_tumor": empty,
                "tumor_core": empty.copy(),
                "whole_tumor": wt,
            },
            display,
        )

    monkeypatch.setattr(rc, "_segment_registered", fake_seg)


def test_run_case_tiny_wt_writes_low_note(tmp_path: Path, monkeypatch):
    from pipeline.run_case import run_case

    _patch_tiny_or_plausible(monkeypatch, wt_voxels=1_000)  # 1.0 mL at 1 mm³
    report = run_case(
        input_dir=None,
        output_dir=tmp_path / "tiny",
        allow_random_weights=True,
        synthetic=True,
    )
    html = report.read_text(encoding="utf-8")
    assert "volume-plausibility-banner" in html
    assert "segmentation artifact" in html
    meta = json.loads((report.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["volume_plausibility"] == "low"
    assert "below the smallest confirmed tumor" in meta["volume_plausibility_message"]


def test_run_case_plausible_wt_has_no_note(tmp_path: Path, monkeypatch):
    from pipeline.run_case import run_case

    _patch_tiny_or_plausible(monkeypatch, wt_voxels=50_000)  # 50 mL
    report = run_case(
        input_dir=None,
        output_dir=tmp_path / "ok",
        allow_random_weights=True,
        synthetic=True,
    )
    html = report.read_text(encoding="utf-8")
    assert "volume-plausibility-banner" not in html
    assert "segmentation artifact" not in html
    assert "over-segmentation of normal tissue" not in html
    meta = json.loads((report.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["volume_plausibility"] == "plausible"
    assert meta["volume_plausibility_message"] is None
