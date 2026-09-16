"""Tests for domain-gap report utilities."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import torch

from segmentation.domain_gap_report import (
    MODEL_BRATS,
    MODEL_FINETUNED,
    build_comparison_table,
    evaluate_clinical_case,
    load_finetune_test_records,
    mask_volume_ml,
    plot_domain_gap_bars,
    summarize_comparison,
    volume_errors,
)
from segmentation.finetune import CORRECTED_LABEL_NAMES
from segmentation.model import BRATS_REGIONS, build_brats_model
from segmentation.pseudo_label import REGISTERED_NAMES
from segmentation.train import post_transforms


def _make_corrected_study(tmp_path: Path, name: str) -> Path:
    study = tmp_path / name
    reg = study / "04_registered_1mm"
    reg.mkdir(parents=True)
    for fname in REGISTERED_NAMES.values():
        (reg / fname).write_bytes(b"")
    (reg / CORRECTED_LABEL_NAMES[0]).write_bytes(b"")
    return study


def test_mask_volume_and_volume_errors():
    spacing = (1.0, 1.0, 1.0)
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    assert mask_volume_ml(mask, spacing) == pytest.approx(27 * 0.001)

    pred = mask.copy()
    gt = np.zeros_like(mask)
    gt[2:4, 2:4, 2:4] = True
    err = volume_errors(pred, gt, spacing)
    assert err["vol_abs_error_ml"] == pytest.approx(abs(27 - 8) * 0.001)
    assert err["vol_rel_error_pct"] == pytest.approx(100.0 * err["vol_abs_error_ml"] / err["gt_vol_ml"])


def test_load_finetune_test_records_from_split_json(tmp_path: Path):
    for name in ("c1", "c2", "c3", "c4", "c5"):
        _make_corrected_study(tmp_path, name)
    split_path = tmp_path / "finetune_split.json"
    split_path.write_text(
        json.dumps({"train": ["c1"], "val": ["c2"], "test": ["c3", "c4", "c5"]}),
        encoding="utf-8",
    )
    records = load_finetune_test_records(tmp_path, split_json=split_path)
    assert [r["subject_id"] for r in records] == ["c3", "c4", "c5"]


def test_build_comparison_and_summary():
    brats_df = pd.DataFrame(
        {
            "subject_id": ["a", "b"],
            f"{MODEL_BRATS}_dice_whole_tumor": [0.5, 0.6],
            f"{MODEL_BRATS}_vol_rel_error_pct_whole_tumor": [40.0, 30.0],
            f"{MODEL_BRATS}_dice_mean": [0.55, 0.65],
            f"{MODEL_BRATS}_vol_rel_error_pct_mean": [35.0, 25.0],
        }
    )
    finetuned_df = pd.DataFrame(
        {
            "subject_id": ["a", "b"],
            f"{MODEL_FINETUNED}_dice_whole_tumor": [0.7, 0.75],
            f"{MODEL_FINETUNED}_vol_rel_error_pct_whole_tumor": [20.0, 15.0],
            f"{MODEL_FINETUNED}_dice_mean": [0.72, 0.78],
            f"{MODEL_FINETUNED}_vol_rel_error_pct_mean": [18.0, 12.0],
        }
    )
    comparison = build_comparison_table(brats_df, finetuned_df)
    assert "delta_dice_whole_tumor" in comparison.columns
    assert comparison.loc[0, "delta_dice_whole_tumor"] == pytest.approx(0.2)

    summary = summarize_comparison(comparison)
    assert len(summary) == 3
    improvement = summary.loc[summary["model"].str.contains("improvement")].iloc[0]
    assert improvement["mean_dice_whole_tumor"] == pytest.approx(0.175)


def test_plot_domain_gap_bars(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {
                "model": MODEL_BRATS,
                "n_cases": 2,
                "mean_dice_whole_tumor": 0.5,
                "mean_vol_rel_error_pct_whole_tumor": 30.0,
                "mean_dice_overall": 0.55,
                "mean_vol_rel_error_pct_overall": 25.0,
            },
            {
                "model": MODEL_FINETUNED,
                "n_cases": 2,
                "mean_dice_whole_tumor": 0.7,
                "mean_vol_rel_error_pct_whole_tumor": 15.0,
                "mean_dice_overall": 0.72,
                "mean_vol_rel_error_pct_overall": 12.0,
            },
        ]
    )
    for region in BRATS_REGIONS:
        if region != "whole_tumor":
            summary[f"mean_dice_{region}"] = 0.6
            summary[f"mean_vol_rel_error_pct_{region}"] = 20.0

    out = plot_domain_gap_bars(summary, tmp_path / "bars.png")
    assert out.is_file()


def test_evaluate_clinical_case_on_synthetic_volume():
    model = build_brats_model("segresnet", None)
    model.eval()
    post_pred, post_label = post_transforms()

    batch = {
        "image": torch.randn(1, 4, 32, 32, 32),
        "label": torch.zeros(1, 1, 32, 32, 32),
    }
    batch["label"][:, :, 10:20, 10:20, 10:20] = 3
    batch["label"][:, :, 8:22, 8:22, 8:22] = 2

    metrics = evaluate_clinical_case(
        model,
        batch,
        torch.device("cpu"),
        roi_size=(16, 16, 16),
        use_amp=False,
        post_pred=post_pred,
        post_label=post_label,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )
    assert set(metrics) >= {f"dice_{r}" for r in BRATS_REGIONS}
    assert set(metrics) >= {f"vol_rel_error_pct_{r}" for r in BRATS_REGIONS}
    assert "dice_mean" in metrics
