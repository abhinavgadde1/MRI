"""Tests for BraTS evaluation utilities (synthetic tensors only)."""

import math

import pytest
import torch

from segmentation.evaluate import EvalConfig, _evaluate_case, _metric_value
from segmentation.model import BRATS_REGIONS, build_brats_model
from segmentation.train import post_transforms


def test_metric_value_handles_nan():
    assert math.isnan(_metric_value(torch.tensor([float("nan")]), 0))
    assert _metric_value(torch.tensor([0.8, 0.9]), 1) == pytest.approx(0.9)


def test_evaluate_case_on_synthetic_volume():
    model = build_brats_model("segresnet", None)
    model.eval()
    post_pred, post_label = post_transforms()

    batch = {
        "image": torch.randn(1, 4, 32, 32, 32),
        "label": torch.zeros(1, 1, 32, 32, 32),
    }
    batch["label"][:, :, 10:20, 10:20, 10:20] = 3
    batch["label"][:, :, 8:22, 8:22, 8:22] = 2

    metrics = _evaluate_case(
        model,
        batch,
        torch.device("cpu"),
        roi_size=(16, 16, 16),
        use_amp=False,
        post_pred=post_pred,
        post_label=post_label,
    )
    assert set(metrics) >= {f"dice_{r}" for r in BRATS_REGIONS} | {f"hd95_{r}" for r in BRATS_REGIONS}
    assert "dice_mean" in metrics


def test_evaluate_case_empty_et_is_nan():
    """Empty GT for a region must yield NaN Dice/HD95 (excluded from aggregates)."""
    model = build_brats_model("segresnet", None)
    model.eval()
    post_pred, post_label = post_transforms()

    # Labels 1+2 only → TC and WT nonempty; ET empty.
    batch = {
        "image": torch.randn(1, 4, 32, 32, 32),
        "label": torch.zeros(1, 1, 32, 32, 32),
    }
    batch["label"][:, :, 8:20, 8:20, 8:20] = 2
    batch["label"][:, :, 10:18, 10:18, 10:18] = 1

    metrics = _evaluate_case(
        model,
        batch,
        torch.device("cpu"),
        roi_size=(16, 16, 16),
        use_amp=False,
        post_pred=post_pred,
        post_label=post_label,
    )
    assert math.isnan(metrics["dice_enhancing_tumor"])
    assert math.isnan(metrics["hd95_enhancing_tumor"])
    assert metrics["dice_tumor_core"] == metrics["dice_tumor_core"]  # finite
    assert metrics["dice_whole_tumor"] == metrics["dice_whole_tumor"]


def test_eval_config_defaults():
    cfg = EvalConfig()
    assert cfg.val_fraction == 0.2
    assert cfg.roi_size == (96, 96, 96)
    assert cfg.postprocess == "raw"
