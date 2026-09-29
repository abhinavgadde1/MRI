"""Tests for BraTS training utilities (no real training)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.losses import DiceLoss

from preprocessing.h5_to_nifti import MODALITY_NAMES
from segmentation.train import (
    TrainConfig,
    assert_disjoint_from_heldout,
    assert_loss_covers_all_channels,
    build_brats_loss,
    build_channel_dice_loss,
    post_transforms,
    split_brats_cases,
)


def _write_subject(root: Path, name: str) -> None:
    subject = root / name
    subject.mkdir(parents=True)
    affine = np.eye(4)
    for modality in MODALITY_NAMES:
        data = np.random.randn(8, 8, 8).astype(np.float32)
        nib.save(nib.Nifti1Image(data, affine), str(subject / f"{modality}.nii.gz"))
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 2
    nib.save(nib.Nifti1Image(mask, affine), str(subject / "mask.nii.gz"))


def test_split_brats_cases_80_20(tmp_path: Path):
    for i in range(10):
        _write_subject(tmp_path, f"case_{i:03d}")
    train, val = split_brats_cases(tmp_path, val_fraction=0.2, seed=42)
    assert len(train) + len(val) == 10
    assert len(train) == 8
    assert len(val) == 2


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.val_fraction == 0.2
    assert cfg.model_name == "segresnet"
    assert cfg.amp is True


def test_post_transforms_sigmoid_threshold():
    pred_t, label_t = post_transforms()
    assert len(pred_t.transforms) == 2
    assert pred_t.transforms[0].__class__.__name__ == "Activations"
    assert len(label_t.transforms) == 0


def test_brats_loss_covers_all_three_channels():
    """Guard: include_background must be True and Dice must see 3 channels."""
    loss_fn = build_brats_loss()
    assert loss_fn.dice.include_background is True
    assert_loss_covers_all_channels(loss_fn)

    channel_loss = build_channel_dice_loss()
    assert channel_loss.include_background is True
    logits = torch.randn(2, 3, 8, 8, 8)
    target = torch.randint(0, 2, (2, 3, 8, 8, 8)).float()
    per_ch = channel_loss(logits, target)
    assert per_ch.shape[1] == 3

    # The historical bug: include_background=False drops ET (channel 0).
    buggy = DiceLoss(sigmoid=True, include_background=False, reduction="none")
    buggy_out = buggy(logits, target)
    assert buggy_out.shape[1] == 2
    with pytest.raises(AssertionError, match="include_background=False"):
        assert_loss_covers_all_channels(buggy)


def test_assert_disjoint_from_heldout():
    assert_disjoint_from_heldout(["a", "b"], ["c"], ["d", "e"])
    with pytest.raises(AssertionError, match="Held-out leakage"):
        assert_disjoint_from_heldout(["a", "x"], ["c"], ["x", "y"])


def test_best_epoch_from_v2_metrics_fixture():
    """Regression fixture: metrics.csv epoch 15 beats 14; selection must pick 15."""
    from segmentation.train import best_epoch_from_metrics, update_best_dice

    metrics_csv = (
        Path(__file__).resolve().parents[1]
        / "checkpoints"
        / "brats_v2_loss_fix"
        / "metrics.csv"
    )
    if not metrics_csv.is_file():
        pytest.skip("brats_v2_loss_fix/metrics.csv not present")

    import pandas as pd

    df = pd.read_csv(metrics_csv)
    assert len(df) == 20
    best_epoch = best_epoch_from_metrics(df)
    assert best_epoch == 15
    dice_14 = float(df.loc[df["epoch"] == 14, "val_dice_mean"].iloc[0])
    dice_15 = float(df.loc[df["epoch"] == 15, "val_dice_mean"].iloc[0])
    assert dice_15 > dice_14
    assert abs(dice_14 - 0.2257) < 1e-3
    assert abs(dice_15 - 0.3497) < 1e-3

    best, is_new = update_best_dice(dice_14, -1.0)
    assert is_new and abs(best - dice_14) < 1e-9
    best, is_new = update_best_dice(dice_15, best)
    assert is_new and abs(best - dice_15) < 1e-9
    best, is_new = update_best_dice(0.25, best)
    assert not is_new and abs(best - dice_15) < 1e-9


def test_assert_best_checkpoint_matches_metrics_detects_stale_epoch(tmp_path: Path):
    """Fail when checkpoint epoch is not the metrics argmax (the historical bug)."""
    import shutil

    from segmentation.train import assert_best_checkpoint_matches_metrics

    metrics_csv = (
        Path(__file__).resolve().parents[1]
        / "checkpoints"
        / "brats_v2_loss_fix"
        / "metrics.csv"
    )
    if not metrics_csv.is_file():
        pytest.skip("brats_v2_loss_fix/metrics.csv not present")

    import pandas as pd

    fixture = tmp_path / "metrics.csv"
    shutil.copy(metrics_csv, fixture)
    df = pd.read_csv(fixture)
    wrong_epoch = 14
    wrong_dice = float(df.loc[df["epoch"] == wrong_epoch, "val_dice_mean"].iloc[0])

    bad_ckpt = tmp_path / "best_model.pt"
    torch.save(
        {
            "epoch": wrong_epoch,
            "epoch_1based": True,
            "val_dice": wrong_dice,
            "best_val_dice": wrong_dice,
            "model_state": {},
        },
        bad_ckpt,
    )
    with pytest.raises(AssertionError, match="argmax"):
        assert_best_checkpoint_matches_metrics(bad_ckpt, fixture)

    good_epoch = 15
    good_dice = float(df.loc[df["epoch"] == good_epoch, "val_dice_mean"].iloc[0])
    good_ckpt = tmp_path / "best_model_ok.pt"
    torch.save(
        {
            "epoch": good_epoch,
            "epoch_1based": True,
            "val_dice": good_dice,
            "best_val_dice": good_dice,
            "model_state": {},
        },
        good_ckpt,
    )
    assert_best_checkpoint_matches_metrics(good_ckpt, fixture)


def test_checkpoint_epoch_1based_legacy_compat():
    from segmentation.train import checkpoint_epoch_1based

    assert checkpoint_epoch_1based({"epoch": 14, "epoch_1based": True}) == 14
    assert checkpoint_epoch_1based({"epoch": 14}) == 15  # legacy 0-based
