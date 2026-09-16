"""Tests for BraTS training utilities."""

from segmentation.train import TrainConfig, split_brats_cases


def test_split_brats_cases_80_20():
    train, val = split_brats_cases("data/processed/brats_nifti", val_fraction=0.2, seed=42)
    assert len(train) + len(val) == 369
    assert len(train) == 295
    assert len(val) == 74


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.val_fraction == 0.2
    assert cfg.model_name == "segresnet"
    assert cfg.amp is True
