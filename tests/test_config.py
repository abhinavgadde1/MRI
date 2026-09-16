"""Tests for YAML config loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from config import PROJECT_ROOT, load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.paths.raw_dicom == (PROJECT_ROOT / "MRI DATA").resolve()
    assert cfg.paths.brats == (
        PROJECT_ROOT / "archive" / "BraTS2020_training_data" / "content" / "data"
    ).resolve()
    assert cfg.paths.processed == (PROJECT_ROOT / "data" / "processed").resolve()
    assert cfg.paths.checkpoints == (PROJECT_ROOT / "checkpoints").resolve()
    assert cfg.training.pretrain.batch_size == 2
    assert cfg.training.pretrain.learning_rate == 1e-4
    assert cfg.training.pretrain.epochs == 100
    assert cfg.training.finetune.batch_size == 1
    assert cfg.training.finetune.learning_rate == 1e-5
    assert cfg.training.finetune.epochs == 50
    assert cfg.paths.raw_dicom.is_dir()
    assert cfg.paths.brats.is_dir()


def test_rejects_invalid_hyperparameters(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
paths:
  raw_dicom: "MRI DATA"
  brats: "archive/BraTS2020_training_data/content/data"
  processed: "data/processed"
  checkpoints: "checkpoints"
training:
  pretrain:
    batch_size: 0
    learning_rate: 1.0e-4
    epochs: 10
  finetune:
    batch_size: 1
    learning_rate: 1.0e-5
    epochs: 5
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(bad)
