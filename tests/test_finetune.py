"""Tests for clinical fine-tuning utilities."""

from pathlib import Path

import pytest

from segmentation.finetune import (
    CORRECTED_LABEL_NAMES,
    FinetuneConfig,
    corrected_case_to_datadict,
    discover_corrected_cases,
    freeze_early_encoder,
    split_finetune_sets,
)
from segmentation.model import build_segresnet, build_unet
from segmentation.pseudo_label import REGISTERED_NAMES


def _make_corrected_study(tmp_path: Path, name: str) -> Path:
    study = tmp_path / name
    reg = study / "04_registered_1mm"
    reg.mkdir(parents=True)
    for fname in REGISTERED_NAMES.values():
        (reg / fname).write_bytes(b"")
    (reg / CORRECTED_LABEL_NAMES[0]).write_bytes(b"")
    return study


def test_discover_and_datadict(tmp_path: Path):
    _make_corrected_study(tmp_path, "case_a")
    cases = discover_corrected_cases(tmp_path)
    assert len(cases) == 1
    record = corrected_case_to_datadict(cases[0])
    assert record["subject_id"] == "case_a"
    assert record["label"].endswith("corrected_tumor_label.nii.gz")
    assert len(record["image"]) == 4


def test_split_finetune_sets_holds_out_test():
    records = [{"subject_id": f"c{i}"} for i in range(10)]
    train, val, test = split_finetune_sets(records, n_test_cases=4, seed=0)
    assert len(test) == 4
    assert len(train) + len(val) + len(test) == 10
    ids = {r["subject_id"] for r in train + val + test}
    assert len(ids) == 10


def test_split_requires_minimum_cases():
    records = [{"subject_id": "a"}, {"subject_id": "b"}]
    with pytest.raises(ValueError):
        split_finetune_sets(records)


def test_freeze_encoder_segresnet():
    model = build_segresnet()
    n = freeze_early_encoder(model, "segresnet", n_down_blocks=2)
    assert n > 0
    trainable = [p.requires_grad for p in model.convInit.parameters()]
    assert not any(trainable)


def test_freeze_encoder_unet():
    model = build_unet()
    n = freeze_early_encoder(model, "unet", n_down_blocks=1)
    assert n > 0


def test_finetune_config_defaults():
    cfg = FinetuneConfig()
    assert cfg.learning_rate == 1e-5
    assert cfg.min_test_cases == 3
    assert cfg.max_test_cases == 5
