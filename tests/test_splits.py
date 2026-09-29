"""Tests for BraTS train / held-out split integrity."""

from pathlib import Path

import pytest

from config import PROJECT_ROOT
from segmentation.splits import (
    create_heldout_split,
    overlap_report,
    read_id_list,
    recover_training_cases_from_checkpoint,
    write_id_list,
)


def test_heldout_and_train_lists_disjoint_on_disk():
    train_path = PROJECT_ROOT / "splits" / "train.txt"
    heldout_path = PROJECT_ROOT / "splits" / "heldout.txt"
    if not train_path.is_file() or not heldout_path.is_file():
        pytest.skip("splits not generated yet")
    train = set(read_id_list(train_path))
    heldout = set(read_id_list(heldout_path))
    assert train, "train.txt empty"
    assert heldout, "heldout.txt empty"
    assert train.isdisjoint(heldout)
    assert len(heldout) == 60


def test_create_heldout_split_disjoint(tmp_path: Path):
    train_ids = [f"case_{i:03d}" for i in range(40)]
    # Fake brats root is not used when we only test set logic via exclude — use real if available
    brats = PROJECT_ROOT / "data" / "processed" / "brats_nifti"
    if not brats.is_dir() or not any(brats.iterdir()):
        pytest.skip("BraTS NIfTI cache missing")
    train_out, heldout = create_heldout_split(
        brats_nifti_root=brats,
        train_ids=train_ids,
        exclude_ids=train_ids,
        n_heldout=60,
        seed=42,
    )
    assert set(train_out).isdisjoint(heldout)
    assert len(heldout) == 60
    # Deterministic
    _, heldout2 = create_heldout_split(
        brats_nifti_root=brats,
        train_ids=train_ids,
        exclude_ids=train_ids,
        n_heldout=60,
        seed=42,
    )
    assert heldout == heldout2


def test_recover_and_overlap_if_checkpoint_exists():
    ckpt = PROJECT_ROOT / "checkpoints" / "brats_pretrain" / "best_model.pt"
    eval_csv = PROJECT_ROOT / "checkpoints" / "brats_eval_metrics.csv"
    if not ckpt.is_file() or not eval_csv.is_file():
        pytest.skip("checkpoint or eval CSV missing")
    train_ids, run_val_ids, meta = recover_training_cases_from_checkpoint(ckpt)
    assert meta["n_train"] == len(train_ids)
    assert set(train_ids).isdisjoint(run_val_ids)
    report = overlap_report(eval_csv, train_ids, run_val_ids)
    assert report["n_eval"] == 50
    assert "overlap_eval_vs_train" in report


def test_write_read_id_list(tmp_path: Path):
    path = write_id_list(tmp_path / "ids.txt", ["a", "b", "c"])
    assert read_id_list(path) == ["a", "b", "c"]
