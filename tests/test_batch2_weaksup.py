"""Tests for weakly supervised batch2 losses, balancing, and checkpoint guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from segmentation.batch2_augment import DomainAugmentFlags, DomainAugmentd, apply_zscore_brain
from segmentation.batch2_finetune import assert_safe_checkpoint_output_dir
from segmentation.batch2_weaksup import (
    sample_balanced_pos_neg,
    shuffle_labels,
    classic_label_from_regions,
)
from segmentation.batch2_weaksup_loss import (
    negative_scan_loss,
    positive_scan_loss,
    topk_mean,
    weaksup_batch_loss,
)
from config import PROJECT_ROOT


def test_topk_mean_and_negative_loss_pushes_low():
    logits = torch.full((1, 3, 8, 8, 8), -4.0)
    # Spike a few voxels high on WT
    logits[0, 2, 0, 0, :5] = 8.0
    loss = negative_scan_loss(logits, k=10)
    assert float(loss.item()) > 0.1
    # Zero logits → near-zero top-k probs → lower loss
    low = negative_scan_loss(torch.full_like(logits, -8.0), k=10)
    assert float(low.item()) < float(loss.item())


def test_positive_mil_loss_finite():
    logits = torch.randn(1, 3, 16, 16, 16) * 0.1
    loss = positive_scan_loss(logits, k=50, encourage_nesting=True)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0


def test_weaksup_batch_loss_mixed():
    logits = torch.randn(2, 3, 12, 12, 12)
    is_neg = torch.tensor([True, False])
    is_native = torch.tensor([False, True])
    loss = weaksup_batch_loss(logits, is_negative=is_neg, is_native_t1c=is_native, k=30)
    assert torch.isfinite(loss)


def test_balanced_sampling_equal_counts():
    records = (
        [{"subject_id": f"p{i}", "is_negative": False} for i in range(5)]
        + [{"subject_id": f"n{i}", "is_negative": True} for i in range(20)]
    )
    out = sample_balanced_pos_neg(records, seed=42, epoch=0)
    n_pos = sum(1 for r in out if not r["is_negative"])
    n_neg = sum(1 for r in out if r["is_negative"])
    assert n_pos == n_neg == 5


def test_shuffle_preserves_counts():
    records = (
        [{"subject_id": f"p{i}", "is_negative": False, "cohort": "positive"} for i in range(3)]
        + [{"subject_id": f"n{i}", "is_negative": True, "cohort": "negative"} for i in range(7)]
    )
    sh = shuffle_labels(records, seed=0)
    assert sum(1 for r in sh if r["is_negative"]) == 7
    assert sum(1 for r in sh if not r["is_negative"]) == 3


def test_refuse_brats_scale_full(tmp_path: Path, monkeypatch):
    protected = PROJECT_ROOT / "checkpoints" / "brats_scale_full"
    with pytest.raises(ValueError):
        assert_safe_checkpoint_output_dir(protected)


def test_weaksup_fold_ckpt_path_allowed(tmp_path: Path):
    out = assert_safe_checkpoint_output_dir(tmp_path / "batch2_weaksup_fold_0")
    assert out.name == "batch2_weaksup_fold_0"


def test_zscore_and_t1c_hide_flags():
    flags = DomainAugmentFlags(
        thick_slice=False,
        intensity=False,
        modality_dropout=False,
        weaksup_t1c_hide=True,
        weaksup_t1c_hide_neg_always=True,
        zscore_brain=True,
        inplane_rescale=False,
        blur=False,
    )
    names = flags.enabled_names()
    assert any("weaksup_t1c_hide" in n for n in names)
    assert any("zscore" in n for n in names)

    img = torch.randn(4, 8, 8, 8)
    img[:, :2, :2, :2] = 0  # background corner
    out = apply_zscore_brain(img)
    assert out.shape == img.shape

    aug = DomainAugmentd(keys=["image"], flags=flags)
    d = aug({"image": img.clone(), "is_negative": True, "label": torch.zeros(1, 8, 8, 8)})
    # Negatives: T1c replaced by T1 → channels equal
    assert torch.allclose(d["image"][2], d["image"][1])


def test_classic_label_nesting():
    regs = np.zeros((3, 4, 4, 4), dtype=np.float32)
    regs[2, 1:3, 1:3, 1:3] = 1  # WT
    regs[1, 1:2, 1:2, 1:2] = 1  # TC
    regs[0, 1, 1, 1] = 1  # ET
    lab = classic_label_from_regions(regs)
    assert lab[1, 1, 1] == 4
    assert int((lab == 2).sum()) >= 1


def test_topk_mean_shape():
    t = torch.arange(100, dtype=torch.float32).reshape(1, 10, 10)
    m = topk_mean(t, k=5)
    assert m.shape == (1,)
    assert float(m.item()) == pytest.approx(float(torch.arange(95, 100).float().mean()))
