"""Unit tests for experimental Claude-draft fine-tune helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from segmentation.experimental_draft_ft import (
    CKPT_DIR,
    DRAFT_STUDIES,
    WT_ONLY_STUDIES,
    channel_weighted_dicece,
    write_model_card,
)


def test_channel_weighted_dicece_masks_tc_et():
    """Zero weights on ET/TC must ignore those channels (WT-only drafts)."""
    torch.manual_seed(0)
    b, c, s = 1, 3, 8
    logits = torch.randn(b, c, s, s, s)
    labels = (torch.rand(b, c, s, s, s) > 0.7).float()
    # Perfect WT logits, garbage ET/TC
    logits_wt = logits.clone()
    logits_wt[:, 2] = 20.0 * labels[:, 2] - 20.0 * (1.0 - labels[:, 2])
    logits_wt[:, 0] = 20.0  # wrong ET
    logits_wt[:, 1] = 20.0  # wrong TC

    wt_only = torch.tensor([0.0, 0.0, 1.0])
    full = torch.tensor([1.0, 1.0, 1.0])
    loss_wt = channel_weighted_dicece(logits_wt, labels, wt_only)
    loss_full = channel_weighted_dicece(logits_wt, labels, full)
    assert loss_wt.item() < 0.2
    assert loss_full.item() > loss_wt.item()


def test_wt_only_study_ids():
    assert WT_ONLY_STUDIES == frozenset({"batch2_01", "batch2_12"})
    assert set(DRAFT_STUDIES) >= WT_ONLY_STUDIES


def test_ckpt_dir_not_brats():
    assert CKPT_DIR.name == "experimental_draft_ft"
    assert not str(CKPT_DIR).endswith("brats_scale_full")


def test_model_card_disclaimer(tmp_path: Path):
    path = tmp_path / "MODEL_CARD.md"
    write_model_card(path)
    text = path.read_text(encoding="utf-8")
    assert "experimental" in text.lower()
    assert "unverified" in text.lower()
    assert "not for clinical" in text.lower()
    assert "not the default" in text.lower()
