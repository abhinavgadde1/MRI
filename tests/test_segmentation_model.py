"""Tests for BraTS segmentation model configuration."""

import torch

from segmentation.model import (
    BRATS_REGIONS,
    IN_CHANNELS,
    OUT_CHANNELS,
    BraTSModelConfig,
    brats_label_to_regions,
    build_brats_model,
    build_segresnet,
    build_unet,
)


def test_brats_region_mapping():
    # Single voxel labels in a tiny volume
    label = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)  # (1,1,2,2)
    regions = brats_label_to_regions(label)
    assert regions.shape == (1, 3, 2, 2)
    # index (0,0,1,1) is label 3 -> ET=1, TC=1, WT=1
    assert regions[0, 0, 1, 1] == 1.0
    assert regions[0, 1, 1, 1] == 1.0
    assert regions[0, 2, 1, 1] == 1.0
    # label 2 -> edema only in WT
    assert regions[0, 0, 1, 0] == 0.0
    assert regions[0, 1, 1, 0] == 0.0
    assert regions[0, 2, 1, 0] == 1.0


def test_model_io_shapes():
    cfg = BraTSModelConfig(init_filters=8, channels=(8, 16, 32, 64, 128))
    x = torch.randn(2, IN_CHANNELS, 32, 32, 32)

    segresnet = build_segresnet(cfg)
    assert segresnet(x).shape == (2, OUT_CHANNELS, 32, 32, 32)

    unet = build_unet(cfg)
    assert unet(x).shape == (2, OUT_CHANNELS, 32, 32, 32)

    default = build_brats_model("segresnet")
    assert default(x).shape == (2, OUT_CHANNELS, 32, 32, 32)


def test_region_names():
    assert BRATS_REGIONS == ("enhancing_tumor", "tumor_core", "whole_tumor")
    assert OUT_CHANNELS == 3
    assert IN_CHANNELS == 4
