"""MONAI segmentation models for multi-modal BraTS tumor segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from monai.networks.nets import SegResNet, UNet

from preprocessing.h5_to_nifti import MODALITY_NAMES

# Input channel order matches cached NIfTIs / dataset stacking.
INPUT_MODALITIES: tuple[str, ...] = MODALITY_NAMES  # FLAIR, T1, T1c, T2
IN_CHANNELS = len(INPUT_MODALITIES)

# BraTS evaluation regions (3 output channels, multi-label style).
# Derived from single-channel labels 0–3 stored in mask.nii.gz:
#   0 = background
#   1 = necrotic / non-enhancing tumor core
#   2 = edema (peritumoral)
#   3 = enhancing tumor
BRATS_REGIONS: tuple[str, ...] = (
    "enhancing_tumor",   # ET: label == 3
    "tumor_core",        # TC: labels 1 + 3
    "whole_tumor",       # WT: labels 1 + 2 + 3 (includes edema)
)
OUT_CHANNELS = len(BRATS_REGIONS)

ModelName = Literal["segresnet", "unet"]


@dataclass(frozen=True)
class BraTSModelConfig:
    """Architecture hyperparameters for BraTS segmentation."""

    spatial_dims: int = 3
    in_channels: int = IN_CHANNELS
    out_channels: int = OUT_CHANNELS
    # SegResNet
    init_filters: int = 16
    blocks_down: tuple[int, ...] = (1, 2, 2, 4)
    blocks_up: tuple[int, ...] = (1, 1, 1)
    dropout_prob: float = 0.2
    # UNet
    channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2


def brats_label_to_regions(label: torch.Tensor) -> torch.Tensor:
    """Convert single-channel BraTS labels (0–3) to 3 binary region channels.

    Parameters
    ----------
    label:
        Tensor shaped ``(B, 1, *spatial)`` or ``(B, *spatial)`` with integer
        values in ``{0, 1, 2, 3}``.

    Returns
    -------
    torch.Tensor
        Tensor shaped ``(B, 3, *spatial)`` with channels ``[ET, TC, WT]``.
    """
    if label.ndim < 3:
        raise ValueError(f"Label must be at least 3D, got shape {tuple(label.shape)}")
    if label.shape[1] != 1:
        label = label.unsqueeze(1)
    if label.shape[1] != 1:
        raise ValueError(f"Expected one label channel, got shape {tuple(label.shape)}")

    lbl = label.long()
    necrotic = lbl == 1
    edema = lbl == 2
    enhancing = lbl == 3

    et = enhancing
    tc = necrotic | enhancing
    wt = necrotic | edema | enhancing

    return torch.cat([et, tc, wt], dim=1).float()


def build_segresnet(config: BraTSModelConfig | None = None) -> SegResNet:
    """Build a 3D SegResNet for 4-modality input and 3 BraTS region outputs."""
    cfg = config or BraTSModelConfig()
    return SegResNet(
        spatial_dims=cfg.spatial_dims,
        init_filters=cfg.init_filters,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        blocks_down=cfg.blocks_down,
        blocks_up=cfg.blocks_up,
        dropout_prob=cfg.dropout_prob,
    )


def build_unet(config: BraTSModelConfig | None = None) -> UNet:
    """Build a 3D U-Net for 4-modality input and 3 BraTS region outputs."""
    cfg = config or BraTSModelConfig()
    return UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        channels=cfg.channels,
        strides=cfg.strides,
        num_res_units=cfg.num_res_units,
    )


def build_brats_model(
    name: ModelName = "segresnet",
    config: BraTSModelConfig | None = None,
) -> torch.nn.Module:
    """Factory for BraTS-configured SegResNet (default) or UNet."""
    builders = {
        "segresnet": build_segresnet,
        "unet": build_unet,
    }
    try:
        builder = builders[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose from {list(builders)}") from exc
    return builder(config)
