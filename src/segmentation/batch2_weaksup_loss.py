"""Scan-level (MIL) losses for weakly supervised batch2 fine-tuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from segmentation.model import BRATS_REGIONS

# Channel order: ET, TC, WT
ET_CH, TC_CH, WT_CH = 0, 1, 2


def topk_mean(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of top-k values over spatial dims. ``probs`` is (B, ...) or (...)."""
    flat = probs.reshape(probs.shape[0], -1) if probs.ndim >= 2 else probs.reshape(1, -1)
    k_use = min(int(k), flat.shape[1])
    if k_use <= 0:
        return flat.new_zeros(flat.shape[0])
    vals, _ = torch.topk(flat, k=k_use, dim=1)
    return vals.mean(dim=1)


def predicted_volume_ml(
    mask_prob: torch.Tensor,
    *,
    threshold: float = 0.5,
    voxel_ml: float = 0.001,
) -> torch.Tensor:
    """Soft-ish volume from thresholded probs; returns (B,) mL."""
    binary = (mask_prob >= threshold).float()
    return binary.reshape(binary.shape[0], -1).sum(dim=1) * float(voxel_ml)


def n_large_components(
    mask: torch.Tensor,
    *,
    min_voxels: int = 500,
) -> int:
    """Count connected components with ≥ ``min_voxels`` (single volume, no batch)."""
    arr = mask.detach().float().cpu().numpy() > 0.5
    if arr.ndim == 4:
        arr = arr[0]
    labeled, n = ndi.label(arr)
    if n == 0:
        return 0
    counts = ndi.sum(arr, labeled, index=list(range(1, n + 1)))
    return int(sum(1 for c in counts if c >= min_voxels))


def negative_scan_loss(
    logits: torch.Tensor,
    *,
    k: int = 200,
    volume_weight: float = 0.1,
    voxel_ml: float = 0.001,
) -> torch.Tensor:
    """Push top-k WT/TC/ET probs → 0; small penalty on predicted WT volume."""
    probs = torch.sigmoid(logits)
    losses = []
    for ch in (WT_CH, TC_CH, ET_CH):
        top = topk_mean(probs[:, ch], k)
        losses.append(F.binary_cross_entropy(top, torch.zeros_like(top)))
    vol = predicted_volume_ml(probs[:, WT_CH], voxel_ml=voxel_ml)
    # Soft volume: encourage near-zero (BCE against 0 on normalized vol proxy)
    vol_pen = (vol / 50.0).clamp(0, 5).mean()  # ~50 mL scale
    return sum(losses) / len(losses) + float(volume_weight) * vol_pen


def positive_scan_loss(
    logits: torch.Tensor,
    *,
    k: int = 200,
    max_wt_ml: float = 250.0,
    max_large_components: int = 3,
    encourage_nesting: bool = False,
    voxel_ml: float = 0.001,
    size_weight: float = 0.2,
    component_weight: float = 0.1,
    nesting_weight: float = 0.1,
) -> torch.Tensor:
    """MIL top-k WT → 1, plus size / component / optional nesting priors."""
    probs = torch.sigmoid(logits)
    top = topk_mean(probs[:, WT_CH], k)
    mil = F.binary_cross_entropy(top, torch.ones_like(top))

    vol = predicted_volume_ml(probs[:, WT_CH], voxel_ml=voxel_ml)
    # Penalize excess above max_wt_ml
    excess = F.relu(vol - float(max_wt_ml)) / float(max_wt_ml)
    size_pen = excess.mean()

    # Component penalty (non-diff; stop-grad bridge via volume soft proxy)
    with torch.no_grad():
        n_large = 0
        for b in range(probs.shape[0]):
            n_large += max(
                0,
                n_large_components(probs[b, WT_CH], min_voxels=500)
                - int(max_large_components),
            )
        comp_pen = float(n_large) / max(1, probs.shape[0])

    loss = mil + float(size_weight) * size_pen + float(component_weight) * comp_pen

    if encourage_nesting:
        # Soft nesting: ET ⊆ TC ⊆ WT (penalize ET outside TC, TC outside WT)
        et, tc, wt = probs[:, ET_CH], probs[:, TC_CH], probs[:, WT_CH]
        nest = ((et * (1.0 - tc)).mean() + (tc * (1.0 - wt)).mean())
        loss = loss + float(nesting_weight) * nest

    return loss


def weaksup_batch_loss(
    logits: torch.Tensor,
    *,
    is_negative: torch.Tensor,
    is_native_t1c: torch.Tensor | None = None,
    k: int = 200,
) -> torch.Tensor:
    """Dispatch per-sample weak losses then mean."""
    losses = []
    for i in range(logits.shape[0]):
        li = logits[i : i + 1]
        if bool(is_negative[i].item()):
            losses.append(negative_scan_loss(li, k=k))
        else:
            nest = False
            if is_native_t1c is not None:
                nest = bool(is_native_t1c[i].item())
            losses.append(positive_scan_loss(li, k=k, encourage_nesting=nest))
    return torch.stack(losses).mean()
