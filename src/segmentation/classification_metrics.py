"""Voxel-level binary classification metrics (confusion matrix + derived scores)."""

from __future__ import annotations

from typing import Any

import numpy as np

REGION_SHORT: dict[str, str] = {
    "enhancing_tumor": "ET",
    "tumor_core": "TC",
    "whole_tumor": "WT",
}


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    """Return TP/FP/FN/TN voxel counts for one binary mask pair."""
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch pred {p.shape} vs gt {g.shape}")
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, np.logical_not(g)).sum())
    fn = int(np.logical_and(np.logical_not(p), g).sum())
    tn = int(np.logical_and(np.logical_not(p), np.logical_not(g)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def scores_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, float]:
    """Accuracy / precision / recall / specificity / F1 / Dice from CM counts.

    F1 and binary Dice are identical: ``2 TP / (2 TP + FP + FN)``.
    Undefined ratios (0/0) are returned as NaN.
    """
    tp_f, fp_f, fn_f, tn_f = float(tp), float(fp), float(fn), float(tn)
    total = tp_f + fp_f + fn_f + tn_f
    acc = (tp_f + tn_f) / total if total > 0 else float("nan")
    prec = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else float("nan")
    rec = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else float("nan")
    spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else float("nan")
    denom = 2.0 * tp_f + fp_f + fn_f
    # Binary Dice and F1 are the same formula; use it even when precision is
    # undefined (empty prediction, nonempty GT → both 0).
    dice = (2.0 * tp_f / denom) if denom > 0 else float("nan")
    f1 = dice
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "dice": dice,
    }


def prevalence(gt_positive: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    return float(gt_positive) / float(total)


def nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def format_float(value: float, digits: int = 4) -> str:
    if value != value:  # NaN
        return "—"
    return f"{value:.{digits}f}"


def row_from_counts(
    *,
    model: str,
    region: str,
    aggregation: str,
    n_cases: int,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = scores_from_counts(tp, fp, fn, tn)
    total = tp + fp + fn + tn
    row: dict[str, Any] = {
        "model": model,
        "region": region,
        "aggregation": aggregation,
        "n_cases": n_cases,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_voxels": total,
        "gt_positive": tp + fn,
        "pred_positive": tp + fp,
        "prevalence": prevalence(tp + fn, total),
        **scores,
    }
    if extra:
        row.update(extra)
    return row
