"""Bland–Altman agreement statistics and tidy plot frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BlandAltmanResult:
    mean_diff: float
    std_diff: float
    loa_lower: float
    loa_upper: float
    n: int


def bland_altman_stats(method_a: np.ndarray | list[float], method_b: np.ndarray | list[float]) -> BlandAltmanResult:
    """Compute Bland–Altman mean difference and 95% limits of agreement."""
    a = np.asarray(method_a, dtype=float).ravel()
    b = np.asarray(method_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("method_a and method_b must have the same shape")
    if a.size < 2:
        raise ValueError("Need at least two paired measurements")

    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    return BlandAltmanResult(
        mean_diff=mean_diff,
        std_diff=std_diff,
        loa_lower=mean_diff - 1.96 * std_diff,
        loa_upper=mean_diff + 1.96 * std_diff,
        n=int(a.size),
    )


def bland_altman_plot_frame(
    method_a: np.ndarray | list[float],
    method_b: np.ndarray | list[float],
    *,
    label_a: str = "method_a",
    label_b: str = "method_b",
) -> pd.DataFrame:
    """Return a DataFrame of means and differences for Bland–Altman plotting."""
    a = np.asarray(method_a, dtype=float).ravel()
    b = np.asarray(method_b, dtype=float).ravel()
    return pd.DataFrame(
        {
            "mean": (a + b) / 2.0,
            "diff": a - b,
            label_a: a,
            label_b: b,
        }
    )
