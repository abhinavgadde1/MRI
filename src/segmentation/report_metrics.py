"""Bootstrap summary tables for BraTS segmentation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from segmentation.model import BRATS_REGIONS

# Short names for the paper table
REGION_SHORT = {
    "enhancing_tumor": "ET",
    "tumor_core": "TC",
    "whole_tumor": "WT",
}


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values`` (NaNs dropped)."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(vals, size=vals.size, replace=True)
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Mean paired difference (b−a) with percentile bootstrap 95% CI."""
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    diff = b - a
    mean_diff = float(np.mean(diff))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, diff.size, size=diff.size)
        boots[i] = float(np.mean(diff[idx]))
    return mean_diff, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarize_segmentation_csv(
    metrics_csv: str | Path,
    output_csv: str | Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Write mean±SD, median, and bootstrap 95% CI for Dice/HD95 by region."""
    df = pd.read_csv(metrics_csv)
    rows: list[dict] = []
    for region in BRATS_REGIONS:
        short = REGION_SHORT[region]
        for metric, col in (("Dice", f"dice_{region}"), ("HD95", f"hd95_{region}")):
            series = df[col].to_numpy(dtype=float)
            finite = series[np.isfinite(series)]
            mean = float(np.mean(finite)) if finite.size else float("nan")
            std = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            median = float(np.median(finite)) if finite.size else float("nan")
            lo, hi = _bootstrap_ci(series, n_resamples=n_bootstrap, seed=seed)
            rows.append(
                {
                    "region": short,
                    "metric": metric,
                    "n": int(finite.size),
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "mean_pm_sd": f"{mean:.4f} ± {std:.4f}" if finite.size else "",
                    "ci95": f"[{lo:.4f}, {hi:.4f}]" if finite.size else "",
                }
            )

    out = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap segmentation summary table.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "brats_eval_heldout.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "segmentation_table.csv",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    table = summarize_segmentation_csv(
        args.metrics_csv,
        args.output_csv,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(table.to_string(index=False))
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
