"""Post-training held-out comparison: v1 vs v2 (inference only)."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from scipy import stats as sp_stats

from config import PROJECT_ROOT, load_config
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS, brats_label_to_regions
from segmentation.postprocess import fragmentation_stats
from segmentation.report_metrics import REGION_SHORT, _bootstrap_ci
from segmentation.splits import read_id_list
from validation.bland_altman import bland_altman_stats
from validation.ellipsoid import voxel_volume_ml_from_binary
from validation.ellipsoid_batch import (
    _bootstrap_agreement_cis,
    icc_2_1,
)


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


def build_v1_v2_table(
    v1_csv: Path,
    v2_csv: Path,
    output_csv: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Side-by-side wide table: v1-lcc vs v2-lcc Dice/HD95 with bootstrap CIs."""
    v1 = pd.read_csv(v1_csv)
    v2 = pd.read_csv(v2_csv)
    rows: list[dict[str, Any]] = []
    for region in BRATS_REGIONS:
        short = REGION_SHORT[region]
        for metric, col in (("Dice", f"dice_{region}"), ("HD95", f"hd95_{region}")):
            row: dict[str, Any] = {"region": short, "metric": metric}
            for prefix, df in (("v1_lcc", v1), ("v2_lcc", v2)):
                series = df[col].to_numpy(dtype=float)
                finite = series[np.isfinite(series)]
                mean = float(np.mean(finite)) if finite.size else float("nan")
                std = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
                median = float(np.median(finite)) if finite.size else float("nan")
                lo, hi = _bootstrap_ci(series, n_resamples=n_bootstrap, seed=seed)
                row[f"{prefix}_n"] = int(finite.size)
                row[f"{prefix}_mean"] = mean
                row[f"{prefix}_std"] = std
                row[f"{prefix}_median"] = median
                row[f"{prefix}_ci95_low"] = lo
                row[f"{prefix}_ci95_high"] = hi
                row[f"{prefix}_mean_pm_sd"] = (
                    f"{mean:.4f} ± {std:.4f}" if finite.size else ""
                )
                row[f"{prefix}_ci95"] = f"[{lo:.4f}, {hi:.4f}]" if finite.size else ""
            rows.append(row)
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def paired_v1_v2(
    v1_csv: Path,
    v2_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    v1 = pd.read_csv(v1_csv).set_index("subject_id")
    v2 = pd.read_csv(v2_csv).set_index("subject_id")
    common = sorted(set(v1.index) & set(v2.index))
    rows: list[dict[str, Any]] = []
    for region in BRATS_REGIONS:
        col = f"dice_{region}"
        a = v1.loc[common, col].to_numpy(dtype=float)
        b = v2.loc[common, col].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        a_m, b_m = a[mask], b[mask]
        mean_d, lo, hi = _bootstrap_mean_diff_ci(a_m, b_m)
        delta = b_m - a_m
        improved = int((delta > 0.01).sum())
        worsened = int((delta < -0.01).sum())
        unchanged = int(len(delta) - improved - worsened)
        rows.append(
            {
                "region": REGION_SHORT[region],
                "n": int(a_m.size),
                "mean_delta_dice": mean_d,
                "ci95_low": lo,
                "ci95_high": hi,
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    return out


@torch.no_grad()
def run_v2_volume_and_worst(
    checkpoint: Path,
    subjects_file: Path,
    brats_nifti_root: Path,
    *,
    volume_csv: Path,
    pipeline_dir: Path,
    worst_csv: Path,
    v2_lcc_csv: Path,
    device: str | None = None,
    n_worst: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single held-out pass: region volumes, pipeline WT agreement, worst WT CCs."""
    from monai.inferers import sliding_window_inference
    from segmentation.postprocess import apply_channel_postprocess
    from validation.ellipsoid import diameters_from_binary, ellipsoid_volume_cm3
    from validation.ellipsoid_batch import _plot_bland_altman

    subject_ids = read_id_list(subjects_file)
    records = [subject_to_datadict(brats_nifti_root / sid) for sid in subject_ids]
    device_t = torch.device(
        device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    model = load_model_from_checkpoint(checkpoint, device_t)
    model.float().eval()
    loader = DataLoader(
        Dataset(data=records, transform=get_val_transforms()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    spacing = (1.0, 1.0, 1.0)

    gt_vols = {r: [] for r in BRATS_REGIONS}
    pr_vols = {r: [] for r in BRATS_REGIONS}
    empty_pred = {r: 0 for r in BRATS_REGIONS}

    pipe_gt: list[float] = []
    pipe_pred: list[float] = []
    pipe_ell: list[float] = []

    # Prefetch worst subject ids for CC stats
    v2 = pd.read_csv(v2_lcc_csv)
    worst_ids = set(v2.nsmallest(n_worst, "dice_whole_tumor")["subject_id"].tolist())
    worst_rows: dict[str, dict[str, Any]] = {}

    for batch in loader:
        sid = batch["subject_id"][0]
        label = brats_label_to_regions(batch["label"]).squeeze(0).cpu().numpy()
        gt_wt_raw = batch["label"].squeeze(0).squeeze(0).detach().cpu().numpy() != 0
        inputs = batch["image"].to(device_t, dtype=torch.float32)
        logits = sliding_window_inference(
            inputs, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
        )
        pred_raw = (torch.sigmoid(logits) > 0.5).squeeze(0).detach().cpu().numpy()
        pred_lcc = apply_channel_postprocess(pred_raw.astype(np.float32), mode="lcc")

        for i, region in enumerate(BRATS_REGIONS):
            gt = label[i].astype(bool)
            pr = pred_lcc[i].astype(bool)
            gt_ml = voxel_volume_ml_from_binary(gt, spacing) if np.any(gt) else 0.0
            pr_ml = voxel_volume_ml_from_binary(pr, spacing) if np.any(pr) else 0.0
            if not np.any(pr):
                empty_pred[region] += 1
            if np.any(gt):
                gt_vols[region].append(gt_ml)
                pr_vols[region].append(pr_ml)

        # Pipeline WT (LCC pred voxel + radiologist ABC/2 vs GT)
        pr_wt = pred_lcc[2].astype(bool)
        if np.any(gt_wt_raw) and np.any(pr_wt):
            gt_ml = voxel_volume_ml_from_binary(gt_wt_raw, spacing)
            pr_ml = voxel_volume_ml_from_binary(pr_wt, spacing)
            diameters, _ = diameters_from_binary(pr_wt, spacing, method="radiologist")
            ell_ml = ellipsoid_volume_cm3(diameters, formula="abc_over_2")
            pipe_gt.append(gt_ml)
            pipe_pred.append(pr_ml)
            pipe_ell.append(ell_ml)

        if sid in worst_ids:
            gt_stats = fragmentation_stats(gt_wt_raw, spacing)
            raw_stats = fragmentation_stats(pred_raw[2].astype(bool), spacing)
            row = v2.loc[v2["subject_id"] == sid].iloc[0]
            worst_rows[sid] = {
                "subject_id": sid,
                "dice_whole_tumor": float(row["dice_whole_tumor"]),
                "hd95_whole_tumor": float(row["hd95_whole_tumor"]),
                "n_components_raw": int(raw_stats["n_components"]),
                "gt_n_components": int(gt_stats["n_components"]),
            }

    # Region volume agreement
    vol_rows: list[dict[str, Any]] = []
    for region in BRATS_REGIONS:
        gt = np.asarray(gt_vols[region], dtype=float)
        pr = np.asarray(pr_vols[region], dtype=float)
        if gt.size < 2:
            vol_rows.append(
                {
                    "region": REGION_SHORT[region],
                    "n": int(gt.size),
                    "n_pred_empty": empty_pred[region],
                    "mean_bias_ml": float("nan"),
                }
            )
            continue
        ba = bland_altman_stats(pr, gt)
        ratio = pr / np.maximum(gt, 1e-12)
        spearman_r, spearman_p = sp_stats.spearmanr(gt, pr)
        cis = _bootstrap_agreement_cis(gt, pr, n_resamples=1000, seed=42)
        vol_rows.append(
            {
                "region": REGION_SHORT[region],
                "n": int(gt.size),
                "n_pred_empty": empty_pred[region],
                "mean_bias_ml": float(np.mean(pr - gt)),
                "loa_lower_ml": ba.loa_lower,
                "loa_upper_ml": ba.loa_upper,
                "median_ratio": float(np.median(ratio)),
                "iqr_ratio_low": float(np.percentile(ratio, 25)),
                "iqr_ratio_high": float(np.percentile(ratio, 75)),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "icc_2_1": icc_2_1(gt, pr),
                **cis,
            }
        )
    vols = pd.DataFrame(vol_rows)
    volume_csv.parent.mkdir(parents=True, exist_ok=True)
    vols.to_csv(volume_csv, index=False)

    # Pipeline-level summary (same schema as lcc_followup)
    pipeline_dir = Path(pipeline_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pipeline_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(pipe_gt, dtype=float)
    pred = np.asarray(pipe_pred, dtype=float)
    ell = np.asarray(pipe_ell, dtype=float)
    pipe_rows = []
    for name, measured in (
        ("pred_voxel_vs_gt_voxel", pred),
        ("pred_ellipsoid_radiologist_abc_over_2_vs_gt_voxel", ell),
    ):
        ba = bland_altman_stats(measured, gt)
        ratio = measured / gt
        spearman_r, spearman_p = sp_stats.spearmanr(gt, measured)
        cis = _bootstrap_agreement_cis(gt, measured, n_resamples=1000, seed=42)
        pipe_rows.append(
            {
                "comparison": name,
                "n": int(gt.size),
                "mean_bias_ml": float(np.mean(measured - gt)),
                "loa_lower_ml": ba.loa_lower,
                "loa_upper_ml": ba.loa_upper,
                "median_ratio": float(np.median(ratio)),
                "iqr_ratio_low": float(np.percentile(ratio, 25)),
                "iqr_ratio_high": float(np.percentile(ratio, 75)),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "icc_2_1": icc_2_1(gt, measured),
                **cis,
            }
        )
        stem = "pipeline_voxel" if "pred_voxel" in name else "pipeline_ellipsoid"
        _plot_bland_altman(
            gt,
            measured,
            title=f"Pipeline LCC | {name} (mL)",
            out_path=fig_dir / f"{stem}_ml.png",
            percent=False,
        )
        _plot_bland_altman(
            gt,
            measured,
            title=f"Pipeline LCC | {name} (log-ratio %)",
            out_path=fig_dir / f"{stem}_pct.png",
            percent=True,
        )
    pipeline = pd.DataFrame(pipe_rows)
    pipeline.to_csv(pipeline_dir / "pipeline_level.csv", index=False)

    # Worst WT ordered by Dice ascending
    worst_list = [
        worst_rows[sid]
        for sid in v2.nsmallest(n_worst, "dice_whole_tumor")["subject_id"]
        if sid in worst_rows
    ]
    worst = pd.DataFrame(worst_list)
    worst.to_csv(worst_csv, index=False)
    return vols, pipeline, worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only rebuild table/paired from existing metrics CSVs",
    )
    args = parser.parse_args(argv)
    app = load_config()
    v1 = PROJECT_ROOT / "checkpoints/brats_eval_heldout/lcc/metrics.csv"
    v2 = PROJECT_ROOT / "checkpoints/brats_eval_heldout_v2/lcc/metrics.csv"
    ckpt = PROJECT_ROOT / "checkpoints/brats_v2_loss_fix/best_model.pt"
    subjects = PROJECT_ROOT / "splits/heldout.txt"
    brats = app.paths.brats_nifti

    table = build_v1_v2_table(
        v1, v2, PROJECT_ROOT / "results/segmentation_table_v2.csv"
    )
    print("\n=== v1-lcc vs v2-lcc ===")
    print(table.to_string(index=False))
    paired = paired_v1_v2(
        v1, v2, PROJECT_ROOT / "results/v1_vs_v2_paired_dice.csv"
    )
    print("\n=== Paired ΔDice (v2−v1) ===")
    print(paired.to_string(index=False))

    if not args.skip_inference:
        vols, pipeline, worst = run_v2_volume_and_worst(
            ckpt,
            subjects,
            brats,
            volume_csv=PROJECT_ROOT / "results/v2_region_volumes.csv",
            pipeline_dir=PROJECT_ROOT / "results/ellipsoid/brats_v2_loss_fix",
            worst_csv=PROJECT_ROOT / "results/v2_worst_wt_cases.csv",
            v2_lcc_csv=v2,
            device=args.device,
        )
        print("\n=== v2 pred vs GT voxel volumes (LCC) ===")
        print(vols.to_string(index=False))
        print("\n=== Pipeline-level (v2 LCC) ===")
        print(pipeline.to_string(index=False))
        print("\n=== 5 worst WT Dice (v2-lcc) ===")
        print(worst.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
