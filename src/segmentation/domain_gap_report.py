"""Compare BraTS-pretrained vs fine-tuned models on held-out clinical test cases."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

from config import load_config
from segmentation.dataset import get_val_transforms
from segmentation.evaluate import _metric_value, load_model_from_checkpoint
from segmentation.finetune import (
    FinetuneConfig,
    build_corrected_file_list,
    corrected_case_to_datadict,
    split_finetune_sets,
)
from segmentation.model import BRATS_REGIONS, brats_label_to_regions
from segmentation.sanity_check import dice_numpy
from segmentation.train import post_transforms

logger = logging.getLogger(__name__)

MODEL_BRATS = "brats_pretrained"
MODEL_FINETUNED = "clinical_finetuned"


@dataclass
class DomainGapConfig:
    """Settings for clinical domain-gap evaluation."""

    roi_size: tuple[int, int, int] = (96, 96, 96)
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    amp: bool = True
    n_test_cases: int = 4
    min_test_cases: int = 3
    max_test_cases: int = 5
    val_fraction: float = 0.15
    seed: int = 42


def mask_volume_ml(mask: np.ndarray, voxel_spacing_mm: tuple[float, float, float]) -> float:
    """Volume in milliliters from a binary mask and voxel spacing (mm)."""
    voxel_ml = float(np.prod(voxel_spacing_mm) / 1000.0)
    return float(np.count_nonzero(mask.astype(bool)) * voxel_ml)


def volume_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
) -> dict[str, float]:
    """Absolute and relative volume error for one binary region."""
    pred_vol = mask_volume_ml(pred, voxel_spacing_mm)
    gt_vol = mask_volume_ml(gt, voxel_spacing_mm)
    abs_err = abs(pred_vol - gt_vol)
    rel_err_pct = 100.0 * abs_err / gt_vol if gt_vol > 0.0 else float("nan")
    return {
        "pred_vol_ml": pred_vol,
        "gt_vol_ml": gt_vol,
        "vol_abs_error_ml": abs_err,
        "vol_rel_error_pct": rel_err_pct,
    }


def load_finetune_test_records(
    corrected_root: str | Path,
    *,
    split_json: str | Path | None = None,
    config: DomainGapConfig | None = None,
) -> list[dict[str, Any]]:
    """Resolve held-out test cases from ``finetune_split.json`` or the same split logic."""
    config = config or DomainGapConfig()
    records = build_corrected_file_list(corrected_root)
    record_by_id = {r["subject_id"]: r for r in records}

    if split_json is not None:
        split_path = Path(split_json)
        if not split_path.is_file():
            raise FileNotFoundError(f"Fine-tune split manifest not found: {split_path}")
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = manifest.get("test", [])
        if not test_ids:
            raise ValueError(f"No test cases listed in {split_path}")
        missing = [sid for sid in test_ids if sid not in record_by_id]
        if missing:
            raise FileNotFoundError(
                f"Test subjects in split not found under corrected root: {missing}"
            )
        return [record_by_id[sid] for sid in test_ids]

    _, _, test_files = split_finetune_sets(
        records,
        n_test_cases=config.n_test_cases,
        min_test_cases=config.min_test_cases,
        max_test_cases=config.max_test_cases,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    return test_files


def _tensor_region_to_numpy(regions: torch.Tensor) -> np.ndarray:
    """Convert ``(C, *spatial)`` region tensor to numpy."""
    arr = regions.detach().cpu().numpy() if isinstance(regions, torch.Tensor) else np.asarray(regions)
    if arr.ndim < 4:
        raise ValueError(f"Expected region tensor (C, D, H, W), got shape {arr.shape}")
    return arr


@torch.no_grad()
def evaluate_clinical_case(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    roi_size: tuple[int, int, int],
    use_amp: bool,
    post_pred,
    post_label,
    voxel_spacing_mm: tuple[float, float, float],
) -> dict[str, float]:
    """Run sliding-window inference and return Dice + volume metrics per region."""
    inputs = batch["image"].to(device)
    labels = brats_label_to_regions(batch["label"].to(device))

    with torch.autocast(device_type=device.type, enabled=use_amp):
        outputs = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
        )

    pred_regions = _tensor_region_to_numpy(post_pred(decollate_batch(outputs)[0]))
    gt_regions = _tensor_region_to_numpy(post_label(decollate_batch(labels)[0]))

    dice_metric = DiceMetric(include_background=False, reduction="none")
    pred_t = torch.as_tensor(pred_regions).unsqueeze(0)
    label_t = torch.as_tensor(gt_regions).unsqueeze(0)
    dice_metric(y_pred=[pred_t], y=[label_t])
    dice_scores = dice_metric.aggregate()

    metrics: dict[str, float] = {}
    for idx, region in enumerate(BRATS_REGIONS):
        pred_mask = pred_regions[idx].astype(bool)
        gt_mask = gt_regions[idx].astype(bool)
        metrics[f"dice_{region}"] = _metric_value(dice_scores, idx)
        metrics[f"dice_numpy_{region}"] = dice_numpy(pred_mask, gt_mask)
        vol = volume_errors(pred_mask, gt_mask, voxel_spacing_mm)
        metrics[f"pred_vol_ml_{region}"] = vol["pred_vol_ml"]
        metrics[f"gt_vol_ml_{region}"] = vol["gt_vol_ml"]
        metrics[f"vol_abs_error_ml_{region}"] = vol["vol_abs_error_ml"]
        metrics[f"vol_rel_error_pct_{region}"] = vol["vol_rel_error_pct"]

    dice_vals = [
        metrics[f"dice_{r}"]
        for r in BRATS_REGIONS
        if metrics[f"dice_{r}"] == metrics[f"dice_{r}"]
    ]
    vol_rel_vals = [
        metrics[f"vol_rel_error_pct_{r}"]
        for r in BRATS_REGIONS
        if metrics[f"vol_rel_error_pct_{r}"] == metrics[f"vol_rel_error_pct_{r}"]
    ]
    metrics["dice_mean"] = sum(dice_vals) / len(dice_vals) if dice_vals else float("nan")
    metrics["vol_rel_error_pct_mean"] = (
        sum(vol_rel_vals) / len(vol_rel_vals) if vol_rel_vals else float("nan")
    )
    return metrics


def _run_model_on_test_set(
    checkpoint_path: str | Path,
    test_files: Sequence[dict[str, Any]],
    *,
    model_tag: str,
    config: DomainGapConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate one checkpoint on all test cases; prefix columns with ``model_tag``."""
    use_amp = config.amp and device.type == "cuda"
    model = load_model_from_checkpoint(checkpoint_path, device)
    post_pred, post_label = post_transforms()

    test_ds = Dataset(
        data=list(test_files),
        transform=get_val_transforms(pixdim=config.pixdim),
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    rows: list[dict[str, Any]] = []
    for batch in test_loader:
        subject_id = batch.get("subject_id", ["unknown"])[0]
        logger.info("Evaluating %s (%s)", subject_id, model_tag)
        case_metrics = evaluate_clinical_case(
            model,
            batch,
            device,
            config.roi_size,
            use_amp,
            post_pred,
            post_label,
            config.pixdim,
        )
        row: dict[str, Any] = {"subject_id": subject_id, "model": model_tag}
        row.update(case_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    rename = {
        col: f"{model_tag}_{col}"
        for col in df.columns
        if col not in ("subject_id", "model")
    }
    return df.rename(columns=rename)


def build_comparison_table(
    brats_df: pd.DataFrame,
    finetuned_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-model metrics and add before/after deltas (finetuned − BraTS)."""
    brats_cols = [c for c in brats_df.columns if c.startswith(f"{MODEL_BRATS}_")]
    finetuned_cols = [c for c in finetuned_df.columns if c.startswith(f"{MODEL_FINETUNED}_")]

    merged = brats_df[["subject_id", *brats_cols]].merge(
        finetuned_df[["subject_id", *finetuned_cols]],
        on="subject_id",
        how="inner",
    )

    metric_suffixes = [
        c.removeprefix(f"{MODEL_BRATS}_")
        for c in brats_cols
    ]
    for suffix in metric_suffixes:
        brats_key = f"{MODEL_BRATS}_{suffix}"
        finetuned_key = f"{MODEL_FINETUNED}_{suffix}"
        if brats_key in merged.columns and finetuned_key in merged.columns:
            merged[f"delta_{suffix}"] = merged[finetuned_key] - merged[brats_key]

    return merged


def summarize_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level means for key Dice and volume metrics."""
    summary_rows: list[dict[str, Any]] = []
    for model_tag in (MODEL_BRATS, MODEL_FINETUNED):
        row: dict[str, Any] = {"model": model_tag, "n_cases": len(comparison_df)}
        for region in BRATS_REGIONS:
            dice_col = f"{model_tag}_dice_{region}"
            vol_col = f"{model_tag}_vol_rel_error_pct_{region}"
            if dice_col in comparison_df.columns:
                row[f"mean_dice_{region}"] = comparison_df[dice_col].mean(skipna=True)
            if vol_col in comparison_df.columns:
                row[f"mean_vol_rel_error_pct_{region}"] = comparison_df[vol_col].mean(skipna=True)
        dice_mean_col = f"{model_tag}_dice_mean"
        vol_mean_col = f"{model_tag}_vol_rel_error_pct_mean"
        if dice_mean_col in comparison_df.columns:
            row["mean_dice_overall"] = comparison_df[dice_mean_col].mean(skipna=True)
        if vol_mean_col in comparison_df.columns:
            row["mean_vol_rel_error_pct_overall"] = comparison_df[vol_mean_col].mean(skipna=True)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    if len(summary) == 2:
        brats_row = summary.loc[summary["model"] == MODEL_BRATS].iloc[0]
        ft_row = summary.loc[summary["model"] == MODEL_FINETUNED].iloc[0]
        delta: dict[str, Any] = {"model": "improvement (finetuned − brats)"}
        for col in summary.columns:
            if col in ("model", "n_cases"):
                continue
            b_val = brats_row[col]
            f_val = ft_row[col]
            if pd.isna(b_val) or pd.isna(f_val):
                delta[col] = float("nan")
            elif "vol_rel_error" in col:
                delta[col] = b_val - f_val  # lower volume error is better
            else:
                delta[col] = f_val - b_val  # higher Dice is better
        summary = pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)

    return summary


def format_comparison_table(comparison_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    """Human-readable table for reports (per-case + summary)."""
    display_cols = ["subject_id"]
    for region in BRATS_REGIONS:
        display_cols.extend(
            [
                f"{MODEL_BRATS}_dice_{region}",
                f"{MODEL_FINETUNED}_dice_{region}",
                f"delta_dice_{region}",
                f"{MODEL_BRATS}_vol_rel_error_pct_{region}",
                f"{MODEL_FINETUNED}_vol_rel_error_pct_{region}",
                f"delta_vol_rel_error_pct_{region}",
            ]
        )
    display_cols.extend(
        [
            f"{MODEL_BRATS}_dice_mean",
            f"{MODEL_FINETUNED}_dice_mean",
            "delta_dice_mean",
        ]
    )
    present = [c for c in display_cols if c in comparison_df.columns]
    case_table = comparison_df[present].to_string(index=False, float_format=lambda x: f"{x:.4f}")

    summary_table = summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}")
    return (
        "Domain adaptation — held-out clinical test cases\n"
        "================================================\n\n"
        "Per-case metrics (Dice ↑ better; volume rel. error ↓ better)\n"
        f"{case_table}\n\n"
        "Summary\n"
        f"{summary_table}\n"
    )


def plot_domain_gap_bars(
    summary_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Domain adaptation: BraTS-pretrained vs fine-tuned (clinical test holdout)",
) -> Path:
    """Grouped bar chart of mean Dice and mean volume error before/after fine-tuning."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_rows = summary_df[summary_df["model"].isin((MODEL_BRATS, MODEL_FINETUNED))]
    if model_rows.empty:
        raise ValueError("Summary must contain rows for both BraTS and fine-tuned models")

    dice_metrics = [f"mean_dice_{r}" for r in BRATS_REGIONS] + ["mean_dice_overall"]
    vol_metrics = [f"mean_vol_rel_error_pct_{r}" for r in BRATS_REGIONS] + [
        "mean_vol_rel_error_pct_overall"
    ]
    dice_labels = [r.replace("_", " ").title() for r in BRATS_REGIONS] + ["Mean Dice"]
    vol_labels = [f"{r} vol % err" for r in BRATS_REGIONS] + ["Mean vol % err"]

    brats = model_rows.loc[model_rows["model"] == MODEL_BRATS].iloc[0]
    finetuned = model_rows.loc[model_rows["model"] == MODEL_FINETUNED].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(dice_metrics))
    width = 0.35

    for ax, metrics, labels, ylabel, subtitle in (
        (
            axes[0],
            dice_metrics,
            dice_labels,
            "Dice coefficient",
            "Segmentation overlap (higher is better)",
        ),
        (
            axes[1],
            vol_metrics,
            vol_labels,
            "Relative volume error (%)",
            "Whole-tumor volume error (lower is better)",
        ),
    ):
        brats_vals = [float(brats.get(m, float("nan"))) for m in metrics]
        ft_vals = [float(finetuned.get(m, float("nan"))) for m in metrics]
        ax.bar(x - width / 2, brats_vals, width, label="BraTS-pretrained", color="#5B7C99")
        ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned", color="#2E8B57")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.set_ylim(0, 1.05 if "Dice" in ylabel else None)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart -> %s", output_path)
    return output_path


def run_domain_gap_report(
    brats_checkpoint: str | Path,
    finetuned_checkpoint: str | Path,
    corrected_root: str | Path,
    output_dir: str | Path,
    *,
    split_json: str | Path | None = None,
    config: DomainGapConfig | None = None,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Run both checkpoints on the fine-tune test holdout and write table + chart.

    Returns
    -------
    comparison_df, summary_df, comparison_csv_path, bar_chart_path
    """
    config = config or DomainGapConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_files = load_finetune_test_records(
        corrected_root,
        split_json=split_json,
        config=config,
    )
    logger.info("Domain-gap evaluation on %d held-out test cases", len(test_files))

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    brats_df = _run_model_on_test_set(
        brats_checkpoint,
        test_files,
        model_tag=MODEL_BRATS,
        config=config,
        device=device_t,
    )
    finetuned_df = _run_model_on_test_set(
        finetuned_checkpoint,
        test_files,
        model_tag=MODEL_FINETUNED,
        config=config,
        device=device_t,
    )

    comparison_df = build_comparison_table(brats_df, finetuned_df)
    summary_df = summarize_comparison(comparison_df)

    comparison_csv = output_dir / "domain_gap_comparison.csv"
    summary_csv = output_dir / "domain_gap_summary.csv"
    table_txt = output_dir / "domain_gap_table.txt"
    bar_png = output_dir / "domain_gap_bars.png"

    comparison_df.to_csv(comparison_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    table_txt.write_text(
        format_comparison_table(comparison_df, summary_df) + "\n",
        encoding="utf-8",
    )
    plot_domain_gap_bars(summary_df, bar_png)

    logger.info("Saved comparison CSV -> %s", comparison_csv)
    logger.info("Saved summary CSV -> %s", summary_csv)
    logger.info("Saved formatted table -> %s", table_txt)
    print(format_comparison_table(comparison_df, summary_df))

    return comparison_df, summary_df, comparison_csv, bar_png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare BraTS vs fine-tuned checkpoints on clinical test holdout.",
    )
    parser.add_argument(
        "--brats-checkpoint",
        type=Path,
        default=None,
        help="BraTS-only pretrained checkpoint (default: checkpoints/brats_pretrain/best_model.pt)",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint (default: checkpoints/clinical_finetune/best_model.pt)",
    )
    parser.add_argument(
        "--corrected-root",
        type=Path,
        default=None,
        help="Manually corrected patient studies root",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="finetune_split.json from fine-tuning (default: alongside finetuned checkpoint)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Project config.yaml")
    parser.add_argument("--test-cases", type=int, default=4, help="Test holdout if no split JSON")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)

    brats_ckpt = args.brats_checkpoint or (
        app_cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt"
    )
    finetuned_ckpt = args.finetuned_checkpoint or (
        app_cfg.paths.checkpoints / "clinical_finetune" / "best_model.pt"
    )
    corrected_root = args.corrected_root or (
        app_cfg.paths.processed / "real_patients" / "corrected"
    )
    output_dir = args.output_dir or (app_cfg.paths.checkpoints / "domain_gap_report")
    split_json = args.split_json
    if split_json is None and finetuned_ckpt.parent.is_dir():
        candidate = finetuned_ckpt.parent / "finetune_split.json"
        if candidate.is_file():
            split_json = candidate

    config = DomainGapConfig(
        amp=not args.no_amp,
        n_test_cases=max(3, min(5, args.test_cases)),
        seed=FinetuneConfig().seed,
    )

    run_domain_gap_report(
        brats_ckpt,
        finetuned_ckpt,
        corrected_root,
        output_dir,
        split_json=split_json,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
