"""Evaluate a trained BraTS segmentation checkpoint on the validation set."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from config import load_config
from segmentation.dataset import get_val_transforms
from segmentation.model import BRATS_REGIONS, build_brats_model
from segmentation.model import brats_label_to_regions
from segmentation.train import split_brats_cases, post_transforms

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation settings aligned with training defaults."""

    val_fraction: float = 0.2
    seed: int = 42
    roi_size: tuple[int, int, int] = (96, 96, 96)
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    amp: bool = True
    max_cases: int | None = None


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained model from ``best_model.pt`` / ``last_model.pt``."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "segresnet") if isinstance(ckpt, dict) else "segresnet"
    model = build_brats_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _metric_value(tensor: torch.Tensor, index: int) -> float:
    """Extract one scalar metric, mapping NaN to ``float('nan')``."""
    flat = tensor.detach().flatten()
    if flat.numel() <= index:
        return float("nan")
    value = float(flat[index].item())
    if value != value:  # NaN
        return float("nan")
    return value


@torch.no_grad()
def _evaluate_case(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    roi_size: tuple[int, int, int],
    use_amp: bool,
    post_pred,
    post_label,
) -> dict[str, float]:
    """Run inference on one case and return per-region Dice / HD95."""
    inputs = batch["image"].to(device)
    labels = brats_label_to_regions(batch["label"].to(device))

    with torch.autocast(device_type=device.type, enabled=use_amp):
        outputs = sliding_window_inference(
            inputs,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
        )

    pred = post_pred(decollate_batch(outputs)[0])
    label = post_label(decollate_batch(labels)[0])

    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="none",
    )
    dice_metric(y_pred=[pred], y=[label])
    hd_metric(y_pred=[pred], y=[label])

    dice_scores = dice_metric.aggregate()
    hd_scores = hd_metric.aggregate()

    metrics: dict[str, float] = {}
    for idx, region in enumerate(BRATS_REGIONS):
        metrics[f"dice_{region}"] = _metric_value(dice_scores, idx)
        metrics[f"hd95_{region}"] = _metric_value(hd_scores, idx)

    dice_vals = [metrics[f"dice_{r}"] for r in BRATS_REGIONS if metrics[f"dice_{r}"] == metrics[f"dice_{r}"]]
    hd_vals = [metrics[f"hd95_{r}"] for r in BRATS_REGIONS if metrics[f"hd95_{r}"] == metrics[f"hd95_{r}"]]
    metrics["dice_mean"] = sum(dice_vals) / len(dice_vals) if dice_vals else float("nan")
    metrics["hd95_mean"] = sum(hd_vals) / len(hd_vals) if hd_vals else float("nan")
    return metrics


def evaluate_validation_set(
    checkpoint_path: str | Path,
    output_csv: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    config: EvalConfig | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Load a checkpoint, infer on the BraTS validation split, and save metrics.

    Reports per-class Dice and 95th-percentile Hausdorff distance (HD95) for
    enhancing tumor, tumor core, and whole tumor. Writes one CSV row per case
    plus logs dataset-level means.
    """
    config = config or EvalConfig()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = config.amp and device_t.type == "cuda"
    model = load_model_from_checkpoint(checkpoint_path, device_t)
    post_pred, post_label = post_transforms()

    _, val_files = split_brats_cases(
        brats_nifti_root,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    if config.max_cases is not None:
        val_files = val_files[: config.max_cases]

    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms(pixdim=config.pixdim),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    rows: list[dict[str, Any]] = []
    for batch in val_loader:
        subject_id = batch.get("subject_id", ["unknown"])[0]
        logger.info("Evaluating %s", subject_id)
        case_metrics = _evaluate_case(
            model,
            batch,
            device_t,
            config.roi_size,
            use_amp,
            post_pred,
            post_label,
        )
        rows.append({"subject_id": subject_id, **case_metrics})

    df = pd.DataFrame(rows)
    column_order = (
        ["subject_id"]
        + [f"dice_{r}" for r in BRATS_REGIONS]
        + [f"hd95_{r}" for r in BRATS_REGIONS]
        + ["dice_mean", "hd95_mean"]
    )
    df = df.reindex(columns=column_order)
    df.to_csv(output_csv, index=False)

    summary = {
        f"mean_dice_{r}": df[f"dice_{r}"].mean(skipna=True) for r in BRATS_REGIONS
    }
    summary.update({f"mean_hd95_{r}": df[f"hd95_{r}"].mean(skipna=True) for r in BRATS_REGIONS})
    summary["mean_dice_overall"] = df["dice_mean"].mean(skipna=True)
    summary["mean_hd95_overall"] = df["hd95_mean"].mean(skipna=True)

    logger.info("Saved per-case metrics -> %s", output_csv)
    for region in BRATS_REGIONS:
        logger.info(
            "%s | Dice=%.4f | HD95=%.2f mm",
            region,
            summary[f"mean_dice_{region}"],
            summary[f"mean_hd95_{region}"],
        )
    logger.info(
        "Overall mean Dice=%.4f | mean HD95=%.2f mm",
        summary["mean_dice_overall"],
        summary["mean_hd95_overall"],
    )
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BraTS segmentation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Project config.yaml")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases (debug)")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    output_csv = args.output_csv or (app_cfg.paths.checkpoints / "brats_eval_metrics.csv")
    evaluate_validation_set(
        args.checkpoint,
        output_csv,
        brats_nifti_root=app_cfg.paths.brats_nifti,
        config=EvalConfig(max_cases=args.max_cases, amp=not args.no_amp),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
