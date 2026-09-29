"""Evaluate a trained BraTS segmentation checkpoint on the validation set."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from config import load_config
from device import get_device
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.model import BRATS_REGIONS, build_brats_model
from segmentation.model import brats_label_to_regions
from segmentation.postprocess import (
    DEFAULT_LCC_MIN_FRAC,
    DEFAULT_LCC_MIN_ML,
    POSTPROCESS_MODES,
    PostprocessMode,
    apply_channel_postprocess,
)
from segmentation.splits import read_id_list
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
    float32: bool = False
    postprocess: PostprocessMode = "raw"
    lcc_min_frac: float = DEFAULT_LCC_MIN_FRAC
    lcc_min_ml: float = DEFAULT_LCC_MIN_ML


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained model from ``best_model.pt`` / ``last_model.pt``."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    *,
    postprocess: PostprocessMode = "raw",
    lcc_min_frac: float = DEFAULT_LCC_MIN_FRAC,
    lcc_min_ml: float = DEFAULT_LCC_MIN_ML,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """Run inference on one case and return per-region Dice / HD95."""
    inputs = batch["image"].to(device=device, dtype=torch.float32)
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
    # MONAI HD95 uses float64 distance transforms — not supported on MPS.
    pred = pred.detach().float().cpu()
    label = label.detach().float().cpu()

    if postprocess != "raw":
        cleaned = apply_channel_postprocess(
            pred.numpy(),
            mode=postprocess,
            min_frac_of_largest=lcc_min_frac,
            min_volume_ml=lcc_min_ml,
            spacing_mm=pixdim,
        )
        pred = torch.from_numpy(cleaned.astype(np.float32))

    dice_metric = DiceMetric(include_background=True, reduction="none")
    hd_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction="none",
    )
    dice_metric(y_pred=[pred], y=[label])
    hd_metric(y_pred=[pred], y=[label])

    dice_scores = dice_metric.aggregate()
    hd_scores = hd_metric.aggregate()

    metrics: dict[str, float] = {}
    for idx, region in enumerate(BRATS_REGIONS):
        # Exclude empty-GT regions from Dice/HD95 (undefined when no foreground).
        if float(label[idx].sum().item()) <= 0.0:
            metrics[f"dice_{region}"] = float("nan")
            metrics[f"hd95_{region}"] = float("nan")
            continue
        metrics[f"dice_{region}"] = _metric_value(dice_scores, idx)
        metrics[f"hd95_{region}"] = _metric_value(hd_scores, idx)

    dice_vals = [metrics[f"dice_{r}"] for r in BRATS_REGIONS if metrics[f"dice_{r}"] == metrics[f"dice_{r}"]]
    hd_vals = [metrics[f"hd95_{r}"] for r in BRATS_REGIONS if metrics[f"hd95_{r}"] == metrics[f"hd95_{r}"]]
    metrics["dice_mean"] = sum(dice_vals) / len(dice_vals) if dice_vals else float("nan")
    metrics["hd95_mean"] = sum(hd_vals) / len(hd_vals) if hd_vals else float("nan")
    return metrics


def _records_from_subject_ids(
    brats_nifti_root: str | Path,
    subject_ids: Sequence[str],
) -> list[dict[str, Any]]:
    root = Path(brats_nifti_root)
    records: list[dict[str, Any]] = []
    for sid in subject_ids:
        subject_dir = root / sid
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        records.append(subject_to_datadict(subject_dir))
    return records


def evaluate_subject_list(
    checkpoint_path: str | Path,
    subject_ids: Sequence[str],
    output_csv: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    config: EvalConfig | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Inference-only evaluation on an explicit subject list (float32 by default)."""
    config = config or EvalConfig(float32=True, amp=False)
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device_t = get_device(device)
    use_amp = (not config.float32) and config.amp and device_t.type == "cuda"
    model = load_model_from_checkpoint(checkpoint_path, device_t)
    model.float()
    post_pred, post_label = post_transforms()

    records = _records_from_subject_ids(brats_nifti_root, subject_ids)
    loader = DataLoader(
        Dataset(data=records, transform=get_val_transforms(pixdim=config.pixdim)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rows: list[dict[str, Any]] = []
    for batch in loader:
        subject_id = batch.get("subject_id", ["unknown"])[0]
        logger.info("Evaluating %s on %s", subject_id, device_t)
        case_metrics = _evaluate_case(
            model,
            batch,
            device_t,
            config.roi_size,
            use_amp,
            post_pred,
            post_label,
            postprocess=config.postprocess,
            lcc_min_frac=config.lcc_min_frac,
            lcc_min_ml=config.lcc_min_ml,
            pixdim=config.pixdim,
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
    logger.info(
        "Saved held-out metrics -> %s (%d cases, postprocess=%s)",
        output_csv,
        len(df),
        config.postprocess,
    )
    for region in BRATS_REGIONS:
        n_region = int(df[f"dice_{region}"].notna().sum())
        mean_dice = float(df[f"dice_{region}"].mean(skipna=True))
        mean_hd = float(df[f"hd95_{region}"].mean(skipna=True))
        logger.info(
            "%s (n=%d) | Dice=%.4f | HD95=%.2f mm",
            region,
            n_region,
            mean_dice,
            mean_hd,
        )
    return df


def evaluate_validation_set(
    checkpoint_path: str | Path,
    output_csv: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    config: EvalConfig | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Load a checkpoint, infer on the BraTS validation split, and save metrics."""
    config = config or EvalConfig()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device_t = get_device(device)
    use_amp = (not config.float32) and config.amp and device_t.type == "cuda"
    model = load_model_from_checkpoint(checkpoint_path, device_t)
    if config.float32:
        model.float()
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
            postprocess=config.postprocess,
            lcc_min_frac=config.lcc_min_frac,
            lcc_min_ml=config.lcc_min_ml,
            pixdim=config.pixdim,
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
        n_region = int(df[f"dice_{region}"].notna().sum())
        logger.info(
            "%s (n=%d) | Dice=%.4f | HD95=%.2f mm",
            region,
            n_region,
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
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=None,
        help="Text file of subject IDs (one per line) for held-out evaluation",
    )
    parser.add_argument("--float32", action="store_true", help="Force float32, disable AMP")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--postprocess",
        choices=POSTPROCESS_MODES,
        default="raw",
        help="raw | lcc | lcc_min (largest + CCs >= max(5%% of LCC, 1 mL), then fill holes)",
    )
    parser.add_argument(
        "--lcc-min-frac",
        type=float,
        default=DEFAULT_LCC_MIN_FRAC,
        help="For lcc_min: keep secondary CCs ≥ this fraction of the largest",
    )
    parser.add_argument(
        "--lcc-min-ml",
        type=float,
        default=DEFAULT_LCC_MIN_ML,
        help="For lcc_min: absolute minimum secondary CC volume (mL)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    if args.output_csv is not None:
        output_csv = args.output_csv
    elif args.subjects_file is not None:
        output_csv = (
            app_cfg.paths.checkpoints
            / "brats_eval_heldout"
            / args.postprocess
            / "metrics.csv"
        )
    else:
        output_csv = (
            app_cfg.paths.checkpoints
            / "brats_eval"
            / args.postprocess
            / "metrics.csv"
        )

    eval_cfg = EvalConfig(
        max_cases=args.max_cases,
        amp=not args.no_amp and not args.float32,
        float32=args.float32 or bool(args.subjects_file),
        postprocess=args.postprocess,
        lcc_min_frac=args.lcc_min_frac,
        lcc_min_ml=args.lcc_min_ml,
    )

    if args.subjects_file is not None:
        ids = read_id_list(args.subjects_file)
        evaluate_subject_list(
            args.checkpoint,
            ids,
            output_csv,
            brats_nifti_root=app_cfg.paths.brats_nifti,
            config=eval_cfg,
            device=args.device,
        )
    else:
        evaluate_validation_set(
            args.checkpoint,
            output_csv,
            brats_nifti_root=app_cfg.paths.brats_nifti,
            config=eval_cfg,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
