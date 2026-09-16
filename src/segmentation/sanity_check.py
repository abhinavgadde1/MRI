"""Visual and numeric sanity checks for BraTS segmentation predictions."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from skimage.measure import find_contours

from config import load_config
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS, brats_label_to_regions
from segmentation.train import post_transforms

logger = logging.getLogger(__name__)

RegionName = Literal["enhancing_tumor", "tumor_core", "whole_tumor"]
DICE_TOLERANCE = 1e-4


@dataclass
class CaseVolumes:
    """Prediction / ground-truth arrays for one subject in validation space."""

    subject_id: str
    region: RegionName
    background: np.ndarray  # (H, W, D) FLAIR
    pred_mask: np.ndarray  # (H, W, D) bool
    gt_mask: np.ndarray  # (H, W, D) bool
    affine: np.ndarray  # 4x4 shared voxel-to-world mapping


def dice_numpy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Manual Dice: ``2 * |pred ∩ gt| / (|pred| + |gt|)``."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = float(np.logical_and(pred_b, gt_b).sum())
    denom = float(pred_b.sum() + gt_b.sum())
    if denom == 0.0:
        return 1.0
    return 2.0 * intersection / denom


def dice_monai(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice via MONAI ``DiceMetric`` for one binary region."""
    metric = DiceMetric(include_background=False, reduction="mean")
    pred_t = torch.as_tensor(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.as_tensor(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    metric(y_pred=[pred_t], y=[gt_t])
    return float(metric.aggregate().item())


def assert_matching_geometry(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_affine: np.ndarray,
    gt_affine: np.ndarray,
    *,
    context: str = "",
) -> None:
    """Raise a clear error when shape or affine differ before metric comparison."""
    prefix = f"{context}: " if context else ""
    if pred.shape != gt.shape:
        raise ValueError(
            f"{prefix}prediction shape {pred.shape} != ground-truth shape {gt.shape}. "
            "This usually indicates an orientation or resampling mismatch."
        )
    if pred_affine.shape != (4, 4) or gt_affine.shape != (4, 4):
        raise ValueError(f"{prefix}affine must be 4x4; got {pred_affine.shape} and {gt_affine.shape}")
    if not np.allclose(pred_affine, gt_affine, atol=1e-4, rtol=1e-4):
        max_diff = float(np.max(np.abs(pred_affine - gt_affine)))
        raise ValueError(
            f"{prefix}prediction affine differs from ground-truth affine (max |Δ|={max_diff:.6f}). "
            "Volumes are not in the same world coordinate frame."
        )


def assert_dice_agreement(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tolerance: float = DICE_TOLERANCE,
) -> tuple[float, float]:
    """Compare manual and MONAI Dice; print both and assert on mismatch."""
    if not pred.any() and not gt.any():
        return 1.0, 1.0
    manual = dice_numpy(pred, gt)
    monai = dice_monai(pred, gt)
    if abs(manual - monai) > tolerance:
        print(f"Manual Dice: {manual:.6f}")
        print(f"MONAI Dice:  {monai:.6f}")
        print(f"Tolerance:   {tolerance}")
        raise AssertionError(
            f"Dice mismatch: manual={manual:.6f}, monai={monai:.6f}, "
            f"delta={abs(manual - monai):.6f}"
        )
    return manual, monai


def _tensor_to_numpy(obj: Any) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def _extract_affine(label_obj: Any, subject_id: str) -> np.ndarray:
    if hasattr(label_obj, "affine"):
        return np.asarray(label_obj.affine)
    meta = getattr(label_obj, "meta", None)
    if isinstance(meta, dict):
        for key in ("affine", "original_affine"):
            if key in meta:
                return np.asarray(meta[key])
    raise ValueError(
        f"Could not read affine for {subject_id}. "
        "Ensure validation transforms preserve MetaTensor metadata."
    )


def _region_index(region: RegionName) -> int:
    try:
        return BRATS_REGIONS.index(region)
    except ValueError as exc:
        raise ValueError(f"Unknown region {region!r}; choose from {BRATS_REGIONS}") from exc


@torch.no_grad()
def load_case_volumes(
    subject_id: str,
    checkpoint_path: str | Path,
    *,
    brats_nifti_root: str | Path,
    region: RegionName = "whole_tumor",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    device: str | None = None,
) -> CaseVolumes:
    """Run inference for one subject and extract one tumor sub-region."""
    brats_nifti_root = Path(brats_nifti_root)
    subject_dir = brats_nifti_root / subject_id
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_checkpoint(checkpoint_path, device_t)
    post_pred, post_label = post_transforms()

    record = subject_to_datadict(subject_dir)
    data = get_val_transforms()(record)

    image = data["image"].unsqueeze(0).to(device_t)
    label = data["label"].unsqueeze(0).to(device_t)
    gt_affine = _extract_affine(data["label"], subject_id)

    with torch.autocast(device_type=device_t.type, enabled=device_t.type == "cuda"):
        outputs = sliding_window_inference(
            image,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
        )

    pred_regions = post_pred(decollate_batch(outputs)[0])
    gt_regions = post_label(decollate_batch(brats_label_to_regions(label))[0])

    image_np = _tensor_to_numpy(data["image"])
    if image_np.ndim == 4:
        background = image_np[0]  # FLAIR
    else:
        raise ValueError(f"Expected 4-channel image, got shape {image_np.shape}")

    pred_np = _tensor_to_numpy(pred_regions)
    gt_np = _tensor_to_numpy(gt_regions)
    idx = _region_index(region)
    pred_mask = pred_np[idx].astype(bool)
    gt_mask = gt_np[idx].astype(bool)

    assert_matching_geometry(
        pred_mask,
        gt_mask,
        gt_affine,
        gt_affine,
        context=f"{subject_id}/{region}",
    )

    return CaseVolumes(
        subject_id=subject_id,
        region=region,
        background=background,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        affine=gt_affine,
    )


def select_axial_slices(mask: np.ndarray, n_slices: int = 4) -> list[int]:
    """Pick ``n_slices`` evenly spaced axial indices where ``mask`` is non-empty."""
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    occupied = np.where(mask.any(axis=(0, 1)))[0]
    if occupied.size == 0:
        # Fall back to central slices when the region is empty.
        depth = mask.shape[2]
        return [int(z) for z in np.linspace(0, depth - 1, n_slices, dtype=int)]
    if occupied.size <= n_slices:
        return occupied.tolist()
    positions = np.linspace(0, occupied.size - 1, n_slices, dtype=int)
    return sorted(int(occupied[i]) for i in positions)


def plot_case_sanity(
    case: CaseVolumes,
    output_png: str | Path,
    *,
    n_slices: int = 4,
) -> Path:
    """Save a PNG grid of axial slices with pred / GT contours on FLAIR."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    combined = case.pred_mask | case.gt_mask
    slice_indices = select_axial_slices(combined, n_slices=n_slices)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes]

    for ax, z in zip(axes, slice_indices):
        ax.imshow(case.background[:, :, z].T, cmap="gray", origin="lower")
        gt_slice = case.gt_mask[:, :, z].T.astype(float)
        pred_slice = case.pred_mask[:, :, z].T.astype(float)
        for mask_slice, color, label in (
            (gt_slice, "#2ca02c", "GT"),
            (pred_slice, "#d62728", "Pred"),
        ):
            if not mask_slice.any():
                continue
            for contour in find_contours(mask_slice, level=0.5):
                ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.2, label=label)
        ax.set_title(f"z={z}")
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=2)
    dice = dice_numpy(case.pred_mask, case.gt_mask)
    fig.suptitle(
        f"{case.subject_id} | {case.region} | Dice={dice:.3f}",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sanity plot -> %s", output_png)
    return output_png


def run_case_sanity_check(
    subject_id: str,
    checkpoint_path: str | Path,
    output_png: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    region: RegionName = "whole_tumor",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    device: str | None = None,
) -> tuple[float, float]:
    """Full sanity check for one case: Dice agreement + contour PNG."""
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti

    case = load_case_volumes(
        subject_id,
        checkpoint_path,
        brats_nifti_root=brats_nifti_root,
        region=region,
        roi_size=roi_size,
        device=device,
    )
    manual, monai = assert_dice_agreement(case.pred_mask, case.gt_mask)
    logger.info(
        "%s | %s | manual Dice=%.4f | MONAI Dice=%.4f",
        subject_id,
        region,
        manual,
        monai,
    )
    plot_case_sanity(case, output_png)
    return manual, monai


def select_best_worst_cases(
    eval_csv: str | Path,
    *,
    metric: str = "dice_mean",
) -> tuple[str, str]:
    """Return (best_subject_id, worst_subject_id) from an evaluation CSV."""
    df = pd.read_csv(eval_csv)
    if metric not in df.columns:
        raise ValueError(f"Metric column {metric!r} not in {eval_csv}; columns={list(df.columns)}")
    ranked = df.dropna(subset=[metric, "subject_id"]).sort_values(metric, ascending=False)
    if ranked.empty:
        raise ValueError(f"No valid rows in evaluation CSV: {eval_csv}")
    best = str(ranked.iloc[0]["subject_id"])
    worst = str(ranked.iloc[-1]["subject_id"])
    return best, worst


def run_best_worst_sanity_checks(
    checkpoint_path: str | Path,
    eval_csv: str | Path,
    output_dir: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    region: RegionName = "whole_tumor",
    metric: str = "dice_mean",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    device: str | None = None,
) -> pd.DataFrame:
    """Run sanity checks on the best- and worst-Dice validation cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_id, worst_id = select_best_worst_cases(eval_csv, metric=metric)

    rows: list[dict[str, Any]] = []
    for tag, subject_id in (("best", best_id), ("worst", worst_id)):
        png = output_dir / f"sanity_{tag}_{subject_id}_{region}.png"
        manual, monai = run_case_sanity_check(
            subject_id,
            checkpoint_path,
            png,
            brats_nifti_root=brats_nifti_root,
            region=region,
            roi_size=roi_size,
            device=device,
        )
        rows.append(
            {
                "tag": tag,
                "subject_id": subject_id,
                "region": region,
                "dice_manual": manual,
                "dice_monai": monai,
                "plot_path": str(png),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "sanity_check_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Best case: %s | Worst case: %s | summary -> %s", best_id, worst_id, summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segmentation sanity checks with contour plots.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, default=None, help="Evaluation CSV for best/worst")
    parser.add_argument("--subject-id", type=str, default=None, help="Single subject (overrides CSV)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--region", choices=BRATS_REGIONS, default="whole_tumor")
    parser.add_argument("--metric", default="dice_mean", help="Column for best/worst selection")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    output_dir = args.output_dir or (app_cfg.paths.checkpoints / "sanity_checks")

    if args.subject_id is not None:
        run_case_sanity_check(
            args.subject_id,
            args.checkpoint,
            Path(output_dir) / f"sanity_{subject_id}_{args.region}.png",
            brats_nifti_root=app_cfg.paths.brats_nifti,
            region=args.region,
        )
    elif args.eval_csv is not None:
        run_best_worst_sanity_checks(
            args.checkpoint,
            args.eval_csv,
            output_dir,
            brats_nifti_root=app_cfg.paths.brats_nifti,
            region=args.region,
            metric=args.metric,
        )
    else:
        parser.error("Provide --eval-csv for best/worst checks or --subject-id for one case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
