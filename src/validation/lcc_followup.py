"""Inference-only LCC follow-up: per-case raw vs lcc, overlays, pipeline volumes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from scipy import ndimage as ndi

from config import PROJECT_ROOT, load_config
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.postprocess import fragmentation_stats, keep_largest_cc
from segmentation.splits import read_id_list
from validation.bland_altman import bland_altman_stats
from validation.ellipsoid import diameters_from_binary, ellipsoid_volume_cm3
from validation.ellipsoid_batch import (
    _bootstrap_agreement_cis,
    _plot_bland_altman,
    _predict_wt_binary,
    checkpoint_tag,
    icc_2_1,
)

logger = logging.getLogger(__name__)
DPI = 300


def build_lcc_vs_raw_per_case(
    *,
    raw_metrics_csv: Path,
    lcc_metrics_csv: Path,
    fragmentation_csv: Path,
    raw_volumes_csv: Path,
    lcc_volumes_csv: Path,
    output_csv: Path,
    subjects_file: Path,
    brats_nifti_root: Path,
    checkpoint: Path,
    device: str | None = None,
) -> pd.DataFrame:
    """Merge Dice/HD95/volumes and compute GT connected-component stats."""
    raw = pd.read_csv(raw_metrics_csv)
    lcc = pd.read_csv(lcc_metrics_csv)
    frag = pd.read_csv(fragmentation_csv)
    raw_vol = pd.read_csv(raw_volumes_csv)
    lcc_vol = pd.read_csv(lcc_volumes_csv)

    gt_vol = raw_vol[raw_vol["source"] == "gt"][["subject_id", "voxel_volume_ml"]].rename(
        columns={"voxel_volume_ml": "volume_gt_ml"}
    )
    pred_raw_vol = raw_vol[raw_vol["source"] == "pred"][
        ["subject_id", "voxel_volume_ml"]
    ].rename(columns={"voxel_volume_ml": "volume_raw_ml"})
    pred_lcc_vol = lcc_vol[lcc_vol["source"] == "pred"][
        ["subject_id", "voxel_volume_ml"]
    ].rename(columns={"voxel_volume_ml": "volume_lcc_ml"})

    df = raw[["subject_id", "dice_whole_tumor", "hd95_whole_tumor"]].rename(
        columns={"dice_whole_tumor": "dice_raw", "hd95_whole_tumor": "hd95_raw"}
    )
    df = df.merge(
        lcc[["subject_id", "dice_whole_tumor", "hd95_whole_tumor"]].rename(
            columns={"dice_whole_tumor": "dice_lcc", "hd95_whole_tumor": "hd95_lcc"}
        ),
        on="subject_id",
    )
    df["delta_dice"] = df["dice_lcc"] - df["dice_raw"]
    df = df.merge(
        frag[["subject_id", "n_components"]].rename(
            columns={"n_components": "n_components_raw"}
        ),
        on="subject_id",
    )
    df = df.merge(gt_vol, on="subject_id")
    df = df.merge(pred_raw_vol, on="subject_id")
    df = df.merge(pred_lcc_vol, on="subject_id")

    # GT CC stats on the same 1 mm grid used for eval.
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
    # Transforms only — no checkpoint needed for GT stats.
    loader = DataLoader(
        Dataset(data=records, transform=get_val_transforms()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    gt_rows: list[dict[str, Any]] = []
    for batch in loader:
        sid = batch["subject_id"][0]
        gt_wt = batch["label"].squeeze(0).squeeze(0).detach().cpu().numpy() != 0
        stats = fragmentation_stats(gt_wt, (1.0, 1.0, 1.0))
        gt_rows.append(
            {
                "subject_id": sid,
                "gt_n_components": stats["n_components"],
                "gt_frac_outside_largest": stats["frac_outside_largest"],
            }
        )
    gt_df = pd.DataFrame(gt_rows)
    df = df.merge(gt_df, on="subject_id")

    cols = [
        "subject_id",
        "dice_raw",
        "dice_lcc",
        "delta_dice",
        "hd95_raw",
        "hd95_lcc",
        "n_components_raw",
        "gt_n_components",
        "volume_gt_ml",
        "volume_raw_ml",
        "volume_lcc_ml",
    ]
    # Keep gt_frac for the worst-case report (extra column ok in file).
    out = df[cols + ["gt_frac_outside_largest"]].copy()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    improved = int((out["delta_dice"] > 0.01).sum())
    worsened = int((out["delta_dice"] < -0.01).sum())
    unchanged = int(len(out) - improved - worsened)
    print("\n=== LCC vs raw (WT Dice) ===")
    print(f"improved (ΔDice > +0.01):  {improved}")
    print(f"worsened (ΔDice < -0.01):  {worsened}")
    print(f"unchanged (|ΔDice|≤0.01): {unchanged}")
    worst8 = out.nsmallest(8, "delta_dice")
    print("\n8 most negative ΔDice:")
    print(
        worst8[
            [
                "subject_id",
                "delta_dice",
                "dice_raw",
                "dice_lcc",
                "gt_n_components",
                "gt_frac_outside_largest",
                "n_components_raw",
            ]
        ].to_string(index=False)
    )
    return out


def save_worst_lcc_overlays(
    per_case: pd.DataFrame,
    *,
    checkpoint: Path,
    subjects_file: Path,
    brats_nifti_root: Path,
    output_dir: Path,
    device: str | None = None,
    n_worst: int = 2,
) -> list[Path]:
    """Save 3-plane overlays for the worst ΔDice cases (GT / LCC / removed)."""
    worst = per_case.nsmallest(n_worst, "delta_dice")
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for _, row in worst.iterrows():
        sid = str(row["subject_id"])
        rec = subject_to_datadict(brats_nifti_root / sid)
        batch = next(
            iter(
                DataLoader(
                    Dataset([rec], transform=get_val_transforms()),
                    batch_size=1,
                )
            )
        )
        gt_wt = batch["label"].squeeze(0).squeeze(0).detach().cpu().numpy() != 0
        pred_raw = _predict_wt_binary(
            model, batch["image"], device_t, (96, 96, 96), postprocess="raw"
        )
        pred_lcc = keep_largest_cc(pred_raw)
        # Match eval lcc = LCC + hole fill for display of "kept"; removed = raw - lcc core
        from segmentation.postprocess import lcc_then_fill

        pred_lcc_filled = lcc_then_fill(pred_raw)
        removed = pred_raw & ~pred_lcc_filled
        img = batch["image"][0, 0].detach().cpu().numpy()
        out_path = output_dir / f"worst_delta_dice_{sid}.png"
        _overlay_gt_lcc_removed(
            img,
            gt_wt,
            pred_lcc_filled,
            removed,
            out_path,
            subject_id=sid,
            delta_dice=float(row["delta_dice"]),
            gt_n_cc=int(row["gt_n_components"]),
            gt_frac_out=float(row["gt_frac_outside_largest"]),
        )
        paths.append(out_path)
        print(f"Overlay → {out_path}")
    return paths


def _overlay_gt_lcc_removed(
    image_3d: np.ndarray,
    gt_wt: np.ndarray,
    pred_lcc: np.ndarray,
    removed: np.ndarray,
    out_path: Path,
    *,
    subject_id: str,
    delta_dice: float,
    gt_n_cc: int,
    gt_frac_out: float,
) -> None:
    if np.any(gt_wt):
        cx, cy, cz = [int(np.round(c)) for c in ndi.center_of_mass(gt_wt)]
    else:
        cx, cy, cz = [s // 2 for s in pred_lcc.shape]
    cx = int(np.clip(cx, 0, pred_lcc.shape[0] - 1))
    cy = int(np.clip(cy, 0, pred_lcc.shape[1] - 1))
    cz = int(np.clip(cz, 0, pred_lcc.shape[2] - 1))

    def _plane(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
        if axis == 0:
            return vol[index, :, :]
        if axis == 1:
            return vol[:, index, :]
        return vol[:, :, index]

    def _norm(sl: np.ndarray) -> np.ndarray:
        sl = sl.astype(float)
        p2, p98 = np.percentile(sl, (2, 98))
        if p98 <= p2:
            return np.zeros_like(sl)
        return np.clip((sl - p2) / (p98 - p2), 0, 1)

    titles = ("axial (z)", "coronal (y)", "sagittal (x)")
    axes_idx = (2, 1, 0)
    indices = (cz, cy, cx)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, axis, idx in zip(axs, titles, axes_idx, indices):
        base = _norm(_plane(image_3d, axis, idx))
        rgb = np.stack([base, base, base], axis=-1)
        g = _plane(gt_wt, axis, idx)
        l = _plane(pred_lcc, axis, idx)
        r = _plane(removed, axis, idx)
        rgb[g] = rgb[g] * 0.4 + np.array([0.1, 0.85, 0.2]) * 0.6
        rgb[l] = rgb[l] * 0.35 + np.array([0.1, 0.75, 0.95]) * 0.65
        rgb[r] = np.array([0.95, 0.15, 0.1])
        ax.imshow(np.rot90(rgb), origin="lower")
        ax.set_title(f"{title} @{idx}")
        ax.axis("off")
    fig.suptitle(
        f"{subject_id}  ΔDice={delta_dice:+.3f}  "
        f"GT CCs={gt_n_cc}  GT outside LCC={gt_frac_out:.3f}\n"
        "green=GT WT, cyan=LCC pred, red=removed by LCC",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def run_pipeline_level_lcc(
    *,
    checkpoint: Path,
    subjects_file: Path,
    brats_nifti_root: Path,
    output_dir: Path,
    device: str | None = None,
) -> pd.DataFrame:
    """Pipeline volume error: lcc pred voxel & radiologist ABC/2 vs GT voxel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

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

    gt_vols: list[float] = []
    pred_vols: list[float] = []
    ellip_vols: list[float] = []
    sids: list[str] = []

    for batch in loader:
        sid = batch["subject_id"][0]
        gt_wt = batch["label"].squeeze(0).squeeze(0).detach().cpu().numpy() != 0
        if not np.any(gt_wt):
            continue
        pred_lcc = _predict_wt_binary(
            model,
            batch["image"],
            device_t,
            (96, 96, 96),
            postprocess="lcc",
            spacing_mm=spacing,
        )
        if not np.any(pred_lcc):
            continue
        from validation.ellipsoid import voxel_volume_ml_from_binary

        gt_ml = voxel_volume_ml_from_binary(gt_wt, spacing)
        pred_ml = voxel_volume_ml_from_binary(pred_lcc, spacing)
        diameters, _ = diameters_from_binary(pred_lcc, spacing, method="radiologist")
        ell_ml = ellipsoid_volume_cm3(diameters, formula="abc_over_2")
        sids.append(sid)
        gt_vols.append(gt_ml)
        pred_vols.append(pred_ml)
        ellip_vols.append(ell_ml)

    gt = np.asarray(gt_vols, dtype=float)
    pred = np.asarray(pred_vols, dtype=float)
    ell = np.asarray(ellip_vols, dtype=float)

    rows = []
    for name, measured in (
        ("pred_voxel_vs_gt_voxel", pred),
        ("pred_ellipsoid_radiologist_abc_over_2_vs_gt_voxel", ell),
    ):
        ba = bland_altman_stats(measured, gt)
        ratio = measured / gt
        from scipy import stats as sp_stats

        spearman_r, spearman_p = sp_stats.spearmanr(gt, measured)
        cis = _bootstrap_agreement_cis(gt, measured, n_resamples=1000, seed=42)
        rows.append(
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

    summary = pd.DataFrame(rows)
    out_csv = output_dir / "pipeline_level.csv"
    summary.to_csv(out_csv, index=False)
    print("\n=== Pipeline-level volume error (LCC pred vs GT voxel) ===")
    print(summary.to_string(index=False))
    print(f"Wrote {out_csv}")
    return summary


def summarize_wt_metrics(metrics_csv: Path, label: str) -> dict[str, Any]:
    df = pd.read_csv(metrics_csv)
    d = df["dice_whole_tumor"].to_numpy(dtype=float)
    h = df["hd95_whole_tumor"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    h = h[np.isfinite(h)]
    return {
        "mode": label,
        "n": int(d.size),
        "dice_mean_sd": f"{np.mean(d):.3f} ± {np.std(d, ddof=1):.3f}",
        "dice_median": f"{np.median(d):.3f}",
        "hd95_mean_sd": f"{np.mean(h):.1f} ± {np.std(h, ddof=1):.1f}",
        "hd95_median": f"{np.median(h):.1f}",
        "dice_mean": float(np.mean(d)),
        "hd95_mean": float(np.mean(h)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCC vs raw follow-up analysis.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "brats_pretrain" / "best_model.pt",
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=PROJECT_ROOT / "splits" / "heldout.txt",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app_cfg = load_config(args.config)
    tag = checkpoint_tag(args.checkpoint)
    ellip_root = PROJECT_ROOT / "results" / "ellipsoid" / tag

    per_case = build_lcc_vs_raw_per_case(
        raw_metrics_csv=PROJECT_ROOT
        / "checkpoints"
        / "brats_eval_heldout"
        / "raw"
        / "metrics.csv",
        lcc_metrics_csv=PROJECT_ROOT
        / "checkpoints"
        / "brats_eval_heldout"
        / "lcc"
        / "metrics.csv",
        fragmentation_csv=ellip_root / "fragmentation.csv",
        raw_volumes_csv=ellip_root / "raw" / "case_volumes.csv",
        lcc_volumes_csv=ellip_root / "lcc" / "case_volumes.csv",
        output_csv=PROJECT_ROOT / "results" / "lcc_vs_raw_per_case.csv",
        subjects_file=args.subjects_file,
        brats_nifti_root=app_cfg.paths.brats_nifti,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    save_worst_lcc_overlays(
        per_case,
        checkpoint=args.checkpoint,
        subjects_file=args.subjects_file,
        brats_nifti_root=app_cfg.paths.brats_nifti,
        output_dir=ellip_root / "worst_delta_overlays",
        device=args.device,
    )
    run_pipeline_level_lcc(
        checkpoint=args.checkpoint,
        subjects_file=args.subjects_file,
        brats_nifti_root=app_cfg.paths.brats_nifti,
        output_dir=ellip_root,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
