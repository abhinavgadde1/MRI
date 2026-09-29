"""Post-training held-out comparison: v1 / v2 / v3 (inference only)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from scipy import stats as sp_stats

from config import PROJECT_ROOT, load_config
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS, brats_label_to_regions
from segmentation.postprocess import apply_channel_postprocess, fragmentation_stats, keep_largest_cc
from segmentation.report_metrics import REGION_SHORT, _bootstrap_ci, _bootstrap_mean_diff_ci
from segmentation.splits import read_id_list
from validation.bland_altman import bland_altman_stats
from validation.ellipsoid import diameters_from_binary, ellipsoid_volume_cm3, voxel_volume_ml_from_binary
from validation.ellipsoid_batch import _bootstrap_agreement_cis, _plot_bland_altman, icc_2_1


def build_v1_v2_v3_table(
    v1_csv: Path,
    v2_csv: Path,
    v3_csv: Path,
    output_csv: Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    frames = {
        "v1_lcc": pd.read_csv(v1_csv),
        "v2_lcc": pd.read_csv(v2_csv),
        "v3_lcc": pd.read_csv(v3_csv),
    }
    rows: list[dict[str, Any]] = []
    for region in BRATS_REGIONS:
        short = REGION_SHORT[region]
        for metric, col in (("Dice", f"dice_{region}"), ("HD95", f"hd95_{region}")):
            row: dict[str, Any] = {"region": short, "metric": metric}
            for prefix, df in frames.items():
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


def paired_delta(
    baseline_csv: Path,
    newer_csv: Path,
    output_csv: Path,
    *,
    label: str,
) -> pd.DataFrame:
    """Paired ΔDice (newer − baseline) on matched subject_ids."""
    a = pd.read_csv(baseline_csv).set_index("subject_id")
    b = pd.read_csv(newer_csv).set_index("subject_id")
    common = sorted(set(a.index) & set(b.index))
    rows: list[dict[str, Any]] = []
    for region in BRATS_REGIONS:
        col = f"dice_{region}"
        aa = a.loc[common, col].to_numpy(dtype=float)
        bb = b.loc[common, col].to_numpy(dtype=float)
        mask = np.isfinite(aa) & np.isfinite(bb)
        aa_m, bb_m = aa[mask], bb[mask]
        mean_d, lo, hi = _bootstrap_mean_diff_ci(aa_m, bb_m)
        delta = bb_m - aa_m
        improved = int((delta > 0.01).sum())
        worsened = int((delta < -0.01).sum())
        unchanged = int(len(delta) - improved - worsened)
        rows.append(
            {
                "comparison": label,
                "region": REGION_SHORT[region],
                "n": int(aa_m.size),
                "mean_delta_dice": mean_d,
                "ci95_low": lo,
                "ci95_high": hi,
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
            }
        )
    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def region_n_summary(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    rows = []
    for region in BRATS_REGIONS:
        col = f"dice_{region}"
        n = int(df[col].notna().sum())
        rows.append(
            {
                "region": REGION_SHORT[region],
                "n_nonempty_gt": n,
                "n_empty_gt_excluded": int(len(df) - n),
                "mean_dice": float(df[col].mean(skipna=True)),
                "mean_hd95": float(df[f"hd95_{region}"].mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def diagnose_lcc_zero_wt(
    checkpoint: Path,
    subjects_file: Path,
    brats_nifti_root: Path,
    v3_raw_csv: Path,
    v3_lcc_csv: Path,
    output_dir: Path,
    *,
    device: str | None = None,
) -> pd.DataFrame:
    """Find WT Dice=0 after LCC; confirm FP-blob vs true-tumor discard mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(v3_raw_csv).set_index("subject_id")
    lcc_df = pd.read_csv(v3_lcc_csv).set_index("subject_id")
    zero_ids = lcc_df.index[lcc_df["dice_whole_tumor"].fillna(-1) == 0.0].tolist()

    if not zero_ids:
        empty = pd.DataFrame(
            columns=[
                "subject_id",
                "gt_voxels",
                "gt_ml",
                "gt_n_cc",
                "raw_voxels",
                "raw_ml",
                "raw_n_cc",
                "lcc_voxels",
                "lcc_ml",
                "inter_raw",
                "inter_lcc",
                "dice_raw",
                "dice_lcc",
                "largest_cc_overlap_gt",
                "failure_mode",
            ]
        )
        empty.to_csv(output_dir / "zero_dice_diagnosis.csv", index=False)
        return empty

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
    spacing = (1.0, 1.0, 1.0)
    rows: list[dict[str, Any]] = []

    for sid in zero_ids:
        batch = Dataset(
            data=[subject_to_datadict(brats_nifti_root / sid)],
            transform=get_val_transforms(),
        )[0]
        # Dataset __getitem__ returns dict of tensors; wrap as batch dim
        image = batch["image"].unsqueeze(0).to(device_t, dtype=torch.float32)
        label = batch["label"]
        gt_wt = label.squeeze(0).detach().cpu().numpy() != 0
        logits = sliding_window_inference(
            image, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
        )
        pred_raw = (torch.sigmoid(logits) > 0.5).squeeze(0).detach().cpu().numpy()
        wt_raw = pred_raw[2].astype(bool)
        wt_lcc = keep_largest_cc(wt_raw)
        inter_raw = int(np.logical_and(wt_raw, gt_wt).sum())
        inter_lcc = int(np.logical_and(wt_lcc, gt_wt).sum())
        gt_stats = fragmentation_stats(gt_wt, spacing)
        raw_stats = fragmentation_stats(wt_raw, spacing)
        lcc_stats = fragmentation_stats(wt_lcc, spacing)
        dice_raw = float(raw_df.loc[sid, "dice_whole_tumor"]) if sid in raw_df.index else float(
            "nan"
        )
        dice_lcc = float(lcc_df.loc[sid, "dice_whole_tumor"])
        # Same mode as v1/v2: raw overlaps GT but LCC does not
        if inter_raw > 0 and inter_lcc == 0:
            mode = "largest_cc_keeps_fp_blob_discards_true_tumor"
        elif inter_raw == 0 and inter_lcc == 0:
            mode = "no_raw_overlap_with_gt"
        else:
            mode = "other"
        rows.append(
            {
                "subject_id": sid,
                "gt_voxels": int(gt_wt.sum()),
                "gt_ml": voxel_volume_ml_from_binary(gt_wt, spacing),
                "gt_n_cc": int(gt_stats["n_components"]),
                "raw_voxels": int(wt_raw.sum()),
                "raw_ml": voxel_volume_ml_from_binary(wt_raw, spacing) if wt_raw.any() else 0.0,
                "raw_n_cc": int(raw_stats["n_components"]),
                "lcc_voxels": int(wt_lcc.sum()),
                "lcc_ml": voxel_volume_ml_from_binary(wt_lcc, spacing) if wt_lcc.any() else 0.0,
                "inter_raw": inter_raw,
                "inter_lcc": inter_lcc,
                "dice_raw": dice_raw,
                "dice_lcc": dice_lcc,
                "largest_cc_overlap_gt": inter_lcc,
                "failure_mode": mode,
            }
        )
        _save_gt_vs_lcc_overlay(
            gt_wt, wt_raw, wt_lcc, output_dir / f"gt_vs_lcc_{sid}.png", title=sid
        )
        _save_frag_overlay(wt_raw, wt_lcc, output_dir / f"frag_{sid}.png", title=sid)

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "zero_dice_diagnosis.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "model": "v3 (brats_scale_full)",
                "n_heldout": int(len(lcc_df)),
                "n_lcc_dice_zero": int(len(zero_ids)),
                "rate": f"{len(zero_ids)}/{len(lcc_df)}",
                "cases": "; ".join(zero_ids),
                "mechanism": (
                    "Largest-CC heuristic keeps a larger false-positive blob; "
                    "true-tumor component is discarded (raw∩GT>0, LCC∩GT=0)."
                    if all(r["failure_mode"] == "largest_cc_keeps_fp_blob_discards_true_tumor" for r in rows)
                    else "See zero_dice_diagnosis.csv failure_mode column."
                ),
            }
        ]
    )
    summary.to_csv(output_dir / "lcc_failure_summary.csv", index=False)
    return out


def _mid_slice(mask: np.ndarray) -> int:
    zs = np.where(mask.any(axis=(0, 1)))[0]
    if zs.size == 0:
        return mask.shape[2] // 2
    return int(zs[len(zs) // 2])


def _save_gt_vs_lcc_overlay(
    gt: np.ndarray,
    raw: np.ndarray,
    lcc: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    z = _mid_slice(gt | raw)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m, name in zip(
        axes,
        (gt, raw, lcc),
        ("GT WT", "Pred raw WT", "Pred LCC WT"),
    ):
        ax.imshow(m[:, :, z].T, origin="lower", cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_frag_overlay(
    raw: np.ndarray, lcc: np.ndarray, out_path: Path, *, title: str
) -> None:
    z = _mid_slice(raw)
    rgb = np.zeros((*raw[:, :, z].shape, 3), dtype=float)
    rgb[raw[:, :, z].T] = (0.2, 0.2, 0.8)
    rgb[lcc[:, :, z].T] = (0.9, 0.2, 0.1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb, origin="lower")
    ax.set_title(f"{title}: blue=raw, red=LCC")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_v3_volume_and_pipeline(
    checkpoint: Path,
    subjects_file: Path,
    brats_nifti_root: Path,
    *,
    volume_csv: Path,
    pipeline_dir: Path,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    for batch in loader:
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

        pr_wt = pred_lcc[2].astype(bool)
        if np.any(gt_wt_raw) and np.any(pr_wt):
            gt_ml = voxel_volume_ml_from_binary(gt_wt_raw, spacing)
            pr_ml = voxel_volume_ml_from_binary(pr_wt, spacing)
            diameters, _ = diameters_from_binary(pr_wt, spacing, method="radiologist")
            ell_ml = ellipsoid_volume_cm3(diameters, formula="abc_over_2")
            pipe_gt.append(gt_ml)
            pipe_pred.append(pr_ml)
            pipe_ell.append(ell_ml)

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
            title=f"Pipeline LCC v3 | {name} (mL)",
            out_path=fig_dir / f"{stem}_ml.png",
            percent=False,
        )
        _plot_bland_altman(
            gt,
            measured,
            title=f"Pipeline LCC v3 | {name} (log-ratio %)",
            out_path=fig_dir / f"{stem}_pct.png",
            percent=True,
        )
    pipeline = pd.DataFrame(pipe_rows)
    pipeline.to_csv(pipeline_dir / "pipeline_level.csv", index=False)
    return vols, pipeline


def best_worst_wt(metrics_csv: Path, output_csv: Path, n: int = 5) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    cols = ["subject_id", "dice_whole_tumor", "hd95_whole_tumor", "dice_enhancing_tumor", "dice_tumor_core"]
    worst = df.nsmallest(n, "dice_whole_tumor")[cols].assign(rank="worst")
    best = df.nlargest(n, "dice_whole_tumor")[cols].assign(rank="best")
    out = pd.concat([worst, best], ignore_index=True)
    out.to_csv(output_csv, index=False)
    return out


def recover_wt_check(paired_v3_v2: pd.DataFrame, paired_v3_v1: pd.DataFrame) -> str:
    """Plain-language check: WT recovery without giving back ET/TC gains."""
    def d(df: pd.DataFrame, region: str) -> float:
        return float(df.loc[df["region"] == region, "mean_delta_dice"].iloc[0])

    v3_v2_wt = d(paired_v3_v2, "WT")
    v3_v2_et = d(paired_v3_v2, "ET")
    v3_v2_tc = d(paired_v3_v2, "TC")
    v3_v1_et = d(paired_v3_v1, "ET")
    v3_v1_tc = d(paired_v3_v1, "TC")
    v3_v1_wt = d(paired_v3_v1, "WT")
    lines = [
        f"v3−v2: ET={v3_v2_et:+.4f}, TC={v3_v2_tc:+.4f}, WT={v3_v2_wt:+.4f}",
        f"v3−v1: ET={v3_v1_et:+.4f}, TC={v3_v1_tc:+.4f}, WT={v3_v1_wt:+.4f}",
    ]
    recovers_wt = v3_v2_wt > 0.01
    keeps_et_tc_vs_v1 = v3_v1_et > 0.01 and v3_v1_tc > 0.01
    if recovers_wt and keeps_et_tc_vs_v1:
        lines.append(
            "VERDICT: v3 recovers WT vs v2 while keeping ET/TC gains over v1."
        )
    elif keeps_et_tc_vs_v1 and not recovers_wt:
        lines.append(
            "VERDICT: ET/TC gains over v1 retained, but WT not clearly recovered vs v2 "
            f"(ΔWT={v3_v2_wt:+.4f})."
        )
    else:
        lines.append("VERDICT: mixed — see paired tables.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-inference-extras",
        action="store_true",
        help="Only build tables from existing metrics CSVs (skip volume/LCC diagnose pass)",
    )
    args = parser.parse_args(argv)
    app = load_config()
    v1 = PROJECT_ROOT / "checkpoints/brats_eval_heldout/lcc/metrics.csv"
    v2 = PROJECT_ROOT / "checkpoints/brats_eval_heldout_v2/lcc/metrics.csv"
    v3 = PROJECT_ROOT / "checkpoints/brats_eval_heldout_v3/lcc/metrics.csv"
    v3_raw = PROJECT_ROOT / "checkpoints/brats_eval_heldout_v3/raw/metrics.csv"
    ckpt = PROJECT_ROOT / "checkpoints/brats_scale_full/best_model.pt"
    subjects = PROJECT_ROOT / "splits/heldout.txt"
    brats = app.paths.brats_nifti

    print("\n=== n per region (v3-lcc, empty-GT excluded) ===")
    nsum = region_n_summary(v3)
    print(nsum.to_string(index=False))
    nsum.to_csv(PROJECT_ROOT / "results/v3_region_n.csv", index=False)

    table = build_v1_v2_v3_table(
        v1, v2, v3, PROJECT_ROOT / "results/segmentation_table_v3.csv"
    )
    print("\n=== v1-lcc / v2-lcc / v3-lcc ===")
    print(
        table[
            [
                "region",
                "metric",
                "v1_lcc_n",
                "v1_lcc_mean_pm_sd",
                "v1_lcc_median",
                "v1_lcc_ci95",
                "v2_lcc_n",
                "v2_lcc_mean_pm_sd",
                "v2_lcc_median",
                "v2_lcc_ci95",
                "v3_lcc_n",
                "v3_lcc_mean_pm_sd",
                "v3_lcc_median",
                "v3_lcc_ci95",
            ]
        ].to_string(index=False)
    )

    paired_32 = paired_delta(
        v2, v3, PROJECT_ROOT / "results/v3_vs_v2_paired_dice.csv", label="v3−v2"
    )
    paired_31 = paired_delta(
        v1, v3, PROJECT_ROOT / "results/v3_vs_v1_paired_dice.csv", label="v3−v1"
    )
    paired_all = pd.concat([paired_32, paired_31], ignore_index=True)
    paired_all.to_csv(PROJECT_ROOT / "results/v3_paired_dice.csv", index=False)
    print("\n=== Paired ΔDice ===")
    print(paired_all.to_string(index=False))
    print("\n=== WT recovery check ===")
    print(recover_wt_check(paired_32, paired_31))

    bw = best_worst_wt(
        v3, PROJECT_ROOT / "results/v3_best_worst_wt_cases.csv", n=5
    )
    print("\n=== 5 worst / 5 best WT (v3-lcc) ===")
    print(bw.to_string(index=False))

    if not args.skip_inference_extras:
        diag = diagnose_lcc_zero_wt(
            ckpt,
            subjects,
            brats,
            v3_raw,
            v3,
            PROJECT_ROOT / "results/v3_zero_dice_wt_overlays",
            device=args.device,
        )
        print("\n=== LCC zero-Dice WT (v3) ===")
        print(diag.to_string(index=False) if len(diag) else "(none)")

        vols, pipeline = run_v3_volume_and_pipeline(
            ckpt,
            subjects,
            brats,
            volume_csv=PROJECT_ROOT / "results/v3_region_volumes.csv",
            pipeline_dir=PROJECT_ROOT / "results/ellipsoid/brats_scale_full",
            device=args.device,
        )
        print("\n=== v3 pred vs GT voxel volumes (LCC) ===")
        print(vols.to_string(index=False))
        print("\n=== Pipeline-level (v3 LCC) ===")
        print(pipeline.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
