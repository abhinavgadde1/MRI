"""Batch ellipsoid vs voxel volume validation (GT + predicted WT masks)."""

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
from monai.inferers import sliding_window_inference
from scipy import stats

from config import PROJECT_ROOT, load_config
from reconstruction.measurements import sphericity_index, surface_area_mm2_from_mask
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.postprocess import (
    DEFAULT_LCC_MIN_FRAC,
    DEFAULT_LCC_MIN_ML,
    POSTPROCESS_MODES,
    PostprocessMode,
    apply_binary_postprocess,
    fragmentation_stats,
    keep_largest_cc,
)
from segmentation.splits import read_id_list
from validation.bland_altman import bland_altman_plot_frame, bland_altman_stats
from validation.ellipsoid import (
    DIAMETER_METHODS,
    VOLUME_FORMULAS,
    DiameterMethod,
    VolumeFormula,
    diameters_from_binary,
    ellipsoid_volume_cm3,
    mesh_volume_ml_from_binary,
    mesh_voxel_relative_error,
    voxel_volume_ml_from_binary,
)

logger = logging.getLogger(__name__)

MESH_VOXEL_TOL = 0.02  # 2% relative agreement
DPI = 300


def checkpoint_tag(checkpoint_path: str | Path) -> str:
    """Folder name under ``results/ellipsoid/`` for a checkpoint."""
    path = Path(checkpoint_path)
    if path.stem in {"best_model", "last_model", "model", "checkpoint"}:
        return path.parent.name
    return path.stem


def icc_2_1(method_a: np.ndarray, method_b: np.ndarray) -> float:
    """Two-way random effects, single-measure, absolute-agreement ICC(2,1)."""
    a = np.asarray(method_a, dtype=float).ravel()
    b = np.asarray(method_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("Arrays must match for ICC")
    n = int(a.size)
    if n < 2:
        return float("nan")
    ratings = np.column_stack([a, b])
    k = 2
    grand = float(ratings.mean())
    mean_subj = ratings.mean(axis=1)
    mean_rater = ratings.mean(axis=0)
    ss_rows = float(k * np.sum((mean_subj - grand) ** 2))
    ss_cols = float(n * np.sum((mean_rater - grand) ** 2))
    ss_total = float(np.sum((ratings - grand) ** 2))
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0.0:
        return float("nan")
    return float((msr - mse) / denom)


def _bootstrap_agreement_cis(
    voxel_ml: np.ndarray,
    ellipsoid_ml: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap 95% CIs for bias, median ratio, and ICC(2,1)."""
    nan_cis = {
        "bias_ci95_low": float("nan"),
        "bias_ci95_high": float("nan"),
        "median_ratio_ci95_low": float("nan"),
        "median_ratio_ci95_high": float("nan"),
        "icc_2_1_ci95_low": float("nan"),
        "icc_2_1_ci95_high": float("nan"),
    }
    mask = np.isfinite(voxel_ml) & np.isfinite(ellipsoid_ml) & (voxel_ml > 0)
    v = voxel_ml[mask]
    e = ellipsoid_ml[mask]
    n = int(v.size)
    if n < 2:
        return nan_cis

    rng = np.random.default_rng(seed)
    biases = np.empty(n_resamples, dtype=float)
    med_ratios = np.empty(n_resamples, dtype=float)
    iccs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vv = v[idx]
        ee = e[idx]
        biases[i] = float(np.mean(ee - vv))
        med_ratios[i] = float(np.median(ee / vv))
        iccs[i] = icc_2_1(vv, ee)

    lo = alpha / 2.0
    hi = 1.0 - alpha / 2.0
    return {
        "bias_ci95_low": float(np.quantile(biases, lo)),
        "bias_ci95_high": float(np.quantile(biases, hi)),
        "median_ratio_ci95_low": float(np.quantile(med_ratios, lo)),
        "median_ratio_ci95_high": float(np.quantile(med_ratios, hi)),
        "icc_2_1_ci95_low": float(np.nanquantile(iccs, lo)),
        "icc_2_1_ci95_high": float(np.nanquantile(iccs, hi)),
    }


def _sphericity_from_binary(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> float:
    vol = voxel_volume_ml_from_binary(binary, spacing_mm)
    area = surface_area_mm2_from_mask(binary, spacing_mm)
    return float(sphericity_index(vol, area))


def _agreement_row(
    *,
    source: str,
    method: DiameterMethod,
    formula: VolumeFormula,
    voxel_ml: np.ndarray,
    ellipsoid_ml: np.ndarray,
    bootstrap_gt: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summary stats for one (source, diameter method, formula) combination."""
    mask = np.isfinite(voxel_ml) & np.isfinite(ellipsoid_ml) & (voxel_ml > 0)
    v = voxel_ml[mask]
    e = ellipsoid_ml[mask]
    n = int(v.size)
    empty_cis = {
        "bias_ci95_low": float("nan"),
        "bias_ci95_high": float("nan"),
        "median_ratio_ci95_low": float("nan"),
        "median_ratio_ci95_high": float("nan"),
        "icc_2_1_ci95_low": float("nan"),
        "icc_2_1_ci95_high": float("nan"),
    }
    if n < 2:
        return {
            "source": source,
            "diameter_method": method,
            "formula": formula,
            "n": n,
            "mean_bias_ml": float("nan"),
            "loa_lower_ml": float("nan"),
            "loa_upper_ml": float("nan"),
            "median_ratio": float("nan"),
            "iqr_ratio_low": float("nan"),
            "iqr_ratio_high": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "icc_2_1": float("nan"),
            **empty_cis,
        }

    # Bias = ellipsoid − voxel (positive ⇒ formula overestimates).
    ba = bland_altman_stats(e, v)
    ratio = e / v
    spearman_r, spearman_p = stats.spearmanr(v, e)
    row: dict[str, Any] = {
        "source": source,
        "diameter_method": method,
        "formula": formula,
        "n": n,
        "mean_bias_ml": float(np.mean(e - v)),
        "loa_lower_ml": ba.loa_lower,
        "loa_upper_ml": ba.loa_upper,
        "median_ratio": float(np.median(ratio)),
        "iqr_ratio_low": float(np.percentile(ratio, 25)),
        "iqr_ratio_high": float(np.percentile(ratio, 75)),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "icc_2_1": icc_2_1(v, e),
    }
    if bootstrap_gt and source == "gt":
        row.update(
            _bootstrap_agreement_cis(
                v, e, n_resamples=n_bootstrap, seed=seed
            )
        )
    else:
        row.update(empty_cis)
    return row


def _plot_bland_altman(
    voxel_ml: np.ndarray,
    ellipsoid_ml: np.ndarray,
    *,
    title: str,
    out_path: Path,
    percent: bool,
    sphericity_tercile: np.ndarray | None = None,
) -> None:
    mask = np.isfinite(voxel_ml) & np.isfinite(ellipsoid_ml) & (voxel_ml > 0)
    v = voxel_ml[mask]
    e = ellipsoid_ml[mask]
    if v.size < 2:
        return

    mean = (v + e) / 2.0
    if percent:
        # Percent difference via log-ratio: 100 * ln(e/v) ≈ % when small.
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = 100.0 * np.log(e / v)
        ylabel = "100 × ln(ellipsoid / voxel) (%)"
    else:
        diff = e - v
        ylabel = "Ellipsoid − voxel (mL)"

    finite = np.isfinite(diff)
    mean = mean[finite]
    diff = diff[finite]
    if diff.size < 2:
        return

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    loa_lo = mean_diff - 1.96 * std_diff
    loa_hi = mean_diff + 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if sphericity_tercile is not None:
        terc = sphericity_tercile[mask][finite]
        colors = {"low": "#4C78A8", "mid": "#F58518", "high": "#54A24B"}
        for key, color in colors.items():
            sel = terc == key
            if not np.any(sel):
                continue
            ax.scatter(
                mean[sel],
                diff[sel],
                c=color,
                alpha=0.8,
                label=f"sphericity {key}",
                edgecolors="none",
            )
    else:
        ax.scatter(mean, diff, c="#2E5A88", alpha=0.75, edgecolors="none")

    ax.axhline(mean_diff, color="black", lw=1.2, label=f"bias={mean_diff:.2f}")
    ax.axhline(
        loa_lo,
        color="gray",
        ls="--",
        lw=1.0,
        label=f"LoA [{loa_lo:.2f}, {loa_hi:.2f}]",
    )
    ax.axhline(loa_hi, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Mean volume (mL)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _sphericity_terciles(sphericity: np.ndarray) -> np.ndarray:
    """Assign low/mid/high tercile labels (NaN stays empty string)."""
    out = np.array([""] * len(sphericity), dtype=object)
    finite = np.isfinite(sphericity)
    if finite.sum() < 3:
        out[finite] = "mid"
        return out
    vals = sphericity[finite]
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    labels = np.empty(finite.sum(), dtype=object)
    labels[vals <= q1] = "low"
    labels[(vals > q1) & (vals <= q2)] = "mid"
    labels[vals > q2] = "high"
    # Tie-break edges so each bin is non-empty when possible.
    out[finite] = labels
    return out


@torch.no_grad()
def _predict_wt_binary(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    roi_size: tuple[int, int, int],
    *,
    postprocess: PostprocessMode = "raw",
    lcc_min_frac: float = DEFAULT_LCC_MIN_FRAC,
    lcc_min_ml: float = DEFAULT_LCC_MIN_ML,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return WT channel boolean mask after sigmoid > 0.5 (+ optional cleanup)."""
    inputs = image.to(device=device, dtype=torch.float32)
    logits = sliding_window_inference(
        inputs,
        roi_size=roi_size,
        sw_batch_size=1,
        predictor=model,
    )
    # Channel order: ET=0, TC=1, WT=2
    wt = (torch.sigmoid(logits[:, 2:3]) > 0.5).squeeze(0).squeeze(0)
    binary = wt.detach().cpu().numpy().astype(bool)
    return apply_binary_postprocess(
        binary,
        mode=postprocess,
        min_frac_of_largest=lcc_min_frac,
        min_volume_ml=lcc_min_ml,
        spacing_mm=spacing_mm,
    )


def save_fragmentation_overlay(
    image_3d: np.ndarray,
    gt_wt: np.ndarray,
    pred_raw: np.ndarray,
    out_path: Path,
    *,
    subject_id: str,
) -> Path:
    """3-plane overlay: GT (green), pred LCC (cyan), pred fragments (red)."""
    from scipy import ndimage as ndi

    pred_raw = np.asarray(pred_raw, dtype=bool)
    gt_wt = np.asarray(gt_wt, dtype=bool)
    lcc = keep_largest_cc(pred_raw)
    fragments = pred_raw & ~lcc

    # Pick slices near the GT centroid (fallback: volume center).
    if np.any(gt_wt):
        cx, cy, cz = [int(np.round(c)) for c in ndi.center_of_mass(gt_wt)]
    else:
        cx, cy, cz = [s // 2 for s in pred_raw.shape]

    cx = int(np.clip(cx, 0, pred_raw.shape[0] - 1))
    cy = int(np.clip(cy, 0, pred_raw.shape[1] - 1))
    cz = int(np.clip(cz, 0, pred_raw.shape[2] - 1))

    def _plane(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
        if axis == 0:
            return vol[index, :, :]
        if axis == 1:
            return vol[:, index, :]
        return vol[:, :, index]

    def _norm_img(sl: np.ndarray) -> np.ndarray:
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
        base = _norm_img(_plane(image_3d, axis, idx))
        rgb = np.stack([base, base, base], axis=-1)
        g = _plane(gt_wt, axis, idx)
        l = _plane(lcc, axis, idx)
        f = _plane(fragments, axis, idx)
        # Tint overlays (keep underlying anatomy visible).
        rgb[g] = rgb[g] * 0.4 + np.array([0.1, 0.85, 0.2]) * 0.6
        rgb[l] = rgb[l] * 0.35 + np.array([0.1, 0.75, 0.95]) * 0.65
        rgb[f] = np.array([0.95, 0.15, 0.1])
        ax.imshow(np.rot90(rgb), origin="lower")
        ax.set_title(f"{title} @ {idx}")
        ax.axis("off")

    fig.suptitle(
        f"{subject_id}: green=GT WT, cyan=pred LCC, red=pred fragments",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def run_fragmentation_analysis(
    checkpoint_path: str | Path,
    subjects_file: str | Path,
    output_csv: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    device: str | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    overlay_path: str | Path | None = None,
) -> pd.DataFrame:
    """Analyze RAW predicted-WT connected components on the held-out set."""
    checkpoint_path = Path(checkpoint_path)
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    brats_nifti_root = Path(brats_nifti_root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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
    model = load_model_from_checkpoint(checkpoint_path, device_t)
    model.float().eval()
    loader = DataLoader(
        Dataset(data=records, transform=get_val_transforms(pixdim=pixdim)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    spacing = tuple(float(x) for x in pixdim)

    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    for batch in loader:
        subject_id = batch.get("subject_id", ["unknown"])[0]
        label = batch["label"].squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_wt = label != 0
        pred_raw = _predict_wt_binary(
            model, batch["image"], device_t, roi_size, postprocess="raw"
        )
        stats_row = fragmentation_stats(pred_raw, spacing)
        row = {"subject_id": subject_id, **stats_row}
        rows.append(row)
        payloads[str(subject_id)] = {
            "image": batch["image"][0, 0].detach().cpu().numpy(),
            "gt_wt": gt_wt,
            "pred_raw": pred_raw,
        }

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n=== Predicted WT fragmentation (raw, 26-connectivity) ===")
    print(f"n={len(df)}")
    for col in ("n_components", "frac_largest", "vol_largest_ml", "vol_total_ml", "max_dist_to_lcc_mm"):
        vals = df[col].to_numpy(dtype=float)
        print(
            f"  {col}: median={np.nanmedian(vals):.4g}  max={np.nanmax(vals):.4g}"
        )
    # 5 worst by n_components then (1 - frac_largest)
    ranked = df.sort_values(
        by=["n_components", "frac_largest", "max_dist_to_lcc_mm"],
        ascending=[False, True, False],
    ).head(5)
    print("\n5 worst cases:")
    print(
        ranked[
            [
                "subject_id",
                "n_components",
                "frac_largest",
                "vol_largest_ml",
                "vol_total_ml",
                "max_dist_to_lcc_mm",
            ]
        ].to_string(index=False)
    )

    if overlay_path is not None and payloads:
        worst_id = str(ranked.iloc[0]["subject_id"])
        payload = payloads[worst_id]
        save_fragmentation_overlay(
            payload["image"],
            payload["gt_wt"],
            payload["pred_raw"],
            Path(overlay_path),
            subject_id=worst_id,
        )
        print(f"Worst-case overlay → {overlay_path} ({worst_id})")

    return df


def _case_metrics_from_binary(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    source: str,
    subject_id: str,
) -> dict[str, Any] | None:
    """Per-case volumes / diameters; ``None`` if empty mask."""
    if not np.any(binary):
        return None

    voxel_ml = voxel_volume_ml_from_binary(binary, spacing_mm)
    try:
        mesh_ml = mesh_volume_ml_from_binary(binary, spacing_mm)
        mesh_err = abs(mesh_ml - voxel_ml) / voxel_ml if voxel_ml > 0 else float("nan")
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s/%s mesh volume failed: %s", subject_id, source, exc)
        mesh_ml = float("nan")
        mesh_err = float("nan")

    try:
        sph = _sphericity_from_binary(binary, spacing_mm)
    except Exception:  # noqa: BLE001
        sph = float("nan")

    row: dict[str, Any] = {
        "subject_id": subject_id,
        "source": source,
        "voxel_volume_ml": voxel_ml,
        "mesh_volume_ml": mesh_ml,
        "mesh_voxel_rel_error": mesh_err,
        "sphericity": sph,
        "n_voxels": int(np.count_nonzero(binary)),
    }

    for method in DIAMETER_METHODS:
        try:
            diameters, _center = diameters_from_binary(binary, spacing_mm, method=method)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s/%s/%s diameters failed: %s", subject_id, source, method, exc)
            diameters = (float("nan"), float("nan"), float("nan"))
        row[f"{method}_a_mm"] = diameters[0]
        row[f"{method}_b_mm"] = diameters[1]
        row[f"{method}_c_mm"] = diameters[2]
        for formula in VOLUME_FORMULAS:
            if any(not np.isfinite(d) or d <= 0 for d in diameters):
                vol = float("nan")
            else:
                vol = ellipsoid_volume_cm3(diameters, formula=formula)
            row[f"ellipsoid_{method}_{formula}_ml"] = vol
    return row


def run_ellipsoid_heldout(
    checkpoint_path: str | Path,
    subjects_file: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    postprocess: PostprocessMode = "raw",
    lcc_min_frac: float = DEFAULT_LCC_MIN_FRAC,
    lcc_min_ml: float = DEFAULT_LCC_MIN_ML,
) -> Path:
    """Evaluate ellipsoid formulas on held-out WT GT and predicted masks."""
    checkpoint_path = Path(checkpoint_path)
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    brats_nifti_root = Path(brats_nifti_root)

    tag = checkpoint_tag(checkpoint_path)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "ellipsoid" / tag / postprocess
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
    model = load_model_from_checkpoint(checkpoint_path, device_t)
    model.float().eval()

    loader = DataLoader(
        Dataset(data=records, transform=get_val_transforms(pixdim=pixdim)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    spacing = tuple(float(x) for x in pixdim)

    rows: list[dict[str, Any]] = []
    skipped_empty = {"gt": 0, "pred": 0}

    for batch in loader:
        subject_id = batch.get("subject_id", ["unknown"])[0]
        logger.info(
            "Ellipsoid case %s on %s (postprocess=%s)",
            subject_id,
            device_t,
            postprocess,
        )

        label = batch["label"]
        # label: (1, 1, H, W, D) BraTS ints → WT = nonzero
        gt_np = label.squeeze(0).squeeze(0).detach().cpu().numpy()
        gt_wt = gt_np != 0

        pred_wt = _predict_wt_binary(
            model,
            batch["image"],
            device_t,
            roi_size,
            postprocess=postprocess,
            lcc_min_frac=lcc_min_frac,
            lcc_min_ml=lcc_min_ml,
            spacing_mm=spacing,
        )

        gt_row = _case_metrics_from_binary(
            gt_wt, spacing, source="gt", subject_id=subject_id
        )
        if gt_row is None:
            skipped_empty["gt"] += 1
            logger.warning("Skip %s GT: empty WT mask", subject_id)
        else:
            rows.append(gt_row)

        pred_row = _case_metrics_from_binary(
            pred_wt, spacing, source="pred", subject_id=subject_id
        )
        if pred_row is None:
            skipped_empty["pred"] += 1
            logger.warning("Skip %s pred: empty WT mask", subject_id)
        else:
            rows.append(pred_row)

    df = pd.DataFrame(rows)
    cases_csv = output_dir / "case_volumes.csv"
    df.to_csv(cases_csv, index=False)

    # Mesh vs voxel check (all non-empty cases with finite mesh).
    mesh_ok = df["mesh_voxel_rel_error"].dropna()
    n_mesh = int(mesh_ok.size)
    n_mesh_fail = int((mesh_ok > MESH_VOXEL_TOL).sum())
    mesh_summary = pd.DataFrame(
        [
            {
                "n_checked": n_mesh,
                "n_exceeding_2pct": n_mesh_fail,
                "max_rel_error": float(mesh_ok.max()) if n_mesh else float("nan"),
                "median_rel_error": float(mesh_ok.median()) if n_mesh else float("nan"),
                "tolerance": MESH_VOXEL_TOL,
                "all_within_tolerance": bool(n_mesh_fail == 0 and n_mesh > 0),
            }
        ]
    )
    mesh_summary.to_csv(output_dir / "mesh_vs_voxel_check.csv", index=False)
    if n_mesh_fail:
        logger.warning(
            "Mesh vs voxel: %d/%d cases exceed %.0f%% relative error",
            n_mesh_fail,
            n_mesh,
            100 * MESH_VOXEL_TOL,
        )
    else:
        logger.info(
            "Mesh vs voxel: all %d cases within %.0f%%",
            n_mesh,
            100 * MESH_VOXEL_TOL,
        )

    # Agreement summaries + Bland–Altman plots.
    summary_rows: list[dict[str, Any]] = []
    for source in ("gt", "pred"):
        sub = df[df["source"] == source].copy()
        if sub.empty:
            continue
        terc = _sphericity_terciles(sub["sphericity"].to_numpy(dtype=float))
        voxel = sub["voxel_volume_ml"].to_numpy(dtype=float)

        for method in DIAMETER_METHODS:
            for formula in VOLUME_FORMULAS:
                col = f"ellipsoid_{method}_{formula}_ml"
                ell = sub[col].to_numpy(dtype=float)
                summary_rows.append(
                    _agreement_row(
                        source=source,
                        method=method,
                        formula=formula,
                        voxel_ml=voxel,
                        ellipsoid_ml=ell,
                    )
                )

                stem = f"ba_{source}_{method}_{formula}"
                _plot_bland_altman(
                    voxel,
                    ell,
                    title=f"{source.upper()} WT | {method} | {formula} (mL)",
                    out_path=fig_dir / f"{stem}_ml.png",
                    percent=False,
                    sphericity_tercile=terc,
                )
                _plot_bland_altman(
                    voxel,
                    ell,
                    title=f"{source.upper()} WT | {method} | {formula} (log-ratio %)",
                    out_path=fig_dir / f"{stem}_pct.png",
                    percent=True,
                    sphericity_tercile=terc,
                )

                # Stratified plots by sphericity tercile.
                for key in ("low", "mid", "high"):
                    sel = terc == key
                    if int(sel.sum()) < 2:
                        continue
                    _plot_bland_altman(
                        voxel[sel],
                        ell[sel],
                        title=f"{source.upper()} WT | {method} | {formula} | sph={key}",
                        out_path=fig_dir / f"{stem}_sph_{key}_ml.png",
                        percent=False,
                    )
                    _plot_bland_altman(
                        voxel[sel],
                        ell[sel],
                        title=(
                            f"{source.upper()} WT | {method} | {formula} | "
                            f"sph={key} (log-ratio %)"
                        ),
                        out_path=fig_dir / f"{stem}_sph_{key}_pct.png",
                        percent=True,
                    )

    summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "agreement_summary.csv"
    summary.to_csv(summary_csv, index=False)

    meta = pd.DataFrame(
        [
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_tag": tag,
                "subjects_file": str(subjects_file),
                "n_subjects": len(subject_ids),
                "n_gt_rows": int((df["source"] == "gt").sum()) if len(df) else 0,
                "n_pred_rows": int((df["source"] == "pred").sum()) if len(df) else 0,
                "skipped_empty_gt": skipped_empty["gt"],
                "skipped_empty_pred": skipped_empty["pred"],
                "region": "whole_tumor",
                "mesh_voxel_tolerance": MESH_VOXEL_TOL,
                "postprocess": postprocess,
            }
        ]
    )
    meta.to_csv(output_dir / "run_meta.csv", index=False)

    # Also drop a symlink-friendly copy under results/ellipsoid/ for the tagged run.
    logger.info(
        "Wrote case table → %s (gt=%d pred=%d, postprocess=%s)",
        cases_csv,
        meta.loc[0, "n_gt_rows"],
        meta.loc[0, "n_pred_rows"],
        postprocess,
    )
    print(summary.to_string(index=False))
    print(f"Output dir: {output_dir}")
    print(f"Cases CSV: {cases_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(mesh_summary.to_string(index=False))
    return output_dir


def run_ellipsoid_batch(
    brats_nifti_root: str | Path,
    output_dir: str | Path,
    *,
    max_cases: int = 50,
    whole_tumor: bool = True,
    label: int = 2,
) -> Path:
    """Legacy BraTS-GT-only batch (PCA + ABC/2) kept for backward compatibility."""
    from validation.ellipsoid import compare_to_ellipsoid

    root = Path(brats_nifti_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = sorted(root.glob("*/mask.nii.gz"))[:max_cases]
    if not masks:
        raise FileNotFoundError(f"No mask.nii.gz under {root}")

    rows: list[dict] = []
    use_label = None if whole_tumor else label
    for mask_path in masks:
        subject_id = mask_path.parent.name
        try:
            cmp = compare_to_ellipsoid(mask_path, label=use_label)
            rows.append(
                {
                    "subject_id": subject_id,
                    "mask_volume_ml": cmp.mask_volume_ml,
                    "ellipsoid_volume_ml": cmp.ellipsoid_volume_ml,
                    "volume_difference_ml": cmp.volume_difference_ml,
                    "volume_ratio": cmp.volume_ratio,
                    "diameter_a_mm": cmp.diameters_mm[0],
                    "diameter_b_mm": cmp.diameters_mm[1],
                    "diameter_c_mm": cmp.diameters_mm[2],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip %s: %s", subject_id, exc)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "ellipsoid_vs_voxel_volumes.csv"
    df.to_csv(csv_path, index=False)

    if len(df) < 2:
        raise RuntimeError(f"Need ≥2 successful cases for Bland–Altman; got {len(df)}")

    stats_ba = bland_altman_stats(df["mask_volume_ml"], df["ellipsoid_volume_ml"])
    frame = bland_altman_plot_frame(
        df["mask_volume_ml"],
        df["ellipsoid_volume_ml"],
        label_a="voxel_mask_ml",
        label_b="ellipsoid_ABC_over_2_ml",
    )
    frame.to_csv(output_dir / "bland_altman_points.csv", index=False)

    summary = {
        "n": stats_ba.n,
        "mean_diff_mask_minus_ellipsoid_ml": stats_ba.mean_diff,
        "std_diff_ml": stats_ba.std_diff,
        "loa_lower_ml": stats_ba.loa_lower,
        "loa_upper_ml": stats_ba.loa_upper,
        "mean_mask_ml": float(df["mask_volume_ml"].mean()),
        "mean_ellipsoid_ml": float(df["ellipsoid_volume_ml"].mean()),
        "mean_ratio_mask_over_ellipsoid": float(df["volume_ratio"].mean()),
    }
    pd.Series(summary).to_csv(output_dir / "bland_altman_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(frame["mean"], frame["diff"], c="#2E5A88", alpha=0.75)
    ax.axhline(stats_ba.mean_diff, color="black", linestyle="-", label=f"mean diff={stats_ba.mean_diff:.1f}")
    ax.axhline(stats_ba.loa_lower, color="gray", linestyle="--", label=f"LoA {stats_ba.loa_lower:.1f}")
    ax.axhline(stats_ba.loa_upper, color="gray", linestyle="--", label=f"LoA {stats_ba.loa_upper:.1f}")
    ax.set_xlabel("Mean volume (ml)")
    ax.set_ylabel("Voxel mask − ellipsoid A×B×C/2 (ml)")
    ax.set_title("Bland–Altman: voxel vs clinical ellipsoid volume")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = output_dir / "bland_altman_ellipsoid.png"
    fig.savefig(png_path, dpi=DPI)
    plt.close(fig)

    logger.info("Wrote %s | %s | n=%d", csv_path, png_path, stats_ba.n)
    print(pd.Series(summary).to_string())
    return csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ellipsoid vs voxel volume on held-out WT masks (GT + predicted). "
            "Outputs → results/ellipsoid/<checkpoint_name>/<postprocess>/"
        ),
    )
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
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--brats-nifti", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results/ellipsoid/<checkpoint_tag>/<postprocess>/",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--postprocess",
        choices=POSTPROCESS_MODES,
        default="raw",
        help="raw | lcc | lcc_min on predicted WT",
    )
    parser.add_argument("--lcc-min-frac", type=float, default=DEFAULT_LCC_MIN_FRAC)
    parser.add_argument("--lcc-min-ml", type=float, default=DEFAULT_LCC_MIN_ML)
    parser.add_argument(
        "--fragmentation-only",
        action="store_true",
        help="Only compute raw predicted-WT connected-component stats + overlay",
    )
    parser.add_argument(
        "--legacy-gt-only",
        action="store_true",
        help="Old path: PCA+ABC/2 on GT masks only (no checkpoint).",
    )
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--edema-only", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    brats = args.brats_nifti or app_cfg.paths.brats_nifti

    if args.legacy_gt_only:
        out = args.output_dir or (app_cfg.paths.checkpoints / "ellipsoid_validation")
        run_ellipsoid_batch(
            brats,
            out,
            max_cases=args.max_cases,
            whole_tumor=not args.edema_only,
        )
        return 0

    tag = checkpoint_tag(args.checkpoint)
    tag_root = PROJECT_ROOT / "results" / "ellipsoid" / tag

    if args.fragmentation_only:
        run_fragmentation_analysis(
            args.checkpoint,
            args.subjects_file,
            tag_root / "fragmentation.csv",
            brats_nifti_root=brats,
            device=args.device,
            overlay_path=tag_root / "worst_raw_overlay.png",
        )
        return 0

    out = args.output_dir or (tag_root / args.postprocess)
    run_ellipsoid_heldout(
        args.checkpoint,
        args.subjects_file,
        brats_nifti_root=brats,
        output_dir=out,
        device=args.device,
        postprocess=args.postprocess,
        lcc_min_frac=args.lcc_min_frac,
        lcc_min_ml=args.lcc_min_ml,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
