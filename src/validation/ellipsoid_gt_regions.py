"""GT-only ellipsoid validation for ET, TC, and WT on held-out BraTS cases.

No model predictions — diameters and formula volumes are computed from
ground-truth region masks only.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from monai.data import DataLoader, Dataset

from config import PROJECT_ROOT, load_config
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.model import BRATS_REGIONS
from segmentation.splits import read_id_list
from validation.ellipsoid import DIAMETER_METHODS, VOLUME_FORMULAS
from validation.ellipsoid_batch import (
    _agreement_row,
    _case_metrics_from_binary,
    _plot_bland_altman,
    _sphericity_terciles,
)

logger = logging.getLogger(__name__)

REGION_SHORT = {
    "enhancing_tumor": "ET",
    "tumor_core": "TC",
    "whole_tumor": "WT",
}


def brats_label_to_region_masks(label: np.ndarray) -> dict[str, np.ndarray]:
    """Map BraTS int labels {0,1,2,3} → binary ET / TC / WT masks.

    Conventions match ``segmentation.model.brats_label_to_regions``:
    ET = {3}, TC = {1,3}, WT = {1,2,3}.
    """
    lbl = np.asarray(label)
    necrotic = lbl == 1
    edema = lbl == 2
    enhancing = lbl == 3
    return {
        "enhancing_tumor": enhancing,
        "tumor_core": necrotic | enhancing,
        "whole_tumor": necrotic | edema | enhancing,
    }


def assert_region_nesting(label: np.ndarray) -> None:
    """Assert ET ⊆ TC ⊆ WT for a BraTS integer label volume."""
    masks = brats_label_to_region_masks(label)
    et, tc, wt = (
        masks["enhancing_tumor"],
        masks["tumor_core"],
        masks["whole_tumor"],
    )
    if not np.all(et <= tc):
        raise AssertionError("ET is not a subset of TC")
    if not np.all(tc <= wt):
        raise AssertionError("TC is not a subset of WT")
    # Volume consistency: voxels in ET count in TC and WT; TC ⊆ WT.
    et_n, tc_n, wt_n = int(et.sum()), int(tc.sum()), int(wt.sum())
    if not (et_n <= tc_n <= wt_n):
        raise AssertionError(
            f"Region voxel counts violate ET≤TC≤WT: ET={et_n} TC={tc_n} WT={wt_n}"
        )


def run_gt_only_ellipsoid(
    subjects_file: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Ellipsoid vs voxel volume on GT ET/TC/WT masks for held-out subjects."""
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    brats_nifti_root = Path(brats_nifti_root)
    output_dir = Path(
        output_dir or (PROJECT_ROOT / "results" / "ellipsoid" / "gt_only")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = read_id_list(subjects_file)
    records = [subject_to_datadict(brats_nifti_root / sid) for sid in subject_ids]
    loader = DataLoader(
        Dataset(data=records, transform=get_val_transforms(pixdim=pixdim)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    spacing = tuple(float(x) for x in pixdim)

    case_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {r: 0 for r in BRATS_REGIONS}

    for batch in loader:
        subject_id = batch.get("subject_id", ["unknown"])[0]
        label_np = batch["label"].squeeze(0).squeeze(0).detach().cpu().numpy()
        assert_region_nesting(label_np)
        masks = brats_label_to_region_masks(label_np)

        for region in BRATS_REGIONS:
            row = _case_metrics_from_binary(
                masks[region],
                spacing,
                source="gt",
                subject_id=subject_id,
            )
            if row is None:
                skipped[region] += 1
                logger.info("Skip %s %s: empty GT", subject_id, region)
                continue
            row["region"] = region
            row["region_short"] = REGION_SHORT[region]
            case_rows.append(row)

    cases = pd.DataFrame(case_rows)
    cases_csv = output_dir / "case_volumes.csv"
    cases.to_csv(cases_csv, index=False)

    # Agreement: region × diameter method × formula
    summary_rows: list[dict[str, Any]] = []
    for region in BRATS_REGIONS:
        sub = cases[cases["region"] == region].copy()
        if sub.empty:
            continue
        voxel = sub["voxel_volume_ml"].to_numpy(dtype=float)
        short = REGION_SHORT[region]
        for method in DIAMETER_METHODS:
            for formula in VOLUME_FORMULAS:
                col = f"ellipsoid_{method}_{formula}_ml"
                ell = sub[col].to_numpy(dtype=float)
                row = _agreement_row(
                    source="gt",
                    method=method,
                    formula=formula,
                    voxel_ml=voxel,
                    ellipsoid_ml=ell,
                    bootstrap_gt=True,
                    n_bootstrap=1000,
                    seed=42,
                )
                row["region"] = region
                row["region_short"] = short
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    # Put region columns first for readability.
    cols = ["region_short", "region", "diameter_method", "formula", "n"] + [
        c
        for c in summary.columns
        if c not in {"region_short", "region", "diameter_method", "formula", "n", "source"}
    ]
    summary = summary[cols]
    summary_csv = output_dir / "agreement_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Key scientific comparison: radiologist diameters × both formulas × region.
    rad = summary[summary["diameter_method"] == "radiologist"].copy()
    compare_cols = [
        "region_short",
        "formula",
        "n",
        "mean_bias_ml",
        "loa_lower_ml",
        "loa_upper_ml",
        "median_ratio",
        "iqr_ratio_low",
        "iqr_ratio_high",
        "spearman_r",
        "icc_2_1",
        "bias_ci95_low",
        "bias_ci95_high",
        "median_ratio_ci95_low",
        "median_ratio_ci95_high",
        "icc_2_1_ci95_low",
        "icc_2_1_ci95_high",
    ]
    region_compare = rad[compare_cols].sort_values(["formula", "region_short"])
    # Stable scientific order: ET (compact) → TC → WT (irregular)
    order = {"ET": 0, "TC": 1, "WT": 2}
    region_compare["_ord"] = region_compare["region_short"].map(order)
    region_compare = (
        region_compare.sort_values(["formula", "_ord"]).drop(columns="_ord")
    )
    compare_csv = output_dir / "region_formula_comparison.csv"
    region_compare.to_csv(compare_csv, index=False)

    # Bland–Altman: radiologist × abc_over_2 for each region (± sphericity terciles).
    for region in BRATS_REGIONS:
        sub = cases[cases["region"] == region].copy()
        if len(sub) < 2:
            continue
        short = REGION_SHORT[region]
        voxel = sub["voxel_volume_ml"].to_numpy(dtype=float)
        ell = sub["ellipsoid_radiologist_abc_over_2_ml"].to_numpy(dtype=float)
        terc = _sphericity_terciles(sub["sphericity"].to_numpy(dtype=float))
        stem = f"ba_gt_{short.lower()}_radiologist_abc_over_2"
        _plot_bland_altman(
            voxel,
            ell,
            title=f"GT {short} | radiologist | ABC/2 (mL)",
            out_path=fig_dir / f"{stem}_ml.png",
            percent=False,
            sphericity_tercile=terc,
        )
        _plot_bland_altman(
            voxel,
            ell,
            title=f"GT {short} | radiologist | ABC/2 (log-ratio %)",
            out_path=fig_dir / f"{stem}_pct.png",
            percent=True,
            sphericity_tercile=terc,
        )
        for key in ("low", "mid", "high"):
            sel = terc == key
            if int(sel.sum()) < 2:
                continue
            _plot_bland_altman(
                voxel[sel],
                ell[sel],
                title=f"GT {short} | radiologist | ABC/2 | sph={key}",
                out_path=fig_dir / f"{stem}_sph_{key}_ml.png",
                percent=False,
            )
            _plot_bland_altman(
                voxel[sel],
                ell[sel],
                title=f"GT {short} | radiologist | ABC/2 | sph={key} (log-ratio %)",
                out_path=fig_dir / f"{stem}_sph_{key}_pct.png",
                percent=True,
            )

    # Mean voxel volume per region (context for the scientific comparison).
    vol_stats = (
        cases.groupby("region_short", sort=False)
        .agg(
            n=("voxel_volume_ml", "size"),
            mean_voxel_ml=("voxel_volume_ml", "mean"),
            median_voxel_ml=("voxel_volume_ml", "median"),
            mean_sphericity=("sphericity", "mean"),
            median_sphericity=("sphericity", "median"),
        )
        .reset_index()
    )
    vol_stats.to_csv(output_dir / "region_volume_stats.csv", index=False)

    meta = pd.DataFrame(
        [
            {
                "subjects_file": str(subjects_file),
                "n_subjects": len(subject_ids),
                "n_et": int((cases["region"] == "enhancing_tumor").sum()),
                "n_tc": int((cases["region"] == "tumor_core").sum()),
                "n_wt": int((cases["region"] == "whole_tumor").sum()),
                "skipped_empty_et": skipped["enhancing_tumor"],
                "skipped_empty_tc": skipped["tumor_core"],
                "skipped_empty_wt": skipped["whole_tumor"],
                "pixdim": str(spacing),
            }
        ]
    )
    meta.to_csv(output_dir / "run_meta.csv", index=False)

    print("\n=== Region volume context ===")
    print(vol_stats.to_string(index=False))
    print("\n=== Agreement (all region × method × formula) ===")
    show = summary[
        [
            "region_short",
            "diameter_method",
            "formula",
            "n",
            "mean_bias_ml",
            "loa_lower_ml",
            "loa_upper_ml",
            "median_ratio",
            "iqr_ratio_low",
            "iqr_ratio_high",
            "spearman_r",
            "icc_2_1",
            "bias_ci95_low",
            "bias_ci95_high",
        ]
    ]
    print(show.to_string(index=False))
    print("\n=== Summary: radiologist formulas across ET / TC / WT ===")
    print(region_compare.to_string(index=False))
    print(f"\nOutput dir: {output_dir}")
    print(f"Cases: {cases_csv}")
    print(f"Agreement: {summary_csv}")
    print(f"Region comparison: {compare_csv}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GT-only ellipsoid validation for ET/TC/WT on held-out cases."
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=PROJECT_ROOT / "splits" / "heldout.txt",
    )
    parser.add_argument("--brats-nifti", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "ellipsoid" / "gt_only",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = load_config(args.config)
    run_gt_only_ellipsoid(
        args.subjects_file,
        brats_nifti_root=args.brats_nifti or app.paths.brats_nifti,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
