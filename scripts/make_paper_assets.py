#!/usr/bin/env python3
"""Assemble submission-ready tables/figures under results/paper/.

Primary model: v3 (checkpoints/brats_scale_full). Ablation: v1 → v2 → v3.
Reformats verified CSVs; generates a few held-out overlays for v3 extremes /
case 257 when missing. Does not retrain or re-run full held-out eval / ellipsoid
batch jobs.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper"
FIG = OUT / "figures"
TAB = OUT / "tables"
V3_CKPT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
V3_LCC = ROOT / "checkpoints" / "brats_eval_heldout_v3" / "lcc" / "metrics.csv"
V3_OVERLAY_DIR = ROOT / "results" / "v3_paper_overlays"


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(r[c]) for c in cols) + " |"
        for _, r in df.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def _ensure_dirs() -> None:
    for d in (OUT, FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required source missing: {path}")
    return pd.read_csv(path)


def table1_dataset_preprocessing() -> Path:
    """Dataset & preprocessing summary from splits + batch_summary_patients.csv."""
    train_n = len(
        (ROOT / "splits" / "train.txt").read_text().strip().splitlines()
    )
    held_n = len(
        (ROOT / "splits" / "heldout.txt").read_text().strip().splitlines()
    )
    val_n = len(
        (ROOT / "checkpoints" / "brats_pretrain" / "train_run_val_cases.txt")
        .read_text()
        .strip()
        .splitlines()
    )
    batch = _read_csv(ROOT / "data" / "processed" / "batch_summary_patients.csv")
    n_success = int((batch["status"] == "success").sum())
    n_failed = int((batch["status"] == "failed").sum())
    ok = batch[batch["status"] == "success"]
    n_t1c_native = int(
        ok["modalities_found"].fillna("").str.contains("T1c").sum()
    )
    n_t1c_standin = n_success - n_t1c_native
    failed = batch[batch["status"] == "failed"]
    fail_reasons = (
        failed["error"].fillna("unknown").astype(str).str.split(":").str[0].value_counts()
    )
    fail_note = "; ".join(f"{k}×{v}" for k, v in fail_reasons.items())

    # Seed from train configs / checkpoints
    v2_cfg = json.loads(
        (ROOT / "checkpoints" / "brats_v2_loss_fix" / "train_config.json").read_text()
    )
    seed = int(v2_cfg.get("seed", 42))
    pixdim = v2_cfg.get("pixdim", [1.0, 1.0, 1.0])

    rows = [
        {"item": "BraTS train cases", "value": train_n},
        {"item": "BraTS val cases (train-time)", "value": val_n},
        {"item": "BraTS held-out cases", "value": held_n},
        {"item": "Split seed", "value": seed},
        {
            "item": "Real patients processed (success)",
            "value": n_success,
        },
        {
            "item": "Real patients failed",
            "value": f"{n_failed} ({fail_note})",
        },
        {
            "item": "T1c native in DICOM (success cohort)",
            "value": n_t1c_native,
        },
        {
            "item": "T1→T1c stand-in (success cohort)",
            "value": n_t1c_standin,
        },
        {
            "item": "Target spacing",
            "value": f"{pixdim[0]}×{pixdim[1]}×{pixdim[2]} mm isotropic",
        },
        {
            "item": "Bias correction",
            "value": "N4 (SimpleITK)",
        },
        {
            "item": "Skull stripping",
            "value": "HD-BET if available, else SimpleITK Otsu fallback",
        },
        {
            "item": "Registration",
            "value": "Rigid to T1 (ITK), then 1 mm isotropic resample",
        },
        {
            "item": "Source log",
            "value": "data/processed/batch_summary_patients.csv",
        },
    ]
    df = pd.DataFrame(rows)
    out = TAB / "table1_dataset_preprocessing.csv"
    df.to_csv(out, index=False)
    (TAB / "table1_dataset_preprocessing.md").write_text(
        _df_to_md(df) + "\n",
        encoding="utf-8",
    )
    return out


def table2_segmentation() -> Path:
    """v1/v2/v3 held-out LCC Dice/HD95 + paired ΔDice (v3−v1, v3−v2)."""
    seg = _read_csv(ROOT / "results" / "segmentation_table_v3.csv")
    paired_v1 = _read_csv(ROOT / "results" / "v3_vs_v1_paired_dice.csv")
    paired_v2 = _read_csv(ROOT / "results" / "v3_vs_v2_paired_dice.csv")

    pretty_rows = []
    for _, r in seg.iterrows():
        pretty_rows.append(
            {
                "region": r["region"],
                "metric": r["metric"],
                "n": int(r["v3_lcc_n"]),
                "v1_mean_pm_sd": r["v1_lcc_mean_pm_sd"],
                "v1_median": f"{float(r['v1_lcc_median']):.4f}",
                "v1_ci95": r["v1_lcc_ci95"],
                "v2_mean_pm_sd": r["v2_lcc_mean_pm_sd"],
                "v2_median": f"{float(r['v2_lcc_median']):.4f}",
                "v2_ci95": r["v2_lcc_ci95"],
                "v3_mean_pm_sd": r["v3_lcc_mean_pm_sd"],
                "v3_median": f"{float(r['v3_lcc_median']):.4f}",
                "v3_ci95": r["v3_lcc_ci95"],
            }
        )
    pretty = pd.DataFrame(pretty_rows)
    out_main = TAB / "table2_segmentation_v1_v2_v3.csv"
    pretty.to_csv(out_main, index=False)
    # Keep legacy filename as a pointer to the three-way table for older links.
    pretty.to_csv(TAB / "table2_segmentation_v1_vs_v2.csv", index=False)
    shutil.copy2(
        ROOT / "results" / "segmentation_table_v3.csv", TAB / "table2_segmentation_raw.csv"
    )

    paired_v1.to_csv(TAB / "table2_paired_delta_dice_v3_vs_v1.csv", index=False)
    paired_v2.to_csv(TAB / "table2_paired_delta_dice_v3_vs_v2.csv", index=False)
    paired_all = pd.concat([paired_v2, paired_v1], ignore_index=True)
    paired_all.to_csv(TAB / "table2_paired_delta_dice.csv", index=False)

    md = [
        "## Segmentation (held-out, LCC) — v1 / v2 / v3",
        "",
        "Primary model: **v3** (`brats_scale_full`). Ablation: v1 → v2 → v3.",
        "",
        _df_to_md(pretty),
        "",
        "## Paired ΔDice (v3 − v2)",
        "",
        _df_to_md(paired_v2),
        "",
        "## Paired ΔDice (v3 − v1)",
        "",
        _df_to_md(paired_v1),
        "",
    ]
    (TAB / "table2_segmentation_v1_v2_v3.md").write_text("\n".join(md), encoding="utf-8")
    (TAB / "table2_segmentation_v1_vs_v2.md").write_text("\n".join(md), encoding="utf-8")
    return out_main


def table3_ellipsoid_gt() -> Path:
    """GT-only ellipsoid table (unchanged) + v3 pipeline-level sub-table."""
    src = ROOT / "results" / "ellipsoid" / "gt_only" / "region_formula_comparison.csv"
    df = _read_csv(src)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "region": r["region_short"],
                "formula": "ABC/2" if r["formula"] == "abc_over_2" else "π/6×ABC",
                "n": int(r["n"]),
                "bias_ml": f"{r['mean_bias_ml']:.2f}",
                "loa_ml": f"[{r['loa_lower_ml']:.1f}, {r['loa_upper_ml']:.1f}]",
                "bias_ci95": f"[{r['bias_ci95_low']:.1f}, {r['bias_ci95_high']:.1f}]",
                "median_ratio": f"{r['median_ratio']:.2f}",
                "iqr_ratio": f"[{r['iqr_ratio_low']:.2f}, {r['iqr_ratio_high']:.2f}]",
                "spearman": f"{r['spearman_r']:.2f}",
                "icc_2_1": f"{r['icc_2_1']:.2f}",
                "icc_ci95": f"[{r['icc_2_1_ci95_low']:.2f}, {r['icc_2_1_ci95_high']:.2f}]",
            }
        )
    pretty = pd.DataFrame(rows)
    out = TAB / "table3_ellipsoid_gt_radiologist.csv"
    pretty.to_csv(out, index=False)
    shutil.copy2(src, TAB / "table3_ellipsoid_gt_radiologist_raw.csv")
    shutil.copy2(
        ROOT / "results" / "ellipsoid" / "gt_only" / "agreement_summary.csv",
        TAB / "table3_ellipsoid_agreement_full.csv",
    )

    pipe_src = ROOT / "results" / "ellipsoid" / "brats_scale_full" / "pipeline_level.csv"
    pipe = _read_csv(pipe_src)
    pipe_rows = []
    label = {
        "pred_voxel_vs_gt_voxel": "pred voxel vs GT voxel",
        "pred_ellipsoid_radiologist_abc_over_2_vs_gt_voxel": (
            "pred radiologist ABC/2 vs GT voxel"
        ),
    }
    for _, r in pipe.iterrows():
        pipe_rows.append(
            {
                "comparison": label.get(r["comparison"], r["comparison"]),
                "n": int(r["n"]),
                "bias_ml": f"{r['mean_bias_ml']:.2f}",
                "loa_ml": f"[{r['loa_lower_ml']:.1f}, {r['loa_upper_ml']:.1f}]",
                "bias_ci95": f"[{r['bias_ci95_low']:.1f}, {r['bias_ci95_high']:.1f}]",
                "median_ratio": f"{r['median_ratio']:.2f}",
                "iqr_ratio": f"[{r['iqr_ratio_low']:.2f}, {r['iqr_ratio_high']:.2f}]",
                "spearman": f"{r['spearman_r']:.2f}",
                "icc_2_1": f"{r['icc_2_1']:.2f}",
                "icc_ci95": f"[{r['icc_2_1_ci95_low']:.2f}, {r['icc_2_1_ci95_high']:.2f}]",
            }
        )
    pipe_pretty = pd.DataFrame(pipe_rows)
    pipe_pretty.to_csv(TAB / "table3_pipeline_v3.csv", index=False)
    shutil.copy2(pipe_src, TAB / "table3_pipeline_v3_raw.csv")

    md = [
        "## Table 3a — GT-only ellipsoid (radiologist diameters; unchanged by checkpoint)",
        "",
        _df_to_md(pretty),
        "",
        "## Table 3b — Pipeline-level agreement (v3-lcc, held-out WT)",
        "",
        _df_to_md(pipe_pretty),
        "",
    ]
    (TAB / "table3_ellipsoid_gt_radiologist.md").write_text("\n".join(md), encoding="utf-8")
    return out


def table4_lcc_failures() -> Path:
    """LCC catastrophic failure rate (Dice→0) for v1, v2, and v3 held-out."""
    v1 = _read_csv(ROOT / "results" / "lcc_vs_raw_per_case.csv")
    v2 = _read_csv(ROOT / "checkpoints" / "brats_eval_heldout_v2" / "lcc" / "metrics.csv")
    v3 = _read_csv(V3_LCC)
    v1_zero = v1[v1["dice_lcc"] == 0.0]
    v2_zero = v2[v2["dice_whole_tumor"] == 0.0]
    v3_zero = v3[v3["dice_whole_tumor"] == 0.0]

    summary = pd.DataFrame(
        [
            {
                "model": "v1 (brats_pretrain)",
                "n_heldout": len(v1),
                "n_lcc_dice_zero": int(len(v1_zero)),
                "rate": f"{len(v1_zero)}/{len(v1)}",
                "cases": "; ".join(v1_zero["subject_id"].tolist()),
                "mechanism": (
                    "Largest-CC heuristic keeps a larger false-positive blob; "
                    "true-tumor component is discarded (raw Dice already low)."
                ),
            },
            {
                "model": "v2 (brats_v2_loss_fix)",
                "n_heldout": len(v2),
                "n_lcc_dice_zero": int(len(v2_zero)),
                "rate": f"{len(v2_zero)}/{len(v2)}",
                "cases": "; ".join(v2_zero["subject_id"].tolist()),
                "mechanism": (
                    "Same largest-CC failure mode (confirmed on 288/271: raw∩GT>0, "
                    "LCC∩GT=0)."
                ),
            },
            {
                "model": "v3 (brats_scale_full)",
                "n_heldout": len(v3),
                "n_lcc_dice_zero": int(len(v3_zero)),
                "rate": f"{len(v3_zero)}/{len(v3)}",
                "cases": "; ".join(v3_zero["subject_id"].tolist()),
                "mechanism": (
                    "Same largest-CC failure mode (confirmed on 259: raw∩GT>0, "
                    "LCC∩GT=0)."
                ),
            },
        ]
    )
    out = TAB / "table4_lcc_failure_rate.csv"
    summary.to_csv(out, index=False)

    detail_v1 = v1_zero[
        [
            "subject_id",
            "dice_raw",
            "dice_lcc",
            "delta_dice",
            "n_components_raw",
            "gt_n_components",
        ]
    ].copy()
    detail_v1["model"] = "v1"
    detail_v2 = v2_zero[["subject_id", "dice_whole_tumor", "hd95_whole_tumor"]].copy()
    detail_v2 = detail_v2.rename(columns={"dice_whole_tumor": "dice_lcc"})
    detail_v2["model"] = "v2"
    diag_v2 = ROOT / "results" / "v2_zero_dice_wt_overlays" / "zero_dice_diagnosis.csv"
    if diag_v2.is_file():
        d = pd.read_csv(diag_v2)
        detail_v2 = detail_v2.merge(
            d[
                [
                    "subject_id",
                    "raw_n_cc",
                    "raw_voxels",
                    "lcc_voxels",
                    "inter_raw",
                    "inter_lcc",
                    "dice_raw",
                ]
            ],
            on="subject_id",
            how="left",
        )
    detail_v3 = v3_zero[["subject_id", "dice_whole_tumor", "hd95_whole_tumor"]].copy()
    detail_v3 = detail_v3.rename(columns={"dice_whole_tumor": "dice_lcc"})
    detail_v3["model"] = "v3"
    diag_v3 = ROOT / "results" / "v3_zero_dice_wt_overlays" / "zero_dice_diagnosis.csv"
    if diag_v3.is_file():
        d3 = pd.read_csv(diag_v3)
        keep = [
            c
            for c in (
                "subject_id",
                "raw_n_cc",
                "raw_voxels",
                "lcc_voxels",
                "inter_raw",
                "inter_lcc",
                "dice_raw",
                "failure_mode",
            )
            if c in d3.columns
        ]
        detail_v3 = detail_v3.merge(d3[keep], on="subject_id", how="left")
    detail = pd.concat([detail_v1, detail_v2, detail_v3], ignore_index=True, sort=False)
    detail.to_csv(TAB / "table4_lcc_failure_cases.csv", index=False)
    (TAB / "table4_lcc_failure_rate.md").write_text(
        _df_to_md(summary) + "\n", encoding="utf-8"
    )
    return out


def table5_seg_vs_formula_error() -> Path:
    """Decomposition: segmentation volume error vs GT ellipsoid-formula error (v3)."""
    from scipy import stats as sp_stats

    cases = _read_csv(
        ROOT / "results" / "ellipsoid" / "brats_scale_full" / "lcc" / "case_volumes.csv"
    )
    ecol = "ellipsoid_radiologist_abc_over_2_ml"
    gt = cases[cases["source"] == "gt"][
        ["subject_id", "voxel_volume_ml", ecol]
    ].rename(
        columns={"voxel_volume_ml": "gt_voxel_ml", ecol: "gt_ell_ml"}
    )
    pr = cases[cases["source"] == "pred"][
        ["subject_id", "voxel_volume_ml", ecol]
    ].rename(
        columns={"voxel_volume_ml": "pred_voxel_ml", ecol: "pred_ell_ml"}
    )
    m = gt.merge(pr, on="subject_id", how="inner")
    m["abs_seg"] = (m["pred_voxel_ml"] - m["gt_voxel_ml"]).abs()
    m["abs_gt_formula"] = (m["gt_ell_ml"] - m["gt_voxel_ml"]).abs()
    m["abs_pipeline_ell"] = (m["pred_ell_ml"] - m["gt_voxel_ml"]).abs()
    r_form, p_form = sp_stats.spearmanr(m["abs_gt_formula"], m["abs_pipeline_ell"])
    r_seg, p_seg = sp_stats.spearmanr(m["abs_seg"], m["abs_pipeline_ell"])

    summary = pd.DataFrame(
        [
            {
                "quantity": "median |pred voxel − GT voxel| (seg error)",
                "value_ml": f"{m['abs_seg'].median():.2f}",
                "mean_ml": f"{m['abs_seg'].mean():.2f}",
                "n": int(len(m)),
            },
            {
                "quantity": "median |GT ABC/2 − GT voxel| (formula error on GT)",
                "value_ml": f"{m['abs_gt_formula'].median():.2f}",
                "mean_ml": f"{m['abs_gt_formula'].mean():.2f}",
                "n": int(len(m)),
            },
            {
                "quantity": "median |pred ABC/2 − GT voxel| (pipeline ellipsoid error)",
                "value_ml": f"{m['abs_pipeline_ell'].median():.2f}",
                "mean_ml": f"{m['abs_pipeline_ell'].mean():.2f}",
                "n": int(len(m)),
            },
            {
                "quantity": (
                    "Spearman(|GT-formula error|, |pipeline ell error|)"
                ),
                "value_ml": f"{r_form:.3f} (p={p_form:.2e})",
                "mean_ml": "",
                "n": int(len(m)),
            },
            {
                "quantity": "Spearman(|seg error|, |pipeline ell error|)",
                "value_ml": f"{r_seg:.3f} (p={p_seg:.2e})",
                "mean_ml": "",
                "n": int(len(m)),
            },
        ]
    )
    out = TAB / "table5_seg_vs_formula_error.csv"
    summary.to_csv(out, index=False)

    ex = m.loc[m["subject_id"] == "BraTS20_Training_257"].iloc[0]
    example = pd.DataFrame(
        [
            {
                "subject_id": "BraTS20_Training_257",
                "role": "illustrative (near-perfect seg, residual formula gap)",
                "gt_voxel_ml": f"{ex['gt_voxel_ml']:.2f}",
                "pred_voxel_ml": f"{ex['pred_voxel_ml']:.2f}",
                "gt_ellipsoid_abc2_ml": f"{ex['gt_ell_ml']:.2f}",
                "pred_ellipsoid_abc2_ml": f"{ex['pred_ell_ml']:.2f}",
                "abs_seg_ml": f"{ex['abs_seg']:.2f}",
                "abs_gt_formula_ml": f"{ex['abs_gt_formula']:.2f}",
            }
        ]
    )
    example.to_csv(TAB / "table5_case257_example.csv", index=False)

    md = [
        "## Table 5 — Segmentation vs ellipsoid-formula error (v3-lcc, WT, n=60)",
        "",
        _df_to_md(summary),
        "",
        "### Illustrative case BraTS20_Training_257",
        "",
        _df_to_md(example),
        "",
    ]
    (TAB / "table5_seg_vs_formula_error.md").write_text("\n".join(md), encoding="utf-8")
    return out


def _compose_h(images: list[Path], out_path: Path, titles: list[str] | None = None) -> None:
    imgs = [Image.open(p).convert("RGB") for p in images]
    h = max(im.height for im in imgs)
    imgs = [
        im.resize((int(im.width * h / im.height), h), Image.Resampling.LANCZOS)
        for im in imgs
    ]
    w = sum(im.width for im in imgs)
    canvas = Image.new("RGB", (w, h + (40 if titles else 0)), (255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 40 if titles else 0))
        x += im.width
    if titles:
        # Titles via matplotlib overlay for crisp text.
        fig, ax = plt.subplots(
            figsize=(canvas.width / 100, canvas.height / 100), dpi=100
        )
        ax.imshow(np.asarray(canvas))
        ax.axis("off")
        # Simple top labels
        xs = np.cumsum([0] + [im.width for im in imgs[:-1]])
        for xi, title, im in zip(xs, titles, imgs):
            ax.text(
                xi + im.width / 2,
                18,
                title,
                ha="center",
                va="top",
                fontsize=11,
                color="black",
                transform=ax.transData,
            )
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    else:
        canvas.save(out_path, dpi=(300, 300))


def figure_ba_multipanel() -> Path | None:
    """Compose ET/TC/WT radiologist ABC/2 log-ratio BA panels."""
    src = ROOT / "results" / "ellipsoid" / "gt_only" / "figures"
    panels = [
        src / "ba_gt_et_radiologist_abc_over_2_pct.png",
        src / "ba_gt_tc_radiologist_abc_over_2_pct.png",
        src / "ba_gt_wt_radiologist_abc_over_2_pct.png",
    ]
    if not all(p.is_file() for p in panels):
        return None
    out = FIG / "fig_ba_radiologist_abc_over_2_logratio_ET_TC_WT.png"
    imgs = [Image.open(p).convert("RGB") for p in panels]
    # Resize to common height
    h = min(im.height for im in imgs)
    imgs = [
        im.resize((int(im.width * h / im.height), h), Image.Resampling.LANCZOS)
        for im in imgs
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    titles = ["ET", "TC", "WT"]
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(np.asarray(im))
        ax.set_title(f"{title} · radiologist · ABC/2", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        "GT-only Bland–Altman (log-ratio %) — radiologist diameters, ABC/2",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Also copy individuals
    for p, tag in zip(panels, titles):
        shutil.copy2(p, FIG / f"fig_ba_{tag.lower()}_radiologist_abc_over_2_pct.png")
    return out


def figure_lcc_failure() -> Path | None:
    """Limitation figure: prefer v3 case 259; then v2 271; then v1 147."""
    candidates = [
        (
            ROOT
            / "results"
            / "v3_zero_dice_wt_overlays"
            / "gt_vs_lcc_BraTS20_Training_259.png",
            "v3_259",
        ),
        (
            ROOT
            / "results"
            / "v2_zero_dice_wt_overlays"
            / "gt_vs_lcc_BraTS20_Training_271.png",
            "v2_271",
        ),
        (
            ROOT
            / "results"
            / "ellipsoid"
            / "brats_pretrain"
            / "worst_delta_overlays"
            / "worst_delta_dice_BraTS20_Training_147.png",
            "v1_147",
        ),
    ]
    for src, tag in candidates:
        if src.is_file():
            out = FIG / f"fig_lcc_failure_{tag}.png"
            shutil.copy2(src, out)
            shutil.copy2(src, FIG / "fig_lcc_failure_limitation.png")
            return out
    return None


def _v3_extremes() -> pd.DataFrame:
    v3 = _read_csv(V3_LCC).sort_values("dice_whole_tumor")
    worst = v3.iloc[0]
    median = v3.iloc[len(v3) // 2]
    best = v3.iloc[-1]
    return pd.DataFrame(
        [
            {
                "rank": "worst",
                "subject_id": worst["subject_id"],
                "dice_wt": float(worst["dice_whole_tumor"]),
            },
            {
                "rank": "median",
                "subject_id": median["subject_id"],
                "dice_wt": float(median["dice_whole_tumor"]),
            },
            {
                "rank": "best",
                "subject_id": best["subject_id"],
                "dice_wt": float(best["dice_whole_tumor"]),
            },
        ]
    )


def _infer_wt_masks(subject_id: str, device: str | None = None):
    """Return (t1_or_flair_3d, gt_wt, pred_raw_wt, pred_lcc_wt) for one BraTS case."""
    import torch
    from monai.data import Dataset
    from monai.inferers import sliding_window_inference

    sys.path.insert(0, str(ROOT / "src"))
    from config import load_config
    from segmentation.dataset import get_val_transforms, subject_to_datadict
    from segmentation.evaluate import load_model_from_checkpoint
    from segmentation.postprocess import apply_channel_postprocess, keep_largest_cc

    brats = load_config().paths.brats_nifti
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
    model = load_model_from_checkpoint(V3_CKPT, device_t)
    model.float().eval()
    batch = Dataset(
        data=[subject_to_datadict(brats / subject_id)],
        transform=get_val_transforms(),
    )[0]
    image = batch["image"]  # (C,H,W,D)
    # Prefer FLAIR (channel 3) else T1c (0) for display
    ch = 3 if image.shape[0] > 3 else 0
    image_3d = image[ch].detach().cpu().numpy()
    gt_wt = batch["label"].squeeze(0).detach().cpu().numpy() != 0
    with torch.no_grad():
        logits = sliding_window_inference(
            image.unsqueeze(0).to(device_t, dtype=torch.float32),
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
        )
    pred = (torch.sigmoid(logits) > 0.5).squeeze(0).detach().cpu().numpy()
    pred_raw = pred[2].astype(bool)
    pred_lcc = keep_largest_cc(pred_raw)
    return image_3d, gt_wt, pred_raw, pred_lcc


def ensure_v3_overlays() -> list[Path]:
    """Generate best/median/worst + case-257 paper overlays if missing."""
    sys.path.insert(0, str(ROOT / "src"))
    from segmentation.postprocess import fragmentation_stats
    from validation.lcc_followup import _overlay_gt_lcc_removed

    V3_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    extremes = _v3_extremes()
    extremes = extremes.copy()
    extremes["model"] = "v3-lcc"
    extremes["overlay_available"] = True
    extremes.to_csv(TAB / "fig_a_wt_extremes_case_ids.csv", index=False)

    targets = [
        ("worst", extremes.loc[extremes["rank"] == "worst", "subject_id"].iloc[0]),
        ("median", extremes.loc[extremes["rank"] == "median", "subject_id"].iloc[0]),
        ("best", extremes.loc[extremes["rank"] == "best", "subject_id"].iloc[0]),
        ("case257", "BraTS20_Training_257"),
    ]
    dice_map = _read_csv(V3_LCC).set_index("subject_id")["dice_whole_tumor"].to_dict()

    cache: dict[str, tuple] = {}
    for rank, sid in targets:
        if sid not in cache:
            print(f"  overlay inference {sid} ({rank}) ...")
            cache[sid] = _infer_wt_masks(sid)

    for rank, sid in targets:
        image_3d, gt_wt, pred_raw, pred_lcc = cache[sid]
        dice = float(dice_map.get(sid, float("nan")))
        out = V3_OVERLAY_DIR / f"gt_pred_{rank}_{sid}.png"
        removed = pred_raw & ~pred_lcc
        gt_stats = fragmentation_stats(gt_wt, (1.0, 1.0, 1.0))
        _overlay_gt_lcc_removed(
            image_3d,
            gt_wt,
            pred_lcc,
            removed,
            out,
            subject_id=sid,
            delta_dice=dice,
            gt_n_cc=int(gt_stats["n_components"]),
            gt_frac_out=float(gt_stats.get("frac_outside_largest") or 0.0),
        )
        fig_out = FIG / f"fig_seg_wt_{rank}_{sid}.png"
        shutil.copy2(out, fig_out)
        produced.append(fig_out)
        if rank == "worst":
            shutil.copy2(out, FIG / f"fig_seg_wt_worst_{sid}.png")

    # Case 257 decomposition panel
    sid = "BraTS20_Training_257"
    image_3d, gt_wt, pred_raw, pred_lcc = cache[sid]
    cases = _read_csv(
        ROOT / "results" / "ellipsoid" / "brats_scale_full" / "lcc" / "case_volumes.csv"
    )
    ecol = "ellipsoid_radiologist_abc_over_2_ml"
    gt_row = cases[(cases.subject_id == sid) & (cases.source == "gt")].iloc[0]
    pr_row = cases[(cases.subject_id == sid) & (cases.source == "pred")].iloc[0]
    vols = [
        float(gt_row["voxel_volume_ml"]),
        float(pr_row["voxel_volume_ml"]),
        float(gt_row[ecol]),
        float(pr_row[ecol]),
    ]
    labels = ["GT voxel", "pred voxel", "GT ABC/2", "pred ABC/2"]

    z = int(np.round(np.mean(np.where(gt_wt)[2]))) if np.any(gt_wt) else image_3d.shape[2] // 2
    base = image_3d[:, :, z].astype(float)
    p2, p98 = np.percentile(base, (2, 98))
    base = np.clip((base - p2) / max(p98 - p2, 1e-6), 0, 1)
    rgb = np.stack([base, base, base], axis=-1)
    g = gt_wt[:, :, z]
    p = pred_lcc[:, :, z]
    rgb[g & ~p] = rgb[g & ~p] * 0.35 + np.array([0.1, 0.8, 0.2]) * 0.65
    rgb[p & ~g] = rgb[p & ~g] * 0.35 + np.array([0.95, 0.2, 0.2]) * 0.65
    rgb[g & p] = rgb[g & p] * 0.25 + np.array([0.95, 0.9, 0.1]) * 0.75

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    axes[0].imshow(np.rot90(rgb), origin="lower")
    axes[0].set_title(
        f"{sid} axial\ngreen=GT  red=pred  yellow=overlap\n"
        f"|seg|={abs(vols[1] - vols[0]):.2f} mL"
    )
    axes[0].axis("off")
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    axes[1].bar(labels, vols, color=colors)
    axes[1].set_ylabel("Volume (mL)")
    axes[1].set_title(
        "Ellipsoid gap despite near-perfect voxel match\n"
        f"GT voxel={vols[0]:.1f}  pred voxel={vols[1]:.1f}  "
        f"GT ABC/2={vols[2]:.1f}  pred ABC/2={vols[3]:.1f}"
    )
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Table 5 illustration: formula/diameter error dominates segmentation error",
        fontsize=11,
    )
    fig.tight_layout()
    decomp = V3_OVERLAY_DIR / f"decomp_{sid}.png"
    fig.savefig(decomp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    dest = FIG / "fig_case257_seg_vs_ellipsoid.png"
    shutil.copy2(decomp, dest)
    produced.append(dest)
    produced.append(TAB / "fig_a_wt_extremes_case_ids.csv")
    return produced


def figure_sphericity_from_csv() -> Path:
    """Sphericity vs voxel volume by region from gt_only/case_volumes.csv."""
    cases = _read_csv(
        ROOT / "results" / "ellipsoid" / "gt_only" / "case_volumes.csv"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    colors = {"ET": "#C44E52", "TC": "#4C72B0", "WT": "#55A868"}
    for short, g in cases.groupby("region_short"):
        ax.scatter(
            g["voxel_volume_ml"],
            g["sphericity"],
            s=28,
            alpha=0.75,
            label=f"{short} (n={len(g)})",
            c=colors.get(short, "gray"),
            edgecolors="none",
        )
    ax.set_xlabel("GT voxel volume (mL)")
    ax.set_ylabel("Sphericity")
    ax.set_title("GT region sphericity vs volume (held-out)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig_sphericity_vs_volume_gt.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def figure_seg_extremes_placeholder() -> Path:
    """Ensure v3 best/median/worst (+ case 257) overlays exist."""
    ensure_v3_overlays()
    return TAB / "fig_a_wt_extremes_case_ids.csv"


def write_needs_rerun() -> Path:
    text = """# Needs re-run (not generated by `make paper`)

`make paper` aggregates verified CSVs and generates the few v3 overlays listed
below. Full held-out eval / ellipsoid batch / training are **not** re-run.

## Generated by this `make paper` (v3)

- Best / median / worst WT overlays (`fig_seg_wt_{worst,median,best}_*.png`)
- Case 257 seg-vs-ellipsoid decomposition (`fig_case257_seg_vs_ellipsoid.png`)
- Tables 2–5 from existing `results/` CSVs

## Still optional / not generated

- A **3D mesh render** with sphericity annotation (would require marching-cubes +
  Plotly/matplotlib 3D export for a chosen case). CSV scatter
  `fig_sphericity_vs_volume_gt.png` is present.

## Not flagged (already on disk)

- GT-only Bland–Altman multipanel
- v3 held-out metrics + ellipsoid `pipeline_level.csv`
"""
    out = OUT / "NEEDS_RERUN.md"
    out.write_text(text, encoding="utf-8")
    return out


def write_limitations() -> Path:
    text = """# Limitations

1. **Early models used a small BraTS subset.** v1/v2 were trained on a 40-case
   train / 10-case val split (seed 42) under Mac CPU/MPS and disk constraints.
   The **primary reported model (v3)** scales to 289 train / 20 val cases
   (excluding the 60 held-out) over 20 epochs.

2. **No manual mask correction.** Clinical fine-tuning / corrected masks were
   not used for the held-out comparison; results reflect BraTS-pretrained (or
   loss-fixed / scaled) weights only.

3. **T1-as-T1c stand-in.** Among successfully processed real-patient studies,
   none had a native contrast T1c series in the DICOM modality probe; T1 was
   copied to the T1c slot for model input (`prepare_correction_set.fill_missing_t1c_from_t1`).
   Three studies failed entirely (missing T1; FLAIR+T2 only).

4. **No clinical ground-truth volumes.** Ellipsoid formulas (ABC/2, π/6×ABC) were
   validated against BraTS **voxel** volumes from GT masks, not against
   radiologist-measured clinical volumes.

5. **WT/ET–TC trade-off was specific to the 40-case v2 run (ablation finding).**
   Best checkpoints are chosen by **mean** val Dice over ET/TC/WT. On the small
   v2 run, improving ET/TC came with a held-out WT regression vs v1 (paired
   ΔDice ≈ −0.18). Scaling to **v3 (289 train cases)** removed that trade-off:
   v3 beats both v1 and v2 on ET, TC, and WT simultaneously. This is documented
   as an ablation result, **not** a limitation of the final reported model.

6. **LCC largest-component failure.** Post-process `lcc` keeps the largest
   26-connected component. Failure rate on held-out: v1 2/60, v2 2/60, v3 1/60
   (case 259). Mechanism unchanged: a large false-positive blob is retained and
   the true tumor component is discarded (raw∩GT>0, LCC∩GT=0).

7. **v2 training instability (historical).** After switching to DiceCELoss over
   all channels, epoch-to-epoch val Dice was noisy under ReduceLROnPlateau
   inherited from v1. v3 replaced that schedule with linear warmup + cosine
   annealing and gradient clipping (1.0).

8. **Ellipsoid-formula error dominates once segmentation is strong.** For v3-lcc
   WT, predicted-vs-GT **voxel** ICC is ~0.93 (median |seg error| ~5.9 mL), but
   predicted radiologist ABC/2 vs GT voxel ICC remains ~0.55 — nearly identical
   to GT-only ABC/2 vs GT voxel (~0.58). Pipeline ellipsoid error correlates with
   GT-formula error (Spearman ~0.47) far more than with segmentation error
   (~0.15). Further improving the segmentation model would therefore **not**
   proportionally improve clinical volume estimates that use the ellipsoid
   method specifically (see Table 5; case 257).
"""
    out = OUT / "LIMITATIONS.md"
    out.write_text(text, encoding="utf-8")
    return out


def write_environment() -> Path:
    """pip freeze, Python, device, seed audit."""
    lines: list[str] = []
    lines.append(f"python: {sys.version.replace(chr(10), ' ')}")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"executable: {sys.executable}")

    # Device
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        lines.append(f"torch: {torch.__version__}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"torch: unavailable ({exc})")
    lines.append(f"device: {device}")

    # Seeds — confirm 42 across production paths; flag exceptions
    lines.append("")
    lines.append("=== seeds ===")
    lines.append("production_default: 42")
    lines.append("confirmed:")
    lines.append("  - splits / TrainConfig.seed = 42 (train.py, splits.py)")
    lines.append("  - checkpoints/brats_v2_loss_fix/train_config.json seed = 42")
    lines.append("  - checkpoints/brats_scale_full/train_config.json seed = 42")
    lines.append("  - brats_pretrain best_model.pt train_config.seed = 42")
    lines.append("  - bootstrap CIs (report_metrics, compare_v1_v2/v3, ellipsoid_batch) seed = 42")
    lines.append("  - preprocessing.registration ITK sampling seed = 42")
    lines.append("  - EvaluateConfig / FinetuneConfig / DomainGapReportConfig defaults = 42")
    lines.append("exceptions_or_non_production:")
    lines.append("  - tests/test_finetune.py uses seed=0 for a unit-test split only")
    lines.append("  - tests/test_measurements.py and tests/test_postprocess.py use np.random.default_rng(0)")
    lines.append("  - no other production script was found using a non-42 seed")

    # Training configs (v1 / v2 / v3)
    lines.append("")
    lines.append("=== training configs ===")
    lines.append("primary_model: v3 (checkpoints/brats_scale_full/best_model.pt)")
    lines.append("ablation: v1 (brats_pretrain) → v2 (brats_v2_loss_fix) → v3 (brats_scale_full)")
    lines.append("")
    lines.append("v1 (brats_pretrain):")
    lines.append("  train/val: 40 / 10 (max_cases subset, seed 42)")
    lines.append("  epochs: 100 (best by mean val Dice; see metrics.csv)")
    lines.append("  loss: DiceCELoss (pre-fix history had ET channel skip; see paper ablation)")
    lines.append("  schedule: ReduceLROnPlateau (legacy)")
    lines.append("")
    lines.append("v2 (brats_v2_loss_fix):")
    lines.append("  train/val: 40 / 10 (same subset scale as v1)")
    lines.append("  epochs: 100 budget; DiceCELoss(include_background=True) over ET/TC/WT")
    lines.append("  schedule: ReduceLROnPlateau (inherited)")
    v2_cfg = ROOT / "checkpoints" / "brats_v2_loss_fix" / "train_config.json"
    if v2_cfg.is_file():
        cfg = json.loads(v2_cfg.read_text())
        lines.append(
            f"  recorded: seed={cfg.get('seed')} lr={cfg.get('learning_rate')} "
            f"batch={cfg.get('batch_size')} model={cfg.get('model_name')}"
        )
    lines.append("")
    lines.append("v3 (brats_scale_full) — PRIMARY:")
    lines.append("  train/val: 289 / 20 (splits/scaleup/cfg_c_all_*; held-out 60 excluded)")
    lines.append("  epochs: 20")
    lines.append("  model: SegResNet; batch_size=1; lr=1e-4; amp=False")
    lines.append("  schedule: cosine_warmup (warmup_frac=0.05) + CosineAnnealing; eta_min=1e-6")
    lines.append("  grad_clip_max_norm: 1.0")
    lines.append("  best_epoch (val): 15 (val_dice_mean≈0.711)")
    v3_cfg = ROOT / "checkpoints" / "brats_scale_full" / "train_config.json"
    if v3_cfg.is_file():
        cfg = json.loads(v3_cfg.read_text())
        lines.append(
            f"  recorded: seed={cfg.get('seed')} scheduler_type={cfg.get('scheduler_type')} "
            f"warmup_frac={cfg.get('warmup_frac')} grad_clip={cfg.get('grad_clip_max_norm')} "
            f"max_epochs={cfg.get('max_epochs')}"
        )
    n_train = ROOT / "checkpoints" / "brats_scale_full" / "train_cases.txt"
    n_val = ROOT / "checkpoints" / "brats_scale_full" / "train_run_val_cases.txt"
    if n_train.is_file() and n_val.is_file():
        lines.append(
            f"  lists: train_cases={len(n_train.read_text().split())} "
            f"val_cases={len(n_val.read_text().split())}"
        )

    # pip freeze
    lines.append("")
    lines.append("=== pip freeze ===")
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        lines.append(freeze.rstrip())
    except Exception as exc:  # noqa: BLE001
        lines.append(f"pip freeze failed: {exc}")

    out = OUT / "environment.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_manifest(produced: list[Path]) -> Path:
    # Include all files under results/paper/ so the manifest matches disk.
    all_files = sorted(
        str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()
    )
    text = (
        "# results/paper manifest\n\n"
        "Primary model: **v3** (`brats_scale_full`). Ablation: v1 → v2 → v3.\n\n"
        + "\n".join(f"- `{r}`" for r in all_files)
        + "\n"
    )
    out = OUT / "MANIFEST.md"
    out.write_text(text, encoding="utf-8")
    return out


def main() -> int:
    _ensure_dirs()
    produced: list[Path] = []

    produced.append(table1_dataset_preprocessing())
    produced.append(table2_segmentation())
    produced.append(table3_ellipsoid_gt())
    produced.append(table4_lcc_failures())
    produced.append(table5_seg_vs_formula_error())

    ba = figure_ba_multipanel()
    if ba:
        produced.append(ba)
    lcc = figure_lcc_failure()
    if lcc:
        produced.append(lcc)
    produced.append(figure_sphericity_from_csv())
    print("Generating v3 WT overlays (best/median/worst + case 257) ...")
    for p in ensure_v3_overlays():
        produced.append(p)

    produced.append(write_needs_rerun())
    produced.append(write_limitations())
    produced.append(write_environment())

    for src, name in (
        (
            ROOT / "results" / "ellipsoid" / "brats_pretrain" / "pipeline_level.csv",
            "ref_pipeline_level_v1.csv",
        ),
        (
            ROOT / "results" / "ellipsoid" / "brats_v2_loss_fix" / "pipeline_level.csv",
            "ref_pipeline_level_v2.csv",
        ),
        (
            ROOT / "results" / "ellipsoid" / "brats_scale_full" / "pipeline_level.csv",
            "ref_pipeline_level_v3.csv",
        ),
        (
            ROOT / "results" / "v1_vs_v2_paired_dice.csv",
            "ref_v1_vs_v2_paired_dice.csv",
        ),
        (
            ROOT / "results" / "v3_paired_dice.csv",
            "ref_v3_paired_dice.csv",
        ),
    ):
        if src.is_file():
            dst = TAB / name
            shutil.copy2(src, dst)
            produced.append(dst)

    produced.append(write_manifest(produced))

    print(f"Wrote paper assets → {OUT}")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
