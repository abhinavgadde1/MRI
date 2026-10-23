#!/usr/bin/env python3
"""Voxel-level classification metrics on the BraTS held-out set (v1/v2/v3, LCC).

Held-out eval folders only store Dice/HD95 CSVs — predicted masks were not
saved. This script re-runs LCC inference, writes per-case confusion counts,
and assembles ``results/paper/tables/table6_classification_metrics.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from device import get_device  # noqa: E402
from segmentation.classification_metrics import (  # noqa: E402
    REGION_SHORT,
    confusion_counts,
    format_float,
    nanmean,
    scores_from_counts,
)
from segmentation.dataset import get_val_transforms  # noqa: E402
from segmentation.evaluate import (  # noqa: E402
    _records_from_subject_ids,
    load_model_from_checkpoint,
)
from segmentation.model import BRATS_REGIONS, brats_label_to_regions  # noqa: E402
from segmentation.postprocess import apply_channel_postprocess  # noqa: E402
from segmentation.splits import read_id_list  # noqa: E402
from segmentation.train import post_transforms  # noqa: E402

logger = logging.getLogger("table6")

MODELS: dict[str, tuple[Path, Path]] = {
    "v1": (
        ROOT / "checkpoints" / "brats_pretrain" / "best_model.pt",
        ROOT / "checkpoints" / "brats_eval_heldout" / "lcc" / "metrics.csv",
    ),
    "v2": (
        ROOT / "checkpoints" / "brats_v2_loss_fix" / "best_model.pt",
        ROOT / "checkpoints" / "brats_eval_heldout_v2" / "lcc" / "metrics.csv",
    ),
    "v3": (
        ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt",
        ROOT / "checkpoints" / "brats_eval_heldout_v3" / "lcc" / "metrics.csv",
    ),
}

ROI = (96, 96, 96)
PIXDIM = (1.0, 1.0, 1.0)
DICE_MATCH_ATOL = 1e-4
TAB = ROOT / "results" / "paper" / "tables"
CACHE = ROOT / "results" / "table6_per_case_counts.csv"
CACHE_DIR = ROOT / "results" / "table6_cache"


def _model_cache(model_name: str) -> Path:
    return CACHE_DIR / f"{model_name}_per_case.csv"


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(r[c]) for c in cols) + " |"
        for _, r in df.iterrows()
    ]
    return "\n".join([header, sep, *rows])


@torch.no_grad()
def infer_lcc_case(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    post_pred,
    post_label,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_bool C×…, gt_bool C×…, brain_bool …) after LCC.

    Matches ``evaluate.evaluate_subject_list`` (float32, AMP off, LCC).
    """
    inputs = batch["image"].to(device=device, dtype=torch.float32)
    labels = brats_label_to_regions(batch["label"].to(device))

    with torch.autocast(device_type=device.type, enabled=False):
        outputs = sliding_window_inference(
            inputs,
            roi_size=ROI,
            sw_batch_size=1,
            predictor=model,
        )
    pred = post_pred(decollate_batch(outputs)[0]).detach().float().cpu().numpy()
    label = post_label(decollate_batch(labels)[0]).detach().float().cpu().numpy()
    pred = apply_channel_postprocess(pred, mode="lcc", spacing_mm=PIXDIM)
    pred_b = np.asarray(pred) > 0.5
    gt_b = np.asarray(label) > 0.5
    image_np = batch["image"][0].detach().cpu().numpy()
    brain = np.any(np.abs(image_np) > 0, axis=0)
    return pred_b.astype(bool), gt_b.astype(bool), brain.astype(bool)


def evaluate_model(
    model_name: str,
    checkpoint: Path,
    subject_ids: list[str],
    brats_root: Path,
    device: torch.device,
) -> pd.DataFrame:
    cache_path = _model_cache(model_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        rows.extend(cached.to_dict("records"))
        done = set(str(s) for s in cached["subject_id"].tolist())
        logger.info("%s: resuming with %d/%d cases cached", model_name, len(done), len(subject_ids))

    remaining = [s for s in subject_ids if s not in done]
    if not remaining:
        return pd.DataFrame(rows)

    model = load_model_from_checkpoint(checkpoint, device)
    model.float().eval()
    post_pred, post_label = post_transforms()
    records = _records_from_subject_ids(brats_root, remaining)
    loader = DataLoader(
        Dataset(data=records, transform=get_val_transforms(pixdim=PIXDIM)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for batch in loader:
        sid = str(batch.get("subject_id", ["unknown"])[0])
        logger.info("%s %s on %s", model_name, sid, device)
        pred, gt, brain = infer_lcc_case(model, batch, device, post_pred, post_label)
        n_vox = int(pred[0].size)
        n_brain = int(brain.sum())
        for ch, region in enumerate(BRATS_REGIONS):
            cm = confusion_counts(pred[ch], gt[ch])
            scores = scores_from_counts(**cm)
            gt_pos = cm["tp"] + cm["fn"]
            rows.append(
                {
                    "model": model_name,
                    "subject_id": sid,
                    "region": REGION_SHORT[region],
                    "region_full": region,
                    **cm,
                    "n_voxels": n_vox,
                    "n_brain_voxels": n_brain,
                    "gt_positive": gt_pos,
                    "pred_positive": cm["tp"] + cm["fp"],
                    "prevalence_volume": gt_pos / n_vox if n_vox else float("nan"),
                    "prevalence_brain": gt_pos / n_brain if n_brain else float("nan"),
                    **scores,
                    "empty_gt": gt_pos == 0,
                }
            )
        pd.DataFrame(rows).to_csv(cache_path, index=False)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


def load_reported_dice() -> pd.DataFrame:
    frames = []
    for model, (_ckpt, csv_path) in MODELS.items():
        df = pd.read_csv(csv_path)
        long = df.melt(
            id_vars=["subject_id"],
            value_vars=[
                "dice_enhancing_tumor",
                "dice_tumor_core",
                "dice_whole_tumor",
            ],
            var_name="region_full",
            value_name="dice_reported",
        )
        long["region_full"] = long["region_full"].str.replace("dice_", "", regex=False)
        long["region"] = long["region_full"].map(REGION_SHORT)
        long["model"] = model
        frames.append(long[["model", "subject_id", "region", "dice_reported"]])
    return pd.concat(frames, ignore_index=True)


def aggregate(per_case: pd.DataFrame, reported: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = per_case.merge(reported, on=["model", "subject_id", "region"], how="left")
    summary_rows: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []

    for model in MODELS:
        for region in ("ET", "TC", "WT"):
            sub = merged[(merged["model"] == model) & (merged["region"] == region)]
            n = len(sub)
            # Pooled voxels
            tp = int(sub["tp"].sum())
            fp = int(sub["fp"].sum())
            fn = int(sub["fn"].sum())
            tn = int(sub["tn"].sum())
            pooled = scores_from_counts(tp, fp, fn, tn)
            n_vox = int(sub["n_voxels"].sum())
            n_brain = int(sub["n_brain_voxels"].sum())
            gt_pos = tp + fn
            summary_rows.append(
                {
                    "model": model,
                    "region": region,
                    "aggregation": "pooled_voxels",
                    "n_cases": n,
                    "n_cases_with_gt": int((sub["gt_positive"] > 0).sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "n_voxels": n_vox,
                    "n_brain_voxels": n_brain,
                    "gt_positive": gt_pos,
                    "prevalence_volume": gt_pos / n_vox if n_vox else float("nan"),
                    "prevalence_brain": gt_pos / n_brain if n_brain else float("nan"),
                    **{k: pooled[k] for k in ("accuracy", "precision", "recall", "specificity", "f1", "dice")},
                    "dice_reported_mean": float("nan"),
                    "f1_vs_reported_dice": "",
                }
            )

            # Per-case mean: skip empty-GT for precision/recall/F1/Dice (same as table 2).
            nonempty = sub[sub["gt_positive"] > 0]
            n_gt = len(nonempty)
            mean_scores = {
                "accuracy": nanmean(sub["accuracy"].tolist()),
                "precision": nanmean(nonempty["precision"].tolist()),
                "recall": nanmean(nonempty["recall"].tolist()),
                "specificity": nanmean(sub["specificity"].tolist()),
                "f1": nanmean(nonempty["f1"].tolist()),
                "dice": nanmean(nonempty["dice"].tolist()),
            }
            reported_mean = float(nonempty["dice_reported"].mean(skipna=True)) if n_gt else float("nan")
            f1_mean = mean_scores["f1"]
            dice_cm_mean = mean_scores["dice"]
            match_f1_dice = bool(
                f1_mean == f1_mean
                and dice_cm_mean == dice_cm_mean
                and abs(f1_mean - dice_cm_mean) <= DICE_MATCH_ATOL
            )
            match_reported = bool(
                f1_mean == f1_mean
                and reported_mean == reported_mean
                and abs(f1_mean - reported_mean) <= DICE_MATCH_ATOL
            )
            flag = "ok" if (match_f1_dice and match_reported) else "MISMATCH"
            if flag != "ok":
                flags.append(
                    {
                        "model": model,
                        "region": region,
                        "f1_mean": f1_mean,
                        "dice_from_cm_mean": dice_cm_mean,
                        "dice_reported_mean": reported_mean,
                        "f1_equals_dice_cm": match_f1_dice,
                        "f1_equals_reported": match_reported,
                    }
                )
            summary_rows.append(
                {
                    "model": model,
                    "region": region,
                    "aggregation": "per_case_mean",
                    "n_cases": n,
                    "n_cases_with_gt": n_gt,
                    "tp": float(sub["tp"].mean()),
                    "fp": float(sub["fp"].mean()),
                    "fn": float(sub["fn"].mean()),
                    "tn": float(sub["tn"].mean()),
                    "n_voxels": float(sub["n_voxels"].mean()),
                    "n_brain_voxels": float(sub["n_brain_voxels"].mean()),
                    "gt_positive": float(sub["gt_positive"].mean()),
                    "prevalence_volume": nanmean(sub["prevalence_volume"].tolist()),
                    "prevalence_brain": nanmean(sub["prevalence_brain"].tolist()),
                    **mean_scores,
                    "dice_reported_mean": reported_mean,
                    "f1_vs_reported_dice": flag,
                }
            )

    return pd.DataFrame(summary_rows), {"mismatches": flags}


def pretty_side_by_side(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide tables: region × metric, columns v1/v2/v3."""
    metric_order_pooled = [
        ("tp", "TP", 0),
        ("fp", "FP", 0),
        ("fn", "FN", 0),
        ("tn", "TN", 0),
        ("accuracy", "Accuracy", 4),
        ("precision", "Precision", 4),
        ("recall", "Recall", 4),
        ("specificity", "Specificity", 4),
        ("f1", "F1 (= Dice)", 4),
        ("prevalence_brain", "Tumor / brain voxels", 4),
        ("prevalence_volume", "Tumor / all voxels", 4),
    ]
    metric_order_case = [
        ("accuracy", "Accuracy", 4),
        ("precision", "Precision", 4),
        ("recall", "Recall", 4),
        ("specificity", "Specificity", 4),
        ("f1", "F1 (= Dice)", 4),
        ("dice_reported_mean", "Reported Dice (table 2)", 4),
        ("f1_vs_reported_dice", "F1 vs reported Dice", None),
        ("prevalence_brain", "Tumor / brain voxels", 4),
        ("prevalence_volume", "Tumor / all voxels", 4),
    ]

    def _wide(agg: str, specs: list[tuple[str, str, int | None]]) -> pd.DataFrame:
        sub = summary[summary["aggregation"] == agg]
        rows = []
        for region in ("ET", "TC", "WT"):
            block = sub[sub["region"] == region].set_index("model")
            n = int(block.loc["v3", "n_cases_with_gt"]) if "v3" in block.index else int(block["n_cases_with_gt"].iloc[0])
            n_all = int(block["n_cases"].iloc[0])
            for col, label, digits in specs:
                rec = {"region": region, "n_cases": n_all, "n_with_gt": n, "metric": label}
                for model in ("v1", "v2", "v3"):
                    val = block.loc[model, col]
                    if digits is None:
                        rec[model] = str(val)
                    elif col in {"tp", "fp", "fn", "tn"} and agg == "pooled_voxels":
                        rec[model] = int(val)
                    else:
                        rec[model] = format_float(float(val), digits)
                rows.append(rec)
        return pd.DataFrame(rows)

    return _wide("pooled_voxels", metric_order_pooled), _wide("per_case_mean", metric_order_case)


def write_outputs(summary: pd.DataFrame, flags: dict[str, Any]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    csv_path = TAB / "table6_classification_metrics.csv"
    summary.to_csv(csv_path, index=False)
    pooled_wide, case_wide = pretty_side_by_side(summary)

    mismatch_lines = []
    if flags["mismatches"]:
        mismatch_lines.append("## F1 vs Dice sanity check — **MISMATCH**")
        mismatch_lines.append("")
        mismatch_lines.append("Per-case-mean F1 did not match the previously reported Dice. Investigate before using these numbers.")
        mismatch_lines.append("")
        mismatch_lines.append(_df_to_md(pd.DataFrame(flags["mismatches"])))
        mismatch_lines.append("")
    else:
        mismatch_lines.append(
            "F1 vs Dice sanity check: **pass**. Per-case-mean F1 matches binary Dice "
            f"from the confusion matrix and matches table 2 held-out LCC Dice within {DICE_MATCH_ATOL:g}."
        )
        mismatch_lines.append("")

    md = [
        "## Table 6 — Voxel classification metrics (held-out, LCC)",
        "",
        "Primary model: **v3** (`brats_scale_full`). Ablation: v1 → v2 → v3. Post-process: **lcc**.",
        "",
        "Predicted masks were not stored from the original held-out runs "
        "(`checkpoints/brats_eval_heldout*/lcc/` contains `metrics.csv` only). "
        "Metrics below were recomputed by re-running LCC inference on the 60 held-out cases.",
        "",
        *mismatch_lines,
        "Accuracy is high even for poor segmentations because tumor voxels are a small "
        "fraction of the volume (see prevalence rows). Precision / recall / F1 are the "
        "informative metrics here.",
        "",
        "### Pooled voxels (all 60 cases concatenated)",
        "",
        _df_to_md(pooled_wide),
        "",
        "### Per-case mean (precision / recall / F1 skip empty-GT cases, same as table 2)",
        "",
        _df_to_md(case_wide),
        "",
        "F1 = 2·Precision·Recall / (Precision+Recall) = 2 TP / (2 TP + FP + FN) = binary Dice.",
        "",
    ]
    md_path = TAB / "table6_classification_metrics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (TAB / "table6_classification_metrics_pooled_wide.csv").write_text(
        pooled_wide.to_csv(index=False), encoding="utf-8"
    )
    (TAB / "table6_classification_metrics_percase_wide.csv").write_text(
        case_wide.to_csv(index=False), encoding="utf-8"
    )
    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Per-case cache: {CACHE}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    ids = read_id_list(ROOT / "splits" / "heldout.txt")
    if args.max_cases is not None:
        ids = ids[: args.max_cases]
    brats_root = Path(load_config().paths.brats_nifti)
    device = get_device(args.device)

    frames = []
    for model, (ckpt, _) in MODELS.items():
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        frames.append(evaluate_model(model, ckpt, ids, brats_root, device))

    per_case = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(CACHE, index=False)
    reported = load_reported_dice()
    summary, flags = aggregate(per_case, reported)
    if flags["mismatches"]:
        logger.error("F1/Dice mismatch: %s", json.dumps(flags, indent=2))
    write_outputs(summary, flags)
    return 1 if flags["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
