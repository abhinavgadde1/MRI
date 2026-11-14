#!/usr/bin/env python3
"""Measure v3+LCC separation of batch2 positives vs tumor-free negatives (no training)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PROJECT_ROOT  # noqa: E402
from device import get_device  # noqa: E402
from preprocessing.h5_to_nifti import MODALITY_NAMES  # noqa: E402
from preprocessing.registration import REGISTERED_NAMES  # noqa: E402
from segmentation.evaluate import load_model_from_checkpoint  # noqa: E402
from segmentation.model import BRATS_REGIONS  # noqa: E402
from segmentation.pos_neg_separation import (  # noqa: E402
    BATCH2_ROOT,
    NEG_ROOT,
    OUT_DIR,
    SCORE_COLS,
    SPLIT_DIR,
    evaluate_score,
    plot_roc_and_distributions,
    shortcut_auc_patient_cv,
    write_summary,
)
from segmentation.postprocess import (  # noqa: E402
    CONN_26,
    apply_channel_postprocess,
    enforce_brats_nesting,
)
from segmentation.pseudo_label import get_patient_inference_transforms  # noqa: E402
from segmentation.splits import read_id_list  # noqa: E402
from segmentation.train import post_transforms  # noqa: E402

CKPT = PROJECT_ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
ROI = (96, 96, 96)


def _n_cc(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    _, n = ndi.label(np.asarray(mask, dtype=bool), structure=CONN_26)
    return int(n)


def _mean_brain_intensity(reg: Path) -> float:
    t1 = reg / REGISTERED_NAMES["T1"]
    if not t1.is_file():
        return float("nan")
    data = np.asanyarray(nib.load(str(t1)).dataobj).astype(np.float32)
    m = data > 0
    return float(data[m].mean()) if np.any(m) else float("nan")


@torch.no_grad()
def score_study(model, device, study_dir: Path, study_id: str) -> dict:
    reg = study_dir / "04_registered_1mm"
    record = {
        "study_id": study_id,
        "image": [str(reg / REGISTERED_NAMES[m]) for m in MODALITY_NAMES],
    }
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(device, dtype=torch.float32)
    t0 = time.perf_counter()
    logits = sliding_window_inference(image, roi_size=ROI, sw_batch_size=1, predictor=model)
    wall = time.perf_counter() - t0
    probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    post_pred, _ = post_transforms()
    pred_raw = post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(np.float32)
    pred_lcc = enforce_brats_nesting(apply_channel_postprocess(pred_raw.copy(), mode="lcc"))

    # spacing from T1c
    t1c = nib.load(str(reg / REGISTERED_NAMES["T1c"]))
    spacing = tuple(float(x) for x in t1c.header.get_zooms()[:3])
    voxel_ml = float(np.prod(spacing) / 1000.0)

    row: dict = {"study_id": study_id, "wall_s": wall, "source": "live_v3"}
    short = {"enhancing_tumor": "ET", "tumor_core": "TC", "whole_tumor": "WT"}
    for i, r in enumerate(BRATS_REGIONS):
        sh = short[r]
        raw = pred_raw[i].astype(bool)
        lcc = pred_lcc[i].astype(bool)
        row[f"{sh}_lcc_ml"] = float(np.count_nonzero(lcc) * voxel_ml)
        row[f"{sh}_raw_n_cc"] = _n_cc(raw)
        ch = probs[i]
        on_lcc = ch[lcc]
        row[f"{sh}_max_prob_on_lcc"] = float(on_lcc.max()) if on_lcc.size else 0.0
    row["mean_brain_intensity"] = _mean_brain_intensity(reg)
    return row


def collect_scores(force: bool = False) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "scores.csv"
    manif = pd.read_csv(SPLIT_DIR / "manifest.csv")
    pos_ids = read_id_list(SPLIT_DIR / "positives.txt")
    neg_ids = (
        manif.loc[manif["cohort"] == "negative", "study_id"].astype(str).tolist()
    )

    if cache.is_file() and not force:
        cached = pd.read_csv(cache)
        have = set(cached["study_id"].astype(str))
        need = [s for s in pos_ids + neg_ids if s not in have]
        if not need and all(c in cached.columns for c in SCORE_COLS):
            print(f"Reusing {cache} ({len(cached)} rows)")
            return cached.merge(
                manif[
                    [
                        c
                        for c in manif.columns
                        if c
                        in {
                            "study_id",
                            "cohort",
                            "patient_id",
                            "t1c_status",
                            "tumor_free_status",
                            "manufacturer",
                            "model",
                            "field_strength_t",
                            "inplane_spacing_T1_mm",
                            "brain_mask_fraction",
                        }
                    ]
                ],
                on="study_id",
                how="left",
                suffixes=("", "_m"),
            )

    device = get_device()
    print(f"Loading {CKPT} on {device} (read-only)")
    model = load_model_from_checkpoint(CKPT, device=device)
    model.eval()

    rows = []
    if cache.is_file() and not force:
        rows = pd.read_csv(cache).to_dict(orient="records")
        have = {str(r["study_id"]) for r in rows}
    else:
        have = set()

    for sid in pos_ids:
        if sid in have:
            continue
        print(f"  POS {sid}", flush=True)
        rows.append(score_study(model, device, BATCH2_ROOT / sid, sid))
        have.add(sid)
    for sid in neg_ids:
        if sid in have:
            continue
        print(f"  NEG {sid}", flush=True)
        study = NEG_ROOT / sid
        if not (study / "04_registered_1mm" / REGISTERED_NAMES["T1"]).is_file():
            rows.append(
                {
                    "study_id": sid,
                    "WT_lcc_ml": float("nan"),
                    "TC_lcc_ml": float("nan"),
                    "ET_lcc_ml": float("nan"),
                    "WT_raw_n_cc": float("nan"),
                    "WT_max_prob_on_lcc": float("nan"),
                    "mean_brain_intensity": float("nan"),
                    "source": "skipped_missing",
                }
            )
            continue
        rows.append(score_study(model, device, study, sid))
        have.add(sid)

    scores = pd.DataFrame(rows)
    scores.to_csv(cache, index=False)
    # merge metadata
    meta_cols = [
        "study_id",
        "cohort",
        "patient_id",
        "t1c_status",
        "tumor_free_status",
        "manufacturer",
        "model",
        "field_strength_t",
        "inplane_spacing_T1_mm",
        "brain_mask_fraction",
    ]
    out = scores.merge(manif[[c for c in meta_cols if c in manif.columns]], on="study_id", how="left")
    # fill cohort for safety
    out.loc[out["study_id"].isin(pos_ids), "cohort"] = "positive"
    out.loc[out["study_id"].isin(neg_ids) & out["cohort"].isna(), "cohort"] = "negative"
    out.to_csv(OUT_DIR / "scores_with_meta.csv", index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force-infer", action="store_true")
    args = p.parse_args(argv)

    df = collect_scores(force=args.force_infer)
    df["cohort"] = df["cohort"].astype(str)
    if "tumor_free_status" not in df.columns:
        df["tumor_free_status"] = "claimed_unverified"
    df["tumor_free_status"] = df["tumor_free_status"].fillna("claimed_unverified")

    pos_all = (df["cohort"] == "positive").to_numpy()
    neg_all = (df["cohort"] == "negative").to_numpy()
    confirmed = df["tumor_free_status"].isin(
        ["radiologist_report", "radiologist_verbal"]
    ).to_numpy()
    n_confirmed = int((neg_all & confirmed).sum())
    confirmed_ran = n_confirmed > 0

    results = []
    # (b) all claimed negatives
    for score in SCORE_COLS:
        results.append(
            evaluate_score(
                df,
                score,
                subset="all_claimed_negatives",
                pos_mask=pos_all,
                neg_mask=neg_all,
            )
        )
    plot_roc_and_distributions(
        df,
        pos_mask=pos_all,
        neg_mask=neg_all,
        scores=SCORE_COLS,
        title_prefix="Positives vs all claimed negatives",
        out_path=OUT_DIR / "roc_distributions_all_claimed.png",
    )

    # (a) confirmed only if any
    if confirmed_ran:
        for score in SCORE_COLS:
            results.append(
                evaluate_score(
                    df,
                    score,
                    subset="radiologist_confirmed_negatives",
                    pos_mask=pos_all,
                    neg_mask=neg_all & confirmed,
                )
            )
        plot_roc_and_distributions(
            df,
            pos_mask=pos_all,
            neg_mask=neg_all & confirmed,
            scores=SCORE_COLS,
            title_prefix="Positives vs radiologist-confirmed negatives",
            out_path=OUT_DIR / "roc_distributions_confirmed.png",
        )
    else:
        print(
            "No radiologist-confirmed negatives yet — running (b) all claimed only."
        )

    # Native / stand-in positive subsets vs all negatives
    native = pos_all & (df["t1c_status"].astype(str) == "native").to_numpy()
    standin = pos_all & (df["t1c_status"].astype(str) == "standin").to_numpy()
    for score in SCORE_COLS:
        results.append(
            evaluate_score(
                df,
                score,
                subset="native_T1c_positives_vs_all_neg",
                pos_mask=native,
                neg_mask=neg_all,
            )
        )
        results.append(
            evaluate_score(
                df,
                score,
                subset="standin_T1c_positives_vs_all_neg",
                pos_mask=standin,
                neg_mask=neg_all,
            )
        )

    # Shortcut features
    # mean_brain_intensity from scores; t1c_status; model; inplane; brain_mask_fraction
    shortcut = shortcut_auc_patient_cv(
        df,
        pos_mask=pos_all,
        neg_mask=neg_all,
        feature_cols=[
            "inplane_spacing_T1_mm",
            "brain_mask_fraction",
            "model",
            "t1c_status",
            "mean_brain_intensity",
        ],
        n_folds=5,
        seed=42,
    )
    if shortcut.get("warning"):
        print(
            "WARNING: Non-tumor features separate groups well "
            f"(CV AUC={shortcut['auc_mean']:.3f}). "
            "A classifier may read the scanner, not the tumor."
        )

    # Save AUC table
    auc_df = pd.DataFrame([r.__dict__ for r in results])
    auc_df.to_csv(OUT_DIR / "auc_table.csv", index=False)
    (OUT_DIR / "shortcut_check.json").write_text(
        json.dumps(shortcut, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(
        results=results,
        shortcut=shortcut,
        confirmed_ran=confirmed_ran,
        n_confirmed=n_confirmed,
        out_path=OUT_DIR / "SUMMARY.md",
    )
    print(f"Wrote {OUT_DIR / 'SUMMARY.md'}")
    print(auc_df[auc_df.subset == 'all_claimed_negatives'][
        ['score', 'auc', 'ci_lo', 'ci_hi', 'sensitivity', 'specificity']
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
