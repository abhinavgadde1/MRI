#!/usr/bin/env python3
"""Re-run the real-patient v3-lcc FP audit on HD-BET-stripped registered volumes.

Diagnostic only. Does not change checkpoints, training, or default pipeline
behavior. Writes to outputs/real_patient_fp_audit_hdbet/ so the original
Otsu-era audit under outputs/real_patient_fp_audit/ is left intact.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "outputs" / "real_patient_fp_audit"))

from run_fp_audit import (  # noqa: E402
    CKPT,
    overlay_png,
    discover_studies,
    infer_one,
)
from device import get_device  # noqa: E402
from pipeline.volume_plausibility import (  # noqa: E402
    MAX_PLAUSIBLE_WT_ML,
    MIN_PLAUSIBLE_WT_ML,
    assess_wt_plausibility,
)
from segmentation.evaluate import load_model_from_checkpoint  # noqa: E402

import torch  # noqa: E402

OUT = ROOT / "outputs" / "real_patient_fp_audit_hdbet"
OLD_CSV = ROOT / "outputs" / "real_patient_fp_audit" / "volumes.csv"
FLOOR = float(MIN_PLAUSIBLE_WT_ML)
CEIL = float(MAX_PLAUSIBLE_WT_ML)
REL_CHANGE = 0.20


def classify(wt_ml: float) -> str:
    return assess_wt_plausibility(wt_ml, floor_ml=FLOOR, max_ml=CEIL).status or "unknown"


def compare(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "study_id",
        "WT_lcc_ml",
        "TC_lcc_ml",
        "ET_lcc_ml",
        "WT_raw_n_cc",
        "WT_max_prob_global",
    ]
    left = old[cols].rename(
        columns={c: f"old_{c}" for c in cols if c != "study_id"}
    )
    right = new[cols].rename(
        columns={c: f"new_{c}" for c in cols if c != "study_id"}
    )
    merged = left.merge(right, on="study_id", how="outer")
    merged["WT_delta_ml"] = merged["new_WT_lcc_ml"] - merged["old_WT_lcc_ml"]
    merged["WT_rel_change"] = merged.apply(
        lambda r: (
            float("nan")
            if pd.isna(r["old_WT_lcc_ml"]) or pd.isna(r["new_WT_lcc_ml"])
            else (
                float("inf")
                if r["old_WT_lcc_ml"] == 0 and r["new_WT_lcc_ml"] != 0
                else (
                    0.0
                    if r["old_WT_lcc_ml"] == 0
                    else (r["new_WT_lcc_ml"] - r["old_WT_lcc_ml"]) / r["old_WT_lcc_ml"]
                )
            )
        ),
        axis=1,
    )
    merged["meaningful_gt20pct"] = merged["WT_rel_change"].abs() > REL_CHANGE
    merged["direction"] = merged["WT_delta_ml"].map(
        lambda d: "increased" if d > 0 else ("decreased" if d < 0 else "unchanged")
    )
    merged["old_band"] = merged["old_WT_lcc_ml"].map(
        lambda v: classify(v) if pd.notna(v) else "missing"
    )
    merged["new_band"] = merged["new_WT_lcc_ml"].map(
        lambda v: classify(v) if pd.notna(v) else "missing"
    )
    return merged.sort_values("new_WT_lcc_ml", ascending=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    studies, skipped = discover_studies()
    (OUT / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    print(f"runnable={len(studies)} skipped={len(skipped)}")
    print(f"floor={FLOOR:.3f} mL  ceiling={CEIL:.3f} mL")

    device = get_device()
    model = load_model_from_checkpoint(CKPT, device)
    model.eval()
    print("device", device, "ckpt", CKPT)

    rows = []
    cache: dict[str, tuple] = {}
    t_all = time.perf_counter()
    for i, study in enumerate(studies, start=1):
        row, flair, wt_lcc, _wt_raw = infer_one(model, device, study)
        rows.append(row)
        cache[study.name] = (flair, wt_lcc)
        print(
            f"[{i}/{len(studies)}] {study.name}  WT_lcc={row['WT_lcc_ml']:.2f} mL  "
            f"raw_cc={row['WT_raw_n_cc']}  maxp={row['WT_max_prob_global']:.3f}  "
            f"{row['infer_seconds']:.1f}s"
        )
        if i % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT / "volumes.csv", index=False)
            if device.type == "mps":
                torch.mps.empty_cache()
    wall = time.perf_counter() - t_all
    df = pd.DataFrame(rows).sort_values("WT_lcc_ml", ascending=True)
    df["below_brats_min_gt_wt"] = df["WT_lcc_ml"] < FLOOR
    df["above_brats_max_gt_wt"] = df["WT_lcc_ml"] > CEIL
    df["zero_lcc_wt"] = df["WT_lcc_ml"] <= 0
    df["plausibility"] = df["WT_lcc_ml"].map(classify)
    df.to_csv(OUT / "volumes.csv", index=False)

    smallest = df.nsmallest(4, "WT_lcc_ml")
    overlay_dir = OUT / "smallest_overlays"
    overlay_meta = []
    for rank, rec in enumerate(smallest.itertuples(index=False), start=1):
        flair, wt_lcc = cache[rec.study_id]
        png = overlay_dir / f"{rank:02d}_{rec.study_id}_wt_lcc.png"
        overlay_png(
            flair,
            wt_lcc,
            png,
            title=(
                f"{rec.study_id} (HD-BET)\nWT LCC {rec.WT_lcc_ml:.2f} mL · "
                f"raw CC={rec.WT_raw_n_cc} · max p={rec.WT_max_prob_global:.2f}"
            ),
        )
        overlay_meta.append({"rank": rank, "study_id": rec.study_id, "png": str(png)})
        print("overlay", png)

    old = pd.read_csv(OLD_CSV)
    cmp = compare(old, df)
    cmp.to_csv(OUT / "comparison.csv", index=False)
    meaningful = cmp[cmp["meaningful_gt20pct"].fillna(False)]
    n_up = int((meaningful["direction"] == "increased").sum())
    n_down = int((meaningful["direction"] == "decreased").sum())

    def counts(series: pd.Series) -> dict[str, int]:
        return {
            "low": int((series == "low").sum()),
            "high": int((series == "high").sum()),
            "plausible": int((series == "plausible").sum()),
        }

    old_band = counts(cmp["old_band"])
    new_band = counts(cmp["new_band"])
    summary = {
        "n_runnable": int(len(df)),
        "n_skipped": skipped,
        "wall_seconds": round(wall, 1),
        "device": str(device),
        "checkpoint": str(CKPT),
        "postprocess": "lcc",
        "skull_strip": "hd-bet",
        "floor_ml": FLOOR,
        "ceiling_ml": CEIL,
        "real_WT_lcc_min_ml": float(df["WT_lcc_ml"].min()),
        "real_WT_lcc_median_ml": float(df["WT_lcc_ml"].median()),
        "real_WT_lcc_max_ml": float(df["WT_lcc_ml"].max()),
        "n_zero_WT_lcc": int(df["zero_lcc_wt"].sum()),
        "n_below_floor": new_band["low"],
        "n_above_ceiling": new_band["high"],
        "n_plausible": new_band["plausible"],
        "old_n_below_floor": old_band["low"],
        "old_n_above_ceiling": old_band["high"],
        "old_n_plausible": old_band["plausible"],
        "n_t1c_standin": int(df["t1c_is_standin"].sum()),
        "n_meaningful_gt20pct": int(meaningful.shape[0]),
        "n_meaningful_increased": n_up,
        "n_meaningful_decreased": n_down,
        "overlays": overlay_meta,
        "clinical_ground_truth": "none_found",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "n_runnable", "n_zero_WT_lcc", "n_below_floor", "n_above_ceiling",
        "n_plausible", "old_n_below_floor", "old_n_above_ceiling", "old_n_plausible",
        "n_meaningful_gt20pct", "n_meaningful_increased", "n_meaningful_decreased",
        "real_WT_lcc_min_ml", "real_WT_lcc_median_ml", "real_WT_lcc_max_ml",
        "wall_seconds",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
