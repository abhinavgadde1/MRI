#!/usr/bin/env python3
"""Baseline FP stats for radiologist-confirmed negatives only (no training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.batch2_radiologist_status import (  # noqa: E402
    REPORT_DIR,
    STATUS_REPORT,
    STATUS_VERBAL,
    bootstrap_fraction_ci,
)

BASELINE_SEL = REPORT_DIR / "baseline_selected_negatives.csv"
BASELINE_UN = REPORT_DIR / "baseline_unused_negatives.csv"
MANIFEST = ROOT / "splits" / "batch2_pos_neg" / "manifest.csv"
NEG_ROOT = ROOT / "data" / "processed" / "real_patients"


def _overlay_from_existing_or_skip(study_id: str, wt_ml: float, out_png: Path) -> bool:
    """Reuse FLAIR mid-slice + saved WT if we can re-infer cheaply is heavy; draw note if missing.

    Prefer copying from prior overlay dir if present; else write a text placeholder panel
    only when volumes CSV exists (actual overlays generated when arrays available via re-infer).
    """
    prior = REPORT_DIR / "baseline_overlays_largest3" / f"{study_id}_WT_lcc.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if prior.is_file():
        out_png.write_bytes(prior.read_bytes())
        return True
    return False


def main() -> int:
    manif = pd.read_csv(MANIFEST)
    confirmed_ids = set(
        manif.loc[
            (manif["cohort"] == "negative")
            & (manif["tumor_free_status"].isin([STATUS_REPORT, STATUS_VERBAL])),
            "study_id",
        ].astype(str)
    )

    frames = []
    for path in (BASELINE_SEL, BASELINE_UN):
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        summary = {
            "n_confirmed": 0,
            "note": "No baseline CSVs found; run run_batch2_neg_baseline_v3.py first.",
        }
        (REPORT_DIR / "baseline_confirmed_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (REPORT_DIR / "BASELINE_CONFIRMED.md").write_text(
            "# Confirmed-negative FP baseline\n\nNo baseline volumes available yet.\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return 0

    vols = pd.concat(frames, ignore_index=True)
    vols["study_id"] = vols["study_id"].astype(str)
    conf = vols[vols["study_id"].isin(confirmed_ids)].copy()
    conf = conf.dropna(subset=["WT_lcc_ml"])

    n = int(len(conf))
    if n == 0:
        summary = {
            "n_confirmed_with_volumes": 0,
            "n_confirmed_in_manifest": len(confirmed_ids),
            "frac_wt_gt_0_5_ml": None,
            "frac_ci95": None,
            "wt_ml_median": None,
            "wt_ml_p90": None,
            "wt_ml_max": None,
            "note": (
                "No radiologist-confirmed negatives yet "
                f"(fill {ROOT / 'data/processed/real_patients/radiologist_reads.csv'})."
            ),
        }
        (REPORT_DIR / "baseline_confirmed_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (REPORT_DIR / "BASELINE_CONFIRMED.md").write_text(
            "# Confirmed-negative FP baseline (v3+LCC)\n\n"
            f"{summary['note']}\n\n"
            "Selection was not changed.\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return 0

    flags = (conf["WT_lcc_ml"].astype(float) > 0.5).to_numpy()
    frac, lo, hi = bootstrap_fraction_ci(flags, n_boot=2000, seed=42)
    w = conf["WT_lcc_ml"].astype(float)
    summary = {
        "n_confirmed_with_volumes": n,
        "n_above_0_5_ml": int(flags.sum()),
        "frac_wt_gt_0_5_ml": frac,
        "frac_ci95": [lo, hi],
        "wt_ml_median": float(w.median()),
        "wt_ml_p90": float(w.quantile(0.9)),
        "wt_ml_max": float(w.max()),
        "disclaimer": (
            "v3+LCC drafts only — not diagnoses. Restricted to radiologist_report / "
            "radiologist_verbal negatives."
        ),
    }
    (REPORT_DIR / "baseline_confirmed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    conf.to_csv(REPORT_DIR / "baseline_confirmed_negatives.csv", index=False)

    # Overlays for 5 largest — re-infer if needed
    top = conf.sort_values("WT_lcc_ml", ascending=False).head(5)
    ov_dir = REPORT_DIR / "baseline_overlays_confirmed_largest5"
    ov_dir.mkdir(parents=True, exist_ok=True)
    need_infer = []
    for _, r in top.iterrows():
        dest = ov_dir / f"{r.study_id}_WT_lcc.png"
        if not _overlay_from_existing_or_skip(str(r.study_id), float(r.WT_lcc_ml), dest):
            need_infer.append(str(r.study_id))

    if need_infer:
        # Live infer overlays for missing
        import torch
        from monai.data import decollate_batch
        from monai.inferers import sliding_window_inference

        from device import get_device
        from preprocessing.h5_to_nifti import MODALITY_NAMES
        from preprocessing.registration import REGISTERED_NAMES
        from segmentation.evaluate import load_model_from_checkpoint
        from segmentation.model import BRATS_REGIONS
        from segmentation.postprocess import apply_channel_postprocess, enforce_brats_nesting
        from segmentation.pseudo_label import get_patient_inference_transforms
        from segmentation.train import post_transforms

        device = get_device()
        ckpt = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
        model = load_model_from_checkpoint(ckpt, device=device)
        model.eval()
        post_pred, _ = post_transforms()
        wt_idx = BRATS_REGIONS.index("whole_tumor")
        for sid in need_infer:
            study = NEG_ROOT / sid
            reg = study / "04_registered_1mm"
            if not (reg / REGISTERED_NAMES["FLAIR"]).is_file():
                continue
            record = {
                "study_id": sid,
                "image": [str(reg / REGISTERED_NAMES[m]) for m in MODALITY_NAMES],
            }
            data = get_patient_inference_transforms()(record)
            image = data["image"].unsqueeze(0).to(device, dtype=torch.float32)
            with torch.no_grad():
                logits = sliding_window_inference(
                    image, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
                )
            pred = post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(
                np.float32
            )
            pred = enforce_brats_nesting(apply_channel_postprocess(pred, mode="lcc"))
            wt = pred[wt_idx].astype(bool)
            flair = data["image"][0].detach().cpu().numpy()
            z = flair.shape[2] // 2
            if np.any(wt):
                zs = np.where(wt.any(axis=(0, 1)))[0]
                z = int(zs[len(zs) // 2])
            sl = flair[:, :, z].astype(float)
            p2, p98 = np.percentile(sl, (2, 98))
            norm = (
                np.zeros_like(sl)
                if p98 <= p2
                else np.clip((sl - p2) / (p98 - p2), 0, 1)
            )
            rgb = np.stack([norm, norm, norm], axis=-1)
            m = wt[:, :, z]
            rgb[m] = rgb[m] * 0.35 + np.array([0.95, 0.2, 0.2]) * 0.65
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(np.rot90(rgb), origin="lower")
            wt_ml = float(conf.loc[conf.study_id == sid, "WT_lcc_ml"].iloc[0])
            ax.set_title(f"{sid}\nWT={wt_ml:.2f} mL (confirmed; v3+LCC draft)")
            ax.axis("off")
            fig.savefig(ov_dir / f"{sid}_WT_lcc.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    md = [
        "# Confirmed-negative FP baseline (v3+LCC)",
        "",
        summary["disclaimer"],
        "",
        f"- n confirmed with volumes: **{n}**",
        f"- fraction WT > 0.5 mL: **{frac:.3f}** (95% bootstrap CI [{lo:.3f}, {hi:.3f}])",
        f"- median WT mL: **{summary['wt_ml_median']:.3f}**",
        f"- 90th percentile WT mL: **{summary['wt_ml_p90']:.3f}**",
        f"- max WT mL: **{summary['wt_ml_max']:.3f}**",
        "",
        f"Overlays (5 largest): `{ov_dir}`",
        "",
    ]
    (REPORT_DIR / "BASELINE_CONFIRMED.md").write_text("\n".join(md), encoding="utf-8")

    # Soften main BASELINE disclaimer
    base = REPORT_DIR / "BASELINE.md"
    if base.is_file():
        t = base.read_text(encoding="utf-8")
        t = t.replace(
            "Negatives are claimed tumor-free, unverified.",
            "See per-scan `tumor_free_status` in the manifest; "
            "confirmed-only stats are in BASELINE_CONFIRMED.md.",
        )
        t = t.replace(
            "# Baseline v3+LCC on claimed tumor-free negatives",
            "# Baseline v3+LCC on negatives (all statuses)",
        )
        base.write_text(t, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
