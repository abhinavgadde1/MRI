#!/usr/bin/env python3
"""Baseline v3+LCC on selected/unused negatives (no training).

Writes under ``outputs/batch2_pos_neg/`` only. Does not modify checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PROJECT_ROOT  # noqa: E402
from device import get_device  # noqa: E402
from preprocessing.h5_to_nifti import MODALITY_NAMES  # noqa: E402
from preprocessing.registration import REGISTERED_NAMES  # noqa: E402
from segmentation.evaluate import load_model_from_checkpoint  # noqa: E402
from segmentation.model import BRATS_REGIONS  # noqa: E402
from segmentation.postprocess import apply_channel_postprocess, enforce_brats_nesting  # noqa: E402
from segmentation.pseudo_label import get_patient_inference_transforms  # noqa: E402
from segmentation.splits import read_id_list  # noqa: E402
from segmentation.train import post_transforms  # noqa: E402

NEG_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients"
SPLIT = PROJECT_ROOT / "splits" / "batch2_pos_neg"
OUT = PROJECT_ROOT / "outputs" / "batch2_pos_neg"
CKPT = PROJECT_ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
ROI = (96, 96, 96)
# Read-only checkpoint — never write here.


@torch.no_grad()
def infer_wt_ml(model, device, study_dir: Path) -> dict:
    reg = study_dir / "04_registered_1mm"
    record = {
        "study_id": study_dir.name,
        "image": [str(reg / REGISTERED_NAMES[m]) for m in MODALITY_NAMES],
    }
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(device, dtype=torch.float32)
    t0 = time.perf_counter()
    logits = sliding_window_inference(image, roi_size=ROI, sw_batch_size=1, predictor=model)
    wall = time.perf_counter() - t0
    post_pred, _ = post_transforms()
    pred_raw = post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(np.float32)
    pred_lcc = enforce_brats_nesting(apply_channel_postprocess(pred_raw.copy(), mode="lcc"))
    # WT channel index
    wt_idx = BRATS_REGIONS.index("whole_tumor")
    wt = pred_lcc[wt_idx].astype(bool)
    spacing = (1.0, 1.0, 1.0)
    # try nifti zooms
    t1c = nib.load(str(reg / REGISTERED_NAMES["T1c"]))
    spacing = tuple(float(x) for x in t1c.header.get_zooms()[:3])
    wt_ml = float(np.count_nonzero(wt) * np.prod(spacing) / 1000.0)
    flair = data["image"][0].detach().cpu().numpy()
    return {
        "study_id": study_dir.name,
        "WT_lcc_ml": wt_ml,
        "pred_above_0_5_ml": bool(wt_ml > 0.5),
        "wall_s": wall,
        "flair": flair,
        "wt_mask": wt,
    }


def save_overlay(flair: np.ndarray, wt: np.ndarray, out_png: Path, title: str) -> None:
    z = flair.shape[2] // 2
    if np.any(wt):
        zs = np.where(wt.any(axis=(0, 1)))[0]
        z = int(zs[len(zs) // 2])
    sl = flair[:, :, z].astype(float)
    p2, p98 = np.percentile(sl, (2, 98))
    norm = np.zeros_like(sl) if p98 <= p2 else np.clip((sl - p2) / (p98 - p2), 0, 1)
    rgb = np.stack([norm, norm, norm], axis=-1)
    m = wt[:, :, z]
    rgb[m] = rgb[m] * 0.35 + np.array([0.95, 0.2, 0.2]) * 0.65
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.rot90(rgb), origin="lower")
    ax.set_title(title)
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_list(ids: list[str], model, device, *, keep_arrays: bool = False) -> list[dict]:
    rows = []
    for sid in ids:
        study_dir = NEG_ROOT / sid
        if not (study_dir / "04_registered_1mm" / REGISTERED_NAMES["T1"]).is_file():
            rows.append(
                {
                    "study_id": sid,
                    "WT_lcc_ml": float("nan"),
                    "pred_above_0_5_ml": False,
                    "wall_s": float("nan"),
                    "skipped": True,
                }
            )
            continue
        out = infer_wt_ml(model, device, study_dir)
        row = {
            "study_id": out["study_id"],
            "WT_lcc_ml": out["WT_lcc_ml"],
            "pred_above_0_5_ml": out["pred_above_0_5_ml"],
            "wall_s": out["wall_s"],
            "skipped": False,
        }
        if keep_arrays:
            row["_flair"] = out["flair"]
            row["_wt"] = out["wt_mask"]
        rows.append(row)
        print(
            f"  {sid}: WT={out['WT_lcc_ml']:.3f} mL above0.5={out['pred_above_0_5_ml']}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reuse-fp-audit", action="store_true",
                   help="Fill unused pool from existing FP audit CSV when possible")
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    selected = read_id_list(SPLIT / "negatives_selected.txt")
    unused = read_id_list(SPLIT / "negatives_unused.txt")
    device = get_device()
    print(f"Loading v3 from {CKPT} on {device} (read-only)")
    model = load_model_from_checkpoint(CKPT, device=device)
    model.eval()

    print(f"=== selected negatives n={len(selected)} ===")
    sel_rows = run_list(selected, model, device, keep_arrays=True)
    sel_df = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in sel_rows]
    )
    sel_df.to_csv(OUT / "baseline_selected_negatives.csv", index=False)

    # overlays for 3 largest WT
    ranked = sorted(
        [r for r in sel_rows if not r.get("skipped")],
        key=lambda r: r["WT_lcc_ml"],
        reverse=True,
    )[:3]
    ov_dir = OUT / "baseline_overlays_largest3"
    for r in ranked:
        save_overlay(
            r["_flair"],
            r["_wt"],
            ov_dir / f"{r['study_id']}_WT_lcc.png",
            title=f"{r['study_id']} WT={r['WT_lcc_ml']:.2f} mL (v3+LCC draft)",
        )
    print(f"Overlays: {ov_dir}")

    # unused: prefer existing audit if requested / available
    audit = PROJECT_ROOT / "outputs" / "real_patient_fp_audit" / "volumes.csv"
    unused_rows = []
    if args.reuse_fp_audit and audit.is_file():
        adf = pd.read_csv(audit)
        by_id = {str(r.study_id): r for r in adf.itertuples()}
        missing = []
        for sid in unused:
            if sid in by_id:
                wt = float(by_id[sid].WT_lcc_ml)
                unused_rows.append(
                    {
                        "study_id": sid,
                        "WT_lcc_ml": wt,
                        "pred_above_0_5_ml": wt > 0.5,
                        "source": "fp_audit_volumes.csv",
                    }
                )
            else:
                missing.append(sid)
        if missing:
            print(f"Running missing unused ({len(missing)}) ...")
            for r in run_list(missing, model, device):
                r["source"] = "live_v3"
                unused_rows.append({k: v for k, v in r.items() if not k.startswith("_")})
    else:
        print(f"=== unused negatives n={len(unused)} ===")
        for r in run_list(unused, model, device):
            r["source"] = "live_v3"
            unused_rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    unused_df = pd.DataFrame(unused_rows)
    unused_df.to_csv(OUT / "baseline_unused_negatives.csv", index=False)

    # summary tables
    def _dist(df: pd.DataFrame) -> dict:
        w = df["WT_lcc_ml"].astype(float)
        return {
            "n": int(len(df)),
            "n_above_0_5_ml": int((w > 0.5).sum()),
            "frac_above_0_5_ml": float((w > 0.5).mean()) if len(df) else float("nan"),
            "wt_ml_mean": float(w.mean()) if len(df) else float("nan"),
            "wt_ml_median": float(w.median()) if len(df) else float("nan"),
            "wt_ml_p90": float(w.quantile(0.9)) if len(df) else float("nan"),
            "wt_ml_max": float(w.max()) if len(df) else float("nan"),
        }

    summary = {
        "selected": _dist(sel_df),
        "unused": _dist(unused_df),
        "disclaimer": "v3+LCC drafts only — not diagnoses. Negatives are claimed tumor-free, unverified.",
    }
    (OUT / "baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        "# Baseline v3+LCC on claimed tumor-free negatives",
        "",
        summary["disclaimer"],
        "",
        "## Selected 20 (per-scan)",
        "",
        "| study_id | WT_lcc_ml | >0.5 mL |",
        "|---|---:|---|",
    ]
    for _, r in sel_df.sort_values("WT_lcc_ml", ascending=False).iterrows():
        md.append(
            f"| {r.study_id} | {r.WT_lcc_ml:.3f} | {'yes' if r.pred_above_0_5_ml else 'no'} |"
        )
    md += [
        "",
        f"Overlays (3 largest): `{ov_dir}`",
        "",
        "## Unused 80 (counts / volume distribution only)",
        "",
        "```json",
        json.dumps(summary["unused"], indent=2),
        "```",
        "",
    ]
    (OUT / "BASELINE.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
