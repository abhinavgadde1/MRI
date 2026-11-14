#!/usr/bin/env python3
"""Run v3+LCC on ingested batch2 studies → outputs/batch2_v3_preview/.

Reports WT/TC/ET mL, raw WT component count, volume_plausibility, and a
mid-tumor 2D overlay per case. Does not train or modify checkpoints.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.volume_plausibility import assess_wt_plausibility  # noqa: E402
from preprocessing.h5_to_nifti import MODALITY_NAMES  # noqa: E402
from preprocessing.registration import REGISTERED_NAMES  # noqa: E402
from segmentation.evaluate import load_model_from_checkpoint  # noqa: E402
from segmentation.model import BRATS_REGIONS  # noqa: E402
from segmentation.postprocess import (  # noqa: E402
    apply_channel_postprocess,
    enforce_brats_nesting,
)
from segmentation.pseudo_label import get_patient_inference_transforms  # noqa: E402
from segmentation.train import post_transforms  # noqa: E402

PATIENTS = ROOT / "data" / "processed" / "real_patients_batch2"
OUT = ROOT / "outputs" / "batch2_v3_preview"
CKPT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
ROI = (96, 96, 96)
VOXEL_ML = 0.001  # 1 mm³ isotropic
# MODALITY_NAMES = (FLAIR, T1, T1c, T2). Previously hardcoded (T1c, T1, T2, FLAIR).


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def n_cc(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    _, n = ndi.label(mask.astype(np.uint8))
    return int(n)


def t1c_is_standin(reg: Path) -> bool:
    t1 = reg / REGISTERED_NAMES["T1"]
    t1c = reg / REGISTERED_NAMES["T1c"]
    if not t1.is_file():
        return False
    if not t1c.is_file():
        return True
    return t1.read_bytes() == t1c.read_bytes()


@torch.no_grad()
def infer_one(model, dev: torch.device, study_dir: Path) -> tuple[dict, np.ndarray, dict]:
    reg = study_dir / "04_registered_1mm"
    record = {
        "study_id": study_dir.name,
        "image": [str(reg / REGISTERED_NAMES[m]) for m in MODALITY_NAMES],
    }
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(dev, dtype=torch.float32)
    t0 = time.perf_counter()
    logits = sliding_window_inference(
        image, roi_size=ROI, sw_batch_size=1, predictor=model
    )
    infer_s = time.perf_counter() - t0
    post_pred, _ = post_transforms()
    pred_raw = (
        post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(np.float32)
    )
    pred_lcc = apply_channel_postprocess(pred_raw.copy(), mode="lcc")
    pred_lcc = enforce_brats_nesting(pred_lcc)
    raw = {r: pred_raw[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    lcc = {r: pred_lcc[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    flair = data["image"][0].detach().cpu().numpy()

    wt_ml = float(lcc["whole_tumor"].sum()) * VOXEL_ML
    plaus = assess_wt_plausibility(wt_ml)
    row = {
        "study_id": study_dir.name,
        "t1c_is_standin": bool(t1c_is_standin(reg)),
        "infer_seconds": round(float(infer_s), 3),
        "WT_ml": round(wt_ml, 3),
        "TC_ml": round(float(lcc["tumor_core"].sum()) * VOXEL_ML, 3),
        "ET_ml": round(float(lcc["enhancing_tumor"].sum()) * VOXEL_ML, 3),
        "raw_component_count": n_cc(raw["whole_tumor"]),
        "lcc_component_count": n_cc(lcc["whole_tumor"]),
        "volume_plausibility": plaus.status,
        "volume_plausibility_message": plaus.message,
    }
    return row, flair, lcc


def save_overlay(
    flair: np.ndarray,
    regions: dict[str, np.ndarray],
    path: Path,
    title: str,
) -> None:
    wt = regions["whole_tumor"]
    if np.any(wt):
        z = int(np.round(ndi.center_of_mass(wt)[2]))
    else:
        z = flair.shape[2] // 2
    z = int(np.clip(z, 0, flair.shape[2] - 1))
    base = flair[:, :, z].astype(float)
    p2, p98 = np.percentile(base, (2, 98))
    if p98 <= p2:
        norm = np.zeros_like(base)
    else:
        norm = np.clip((base - p2) / (p98 - p2), 0, 1)
    rgb = np.stack([norm, norm, norm], axis=-1)
    for region, color in (
        ("whole_tumor", np.array([0.1, 0.75, 0.3])),
        ("tumor_core", np.array([0.15, 0.45, 0.95])),
        ("enhancing_tumor", np.array([0.95, 0.2, 0.2])),
    ):
        m = regions[region][:, :, z]
        if np.any(m):
            rgb[m] = rgb[m] * 0.35 + color * 0.65
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=140)
    ax.imshow(np.rot90(rgb), origin="lower")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    assert CKPT.is_file(), f"Missing checkpoint {CKPT}"
    OUT.mkdir(parents=True, exist_ok=True)
    studies = sorted(
        p
        for p in PATIENTS.iterdir()
        if p.is_dir()
        and p.name.startswith("batch2_")
        and (p / "04_registered_1mm" / REGISTERED_NAMES["T1"]).is_file()
    )
    print(f"cases={len(studies)} ckpt={CKPT} out={OUT}", flush=True)
    dev = device()
    model = load_model_from_checkpoint(CKPT, dev)
    model.eval()
    print("device", dev, flush=True)

    rows: list[dict] = []
    for i, study in enumerate(studies, start=1):
        print(f"==== [{i}/{len(studies)}] {study.name} ====", flush=True)
        row, flair, lcc = infer_one(model, dev, study)
        case_dir = OUT / study.name
        case_dir.mkdir(parents=True, exist_ok=True)
        overlay = case_dir / "mid_tumor_overlay.png"
        save_overlay(
            flair,
            lcc,
            overlay,
            title=(
                f"{row['study_id']}  WT={row['WT_ml']} mL  "
                f"raw_cc={row['raw_component_count']}  "
                f"plaus={row['volume_plausibility']}"
            ),
        )
        row["overlay"] = str(overlay)
        (case_dir / "meta.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(row)
        print(
            f"{row['study_id']}: WT={row['WT_ml']} TC={row['TC_ml']} ET={row['ET_ml']} "
            f"raw_cc={row['raw_component_count']} plaus={row['volume_plausibility']} "
            f"standin={row['t1c_is_standin']}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(OUT / "preview_summary.csv", index=False)
        if dev.type == "mps":
            torch.mps.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "preview_summary.csv", index=False)
    (OUT / "preview_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(df.to_string(index=False), flush=True)
    print("PREVIEW DONE", flush=True)


if __name__ == "__main__":
    main()
