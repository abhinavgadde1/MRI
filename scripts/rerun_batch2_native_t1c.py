#!/usr/bin/env python3
"""Re-ingest batch2 native-T1c studies after classifier fix; re-run v3+LCC.

Does not touch data/processed/real_patients/ or checkpoints. No training.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi

from pipeline.volume_plausibility import assess_wt_plausibility
from preprocessing.h5_to_nifti import MODALITY_NAMES
from preprocessing.registration import REGISTERED_NAMES, run_patient_preprocessing
from preprocessing.skull_strip import hd_bet_available
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS
from segmentation.postprocess import apply_channel_postprocess, enforce_brats_nesting
from segmentation.prepare_correction_set import fill_missing_t1c_from_t1
from segmentation.pseudo_label import get_patient_inference_transforms
from segmentation.train import post_transforms

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "PROJECT"
OUT_PROC = ROOT / "data" / "processed" / "real_patients_batch2"
OUT_PREV = ROOT / "outputs" / "batch2_v3_preview"
OUT_NATIVE = ROOT / "outputs" / "batch2_native_t1c"
CKPT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
FORBIDDEN = ROOT / "data" / "processed" / "real_patients"

# Studies that classify with native post-contrast T1 after the classifier fix.
NATIVE_T1C_IDS = [1, 2, 3, 5, 6, 7, 8, 10, 12, 13]
# Previously ingested with false-positive s3DI_MC as T1c — repair to stand-in.
FALSE_T1C_REPAIR_IDS = [9, 16]
USABLE_IDS = [i for i in range(1, 21) if i != 19]

VOXEL_ML = 0.001
ROI = (96, 96, 96)
# MODALITY_NAMES imported from preprocessing.h5_to_nifti (FLAIR, T1, T1c, T2).
# Previously hardcoded (T1c, T1, T2, FLAIR) — wrong for v3.
# NCC below this on brain-masked T1 vs registered T1c → poor alignment flag.
ALIGN_NCC_FLOOR = 0.15


def _sid(i: int) -> str:
    return f"batch2_{i:02d}"


def mask_fraction(path: Path) -> float | None:
    if not path.is_file():
        return None
    return float(np.mean(nib.load(str(path)).get_fdata() > 0))


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


def reingest_study(i: int, *, expect_native_t1c: bool) -> dict:
    sid = _sid(i)
    src = PROJECT / str(i)
    dst = OUT_PROC / sid
    assert FORBIDDEN.resolve() not in dst.resolve().parents
    if dst.exists():
        shutil.rmtree(dst)
    print(f"\n==== Re-ingest {sid} (expect_native_t1c={expect_native_t1c}) ====", flush=True)
    t0 = time.perf_counter()
    result = run_patient_preprocessing(src, dst, study_name=sid, use_hd_bet=True)
    methods = dict(getattr(result, "skull_strip_methods", {}) or {})
    strip = dst / "03_skull"
    for mod in ("T1", "T1c", "T2", "FLAIR"):
        side = strip / f"{mod}_skullstrip_method.json"
        if side.is_file():
            methods[mod] = json.loads(side.read_text(encoding="utf-8")).get("method")
    if any(v and "otsu" in str(v).lower() for v in methods.values()):
        raise RuntimeError(f"Otsu fallback used: {methods}")
    if any(v and v != "hd-bet" for v in methods.values() if v):
        raise RuntimeError(f"Non-HD-BET methods: {methods}")

    selection = dict(getattr(result, "series_selection", {}) or {})
    sel_path = dst / "series_selection.json"
    if sel_path.is_file():
        selection = json.loads(sel_path.read_text(encoding="utf-8"))

    has_native = "T1c" in selection and "T1" in selection
    # If T1c missing, fill stand-in after registration
    reg = dst / "04_registered_1mm"
    if not (reg / REGISTERED_NAMES["T1c"]).is_file():
        n = fill_missing_t1c_from_t1(OUT_PROC)
        print(f"  filled stand-in T1c (batch fill returned {n})", flush=True)

    native = bool(has_native) and not t1c_is_standin(reg)
    if expect_native_t1c and not native:
        raise RuntimeError(f"{sid}: expected native T1c but stand-in/missing after ingest")
    if not expect_native_t1c and native:
        print(f"  WARN {sid}: unexpected native T1c", flush=True)

    fracs = {
        mod: mask_fraction(strip / f"{mod}_brainmask.nii.gz")
        for mod in ("T1", "T1c", "T2", "FLAIR")
    }
    channels = {}
    for mod in ("T1", "T1c", "T2", "FLAIR"):
        info = selection.get(mod) or {}
        channels[mod] = {
            "series_description": info.get("series_description"),
            "n_slices": info.get("n_slices"),
            "slice_thickness_mm": info.get("slice_thickness_mm"),
            "reason": info.get("reason"),
            "standin": (mod == "T1c" and not native),
        }
    row = {
        "study_id": sid,
        "native_t1c": native,
        "skull_strip_methods": methods,
        "mask_fractions": fracs,
        "channels": channels,
        "duration_sec": round(time.perf_counter() - t0, 2),
        "status": "success",
    }
    (dst / "native_t1c_manifest.json").write_text(
        json.dumps(row, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"SUCCESS {sid} native_t1c={native} methods={methods} "
        f"T1={channels['T1'].get('series_description')} "
        f"T1c={channels['T1c'].get('series_description')} "
        f"in {row['duration_sec']}s",
        flush=True,
    )
    return row


def repair_false_t1c_standin(i: int) -> dict:
    """Drop angio T1c and fill T1→T1c stand-in; re-strip/register T1c only via full reingest."""
    # Full re-ingest is safest so T1c never remains the old angio volume.
    return reingest_study(i, expect_native_t1c=False)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return float("nan")
    x = a[m].astype(np.float64)
    y = b[m].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def alignment_overlay(study_dir: Path, out_png: Path) -> dict:
    """T1 edge contours on registered T1c; NCC brain-masked alignment metric."""
    reg = study_dir / "04_registered_1mm"
    t1 = nib.load(str(reg / REGISTERED_NAMES["T1"])).get_fdata().astype(np.float32)
    t1c = nib.load(str(reg / REGISTERED_NAMES["T1c"])).get_fdata().astype(np.float32)
    # Brain mask from T1 intensity Otsu-ish via percentile
    thr = np.percentile(t1[t1 > 0], 15) if np.any(t1 > 0) else 0
    brain = t1 > thr
    brain = ndi.binary_fill_holes(brain)
    score = ncc(t1, t1c, brain)
    # Mid-brain slice
    if np.any(brain):
        z = int(np.round(ndi.center_of_mass(brain)[2]))
    else:
        z = t1.shape[2] // 2
    z = int(np.clip(z, 0, t1.shape[2] - 1))
    sl_t1 = t1[:, :, z]
    sl_t1c = t1c[:, :, z]
    # Edges of T1
    edges = ndi.sobel(sl_t1)
    edge_mask = edges > np.percentile(edges, 90)

    def _norm(sl: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(sl, (2, 98))
        if p98 <= p2:
            return np.zeros_like(sl)
        return np.clip((sl - p2) / (p98 - p2), 0, 1)

    base = _norm(sl_t1c)
    rgb = np.stack([base, base, base], axis=-1)
    # cyan edges from T1
    rgb[edge_mask] = np.array([0.1, 0.95, 0.95])
    poor = (not np.isfinite(score)) or (score < ALIGN_NCC_FLOOR)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=140)
    ax.imshow(np.rot90(rgb), origin="lower")
    ax.set_title(
        f"{study_dir.name} T1 edges on T1c  NCC={score:.3f}"
        + ("  POOR" if poor else "  OK"),
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return {
        "study_id": study_dir.name,
        "metric": "brain_masked_NCC",
        "ncc": score,
        "poor_alignment": poor,
        "overlay": str(out_png),
        "slice_z": z,
    }


@torch.no_grad()
def infer_one(model, dev, study_dir: Path) -> tuple[dict, np.ndarray, dict]:
    reg = study_dir / "04_registered_1mm"
    record = {
        "study_id": study_dir.name,
        "image": [str(reg / REGISTERED_NAMES[m]) for m in MODALITY_NAMES],
    }
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(dev, dtype=torch.float32)
    logits = sliding_window_inference(
        image, roi_size=ROI, sw_batch_size=1, predictor=model
    )
    post_pred, _ = post_transforms()
    pred_raw = (
        post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(np.float32)
    )
    pred_lcc = apply_channel_postprocess(pred_raw.copy(), mode="lcc")
    pred_lcc = enforce_brats_nesting(pred_lcc)
    raw = {r: pred_raw[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    lcc = {r: pred_lcc[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    flair = data["image"][0].detach().cpu().numpy()
    wt = lcc["whole_tumor"]
    wt_ml = float(wt.sum()) * VOXEL_ML
    plaus = assess_wt_plausibility(wt_ml)
    # Truncation: predicted tumor touches first or last slice
    touches = False
    if np.any(wt):
        zs = np.where(wt.any(axis=(0, 1)))[0]
        touches = bool(zs.min() <= 1 or zs.max() >= wt.shape[2] - 2)
    standin = t1c_is_standin(reg)
    row = {
        "study_id": study_dir.name,
        "t1c_is_standin": standin,
        "native_t1c": not standin,
        "WT_ml": round(wt_ml, 3),
        "TC_ml": round(float(lcc["tumor_core"].sum()) * VOXEL_ML, 3),
        # ET only trustworthy with native T1c — still compute, hide in report for stand-in
        "ET_ml": round(float(lcc["enhancing_tumor"].sum()) * VOXEL_ML, 3),
        "ET_ml_reported": (
            round(float(lcc["enhancing_tumor"].sum()) * VOXEL_ML, 3)
            if not standin
            else None
        ),
        "raw_component_count": n_cc(raw["whole_tumor"]),
        "volume_plausibility": plaus.status,
        "volume_plausibility_message": plaus.message,
        "tumor_touches_edge_slices": touches,
        "et_hidden": standin,
    }
    return row, flair, lcc


def save_overlay(flair, regions, path, title, *, hide_et: bool) -> None:
    wt = regions["whole_tumor"]
    z = (
        int(np.round(ndi.center_of_mass(wt)[2]))
        if np.any(wt)
        else flair.shape[2] // 2
    )
    z = int(np.clip(z, 0, flair.shape[2] - 1))
    base = flair[:, :, z].astype(float)
    p2, p98 = np.percentile(base, (2, 98))
    norm = (
        np.zeros_like(base)
        if p98 <= p2
        else np.clip((base - p2) / (p98 - p2), 0, 1)
    )
    rgb = np.stack([norm, norm, norm], axis=-1)
    pairs = [
        ("whole_tumor", np.array([0.1, 0.75, 0.3])),
        ("tumor_core", np.array([0.15, 0.45, 0.95])),
    ]
    if not hide_et:
        pairs.append(("enhancing_tumor", np.array([0.95, 0.2, 0.2])))
    for region, color in pairs:
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


def before_after_panel(
    before_overlay: Path | None,
    after_overlay: Path,
    out_path: Path,
    title: str,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), dpi=140)
    for ax, path, ttl in (
        (axs[0], before_overlay, "before (T1 stand-in as T1c)"),
        (axs[1], after_overlay, "after (native T1c)"),
    ):
        if path is not None and Path(path).is_file():
            img = plt.imread(str(path))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def mark_standin_studies(native_ids: set[int], repaired: set[int]) -> list[dict]:
    rows = []
    for i in USABLE_IDS:
        sid = _sid(i)
        study = OUT_PROC / sid
        reg = study / "04_registered_1mm"
        if not study.is_dir():
            continue
        if i in native_ids:
            continue
        # Ensure T1c stand-in exists
        if not (reg / REGISTERED_NAMES["T1c"]).is_file() and (
            reg / REGISTERED_NAMES["T1"]
        ).is_file():
            shutil.copy2(reg / REGISTERED_NAMES["T1"], reg / REGISTERED_NAMES["T1c"])
        standin = t1c_is_standin(reg)
        sel = {}
        sel_path = study / "series_selection.json"
        if sel_path.is_file():
            sel = json.loads(sel_path.read_text(encoding="utf-8"))
        # If old false T1c selection lingers without reingest, clear T1c entry
        channels = {}
        for mod in ("T1", "T1c", "T2", "FLAIR"):
            info = sel.get(mod) or {}
            channels[mod] = {
                "series_description": info.get("series_description"),
                "n_slices": info.get("n_slices"),
                "slice_thickness_mm": info.get("slice_thickness_mm"),
                "reason": info.get("reason"),
                "standin": mod == "T1c",
            }
        if standin:
            channels["T1c"] = {
                "series_description": "T1_STANDIN_FROM_T1",
                "n_slices": channels.get("T1", {}).get("n_slices"),
                "slice_thickness_mm": channels.get("T1", {}).get("slice_thickness_mm"),
                "reason": "No native T1c after classifier fix; T1 copied to T1c",
                "standin": True,
            }
        row = {
            "study_id": sid,
            "native_t1c": False,
            "repaired_false_t1c": i in repaired,
            "channels": channels,
            "status": "standin_marked",
        }
        (study / "native_t1c_manifest.json").write_text(
            json.dumps(row, indent=2, default=str) + "\n", encoding="utf-8"
        )
        rows.append(row)
        print(f"MARK stand-in {sid} repaired={i in repaired}", flush=True)
    return rows


def channel_slice_table() -> pd.DataFrame:
    rows = []
    for i in USABLE_IDS:
        sid = _sid(i)
        man = OUT_PROC / sid / "native_t1c_manifest.json"
        if not man.is_file():
            continue
        payload = json.loads(man.read_text(encoding="utf-8"))
        ch = payload.get("channels") or {}
        row = {
            "study_id": sid,
            "native_t1c": payload.get("native_t1c"),
        }
        for mod in ("T1", "T1c", "T2", "FLAIR"):
            info = ch.get(mod) or {}
            row[f"{mod}_desc"] = info.get("series_description")
            row[f"{mod}_n_slices"] = info.get("n_slices")
            row[f"{mod}_thickness_mm"] = info.get("slice_thickness_mm")
            row[f"{mod}_standin"] = info.get("standin")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    assert hd_bet_available(), "HD-BET required"
    assert CKPT.is_file(), f"missing {CKPT}"
    OUT_NATIVE.mkdir(parents=True, exist_ok=True)
    OUT_PREV.mkdir(parents=True, exist_ok=True)

    before_csv = OUT_PREV / "preview_summary.csv"
    before_df = pd.read_csv(before_csv) if before_csv.is_file() else pd.DataFrame()
    before_df.to_csv(OUT_NATIVE / "before_standin_preview_summary.csv", index=False)

    ingest_rows = []
    for i in NATIVE_T1C_IDS:
        ingest_rows.append(reingest_study(i, expect_native_t1c=True))
        (OUT_NATIVE / "ingest_log.json").write_text(
            json.dumps(ingest_rows, indent=2, default=str) + "\n", encoding="utf-8"
        )

    repaired: set[int] = set()
    for i in FALSE_T1C_REPAIR_IDS:
        ingest_rows.append(repair_false_t1c_standin(i))
        repaired.add(i)
        (OUT_NATIVE / "ingest_log.json").write_text(
            json.dumps(ingest_rows, indent=2, default=str) + "\n", encoding="utf-8"
        )

    standin_rows = mark_standin_studies(set(NATIVE_T1C_IDS), repaired)
    # Refresh stand-ins for any study still missing T1c
    n_filled = fill_missing_t1c_from_t1(OUT_PROC)
    print(f"fill_missing_t1c_from_t1 → {n_filled}", flush=True)

    # Alignment overlays for native T1c
    align_rows = []
    for i in NATIVE_T1C_IDS:
        sid = _sid(i)
        png = OUT_NATIVE / "alignment" / f"{sid}_t1_edges_on_t1c.png"
        align_rows.append(alignment_overlay(OUT_PROC / sid, png))
        print(
            f"ALIGN {sid}: NCC={align_rows[-1]['ncc']:.3f} "
            f"poor={align_rows[-1]['poor_alignment']}",
            flush=True,
        )
    pd.DataFrame(align_rows).to_csv(OUT_NATIVE / "alignment_metrics.csv", index=False)

    # v3+LCC on all usable
    print("\n==== v3+LCC all usable ====", flush=True)
    dev = device()
    model = load_model_from_checkpoint(CKPT, dev)
    model.eval()
    after_rows = []
    for i in USABLE_IDS:
        sid = _sid(i)
        study = OUT_PROC / sid
        print(f"==== infer {sid} ====", flush=True)
        row, flair, lcc = infer_one(model, dev, study)
        case_dir = OUT_PREV / sid
        case_dir.mkdir(parents=True, exist_ok=True)
        overlay = case_dir / "mid_tumor_overlay.png"
        hide_et = bool(row["et_hidden"])
        save_overlay(
            flair,
            lcc,
            overlay,
            title=(
                f"{sid} WT={row['WT_ml']} "
                f"ET={'hidden' if hide_et else row['ET_ml']} "
                f"plaus={row['volume_plausibility']} "
                f"{'stand-in' if row['t1c_is_standin'] else 'native T1c'}"
            ),
            hide_et=hide_et,
        )
        row["overlay"] = str(overlay)
        # channel slice info
        man = study / "native_t1c_manifest.json"
        if man.is_file():
            payload = json.loads(man.read_text(encoding="utf-8"))
            row["native_t1c_manifest"] = payload.get("native_t1c")
            ch = payload.get("channels") or {}
            for mod in ("T1", "T1c", "T2", "FLAIR"):
                info = ch.get(mod) or {}
                row[f"{mod}_n_slices"] = info.get("n_slices")
                row[f"{mod}_thickness_mm"] = info.get("slice_thickness_mm")
                row[f"{mod}_desc"] = info.get("series_description")
        (case_dir / "meta.json").write_text(
            json.dumps(row, indent=2, default=str) + "\n", encoding="utf-8"
        )
        after_rows.append(row)
        if i in NATIVE_T1C_IDS and not before_df.empty:
            b = before_df[before_df.study_id == sid]
            before_png = None
            if len(b):
                # previous overlay path if present
                cand = case_dir / "mid_tumor_overlay_before_standin.png"
                # We saved before CSV but overlays were overwritten — copy from
                # batch2_native archive if we stored; else use before volumes only.
                before_png = None
            panel = OUT_NATIVE / "before_after" / f"{sid}_before_after.png"
            # Save a dedicated after overlay and build panel from stored before
            # snapshot if available under outputs/batch2_native_t1c/before_overlays
            before_snap = OUT_NATIVE / "before_overlays" / f"{sid}.png"
            after_snap = OUT_NATIVE / "after_overlays" / f"{sid}.png"
            after_snap.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(overlay, after_snap)
            if before_snap.is_file():
                before_after_panel(
                    before_snap,
                    after_snap,
                    panel,
                    title=f"{sid} stand-in vs native T1c",
                )
            else:
                # Still write a single-panel note image using after only + text stats
                before_after_panel(
                    None,
                    after_snap,
                    panel,
                    title=f"{sid} native T1c (before overlay unavailable)",
                )
        print(
            f"{sid}: WT={row['WT_ml']} TC={row['TC_ml']} "
            f"ET_reported={row['ET_ml_reported']} raw_cc={row['raw_component_count']} "
            f"plaus={row['volume_plausibility']} edge={row['tumor_touches_edge_slices']} "
            f"native={row['native_t1c']}",
            flush=True,
        )
        if dev.type == "mps":
            torch.mps.empty_cache()

    after_df = pd.DataFrame(after_rows)
    after_df.to_csv(OUT_PREV / "preview_summary.csv", index=False)
    after_df.to_csv(OUT_NATIVE / "after_native_preview_summary.csv", index=False)

    # Before/after volume table for native studies
    cmp_rows = []
    for i in NATIVE_T1C_IDS:
        sid = _sid(i)
        a = after_df[after_df.study_id == sid].iloc[0]
        b = before_df[before_df.study_id == sid]
        brow = b.iloc[0] if len(b) else None
        cmp_rows.append(
            {
                "study_id": sid,
                "WT_before": None if brow is None else brow.get("WT_ml"),
                "TC_before": None if brow is None else brow.get("TC_ml"),
                "ET_before": None if brow is None else brow.get("ET_ml"),
                "raw_cc_before": None if brow is None else brow.get("raw_component_count"),
                "plaus_before": None if brow is None else brow.get("volume_plausibility"),
                "WT_after": a.WT_ml,
                "TC_after": a.TC_ml,
                "ET_after": a.ET_ml_reported,
                "raw_cc_after": a.raw_component_count,
                "plaus_after": a.volume_plausibility,
                "plaus_changed_low_to_plausible": (
                    brow is not None
                    and str(brow.get("volume_plausibility")) == "low"
                    and a.volume_plausibility == "plausible"
                ),
                "ET_delta": (
                    None
                    if brow is None or a.ET_ml_reported is None
                    else float(a.ET_ml_reported) - float(brow.get("ET_ml") or 0)
                ),
            }
        )
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT_NATIVE / "native_t1c_before_after.csv", index=False)

    slices_df = channel_slice_table()
    slices_df.to_csv(OUT_NATIVE / "channel_series_table.csv", index=False)

    summary = {
        "n_native_t1c": int(sum(1 for r in ingest_rows if r.get("native_t1c"))),
        "n_standin": int((after_df["t1c_is_standin"] == True).sum()),  # noqa: E712
        "n_plausible_after": int((after_df.volume_plausibility == "plausible").sum()),
        "n_low_to_plausible": int(cmp_df["plaus_changed_low_to_plausible"].sum())
        if len(cmp_df)
        else 0,
        "n_tumor_touches_edge": int(after_df.tumor_touches_edge_slices.sum()),
        "alignment_poor": [r for r in align_rows if r["poor_alignment"]],
        "native_ids": [_sid(i) for i in NATIVE_T1C_IDS],
        "repaired_false_t1c": [_sid(i) for i in FALSE_T1C_REPAIR_IDS],
    }
    (OUT_NATIVE / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    OUT_NATIVE.mkdir(parents=True, exist_ok=True)
    # Snapshot current overlays as "before" for native studies (stand-in era).
    before_dir = OUT_NATIVE / "before_overlays"
    before_dir.mkdir(parents=True, exist_ok=True)
    for i in NATIVE_T1C_IDS:
        sid = _sid(i)
        src = OUT_PREV / sid / "mid_tumor_overlay.png"
        if src.is_file():
            shutil.copy2(src, before_dir / f"{sid}.png")
    main()
