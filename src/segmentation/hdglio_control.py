"""BraTS heldout control test helpers for the Mac-patched HD-GLIO install."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import torch

from config import PROJECT_ROOT
from segmentation.hdglio_labels import (
    brats_regions_from_classic,
    hdglio_to_brats_classic,
)

HELDOUT_LIST = PROJECT_ROOT / "splits" / "heldout.txt"
BRATS_ROOT = PROJECT_ROOT / "data" / "processed" / "brats_nifti"
V3_LCC_METRICS = PROJECT_ROOT / "checkpoints" / "brats_eval_heldout_v3" / "lcc" / "metrics.csv"
CONTROL_OUT = PROJECT_ROOT / "outputs" / "batch2_second_opinion" / "control_test"

# Published HD-GLIO (Kickingereder et al., Lancet Oncol. 2019) CE Dice ~0.91
# on curated multi-center data — used only as a rough reference bar.
PUBLISHED_CE_DICE_REF = 0.91


@dataclass
class HeldoutCase:
    study_id: str
    wt_ml: float
    role: str  # small | median | large | random


def wt_volume_ml(mask_path: Path) -> float:
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
    n = int(np.count_nonzero(data > 0))
    return float(n) * float(np.prod(spacing)) / 1000.0


def select_heldout_control_cases(
    *,
    n: int = 5,
    seed: int = 42,
    heldout_path: Path | None = None,
    brats_root: Path | None = None,
) -> list[HeldoutCase]:
    """Pick n heldout cases with seed; force one small, one median, one large WT."""
    path = heldout_path or HELDOUT_LIST
    root = brats_root or BRATS_ROOT
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    vols: list[tuple[str, float]] = []
    for sid in ids:
        mp = root / sid / "mask.nii.gz"
        if not mp.is_file():
            continue
        vols.append((sid, wt_volume_ml(mp)))
    if len(vols) < n:
        raise RuntimeError(f"Need ≥{n} heldout cases with masks, found {len(vols)}")
    vols_sorted = sorted(vols, key=lambda x: x[1])
    small = vols_sorted[0]
    large = vols_sorted[-1]
    median = vols_sorted[len(vols_sorted) // 2]
    forced = {small[0], median[0], large[0]}
    remaining = [v for v in vols if v[0] not in forced]
    rng = np.random.default_rng(seed)
    n_extra = n - 3
    if n_extra < 0:
        raise ValueError("n must be ≥ 3 to include small/median/large")
    if n_extra > len(remaining):
        raise RuntimeError("not enough remaining heldout cases")
    extra_idx = rng.choice(len(remaining), size=n_extra, replace=False)
    extras = [remaining[int(i)] for i in sorted(extra_idx)]
    out = [
        HeldoutCase(small[0], small[1], "small"),
        HeldoutCase(median[0], median[1], "median"),
        HeldoutCase(large[0], large[1], "large"),
    ]
    for sid, ml in extras:
        out.append(HeldoutCase(sid, ml, "random"))
    return out


def has_identity_affine(img: nib.Nifti1Image, *, atol: float = 1e-5) -> bool:
    return bool(np.allclose(img.affine, np.eye(4), atol=atol))


def regions_from_gt_mask(mask: np.ndarray) -> dict[str, np.ndarray]:
    """BraTS HDF5 restack encoding: 1 necrotic, 2 edema, 3 enhancing."""
    return brats_regions_from_classic(mask)


def dice_hd95_regions(
    pred_classic: np.ndarray,
    gt_classic: np.ndarray,
) -> dict[str, float]:
    """Dice + HD95 for ET/TC/WT (empty-GT → NaN)."""
    pred_r = brats_regions_from_classic(pred_classic)
    gt_r = brats_regions_from_classic(gt_classic)
    order = ("ET", "TC", "WT")
    names = ("enhancing_tumor", "tumor_core", "whole_tumor")
    pred_ch = np.stack([pred_r[k].astype(np.float32) for k in order], axis=0)
    gt_ch = np.stack([gt_r[k].astype(np.float32) for k in order], axis=0)
    pred_t = torch.from_numpy(pred_ch)
    gt_t = torch.from_numpy(gt_ch)
    dice_m = DiceMetric(include_background=True, reduction="none")
    hd_m = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction="none"
    )
    dice_m(y_pred=[pred_t], y=[gt_t])
    hd_m(y_pred=[pred_t], y=[gt_t])
    dice = dice_m.aggregate()
    hd = hd_m.aggregate()
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        if float(gt_t[i].sum().item()) <= 0.0:
            out[f"dice_{name}"] = float("nan")
            out[f"hd95_{name}"] = float("nan")
            continue
        out[f"dice_{name}"] = float(dice[0, i].item())
        out[f"hd95_{name}"] = float(hd[0, i].item())
    return out


def orientation_sanity(
    pred_img: nib.Nifti1Image,
    gt_img: nib.Nifti1Image,
    t1c_img: nib.Nifti1Image,
    out_png: Path,
) -> dict[str, Any]:
    """Check grid match and write a mid-slice overlay (not an assumption)."""
    shape_ok = pred_img.shape == gt_img.shape == t1c_img.shape
    aff_ok = bool(
        np.allclose(pred_img.affine, gt_img.affine, atol=1e-3)
        and np.allclose(pred_img.affine, t1c_img.affine, atol=1e-3)
    )
    ax_pred = "".join(nib.aff2axcodes(pred_img.affine))
    ax_gt = "".join(nib.aff2axcodes(gt_img.affine))
    pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int16)
    gt = np.rint(np.asanyarray(gt_img.dataobj)).astype(np.int16)
    t1c = np.asanyarray(t1c_img.dataobj).astype(np.float32)
    z = t1c.shape[2] // 2
    # Prefer a slice with GT WT if mid is empty
    gt_wt = gt > 0
    if np.any(gt_wt):
        zs = np.where(gt_wt.any(axis=(0, 1)))[0]
        z = int(zs[len(zs) // 2])
    base = t1c[:, :, z]
    p2, p98 = np.percentile(base, (2, 98))
    norm = np.zeros_like(base) if p98 <= p2 else np.clip((base - p2) / (p98 - p2), 0, 1)
    rgb = np.stack([norm, norm, norm], axis=-1)
    pred_e = (pred > 0)[:, :, z]
    gt_e = (gt > 0)[:, :, z]
    # edges
    from scipy import ndimage as ndi

    def edges(m):
        m = np.asarray(m, dtype=bool)
        return m & ~ndi.binary_erosion(m) if np.any(m) else m

    rgb[edges(gt_e)] = (0.15, 0.85, 0.25)
    rgb[edges(pred_e)] = (0.95, 0.25, 0.2)
    overlap = int(np.count_nonzero(pred_e & gt_e))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(np.rot90(rgb), origin="lower")
    ax.set_title(
        f"z={z} green=GT WT red=HD-GLIO\n"
        f"shape_ok={shape_ok} aff_ok={aff_ok} ax={ax_pred}/{ax_gt}"
    )
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {
        "shape_ok": shape_ok,
        "affine_ok": aff_ok,
        "axcodes_pred": ax_pred,
        "axcodes_gt": ax_gt,
        "identity_affine_input": has_identity_affine(t1c_img),
        "mid_slice_overlap_voxels": overlap,
        "overlay_png": str(out_png),
    }


def load_v3_lcc_metrics(path: Path | None = None) -> dict[str, dict[str, float]]:
    import pandas as pd

    p = path or V3_LCC_METRICS
    df = pd.read_csv(p)
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        sid = str(row["subject_id"])
        out[sid] = {c: float(row[c]) for c in df.columns if c != "subject_id"}
    return out


def cases_to_jsonable(cases: list[HeldoutCase]) -> list[dict[str, Any]]:
    return [asdict(c) for c in cases]
