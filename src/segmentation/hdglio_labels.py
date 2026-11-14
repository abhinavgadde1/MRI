"""HD-GLIO output → classic BraTS labels and agreement metrics.

HD-GLIO (Kickingereder et al., Lancet Oncol. 2019) predicts:
  - label 1 = contrast-enhancing tumor
  - label 2 = non-enhancing T2/FLAIR signal abnormality

We map to classic BraTS for comparison with v3 drafts:
  - HD-GLIO 1 → BraTS 4 (ET)
  - HD-GLIO 2 → BraTS 2 (edema / non-enhancing)
  - There is no separate necrotic (1) class from HD-GLIO; TC ≡ ET for this mapping.

All outputs here are **model drafts**, not verified reference masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

DRAFT_DISCLAIMER = (
    "Model drafts only — neither HD-GLIO nor v3 outlines are verified reference masks."
)


def voxel_volume_ml_from_binary(
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Volume in mL from a binary mask and voxel spacing (mm)."""
    n = int(np.count_nonzero(mask))
    return float(n) * float(spacing[0]) * float(spacing[1]) * float(spacing[2]) / 1000.0


def hdglio_to_brats_classic(seg: np.ndarray) -> np.ndarray:
    """Map HD-GLIO labels {0,1,2} → classic BraTS {0,2,4}."""
    s = np.rint(np.asarray(seg)).astype(np.int16)
    out = np.zeros_like(s, dtype=np.int16)
    out[s == 1] = 4  # enhancing
    out[s == 2] = 2  # non-enhancing / edema
    # Preserve unexpected labels as 0 with a note by caller
    return out


def brats_regions_from_classic(label: np.ndarray) -> dict[str, np.ndarray]:
    lbl = np.asarray(label)
    necrotic = lbl == 1
    edema = lbl == 2
    enhancing = (lbl == 3) | (lbl == 4)
    return {
        "ET": enhancing,
        "TC": necrotic | enhancing,
        "WT": necrotic | edema | enhancing,
    }


def assert_same_grid(
    a: nib.Nifti1Image,
    b: nib.Nifti1Image,
    *,
    atol: float = 1e-5,
) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if not np.allclose(a.affine, b.affine, atol=atol, rtol=0.0):
        raise ValueError("affine mismatch")


def resample_seg_to_grid(
    seg_img: nib.Nifti1Image,
    target_img: nib.Nifti1Image,
) -> np.ndarray:
    """Nearest-neighbor resample segmentation to target grid via affines."""
    if seg_img.shape == target_img.shape and np.allclose(
        seg_img.affine, target_img.affine, atol=1e-3
    ):
        return np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int16)
    transform = np.linalg.inv(target_img.affine) @ seg_img.affine
    inv = np.linalg.inv(transform)
    data = np.asanyarray(seg_img.dataobj).astype(np.float32)
    out = ndi.affine_transform(
        data,
        matrix=inv[:3, :3],
        offset=inv[:3, 3],
        output_shape=target_img.shape,
        order=0,
    )
    return np.rint(out).astype(np.int16)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.count_nonzero(a & b)
    denom = np.count_nonzero(a) + np.count_nonzero(b)
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def centroid_mm(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return None
    coords = np.argwhere(m).astype(np.float64)
    c = coords.mean(axis=0)
    return float(c[0] * spacing[0]), float(c[1] * spacing[1]), float(c[2] * spacing[2])


def largest_cc(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return lab == keep


@dataclass
class AgreementRow:
    study_id: str
    region: str
    dice: float
    v3_ml: float
    hdglio_ml: float
    centroid_dist_mm: float | None
    lcc_overlap: bool | None
    agreement_class: str


def classify_agreement(wt_dice: float, *, v3_empty: bool, hd_empty: bool) -> str:
    if v3_empty and hd_empty:
        return "both_empty"
    if v3_empty or hd_empty:
        return "one_finds_no_tumor"
    if wt_dice > 0.7:
        return "both_agree"
    if wt_dice >= 0.3:
        return "partial"
    return "disagree"


def compare_v3_hdglio(
    study_id: str,
    v3_label: np.ndarray,
    hd_label_classic: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[AgreementRow]:
    v3 = brats_regions_from_classic(v3_label)
    hd = brats_regions_from_classic(hd_label_classic)
    rows: list[AgreementRow] = []
    wt_dice = dice(v3["WT"], hd["WT"])
    agr = classify_agreement(
        wt_dice,
        v3_empty=not np.any(v3["WT"]),
        hd_empty=not np.any(hd["WT"]),
    )
    for region in ("ET", "TC", "WT"):
        d = dice(v3[region], hd[region])
        v_ml = voxel_volume_ml_from_binary(v3[region], spacing)
        h_ml = voxel_volume_ml_from_binary(hd[region], spacing)
        c0 = centroid_mm(v3[region], spacing)
        c1 = centroid_mm(hd[region], spacing)
        dist = None
        if c0 is not None and c1 is not None:
            dist = float(np.linalg.norm(np.array(c0) - np.array(c1)))
        l0 = largest_cc(v3[region])
        l1 = largest_cc(hd[region])
        overlap = bool(np.any(l0 & l1)) if (np.any(l0) and np.any(l1)) else None
        rows.append(
            AgreementRow(
                study_id=study_id,
                region=region,
                dice=d,
                v3_ml=v_ml,
                hdglio_ml=h_ml,
                centroid_dist_mm=dist,
                lcc_overlap=overlap,
                agreement_class=agr if region == "WT" else "",
            )
        )
    return rows


def save_hdglio_draft_on_t1c_grid(
    hdglio_seg_path: Path,
    t1c_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Convert HD-GLIO seg to BraTS classic on the exact T1c grid."""
    seg_img = nib.load(str(hdglio_seg_path))
    t1c_img = nib.load(str(t1c_path))
    resampled = False
    if seg_img.shape != t1c_img.shape or not np.allclose(
        seg_img.affine, t1c_img.affine, atol=1e-3
    ):
        data = resample_seg_to_grid(seg_img, t1c_img)
        resampled = True
    else:
        data = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int16)
    classic = hdglio_to_brats_classic(data)
    out_img = nib.Nifti1Image(classic, t1c_img.affine, t1c_img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    # verify
    check = nib.load(str(out_path))
    assert_same_grid(check, t1c_img)
    return {
        "out_path": str(out_path),
        "resampled_nearest": resampled,
        "labels": sorted(int(x) for x in np.unique(classic)),
        "disclaimer": DRAFT_DISCLAIMER,
    }
