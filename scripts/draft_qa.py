#!/usr/bin/env python3
"""QA report for unverified draft masks in batch2 labeling packages.

For every study with ``claude_draft_label.nii.gz`` (or ``--draft-name``), compute
volumes, connected components, brain-edge / midline leak fractions, Dice and
centroid distance vs v3 and HD-GLIO drafts. Flags thin midline-touching
components as possible vessels.

Drafts are **unverified**. Does not train or modify checkpoints / BraTS data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PROJECT_ROOT  # noqa: E402
from segmentation.batch2_label_qc import find_brain_mask  # noqa: E402
from segmentation.hdglio_labels import brats_regions_from_classic  # noqa: E402

LABELING_ROOT = PROJECT_ROOT / "outputs" / "batch2_labeling"
HDGLIO_ROOT = PROJECT_ROOT / "outputs" / "batch2_second_opinion"
OUT_ROOT = PROJECT_ROOT / "outputs" / "batch2_draft_qa"
DEFAULT_DRAFT = "claude_draft_label.nii.gz"
EDGE_MM = 5.0
VESSEL_THICKNESS_MM = 4.0


def _spacing(img: nib.Nifti1Image) -> tuple[float, float, float]:
    return tuple(float(z) for z in img.header.get_zooms()[:3])


def _voxel_ml(spacing: tuple[float, float, float]) -> float:
    return float(np.prod(spacing) / 1000.0)


def _centroid_mm(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    if not np.any(mask):
        return None
    c = ndi.center_of_mass(mask)
    return tuple(float(c[i] * spacing[i]) for i in range(3))  # type: ignore[return-value]


def _dist_mm(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = float(np.count_nonzero(a & b))
    denom = float(np.count_nonzero(a) + np.count_nonzero(b))
    if denom == 0:
        return float("nan")
    return 2.0 * inter / denom


def midline_strip(shape: tuple[int, ...], width_vox: int = 5) -> np.ndarray:
    """Central sagittal strip (axis 0 ≈ L–R on these packages)."""
    m = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    lo = max(0, mid - width_vox // 2)
    hi = min(shape[0], lo + width_vox)
    m[lo:hi, :, :] = True
    return m


def edge_band(brain: np.ndarray, spacing: tuple[float, float, float], mm: float = EDGE_MM) -> np.ndarray:
    """Voxels inside brain within ``mm`` of the brain exterior."""
    if not np.any(brain):
        return np.zeros_like(brain, dtype=bool)
    iters = max(1, int(np.ceil(mm / max(min(spacing), 1e-6))))
    eroded = ndi.binary_erosion(brain, iterations=iters)
    return brain & ~eroded


def component_thickness_mm(
    mask: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    """Rough thickness: 2 × min PCA eigenvalue sqrt (mm), or distance-transform peak."""
    if not np.any(mask):
        return 0.0
    # Distance transform in voxels → mm via mean spacing
    dt = ndi.distance_transform_edt(mask)
    peak = float(dt.max())
    mean_sp = float(np.mean(spacing))
    return 2.0 * peak * mean_sp


def qa_one(pkg: Path, draft_name: str) -> dict:
    study_id = pkg.name
    draft_path = pkg / draft_name
    t1c_path = pkg / "T1c.nii.gz"
    t1c_img = nib.load(str(t1c_path))
    spacing = _spacing(t1c_img)
    vox_ml = _voxel_ml(spacing)
    draft = np.rint(np.asanyarray(nib.load(str(draft_path)).dataobj)).astype(np.int16)
    if draft.shape != t1c_img.shape:
        raise ValueError(f"{study_id}: draft shape {draft.shape} ≠ T1c {t1c_img.shape}")

    regs = brats_regions_from_classic(draft)
    brain, brain_note = find_brain_mask(
        pkg, target_shape=t1c_img.shape, target_affine=np.asarray(t1c_img.affine)
    )
    if brain is None:
        brain = draft != 0
        brain_note = "fallback draft>0 (no brain mask)"

    band = edge_band(brain, spacing, EDGE_MM)
    mid = midline_strip(draft.shape)
    wt = regs["WT"]
    n_wt = int(wt.sum())
    frac_edge = float((wt & band).sum() / max(n_wt, 1))
    frac_mid = float((wt & mid).sum() / max(n_wt, 1))

    labeled, nlab = ndi.label(wt)
    vessel_flags: list[str] = []
    for lab in range(1, nlab + 1):
        m = labeled == lab
        if not np.any(m & mid):
            continue
        thick = component_thickness_mm(m, spacing)
        if thick < VESSEL_THICKNESS_MM:
            vessel_flags.append(f"cc{lab}_thick={thick:.2f}mm")

    # v3 / HD-GLIO
    v3_path = pkg / "v3_draft_label.nii.gz"
    hd_path = HDGLIO_ROOT / study_id / "hdglio_draft_label.nii.gz"
    dice_v3 = {k: float("nan") for k in ("ET", "TC", "WT")}
    dice_hd = {k: float("nan") for k in ("ET", "TC", "WT")}
    cent_draft = _centroid_mm(regs["ET"] if np.any(regs["ET"]) else wt, spacing)
    cent_v3 = None
    cent_hd = None
    if v3_path.is_file():
        v3 = np.rint(np.asanyarray(nib.load(str(v3_path)).dataobj)).astype(np.int16)
        if v3.shape == draft.shape:
            rv = brats_regions_from_classic(v3)
            for k in dice_v3:
                dice_v3[k] = _dice(regs[k], rv[k])
            cent_v3 = _centroid_mm(rv["ET"] if np.any(rv["ET"]) else rv["WT"], spacing)
    if hd_path.is_file():
        hd = np.rint(np.asanyarray(nib.load(str(hd_path)).dataobj)).astype(np.int16)
        if hd.shape == draft.shape:
            rh = brats_regions_from_classic(hd)
            for k in dice_hd:
                dice_hd[k] = _dice(regs[k], rh[k])
            cent_hd = _centroid_mm(rh["ET"] if np.any(rh["ET"]) else rh["WT"], spacing)

    return {
        "study_id": study_id,
        "draft_name": draft_name,
        "draft_status": "draft_unverified",
        "WT_ml": float(regs["WT"].sum() * vox_ml),
        "TC_ml": float(regs["TC"].sum() * vox_ml),
        "ET_ml": float(regs["ET"].sum() * vox_ml),
        "n_cc_WT": int(nlab),
        "frac_within_5mm_brain_edge": frac_edge,
        "frac_midline_strip": frac_mid,
        "possible_vessel_flags": ";".join(vessel_flags) if vessel_flags else "",
        "dice_ET_v3": dice_v3["ET"],
        "dice_TC_v3": dice_v3["TC"],
        "dice_WT_v3": dice_v3["WT"],
        "dice_ET_hdglio": dice_hd["ET"],
        "dice_TC_hdglio": dice_hd["TC"],
        "dice_WT_hdglio": dice_hd["WT"],
        "centroid_dist_v3_mm": _dist_mm(cent_draft, cent_v3),
        "centroid_dist_hdglio_mm": _dist_mm(cent_draft, cent_hd),
        "brain_mask_note": brain_note,
    }


def write_summary(rows: list[dict], out_root: Path) -> Path:
    lines = [
        "# Batch2 draft QA (unverified drafts)",
        "",
        "**All masks here are draft / unverified — not clinical reference.**",
        "",
        f"n_studies={len(rows)}. Edge band = {EDGE_MM} mm inside brain mask; "
        f"midline strip = central ~5 voxels; thin midline CC (<{VESSEL_THICKNESS_MM} mm) "
        "→ `possible vessel`.",
        "",
        "| study | WT mL | TC mL | ET mL | n_CC | edge% | mid% | vessel? | "
        "Dice WT v3 | Dice WT HD | Δcent v3 mm | Δcent HD mm |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['study_id']} | {r['WT_ml']:.2f} | {r['TC_ml']:.2f} | {r['ET_ml']:.2f} | "
            f"{r['n_cc_WT']} | {100 * r['frac_within_5mm_brain_edge']:.1f} | "
            f"{100 * r['frac_midline_strip']:.1f} | "
            f"{r['possible_vessel_flags'] or '—'} | "
            f"{r['dice_WT_v3']:.3f} | {r['dice_WT_hdglio']:.3f} | "
            f"{r['centroid_dist_v3_mm']:.1f} | {r['centroid_dist_hdglio_mm']:.1f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Overlap with v3 / HD-GLIO is descriptive only; none of the three are verified.",
        "- High edge% or midline% suggests possible extracranial / sinus / falx leak.",
        "",
    ]
    path = out_root / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeling-root", type=Path, default=LABELING_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--draft-name", type=str, default=DEFAULT_DRAFT)
    args = parser.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)
    pkgs = sorted(
        p
        for p in args.labeling_root.iterdir()
        if p.is_dir() and (p / args.draft_name).is_file() and (p / "T1c.nii.gz").is_file()
    )
    if not pkgs:
        print(f"No packages with {args.draft_name} under {args.labeling_root}")
        # Still write empty stubs so downstream knows QA ran
        (args.out_root / "per_case.csv").write_text("study_id\n", encoding="utf-8")
        (args.out_root / "SUMMARY.md").write_text(
            "# Batch2 draft QA\n\nNo drafts found.\n", encoding="utf-8"
        )
        return 1

    rows = []
    for pkg in pkgs:
        rows.append(qa_one(pkg, args.draft_name))

    csv_path = args.out_root / "per_case.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = write_summary(rows, args.out_root)
    (args.out_root / "qa_meta.json").write_text(
        json.dumps(
            {
                "draft_name": args.draft_name,
                "n_studies": len(rows),
                "disclaimer": "draft_unverified",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path} and {summary} ({len(rows)} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
