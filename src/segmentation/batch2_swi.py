"""Convert / register SWI for batch2 second-opinion review (heuristic only)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

from config import PROJECT_ROOT
from preprocessing.dicom_loader import convert_series_to_nifti
from preprocessing.registration import rigid_register
from segmentation.batch2_scan_qc import (
    edge_overlay_montage,
    normalized_cross_correlation,
)

logger = logging.getLogger(__name__)

INVENTORY = PROJECT_ROOT / "outputs" / "batch2_inventory" / "series_inventory.csv"
BATCH2_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients_batch2"
OUT_ROOT = PROJECT_ROOT / "outputs" / "batch2_second_opinion"
LABELING = PROJECT_ROOT / "outputs" / "batch2_labeling"
NATIVE_SUMMARY = PROJECT_ROOT / "outputs" / "batch2_native_t1c" / "summary.json"


@dataclass
class SwiConvertInfo:
    study_id: str
    series_dir: str
    n_dicom_total: int
    n_magnitude: int
    n_phase: int
    image_types: list[str]
    chosen: str  # magnitude | minimum_intensity | unknown
    native_nifti: str
    registered_nifti: str | None
    ncc_to_t1c: float | None
    overlay_png: str | None
    notes: str = ""


def native_t1c_ids() -> list[str]:
    data = json.loads(NATIVE_SUMMARY.read_text(encoding="utf-8"))
    return list(data.get("native_ids", []))


def inventory_swi_rows(study_ids: Sequence[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(INVENTORY)
    swi = df[df["SeriesDescription"].astype(str).str.contains("SWI", case=False, na=False)]
    if study_ids is not None:
        swi = swi[swi["study_id"].isin(list(study_ids))]
    return swi.copy()


def _is_magnitude(ds: pydicom.Dataset) -> bool:
    it = [str(x).upper() for x in getattr(ds, "ImageType", [])]
    joined = "\\".join(it)
    if "SW_P" in joined or "PHASE" in joined:
        return False
    if "SW_M" in joined or "M_FFE" in joined or "MAGNITUDE" in joined:
        return True
    # fallback: ORIGINAL PRIMARY without phase
    return "PHASE" not in joined


def _is_phase(ds: pydicom.Dataset) -> bool:
    it = [str(x).upper() for x in getattr(ds, "ImageType", [])]
    joined = "\\".join(it)
    return "SW_P" in joined or "PHASE" in joined


def list_swi_dicoms(series_dir: Path) -> tuple[list[Path], list[Path], list[str]]:
    """Return (magnitude_files, phase_files, image_type_labels)."""
    mag: list[Path] = []
    phase: list[Path] = []
    types: set[str] = set()
    for f in sorted(series_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:  # noqa: BLE001
            continue
        it = "\\".join(str(x) for x in getattr(ds, "ImageType", []))
        types.add(it)
        if _is_phase(ds):
            phase.append(f)
        elif _is_magnitude(ds):
            mag.append(f)
    return mag, phase, sorted(types)


def t1c_path_for_study(study_id: str) -> Path:
    p = LABELING / study_id / "T1c.nii.gz"
    if p.is_file():
        return p
    p = BATCH2_ROOT / study_id / "04_registered_1mm" / "T1c_registered_1mm.nii.gz"
    if p.is_file():
        return p
    raise FileNotFoundError(f"No T1c for {study_id}")


def convert_and_register_swi(study_id: str, out_root: Path | None = None) -> SwiConvertInfo:
    """Convert magnitude SWI → NIfTI under study swi/, rigid-register to T1c."""
    out_root = out_root or OUT_ROOT
    rows = inventory_swi_rows([study_id])
    if rows.empty:
        raise FileNotFoundError(f"No SWI inventory row for {study_id}")
    row = rows.iloc[0]
    series_dir = Path(str(row["series_dir_sample"]))
    if not series_dir.is_absolute():
        series_dir = PROJECT_ROOT / series_dir
    if not series_dir.is_dir():
        raise FileNotFoundError(series_dir)

    mag, phase, types = list_swi_dicoms(series_dir)
    swi_dir = out_root / study_id / "swi"
    swi_dir.mkdir(parents=True, exist_ok=True)
    native = swi_dir / "SWI_native.nii.gz"
    chosen = "magnitude"
    notes = ""
    if mag:
        files = mag
        chosen = "magnitude"
        notes = f"Philips SWIp: used SW_M magnitude ({len(mag)} frames); skipped phase ({len(phase)})"
    elif phase:
        files = phase
        chosen = "unknown"
        notes = "Only phase frames found; converting phase (suboptimal)"
    else:
        files = sorted(p for p in series_dir.iterdir() if p.is_file())
        chosen = "unknown"
        notes = "Could not classify ImageType; converted all files"

    convert_series_to_nifti(files, native)
    # Prefer minimum intensity projection along echoes if 4D — not expected for SW_M stack
    img = nib.load(str(native))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        mip = data.min(axis=-1)
        native_mip = swi_dir / "SWI_native_minIP.nii.gz"
        nib.save(nib.Nifti1Image(mip.astype(np.float32), img.affine, img.header), str(native_mip))
        native = native_mip
        chosen = "minimum_intensity"
        notes += "; collapsed 4D with minIP"

    t1c = t1c_path_for_study(study_id)
    reg = swi_dir / "SWI_registered_to_t1c.nii.gz"
    tfm = swi_dir / "SWI_to_t1c.tfm"
    rigid_register(native, t1c, reg, transform_path=tfm)

    t1c_data = np.asanyarray(nib.load(str(t1c)).dataobj).astype(np.float32)
    swi_data = np.asanyarray(nib.load(str(reg)).dataobj).astype(np.float32)
    ncc = normalized_cross_correlation(t1c_data, swi_data)
    overlay = swi_dir / "SWI_edges_on_T1c.png"
    edge_overlay_montage(t1c_data, swi_data, overlay, title=f"{study_id} SWI→T1c NCC={ncc:.3f}")

    info = SwiConvertInfo(
        study_id=study_id,
        series_dir=str(series_dir),
        n_dicom_total=len(mag) + len(phase),
        n_magnitude=len(mag),
        n_phase=len(phase),
        image_types=types,
        chosen=chosen,
        native_nifti=str(native),
        registered_nifti=str(reg),
        ncc_to_t1c=float(ncc),
        overlay_png=str(overlay),
        notes=notes,
    )
    (swi_dir / "swi_convert_meta.json").write_text(
        json.dumps(asdict(info), indent=2) + "\n", encoding="utf-8"
    )
    return info


def swi_dark_focus_inside_contour(
    swi: np.ndarray,
    contour: np.ndarray,
    brain_mask: np.ndarray | None = None,
    *,
    low_pct: float = 5.0,
) -> dict[str, Any]:
    """Heuristic: compare lowest ``low_pct`` SWI intensity inside contour vs brain.

    Returns label yes/no/not_assessable. Dark SWI ≠ diagnosis.
    """
    swi = np.asarray(swi, dtype=np.float32)
    c = np.asarray(contour, dtype=bool)
    if brain_mask is None:
        brain = swi > 0
    else:
        brain = np.asarray(brain_mask, dtype=bool) & (swi > 0)
    surround = brain & ~c
    if not np.any(c) or not np.any(surround):
        return {
            "label": "not_assessable",
            "reason": "empty contour or empty surround",
            "inside_low": None,
            "surround_low": None,
            "ratio": None,
        }
    inside_vals = swi[c]
    surround_vals = swi[surround]
    # ignore exact zeros (masked)
    inside_vals = inside_vals[inside_vals > 0]
    surround_vals = surround_vals[surround_vals > 0]
    if inside_vals.size < 10 or surround_vals.size < 10:
        return {
            "label": "not_assessable",
            "reason": "too few positive voxels",
            "inside_low": None,
            "surround_low": None,
            "ratio": None,
        }
    inside_low = float(np.percentile(inside_vals, low_pct))
    surround_low = float(np.percentile(surround_vals, low_pct))
    ratio = inside_low / surround_low if surround_low > 1e-6 else float("inf")
    # Dark focus if contour's darkest tail is substantially darker than brain's
    yes = ratio < 0.7
    return {
        "label": "yes" if yes else "no",
        "reason": "heuristic lowest-5pct SWI inside contour vs surround",
        "inside_low": inside_low,
        "surround_low": surround_low,
        "ratio": float(ratio),
    }


def assess_dark_focus_study(study_id: str, out_root: Path | None = None) -> dict[str, Any]:
    out_root = out_root or OUT_ROOT
    swi_path = out_root / study_id / "swi" / "SWI_registered_to_t1c.nii.gz"
    if not swi_path.is_file():
        return {
            "study_id": study_id,
            "swi_dark_focus_inside_either_contour": "not_assessable",
            "detail": "no registered SWI",
        }
    swi = np.asanyarray(nib.load(str(swi_path)).dataobj).astype(np.float32)
    v3_path = LABELING / study_id / "v3_draft_label.nii.gz"
    hd_path = out_root / study_id / "hdglio_draft_label.nii.gz"
    either = np.zeros(swi.shape, dtype=bool)
    detail: dict[str, Any] = {}
    for name, path in (("v3", v3_path), ("hdglio", hd_path)):
        if not path.is_file():
            detail[name] = {"label": "not_assessable", "reason": "missing label"}
            continue
        lab = np.rint(np.asanyarray(nib.load(str(path)).dataobj))
        if lab.shape != swi.shape:
            detail[name] = {"label": "not_assessable", "reason": "shape mismatch"}
            continue
        m = lab > 0
        either |= m
        detail[name] = swi_dark_focus_inside_contour(swi, m)
    if not np.any(either):
        label = "not_assessable"
    else:
        combined = swi_dark_focus_inside_contour(swi, either)
        detail["either"] = combined
        label = combined["label"]
    return {
        "study_id": study_id,
        "swi_dark_focus_inside_either_contour": label,
        "detail": detail,
        "disclaimer": "Heuristic only — dark SWI inside a contour is a flag for review, not a diagnosis.",
    }
