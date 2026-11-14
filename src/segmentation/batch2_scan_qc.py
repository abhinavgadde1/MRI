"""Scan QC for batch2 studies (read-only).

Checks orientation, FOV/coverage, spacing, registration NCC, intensity
sanity, T1c contrast, and SWI presence. Writes only to
``outputs/batch2_second_opinion/`` when invoked by the CLI script.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy import ndimage as ndi

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

BATCH2_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients_batch2"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "batch2_second_opinion"
NCC_WARN_THRESHOLD = 0.45
BRAIN_EXTENT_WARN_MM = 80.0  # SI extent below this → likely truncated
HF_MOTION_WARN = 0.35  # heuristic high-freq energy ratio


@dataclass
class SeriesQC:
    modality: str
    path: str
    shape: tuple[int, ...]
    spacing_mm: tuple[float, float, float]
    slice_thickness_mm: float | None
    slice_gap_mm: float | None
    axcodes: str
    is_axial: bool
    fov_mm: tuple[float, float, float]
    brain_extent_si_mm: float | None
    p1: float | None = None
    p50: float | None = None
    p99: float | None = None
    empty_slice_frac: float | None = None
    saturated_slice_frac: float | None = None
    hf_energy_score: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class StudyScanQC:
    study_id: str
    usable: bool
    native_t1c: bool
    has_swi: bool
    swi_registered_to_t1c: bool | None
    series: list[SeriesQC] = field(default_factory=list)
    ncc_t2_t1: float | None = None
    ncc_flair_t1: float | None = None
    ncc_t1c_t1: float | None = None
    t1c_brighter_than_t1: bool | None = None
    t1c_t1_top1pct_ratio: float | None = None
    flags: list[str] = field(default_factory=list)
    verdict: str = "usable_with_caution"
    verdict_reasons: list[str] = field(default_factory=list)

    def to_flat_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "study_id": self.study_id,
            "usable": self.usable,
            "native_t1c": self.native_t1c,
            "has_swi": self.has_swi,
            "swi_registered_to_t1c": self.swi_registered_to_t1c,
            "ncc_t2_t1": self.ncc_t2_t1,
            "ncc_flair_t1": self.ncc_flair_t1,
            "ncc_t1c_t1": self.ncc_t1c_t1,
            "t1c_brighter_than_t1": self.t1c_brighter_than_t1,
            "t1c_t1_top1pct_ratio": self.t1c_t1_top1pct_ratio,
            "verdict": self.verdict,
            "verdict_reasons": "; ".join(self.verdict_reasons),
            "flags": "; ".join(self.flags),
        }
        for s in self.series:
            prefix = s.modality
            row[f"{prefix}_shape"] = "x".join(map(str, s.shape))
            row[f"{prefix}_spacing_mm"] = ",".join(f"{x:.3f}" for x in s.spacing_mm)
            row[f"{prefix}_thickness_mm"] = s.slice_thickness_mm
            row[f"{prefix}_gap_mm"] = s.slice_gap_mm
            row[f"{prefix}_axcodes"] = s.axcodes
            row[f"{prefix}_axial"] = s.is_axial
            row[f"{prefix}_fov_mm"] = ",".join(f"{x:.1f}" for x in s.fov_mm)
            row[f"{prefix}_brain_extent_si_mm"] = s.brain_extent_si_mm
            row[f"{prefix}_hf_energy"] = s.hf_energy_score
            row[f"{prefix}_flags"] = "; ".join(s.flags)
        return row


def usable_study_ids(
    inventory_csv: Path | None = None,
) -> list[str]:
    import pandas as pd

    path = inventory_csv or (
        PROJECT_ROOT / "outputs" / "batch2_inventory" / "study_qc_table.csv"
    )
    df = pd.read_csv(path)
    # usable_for_pipeline column is boolean-ish
    mask = df["usable_for_pipeline"].astype(str).str.lower().isin({"true", "1", "yes"})
    return sorted(df.loc[mask, "study_id"].astype(str).tolist())


def native_t1c_ids() -> set[str]:
    summary = PROJECT_ROOT / "outputs" / "batch2_native_t1c" / "summary.json"
    if summary.is_file():
        return set(json.loads(summary.read_text(encoding="utf-8")).get("native_ids", []))
    return set()


def _spacing(img: nib.Nifti1Image) -> tuple[float, float, float]:
    z = img.header.get_zooms()[:3]
    return float(z[0]), float(z[1]), float(z[2])


def _load_sidecar(nifti_path: Path) -> dict[str, Any]:
    for cand in (nifti_path.with_suffix("").with_suffix(".json"), Path(str(nifti_path) + ".json")):
        # .nii.gz → try stem.json next to file
        pass
    json_path = Path(str(nifti_path).replace(".nii.gz", ".json"))
    if json_path.is_file():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def normalized_cross_correlation(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Brain-masked NCC (Pearson) between two volumes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            # nearest zoom
            factors = [a.shape[i] / m.shape[i] for i in range(3)]
            m = ndi.zoom(m.astype(np.float32), factors, order=0) > 0.5
        a = a[m]
        b = b[m]
    else:
        a = a.ravel()
        b = b.ravel()
    if a.size < 10:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def high_frequency_energy_score(volume: np.ndarray) -> float:
    """Heuristic: fraction of spectral energy in high frequencies (motion/ghosting proxy)."""
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3 or vol.size == 0:
        return float("nan")
    # Use mid-slice for speed
    sl = vol[:, :, vol.shape[2] // 2]
    if np.allclose(sl, 0):
        return 0.0
    f = np.fft.fftshift(np.fft.fft2(sl))
    mag = np.abs(f) ** 2
    cy, cx = np.array(mag.shape) // 2
    yy, xx = np.ogrid[: mag.shape[0], : mag.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max() + 1e-6
    high = mag[r > 0.4 * rmax].sum()
    total = mag.sum() + 1e-12
    return float(high / total)


def brain_extent_si_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0.0
    zs = np.where(np.any(m, axis=(0, 1)))[0]
    if zs.size == 0:
        return 0.0
    return float((zs.max() - zs.min() + 1) * spacing[2])


def top_percentile_mean(volume: np.ndarray, mask: np.ndarray, pct: float = 99.0) -> float:
    v = np.asarray(volume, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if m.shape != v.shape:
        factors = [v.shape[i] / m.shape[i] for i in range(3)]
        m = ndi.zoom(m.astype(np.float32), factors, order=0) > 0.5
    vals = v[m]
    if vals.size == 0:
        return float("nan")
    thr = np.percentile(vals, pct)
    sel = vals[vals >= thr]
    return float(sel.mean()) if sel.size else float("nan")


def _series_qc_from_nifti(
    modality: str,
    path: Path,
    *,
    brain_mask: np.ndarray | None = None,
) -> SeriesQC:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float64)
    spacing = _spacing(img)
    ax = "".join(aff2axcodes(img.affine))
    # Axial if third code is S/I (RAS/LPS variants)
    is_axial = ax[2] in {"S", "I"}
    fov = tuple(float(data.shape[i] * spacing[i]) for i in range(3))
    meta = _load_sidecar(path)
    thickness = meta.get("slice_thickness") or meta.get("SliceThickness")
    gap = meta.get("spacing_between_slices") or meta.get("SpacingBetweenSlices")
    try:
        thickness_f = float(thickness) if thickness is not None else float(spacing[2])
    except (TypeError, ValueError):
        thickness_f = float(spacing[2])
    try:
        gap_f = float(gap) if gap is not None else None
    except (TypeError, ValueError):
        gap_f = None

    flags: list[str] = []
    if not is_axial:
        flags.append(f"non-axial orientation ({ax})")

    extent = None
    if brain_mask is not None:
        extent = brain_extent_si_mm(brain_mask, spacing)
        if extent < BRAIN_EXTENT_WARN_MM:
            flags.append(f"truncated brain SI extent ({extent:.1f} mm)")

    # Intensity percentiles on brain or nonzero
    if brain_mask is not None and brain_mask.shape == data.shape:
        vals = data[brain_mask > 0]
    else:
        vals = data[data != 0]
    if vals.size:
        p1, p50, p99 = (float(np.percentile(vals, q)) for q in (1, 50, 99))
    else:
        p1 = p50 = p99 = float("nan")
        flags.append("no nonzero voxels")

    # Empty / saturated slices along z
    empty = 0
    sat = 0
    nz = max(1, data.shape[2])
    finite = data[np.isfinite(data)]
    sat_thr = float(np.percentile(finite, 99.9)) if finite.size else 0.0
    for z in range(data.shape[2]):
        sl = data[:, :, z]
        if np.count_nonzero(sl) < 10:
            empty += 1
        if sat_thr > 0 and np.mean(sl >= sat_thr) > 0.05:
            sat += 1
    empty_frac = empty / nz
    sat_frac = sat / nz
    if empty_frac > 0.1:
        flags.append(f"empty slices {empty_frac:.0%}")
    if sat_frac > 0.05:
        flags.append(f"saturated slices {sat_frac:.0%}")

    hf = high_frequency_energy_score(data)
    if hf == hf and hf > HF_MOTION_WARN:
        flags.append(f"high hf_energy={hf:.3f} (motion/ghosting heuristic)")

    return SeriesQC(
        modality=modality,
        path=str(path),
        shape=tuple(int(x) for x in data.shape),
        spacing_mm=spacing,
        slice_thickness_mm=thickness_f,
        slice_gap_mm=gap_f,
        axcodes=ax,
        is_axial=is_axial,
        fov_mm=fov,  # type: ignore[arg-type]
        brain_extent_si_mm=extent,
        p1=p1,
        p50=p50,
        p99=p99,
        empty_slice_frac=empty_frac,
        saturated_slice_frac=sat_frac,
        hf_energy_score=hf,
        flags=flags,
    )


def study_has_swi_in_inventory(study_id: str) -> bool:
    """True if batch2 inventory lists an SWI/SWIp series for this study."""
    inv = PROJECT_ROOT / "outputs" / "batch2_inventory" / "series_inventory.csv"
    if not inv.is_file():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(inv)
        sub = df[df["study_id"].astype(str) == study_id]
        desc = sub["SeriesDescription"].astype(str).str.lower()
        return bool(desc.str.contains("swi").any())
    except Exception:  # noqa: BLE001
        return False


def find_swi_nifti(study_dir: Path) -> Path | None:
    """Locate an SWI series NIfTI under DICOM-derived folders (if converted)."""
    dicom_root = study_dir / "01_dicom_nifti"
    if not dicom_root.is_dir():
        return None
    for path in sorted(dicom_root.rglob("*.nii.gz")):
        name = path.name.lower()
        parent = path.parent.name.lower()
        if "swi" in name or "swi" in parent or "swip" in name:
            return path
        meta = _load_sidecar(path)
        desc = str(
            meta.get("series_description") or meta.get("SeriesDescription") or ""
        ).lower()
        if "swi" in desc:
            return path
    return None


def selected_native_path(study_dir: Path, modality: str) -> Path | None:
    """Preferred native DICOM-derived NIfTI from series_selection.json."""
    sel = study_dir / "series_selection.json"
    if sel.is_file():
        try:
            data = json.loads(sel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        entry = data.get(modality) or {}
        nifti = entry.get("nifti_path")
        if nifti and Path(nifti).is_file():
            return Path(nifti)
    # Fallback: first file under 01_dicom_nifti/<mod>/
    folder = study_dir / "01_dicom_nifti"
    # nested study_id folder
    candidates = list(folder.glob(f"*/{modality}/*.nii.gz")) + list(
        folder.glob(f"{modality}/*.nii.gz")
    )
    return candidates[0] if candidates else None


def registered_path(study_dir: Path, modality: str) -> Path | None:
    reg = study_dir / "04_registered_1mm"
    mapping = {
        "T1": "T1_1mm.nii.gz",
        "T1c": "T1c_registered_1mm.nii.gz",
        "T2": "T2_registered_1mm.nii.gz",
        "FLAIR": "FLAIR_registered_1mm.nii.gz",
    }
    name = mapping.get(modality)
    if name is None:
        return None
    path = reg / name
    return path if path.is_file() else None


def brain_mask_path(study_dir: Path, modality: str = "T1") -> Path | None:
    p = study_dir / "03_skull" / f"{modality}_brainmask.nii.gz"
    return p if p.is_file() else None


def edge_overlay_montage(
    ref: np.ndarray,
    mov: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
) -> Path:
    """Save a simple mid-slice edge overlay PNG (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.feature import canny

    z = ref.shape[2] // 2
    r = ref[:, :, z]
    m = mov[:, :, z]
    # normalize
    def _norm(x):
        x = x.astype(np.float64)
        lo, hi = np.percentile(x, (1, 99))
        if hi <= lo:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0, 1)

    rn, mn = _norm(r), _norm(m)
    edges = canny(mn, sigma=1.2)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(rn.T, origin="lower", cmap="gray")
    ys, xs = np.where(edges.T)
    ax.scatter(xs, ys, s=0.3, c="cyan", marker=".")
    ax.set_title(title or "edges")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def qc_one_study(
    study_id: str,
    *,
    batch2_root: Path | None = None,
    out_root: Path | None = None,
    write_overlays: bool = True,
) -> StudyScanQC:
    root = Path(batch2_root or BATCH2_ROOT)
    study_dir = root / study_id
    out_root = Path(out_root or DEFAULT_OUT)
    native_set = native_t1c_ids()
    result = StudyScanQC(
        study_id=study_id,
        usable=study_dir.is_dir(),
        native_t1c=study_id in native_set,
        has_swi=False,
        swi_registered_to_t1c=None,
    )
    if not study_dir.is_dir():
        result.verdict = "do_not_use"
        result.verdict_reasons.append("study directory missing")
        return result

    # Native (DICOM-derived) series QC
    for mod in ("T1", "T1c", "T2", "FLAIR"):
        path = selected_native_path(study_dir, mod)
        if path is None:
            if mod == "T1c" and not result.native_t1c:
                continue
            result.flags.append(f"missing native {mod}")
            continue
        mask_p = brain_mask_path(study_dir, mod if mod != "T1c" else "T1c")
        mask = None
        if mask_p is not None:
            mask = np.asanyarray(nib.load(str(mask_p)).dataobj) > 0
        sq = _series_qc_from_nifti(f"native_{mod}", path, brain_mask=mask)
        result.series.append(sq)
        result.flags.extend([f"{mod}: {f}" for f in sq.flags])

    # Registered 1 mm series QC + NCC
    reg_vols: dict[str, np.ndarray] = {}
    reg_aff = None
    for mod in ("T1", "T1c", "T2", "FLAIR"):
        path = registered_path(study_dir, mod)
        if path is None:
            result.flags.append(f"missing registered {mod}")
            continue
        img = nib.load(str(path))
        reg_vols[mod] = np.asanyarray(img.dataobj).astype(np.float64)
        reg_aff = img.affine
        mask_p = brain_mask_path(study_dir, "T1")
        mask = None
        if mask_p is not None:
            mimg = nib.load(str(mask_p))
            m = np.asanyarray(mimg.dataobj) > 0
            if m.shape != reg_vols[mod].shape:
                factors = [
                    reg_vols[mod].shape[i] / m.shape[i] for i in range(3)
                ]
                m = ndi.zoom(m.astype(np.float32), factors, order=0) > 0.5
            mask = m
        sq = _series_qc_from_nifti(f"reg_{mod}", path, brain_mask=mask)
        result.series.append(sq)

    t1 = reg_vols.get("T1")
    mask_reg = None
    mp = brain_mask_path(study_dir, "T1")
    if mp is not None and t1 is not None:
        m = np.asanyarray(nib.load(str(mp)).dataobj) > 0
        if m.shape != t1.shape:
            factors = [t1.shape[i] / m.shape[i] for i in range(3)]
            m = ndi.zoom(m.astype(np.float32), factors, order=0) > 0.5
        mask_reg = m

    overlay_dir = out_root / "alignment_overlays" / study_id
    for mod, attr in (("T2", "ncc_t2_t1"), ("FLAIR", "ncc_flair_t1"), ("T1c", "ncc_t1c_t1")):
        if t1 is None or mod not in reg_vols:
            continue
        ncc = normalized_cross_correlation(t1, reg_vols[mod], mask_reg)
        setattr(result, attr, ncc)
        if ncc == ncc and ncc < NCC_WARN_THRESHOLD:
            result.flags.append(f"poor registration {mod}→T1 NCC={ncc:.3f}")
        if write_overlays and t1 is not None:
            edge_overlay_montage(
                t1,
                reg_vols[mod],
                overlay_dir / f"{mod}_edges_on_T1.png",
                title=f"{study_id} {mod}→T1 NCC={ncc:.3f}",
            )

    # T1c contrast sanity
    if t1 is not None and "T1c" in reg_vols and mask_reg is not None:
        t1_top = top_percentile_mean(t1, mask_reg, 99.0)
        t1c_top = top_percentile_mean(reg_vols["T1c"], mask_reg, 99.0)
        ratio = t1c_top / t1_top if t1_top and t1_top == t1_top and t1_top > 0 else float("nan")
        result.t1c_t1_top1pct_ratio = ratio
        result.t1c_brighter_than_t1 = bool(ratio == ratio and ratio > 1.05)
        if result.native_t1c and not result.t1c_brighter_than_t1:
            result.flags.append(
                f"T1c not brighter than T1 in top1% vessels/brain (ratio={ratio:.3f})"
            )

    # SWI (often present in DICOM inventory but not converted into 01_dicom_nifti)
    swi = find_swi_nifti(study_dir)
    inv_swi = study_has_swi_in_inventory(study_id)
    result.has_swi = swi is not None or inv_swi
    if swi is not None:
        result.series.append(_series_qc_from_nifti("native_SWI", swi))
        reg_swi = study_dir / "04_registered_1mm" / "SWI_registered_1mm.nii.gz"
        if reg_swi.is_file() and "T1c" in reg_vols:
            swi_img = nib.load(str(reg_swi))
            result.swi_registered_to_t1c = swi_img.shape == reg_vols["T1c"].shape
        else:
            result.swi_registered_to_t1c = False
            result.flags.append("SWI NIfTI present but not registered to T1c 1mm grid")
    elif inv_swi:
        result.swi_registered_to_t1c = False
        result.flags.append(
            "SWI present in DICOM inventory (e.g. SWIp) but not converted to NIfTI "
            "under 01_dicom_nifti — cannot register to T1c"
        )

    # Orientation consistency among native axial series
    ax_set = {s.axcodes for s in result.series if s.modality.startswith("native_")}
    if len(ax_set) > 1:
        result.flags.append(f"mixed native orientations: {sorted(ax_set)}")

    # Verdict
    reasons: list[str] = []
    hard = []
    if not result.usable:
        hard.append("not usable")
    if any("truncated brain" in f for f in result.flags):
        reasons.append("truncated brain coverage")
    if any("poor registration" in f for f in result.flags):
        reasons.append("poor inter-modality registration")
    if any("non-axial" in f for f in result.flags):
        reasons.append("non-axial series")
    if any("hf_energy" in f for f in result.flags):
        reasons.append("possible motion/ghosting")
    if result.native_t1c and result.t1c_brighter_than_t1 is False:
        reasons.append("T1c contrast sanity failed")
    if not result.native_t1c:
        reasons.append("no verified native T1c (ET unreliable)")

    # Almost all batch2 are thick-slice → usable_with_caution baseline
    sparse = any(
        s.modality.startswith("native_") and s.shape[2] < 40 for s in result.series
    )
    if sparse:
        reasons.append("thick-slice / <40 slices on ≥1 series")

    if hard:
        result.verdict = "do_not_use"
        result.verdict_reasons = hard + reasons
    elif reasons:
        # Native T1c + no hard registration failure → still caution due to thick slices
        result.verdict = "usable_with_caution"
        result.verdict_reasons = reasons
    else:
        result.verdict = "ready"
        result.verdict_reasons = ["no QC flags"]

    # Never mark "ready" if no native T1c — enhancing unreliable for second opinion
    if result.verdict == "ready" and not result.native_t1c:
        result.verdict = "usable_with_caution"
        result.verdict_reasons.append("no native T1c")

    return result


def run_scan_qc(
    *,
    study_ids: list[str] | None = None,
    out_root: Path | None = None,
    write_overlays: bool = True,
) -> list[StudyScanQC]:
    out_root = Path(out_root or DEFAULT_OUT)
    out_root.mkdir(parents=True, exist_ok=True)
    ids = study_ids or usable_study_ids()
    results = [
        qc_one_study(sid, out_root=out_root, write_overlays=write_overlays)
        for sid in ids
    ]
    return results


def write_scan_qc_reports(
    results: list[StudyScanQC],
    out_root: Path | None = None,
) -> tuple[Path, Path]:
    import pandas as pd

    out_root = Path(out_root or DEFAULT_OUT)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [r.to_flat_row() for r in results]
    csv_path = out_root / "scan_qc.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    lines = [
        "# Batch2 scan QC",
        "",
        "Read-only QC of DICOM-derived (native) and registered 1 mm volumes.",
        "Does not modify labels or checkpoints.",
        "",
        "| study_id | native_T1c | SWI | NCC T2 | NCC FLAIR | NCC T1c | T1c>T1 | verdict | reasons |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for r in results:
        def _f(x):
            return f"{x:.3f}" if isinstance(x, float) and x == x else "—"

        lines.append(
            f"| {r.study_id} | {r.native_t1c} | {r.has_swi} | {_f(r.ncc_t2_t1)} | "
            f"{_f(r.ncc_flair_t1)} | {_f(r.ncc_t1c_t1)} | {r.t1c_brighter_than_t1} | "
            f"**{r.verdict}** | {'; '.join(r.verdict_reasons) or '—'} |"
        )
    lines.extend(["", "## Per-study verdicts", ""])
    for r in results:
        lines.append(f"### {r.study_id}: `{r.verdict}`")
        for reason in r.verdict_reasons:
            lines.append(f"- {reason}")
        if r.flags:
            lines.append(f"- flags: {'; '.join(r.flags[:8])}")
        lines.append("")
    md_path = out_root / "scan_qc.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
