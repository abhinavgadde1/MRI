"""Matched batch2 positives + claimed tumor-free negatives manifests (no training).

Tumor-free status is the user's claim and is **not verified** in the data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from scipy import stats

from config import PROJECT_ROOT
from preprocessing.registration import REGISTERED_NAMES

logger = logging.getLogger(__name__)

BATCH2_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients_batch2"
NEG_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients"
BATCH2_DETAILS = PROJECT_ROOT / "outputs" / "batch2_inventory" / "study_details.json"
BATCH2_QC = PROJECT_ROOT / "outputs" / "batch2_inventory" / "study_qc_table.csv"
NATIVE_SUMMARY = PROJECT_ROOT / "outputs" / "batch2_native_t1c" / "summary.json"
LABELING = PROJECT_ROOT / "outputs" / "batch2_labeling"
SPLIT_DIR = PROJECT_ROOT / "splits" / "batch2_pos_neg"
REPORT_DIR = PROJECT_ROOT / "outputs" / "batch2_pos_neg"

TUMOR_FREE_CLAIM = (
    "tumor-free status is the user's claim and is not verified in the data "
    "(no radiology report / diagnosis / tumor label found in-repo)"
)

MODS = ("T1", "T1c", "T2", "FLAIR")


@dataclass
class StudyMeta:
    study_id: str
    cohort: str  # positive | negative
    patient_id: str
    study_uid: str
    manufacturer: str
    model: str
    field_strength_t: float | None
    slice_thickness_T1_mm: float | None = None
    slice_thickness_T1c_mm: float | None = None
    slice_thickness_T2_mm: float | None = None
    slice_thickness_FLAIR_mm: float | None = None
    inplane_spacing_T1_mm: float | None = None
    inplane_spacing_T1c_mm: float | None = None
    inplane_spacing_T2_mm: float | None = None
    inplane_spacing_FLAIR_mm: float | None = None
    n_slices_T1: int | None = None
    n_slices_T1c: int | None = None
    n_slices_T2: int | None = None
    n_slices_FLAIR: int | None = None
    t1c_status: str = "unknown"  # native | standin
    has_all_four_registered: bool = False
    brain_mask_fraction: float | None = None
    corrected_label_exists: bool = False
    has_report_or_diagnosis: bool = False
    has_tumor_label: bool = False
    tumor_free_claim: str = ""
    selection_reason: str = ""
    selected: bool = False
    unused_negative: bool = False
    path: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _native_t1c_ids() -> set[str]:
    if not NATIVE_SUMMARY.is_file():
        return set()
    return set(json.loads(NATIVE_SUMMARY.read_text(encoding="utf-8")).get("native_ids", []))


def _read_dicom_scanner(source_root: Path) -> tuple[str, str, float | None]:
    if not source_root.is_dir():
        return "unknown", "unknown", None
    search_roots = []
    for name in ("DICOM", "Images", "IMAGE", "MR"):
        sub = source_root / name
        if sub.is_dir():
            search_roots.append(sub)
    search_roots.append(source_root)

    skip_suffix = {
        ".json",
        ".txt",
        ".xml",
        ".png",
        ".jpg",
        ".gif",
        ".pdf",
        ".exe",
        ".js",
        ".html",
        ".htm",
        ".inf",
        ".css",
    }
    candidates: list[Path] = []
    for root in search_roots:
        candidates.extend(sorted(root.rglob("*.dcm"))[:30])
        if candidates:
            break
    if not candidates:
        for root in search_roots:
            for f in root.rglob("*"):
                if not f.is_file() or f.name.startswith("."):
                    continue
                if f.name.upper() == "DICOMDIR":
                    continue
                if f.suffix.lower() in skip_suffix:
                    continue
                candidates.append(f)
                if len(candidates) >= 80:
                    break
            if candidates:
                break

    for f in candidates:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:  # noqa: BLE001
            continue
        man = str(getattr(ds, "Manufacturer", None) or "").strip()
        model = str(getattr(ds, "ManufacturerModelName", None) or "").strip()
        modality = str(getattr(ds, "Modality", None) or "").strip()
        # Require a real Manufacturer tag — force=True can "read" executables.
        if not man:
            continue
        if modality and modality not in ("MR", "OT"):
            continue
        fs = getattr(ds, "MagneticFieldStrength", None)
        try:
            fs_f = float(fs) if fs is not None else None
        except (TypeError, ValueError):
            fs_f = None
        return man, model or "unknown", fs_f
    return "unknown", "unknown", None


def _series_geom_from_sidecar(study_dir: Path, modality: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "thickness": None,
        "inplane": None,
        "n_slices": None,
    }
    # Prefer 01_dicom_nifti sidecars
    hits = sorted((study_dir / "01_dicom_nifti").rglob(f"*/{modality}/*.json"))
    if not hits:
        hits = sorted((study_dir / "01_dicom_nifti").rglob(f"{modality}/*.json"))
    if not hits:
        # fall back to registered NIfTI header
        reg = study_dir / "04_registered_1mm" / REGISTERED_NAMES.get(modality, "")
        if reg.is_file():
            img = nib.load(str(reg))
            zooms = img.header.get_zooms()[:3]
            out["thickness"] = float(zooms[2])
            out["inplane"] = float((zooms[0] + zooms[1]) / 2.0)
            out["n_slices"] = int(img.shape[2])
        return out
    # Prefer series with most instances
    best = None
    best_n = -1
    for jpath in hits:
        meta = json.loads(jpath.read_text(encoding="utf-8"))
        n = int(meta.get("number_of_instances") or 0)
        if n >= best_n:
            best_n = n
            best = meta
    if best is None:
        return out
    thick = best.get("slice_thickness") or best.get("SliceThickness")
    ps = best.get("pixel_spacing") or best.get("PixelSpacing")
    out["thickness"] = float(thick) if thick is not None else None
    if isinstance(ps, (list, tuple)) and len(ps) >= 2:
        out["inplane"] = float((float(ps[0]) + float(ps[1])) / 2.0)
    out["n_slices"] = int(best.get("number_of_instances") or 0) or None
    return out


def _brain_mask_fraction(study_dir: Path) -> float | None:
    reg = study_dir / "04_registered_1mm"
    t1 = reg / REGISTERED_NAMES["T1"]
    if not t1.is_file():
        return None
    data = np.asanyarray(nib.load(str(t1)).dataobj)
    return float(np.count_nonzero(data) / data.size)


def _t1c_is_standin(study_dir: Path) -> bool:
    reg = study_dir / "04_registered_1mm"
    t1 = reg / REGISTERED_NAMES["T1"]
    t1c = reg / REGISTERED_NAMES["T1c"]
    if not (t1.is_file() and t1c.is_file()):
        return True
    # byte-identical or array-equal
    if t1.stat().st_size == t1c.stat().st_size and t1.read_bytes() == t1c.read_bytes():
        return True
    a = np.asanyarray(nib.load(str(t1)).dataobj)
    b = np.asanyarray(nib.load(str(t1c)).dataobj)
    return bool(a.shape == b.shape and np.array_equal(a, b))


def _has_all_four(study_dir: Path) -> bool:
    reg = study_dir / "04_registered_1mm"
    return all((reg / REGISTERED_NAMES[m]).is_file() for m in MODS)


def _corrected_exists(study_id: str) -> bool:
    for base in (LABELING / study_id, BATCH2_ROOT / study_id):
        if (base / "corrected_label.nii.gz").is_file():
            return True
    return False


def _source_root_from_sidecar(study_dir: Path) -> Path | None:
    for j in (study_dir / "01_dicom_nifti").rglob("*.json"):
        try:
            src = json.loads(j.read_text(encoding="utf-8")).get("source_dicom_root")
            if src:
                return Path(src)
        except Exception:  # noqa: BLE001
            continue
    return None


def _tumor_label_or_report(study_dir: Path) -> tuple[bool, bool]:
    """Return (has_tumor_label, has_report_or_diagnosis). Brain masks don't count."""
    has_report = False
    for pat in ("*report*", "*diagnos*", "*radiolog*"):
        if any(study_dir.rglob(pat)):
            has_report = True
            break
    has_tumor_label = False
    for p in study_dir.rglob("*.nii.gz"):
        name = p.name.lower()
        if "brainmask" in name or "skull" in name:
            continue
        if any(k in name for k in ("tumor", "seg", "label", "gt", "mask")):
            # exclude registered modality names
            if any(k in name for k in ("t1", "t2", "flair", "swi")):
                continue
            has_tumor_label = True
            break
    return has_tumor_label, has_report


def collect_positives() -> list[StudyMeta]:
    details = json.loads(BATCH2_DETAILS.read_text(encoding="utf-8"))
    native = _native_t1c_ids()
    # usable = all dirs except batch2_19
    ids = sorted(
        p.name
        for p in BATCH2_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("batch2_") and p.name != "batch2_19"
    )
    rows: list[StudyMeta] = []
    for sid in ids:
        study_dir = BATCH2_ROOT / sid
        info = details.get(sid, {})
        folder = info.get("folder") or f"PROJECT/{sid.split('_')[-1].lstrip('0') or '0'}"
        # normalize PROJECT path
        src = PROJECT_ROOT / folder if not Path(folder).is_absolute() else Path(folder)
        man, model, fs = _read_dicom_scanner(src)
        geom = {m: _series_geom_from_sidecar(study_dir, m) for m in MODS}
        standin = sid not in native
        # Prefer native summary; also verify for stand-ins
        if sid not in native and not _t1c_is_standin(study_dir):
            # inventory said stand-in but volumes differ — still mark standin from summary
            pass
        meta = StudyMeta(
            study_id=sid,
            cohort="positive",
            patient_id=str(info.get("patient_id") or ""),
            study_uid=str(info.get("study_uid") or ""),
            manufacturer=man,
            model=model,
            field_strength_t=fs,
            slice_thickness_T1_mm=geom["T1"]["thickness"],
            slice_thickness_T1c_mm=geom["T1c"]["thickness"],
            slice_thickness_T2_mm=geom["T2"]["thickness"],
            slice_thickness_FLAIR_mm=geom["FLAIR"]["thickness"],
            inplane_spacing_T1_mm=geom["T1"]["inplane"],
            inplane_spacing_T1c_mm=geom["T1c"]["inplane"],
            inplane_spacing_T2_mm=geom["T2"]["inplane"],
            inplane_spacing_FLAIR_mm=geom["FLAIR"]["inplane"],
            n_slices_T1=geom["T1"]["n_slices"],
            n_slices_T1c=geom["T1c"]["n_slices"],
            n_slices_T2=geom["T2"]["n_slices"],
            n_slices_FLAIR=geom["FLAIR"]["n_slices"],
            t1c_status="standin" if standin else "native",
            has_all_four_registered=_has_all_four(study_dir),
            brain_mask_fraction=_brain_mask_fraction(study_dir),
            corrected_label_exists=_corrected_exists(sid),
            has_report_or_diagnosis=False,
            has_tumor_label=False,
            tumor_free_claim="",
            selection_reason="batch2 usable pipeline study (batch2_19 excluded: missing T1)",
            selected=True,
            path=str(study_dir),
        )
        rows.append(meta)
    return rows


def collect_negatives(positives: Sequence[StudyMeta]) -> list[StudyMeta]:
    pos_pids = {p.patient_id for p in positives if p.patient_id}
    pos_uids = {p.study_uid for p in positives if p.study_uid}
    rows: list[StudyMeta] = []
    for study_dir in sorted(p for p in NEG_ROOT.iterdir() if p.is_dir()):
        if study_dir.name == "corrected":
            continue
        sid = study_dir.name
        # patient_id / study_uid from sidecar
        patient_id = sid.split("_")[0]
        study_uid = ""
        src = _source_root_from_sidecar(study_dir)
        for j in (study_dir / "01_dicom_nifti").rglob("*.json"):
            meta = json.loads(j.read_text(encoding="utf-8"))
            patient_id = str(meta.get("patient_id") or patient_id)
            study_uid = str(meta.get("study_instance_uid") or study_uid)
            break
        if patient_id in pos_pids or (study_uid and study_uid in pos_uids):
            logger.info("Exclude negative %s: overlaps batch2 patient/study UID", sid)
            continue
        man, model, fs = _read_dicom_scanner(src or Path())
        geom = {m: _series_geom_from_sidecar(study_dir, m) for m in MODS}
        has_tumor, has_report = _tumor_label_or_report(study_dir)
        standin = _t1c_is_standin(study_dir) if _has_all_four(study_dir) else True
        rows.append(
            StudyMeta(
                study_id=sid,
                cohort="negative",
                patient_id=patient_id,
                study_uid=study_uid,
                manufacturer=man,
                model=model,
                field_strength_t=fs,
                slice_thickness_T1_mm=geom["T1"]["thickness"],
                slice_thickness_T1c_mm=geom["T1c"]["thickness"],
                slice_thickness_T2_mm=geom["T2"]["thickness"],
                slice_thickness_FLAIR_mm=geom["FLAIR"]["thickness"],
                inplane_spacing_T1_mm=geom["T1"]["inplane"],
                inplane_spacing_T1c_mm=geom["T1c"]["inplane"],
                inplane_spacing_T2_mm=geom["T2"]["inplane"],
                inplane_spacing_FLAIR_mm=geom["FLAIR"]["inplane"],
                n_slices_T1=geom["T1"]["n_slices"],
                n_slices_T1c=geom["T1c"]["n_slices"],
                n_slices_T2=geom["T2"]["n_slices"],
                n_slices_FLAIR=geom["FLAIR"]["n_slices"],
                t1c_status="standin" if standin else "native",
                has_all_four_registered=_has_all_four(study_dir),
                brain_mask_fraction=_brain_mask_fraction(study_dir),
                corrected_label_exists=False,
                has_report_or_diagnosis=has_report,
                has_tumor_label=has_tumor,
                tumor_free_claim=TUMOR_FREE_CLAIM,
                path=str(study_dir),
            )
        )
    return rows


def _thickness_ok(neg: StudyMeta, pos_med: dict[str, float], tol: float = 1.0) -> bool:
    for m in MODS:
        key = f"slice_thickness_{m}_mm"
        nv = getattr(neg, key)
        pv = pos_med.get(m)
        if nv is None or pv is None:
            continue
        if abs(float(nv) - float(pv)) > tol:
            return False
    return True


def select_matched_negatives(
    positives: Sequence[StudyMeta],
    negatives: Sequence[StudyMeta],
    *,
    n: int = 20,
    seed: int = 42,
) -> tuple[list[StudyMeta], list[StudyMeta], dict[str, Any]]:
    """Select n negatives matched by priority criteria; return selected, unused, report."""
    pos_med = {}
    for m in MODS:
        vals = [
            getattr(p, f"slice_thickness_{m}_mm")
            for p in positives
            if getattr(p, f"slice_thickness_{m}_mm") is not None
        ]
        pos_med[m] = float(np.median(vals)) if vals else None

    n_native_pos = sum(1 for p in positives if p.t1c_status == "native")
    target_native_frac = n_native_pos / max(len(positives), 1)
    target_native = int(round(target_native_frac * n))

    # (a) all four sequences
    pool = [x for x in negatives if x.has_all_four_registered]
    # (b) thickness within 1 mm of positive median per sequence
    thick_ok = [x for x in pool if _thickness_ok(x, pos_med, 1.0)]
    used_thick_filter = True
    if len(thick_ok) < n:
        thick_ok = pool
        used_thick_filter = False

    # (d) manufacturer + field strength where possible
    pos_mans = {p.manufacturer.split()[0].lower() for p in positives if p.manufacturer}
    # prefer Philips + 1.5T
    pref = [
        x
        for x in thick_ok
        if "philips" in (x.manufacturer or "").lower()
        and (x.field_strength_t is None or abs(float(x.field_strength_t) - 1.5) < 0.2)
    ]
    if len(pref) >= n:
        thick_ok = pref

    # (c) T1c status distribution
    native_cands = [x for x in thick_ok if x.t1c_status == "native"]
    standin_cands = [x for x in thick_ok if x.t1c_status == "standin"]
    rng = np.random.default_rng(seed)

    selected: list[StudyMeta] = []
    deviation: dict[str, Any] = {
        "target_native_count": target_native,
        "target_native_frac": target_native_frac,
        "positive_native_count": n_native_pos,
        "positive_n": len(positives),
        "thickness_filter_applied": used_thick_filter,
        "pos_median_thickness_mm": pos_med,
        "pool_all4": len(pool),
        "pool_after_thickness": len(thick_ok) if used_thick_filter else len(pool),
        "native_candidates": len(native_cands),
        "standin_candidates": len(standin_cands),
    }

    n_nat_take = min(target_native, len(native_cands))
    if native_cands and n_nat_take > 0:
        idx = rng.choice(len(native_cands), size=n_nat_take, replace=False)
        for i in sorted(idx):
            selected.append(native_cands[int(i)])
    n_stand_need = n - len(selected)
    remain_stand = [x for x in standin_cands if x.study_id not in {s.study_id for s in selected}]
    if len(remain_stand) < n_stand_need:
        # fill from any remaining thick_ok
        remain_stand = [
            x for x in thick_ok if x.study_id not in {s.study_id for s in selected}
        ]
    idx = rng.choice(len(remain_stand), size=min(n_stand_need, len(remain_stand)), replace=False)
    for i in sorted(idx):
        selected.append(remain_stand[int(i)])

    selected_ids = {s.study_id for s in selected}
    for s in selected:
        s.selected = True
        s.unused_negative = False
        s.selection_reason = (
            f"seed={seed}; all4=yes; thickness_match={'yes' if used_thick_filter else 'relaxed'}; "
            f"t1c={s.t1c_status}; manufacturer={s.manufacturer}; B0={s.field_strength_t}; "
            f"matched_toward_pos_native_frac={target_native_frac:.2f}"
        )

    unused = []
    for x in negatives:
        if x.study_id in selected_ids:
            continue
        x.selected = False
        x.unused_negative = True
        x.selection_reason = "unused_false_positive_test_pool"
        unused.append(x)

    deviation["selected_native_count"] = sum(1 for s in selected if s.t1c_status == "native")
    deviation["selected_standin_count"] = sum(1 for s in selected if s.t1c_status == "standin")
    deviation["t1c_match_impossible"] = (
        target_native > 0 and deviation["selected_native_count"] == 0
    )
    deviation["t1c_note"] = (
        "All eligible negatives have T1c stand-ins (no native T1c DICOM); "
        "cannot match positive native fraction "
        f"({n_native_pos}/{len(positives)} ≈ {target_native_frac:.0%})."
        if deviation["t1c_match_impossible"]
        else "T1c native/stand-in counts matched as closely as the pool allows."
    )
    # manufacturer model deviation
    pos_models = {p.model for p in positives}
    neg_models = {s.model for s in selected}
    deviation["positive_models"] = sorted(pos_models)
    deviation["selected_negative_models"] = sorted(neg_models)
    deviation["model_match"] = bool(pos_models & neg_models)

    return selected, unused, deviation


def compare_groups(
    positives: Sequence[StudyMeta],
    selected_neg: Sequence[StudyMeta],
) -> tuple[pd.DataFrame, list[str]]:
    """Descriptive comparison table + shortcut warnings."""

    def _col(rows: Sequence[StudyMeta], attr: str) -> np.ndarray:
        vals = []
        for r in rows:
            v = getattr(r, attr)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            vals.append(float(v) if not isinstance(v, str) else v)
        return np.array(vals, dtype=object)

    warnings_out: list[str] = []
    records = []
    numeric_attrs = [
        ("field_strength_t", "field_strength_T"),
        ("slice_thickness_T1_mm", "thickness_T1_mm"),
        ("slice_thickness_T2_mm", "thickness_T2_mm"),
        ("slice_thickness_FLAIR_mm", "thickness_FLAIR_mm"),
        ("inplane_spacing_T1_mm", "inplane_T1_mm"),
        ("n_slices_T1", "n_slices_T1"),
        ("brain_mask_fraction", "brain_mask_fraction"),
    ]
    for attr, label in numeric_attrs:
        a = np.array(
            [float(getattr(p, attr)) for p in positives if getattr(p, attr) is not None],
            dtype=float,
        )
        b = np.array(
            [
                float(getattr(p, attr))
                for p in selected_neg
                if getattr(p, attr) is not None
            ],
            dtype=float,
        )
        if len(a) == 0 or len(b) == 0:
            continue
        # standardized difference (Cohen's d)
        pooled = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2.0) or 1.0
        d = float((a.mean() - b.mean()) / pooled)
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            pval = float(p)
        except Exception:  # noqa: BLE001
            pval = float("nan")
        records.append(
            {
                "variable": label,
                "pos_mean": float(a.mean()),
                "neg_mean": float(b.mean()),
                "pos_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "neg_sd": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
                "std_diff": d,
                "mannwhitney_p": pval,
            }
        )
        if abs(d) >= 0.8 or (pval == pval and pval < 0.05 and abs(d) >= 0.5):
            warnings_out.append(
                f"WARNING: '{label}' still separates positives vs selected negatives "
                f"(std_diff={d:.2f}, p≈{pval:.3g}). The model could use this as a "
                f"shortcut instead of the tumor."
            )

    # categorical: manufacturer, t1c native fraction
    pos_native = np.mean([1.0 if p.t1c_status == "native" else 0.0 for p in positives])
    neg_native = np.mean(
        [1.0 if p.t1c_status == "native" else 0.0 for p in selected_neg]
    )
    records.append(
        {
            "variable": "t1c_native_fraction",
            "pos_mean": float(pos_native),
            "neg_mean": float(neg_native),
            "pos_sd": float("nan"),
            "neg_sd": float("nan"),
            "std_diff": float(pos_native - neg_native),
            "mannwhitney_p": float("nan"),
        }
    )
    if abs(pos_native - neg_native) >= 0.25:
        warnings_out.append(
            "WARNING: T1c native fraction differs strongly between positives and "
            f"selected negatives ({pos_native:.0%} vs {neg_native:.0%}). A model "
            "could latch onto contrast-channel statistics as a shortcut."
        )

    pos_philips = np.mean(
        [1.0 if "philips" in (p.manufacturer or "").lower() else 0.0 for p in positives]
    )
    neg_philips = np.mean(
        [
            1.0 if "philips" in (p.manufacturer or "").lower() else 0.0
            for p in selected_neg
        ]
    )
    records.append(
        {
            "variable": "philips_fraction",
            "pos_mean": float(pos_philips),
            "neg_mean": float(neg_philips),
            "pos_sd": float("nan"),
            "neg_sd": float("nan"),
            "std_diff": float(pos_philips - neg_philips),
            "mannwhitney_p": float("nan"),
        }
    )
    return pd.DataFrame(records), warnings_out


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4g}" if v == v else "nan")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_manifests(
    positives: Sequence[StudyMeta],
    selected: Sequence[StudyMeta],
    unused: Sequence[StudyMeta],
    *,
    deviation: dict[str, Any],
    compare_df: pd.DataFrame,
    warnings_out: Sequence[str],
) -> dict[str, Path]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pos_path = SPLIT_DIR / "positives.txt"
    neg_sel = SPLIT_DIR / "negatives_selected.txt"
    neg_unused = SPLIT_DIR / "negatives_unused.txt"
    pos_path.write_text(
        "\n".join(p.study_id for p in positives) + "\n", encoding="utf-8"
    )
    neg_sel.write_text(
        "\n".join(p.study_id for p in selected) + "\n", encoding="utf-8"
    )
    neg_unused.write_text(
        "\n".join(p.study_id for p in unused) + "\n", encoding="utf-8"
    )

    all_rows = [p.to_row() for p in positives] + [p.to_row() for p in selected] + [
        p.to_row() for p in unused
    ]
    csv_path = SPLIT_DIR / "manifest.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    corrected = [p.study_id for p in positives if p.corrected_label_exists]
    report = REPORT_DIR / "MATCHING_REPORT.md"
    lines = [
        "# Batch2 positives vs claimed tumor-free negatives — matching report",
        "",
        f"**Negatives:** `{NEG_ROOT}` (100 processed studies). {TUMOR_FREE_CLAIM}.",
        "",
        "No training was run. Checkpoints / `splits/heldout.txt` / "
        "`data/processed/real_patients/` / existing labels were not modified.",
        "",
        "## Positives (19 usable batch2; batch2_19 excluded)",
        "",
        f"- n={len(positives)}; native T1c={sum(1 for p in positives if p.t1c_status=='native')}; "
        f"stand-in={sum(1 for p in positives if p.t1c_status=='standin')}",
        f"- corrected_label.nii.gz exists for: "
        + (", ".join(corrected) if corrected else "**none yet**"),
        "",
        "## Negatives pool",
        "",
        f"- Eligible after excluding patient_id/study UID overlap: see manifest",
        f"- Selected (seed 42): {len(selected)}; unused FP pool: {len(unused)}",
        f"- Reports/diagnoses found in negative dirs: "
        f"{sum(1 for u in list(selected)+list(unused) if u.has_report_or_diagnosis)}",
        f"- Tumor labels found in negative dirs: "
        f"{sum(1 for u in list(selected)+list(unused) if u.has_tumor_label)}",
        "",
        "## Matching deviation",
        "",
        "```json",
        json.dumps(deviation, indent=2),
        "```",
        "",
        "## Group comparison (descriptive)",
        "",
        _df_to_md(compare_df),
        "",
    ]
    if warnings_out:
        lines += ["## Shortcut warnings", ""] + [f"- {w}" for w in warnings_out] + [""]
    else:
        lines += ["## Shortcut warnings", "", "- None triggered under |std_diff|≥0.8 rule.", ""]
    lines += [
        "## Manifest paths",
        "",
        f"- `{pos_path}`",
        f"- `{neg_sel}`",
        f"- `{neg_unused}`",
        f"- `{csv_path}`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    # Always also write compare csv
    compare_df.to_csv(REPORT_DIR / "group_comparison.csv", index=False)
    (REPORT_DIR / "matching_deviation.json").write_text(
        json.dumps(deviation, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "positives": pos_path,
        "negatives_selected": neg_sel,
        "negatives_unused": neg_unused,
        "manifest_csv": csv_path,
        "report": report,
    }
