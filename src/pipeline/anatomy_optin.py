"""Opt-in SynthSeg anatomy overlay for API jobs (isolated .venv-synthseg).

The main pipeline venv never imports DIPY/SynthSeg. Inference runs as a
subprocess; meshing stays in this process.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from config import PROJECT_ROOT
from preprocessing.registration import REGISTERED_NAMES
from visualization.plotly_3d import (
    CUTAWAY_OPACITY,
    MESH_LABELS,
    binary_mask_to_trimesh,
    brain_surface_mesh,
    build_labeled_mesh_figure,
    choose_sagittal_keep_low_x,
    clip_array_to_sagittal_half,
    mesh3d_from_trimesh,
    normalize_mesh_view,
)

logger = logging.getLogger(__name__)

SYNTHSEG_PYTHON = PROJECT_ROOT / ".venv-synthseg" / "bin" / "python"
SYNTHSEG_SCRIPT = (
    PROJECT_ROOT / "experiments" / "feasibility_parcellation" / "run_synthseg_fullbrain.py"
)
# Native-res overlapping 192³ patches are ~70s infer on this Mac; leave headroom
# for init + meshing + larger FOVs.
ANATOMY_TIMEOUT_S = 240
# Cortex + WM voxel floor; synthetic/non-brain volumes typically fail this.
MIN_CORTEX_WM_VOXELS = 5_000
# Drop slivers instead of meshing unreliable fragments (e.g. letterbox brainstem).
MIN_STRUCTURE_VOXELS = 400
SPACING = (1.0, 1.0, 1.0)
TUMOR_TRACE_NAMES = {
    MESH_LABELS["whole_tumor"],
    MESH_LABELS["tumor_core"],
    MESH_LABELS["enhancing_tumor"],
}

# Distinct from WT green / TC blue / ET red / brain beige, and from each other
# at low opacity (no pink-vs-purple pair).
ANAT_SPEC = [
    ("left_cortex", (3,), "#FF9F1C", 0.14, "Left cerebral cortex"),
    ("right_cortex", (42,), "#7B2CBF", 0.14, "Right cerebral cortex"),
    ("left_wm", (2,), "#00BBF9", 0.12, "Left cerebral white matter"),
    ("right_wm", (41,), "#FEE440", 0.12, "Right cerebral white matter"),
    ("cerebellum", (7, 8, 46, 47), "#BC6C25", 0.20, "Cerebellum"),
    ("brainstem", (16,), "#1B4965", 0.22, "Brainstem"),
]


def try_add_anatomy_mesh(out_dir: Path) -> tuple[str, str]:
    """Run isolated SynthSeg and merge anatomy into ``mesh_3d.html``.

    Returns ``(status, reason)`` where status is ``ok`` or ``unavailable``.
    Never raises — callers keep the tumor-only mesh on failure.
    """
    out_dir = Path(out_dir)
    try:
        return _try_add_anatomy_mesh(out_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Anatomy overlay failed for %s", out_dir)
        return "unavailable", f"{type(exc).__name__}: {exc}"


def _try_add_anatomy_mesh(out_dir: Path) -> tuple[str, str]:
    py = SYNTHSEG_PYTHON
    if not py.is_file():
        return (
            "unavailable",
            ".venv-synthseg is not installed (isolated SynthSeg environment missing)",
        )
    if not SYNTHSEG_SCRIPT.is_file():
        return "unavailable", f"SynthSeg script missing: {SYNTHSEG_SCRIPT}"

    meta_path = out_dir / "meta.json"
    pred_path = out_dir / "pred_label_lcc.nii.gz"
    if not pred_path.is_file():
        return "unavailable", "pred_label_lcc.nii.gz missing"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    reg = Path(meta.get("registered_dir") or "")
    t1_path = reg / REGISTERED_NAMES["T1"] if reg.is_dir() else None
    if t1_path is None or not t1_path.is_file():
        return "unavailable", "T1_1mm.nii.gz not found under registered_dir"

    wt_path = out_dir / "anatomy_wt_mask.nii.gz"
    pred_img = nib.load(str(pred_path))
    pred = np.asanyarray(pred_img.dataobj)
    wt = pred > 0
    nib.save(
        nib.Nifti1Image(wt.astype(np.uint8), pred_img.affine, pred_img.header),
        str(wt_path),
    )

    anat_nii = out_dir / "synthseg_anatomy.nii.gz"
    anat_json = out_dir / "synthseg_anatomy.json"
    cmd = [
        str(py),
        str(SYNTHSEG_SCRIPT),
        "--t1",
        str(t1_path),
        "--wt",
        str(wt_path),
        "--out",
        str(anat_nii),
        "--json",
        str(anat_json),
        "--mode",
        "patches",
    ]
    logger.info("Anatomy subprocess: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ANATOMY_TIMEOUT_S,
            check=False,
            env={**_subprocess_env()},
        )
    except subprocess.TimeoutExpired:
        return "unavailable", f"SynthSeg timed out after {ANATOMY_TIMEOUT_S}s"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = tail[-1] if tail else f"exit {proc.returncode}"
        logger.warning("SynthSeg stderr (tail): %s", "\n".join(tail[-20:]))
        return "unavailable", f"SynthSeg failed: {reason[:300]}"

    if not anat_nii.is_file():
        return "unavailable", "SynthSeg finished but wrote no anatomy NIfTI"

    labels = np.asanyarray(nib.load(str(anat_nii)).dataobj)
    cortex_wm = int(np.isin(labels, (2, 3, 41, 42)).sum())
    if cortex_wm < MIN_CORTEX_WM_VOXELS:
        return (
            "unavailable",
            "SynthSeg produced too little cortex/WM (typical on synthetic or non-brain scans)",
        )

    hide_et = bool(meta.get("et_hidden"))
    sidecar = _read_json(anat_json)
    laterality = laterality_report(labels)
    n_traces, structures = _write_combined_mesh(
        out_dir,
        t1_path=t1_path,
        pred=pred,
        labels=labels,
        hide_et=hide_et,
        case_id=str(meta.get("case_id") or out_dir.name),
        postprocess=str(meta.get("postprocess") or "lcc"),
    )
    if n_traces == 0:
        _patch_meta(
            meta_path,
            anatomy_status="unavailable",
            anatomy_reason="SynthSeg labels did not produce usable anatomy meshes",
            anatomy_structures=structures,
            anatomy_laterality=laterality,
            anatomy_min_structure_voxels=MIN_STRUCTURE_VOXELS,
            anatomy_approach=sidecar.get("approach"),
            anatomy_infer_seconds=sidecar.get("infer_seconds"),
        )
        return (
            "unavailable",
            "SynthSeg labels did not produce usable anatomy meshes",
        )
    skipped = [s["key"] for s in structures if not s["rendered"]]
    reason = f"cortex+WM voxels={cortex_wm}"
    if skipped:
        reason += f"; skipped {', '.join(skipped)} (below threshold, not rendered)"
    views = ["full"]
    if (out_dir / "mesh_3d_cutaway.html").is_file():
        views.append("cutaway")
    _patch_meta(
        meta_path,
        anatomy_status="ok",
        anatomy_reason=reason,
        anatomy_structures=structures,
        anatomy_laterality=laterality,
        anatomy_min_structure_voxels=MIN_STRUCTURE_VOXELS,
        anatomy_approach=sidecar.get("approach"),
        anatomy_infer_seconds=sidecar.get("infer_seconds"),
        anatomy_views=views,
    )
    return "ok", reason


def plan_anatomy_structures(
    labels: np.ndarray,
    wt: np.ndarray,
    *,
    min_voxels: int = MIN_STRUCTURE_VOXELS,
) -> list[dict[str, Any]]:
    """Decide which ANAT_SPEC structures are large enough to mesh."""
    wt_bool = np.asarray(wt, dtype=bool)
    rows: list[dict[str, Any]] = []
    for key, codes, color, opacity, title in ANAT_SPEC:
        mask = np.isin(labels, codes) & np.logical_not(wt_bool)
        n = int(mask.sum())
        if n == 0:
            reason = "empty after WT mask, not rendered"
            rendered = False
        elif n < int(min_voxels):
            reason = "below threshold, not rendered"
            rendered = False
            logger.info(
                "Anatomy %s: %d voxels < %d — below threshold, not rendered",
                key,
                n,
                min_voxels,
            )
        else:
            reason = None
            rendered = True
        rows.append(
            {
                "key": key,
                "title": title,
                "voxels": n,
                "color": color,
                "opacity": opacity,
                "rendered": rendered,
                "reason": reason,
            }
        )
    return rows


def laterality_report(labels: np.ndarray) -> dict[str, Any]:
    """Check that left/right cortex and WM masks are spatially distinct.

    SynthSeg assigns one label per voxel, so left/right never share voxels.
    A true labeling bug would show as centroids on the same side of the
    mid-plane or a very small X separation.
    """
    lab = np.asarray(labels)
    out: dict[str, Any] = {
        "label_convention": (
            "SynthSeg 3/2 = left cortex/WM, 42/41 = right. "
            "Discrete labels never share a voxel."
        )
    }
    for name, left_code, right_code in (("cortex", 3, 42), ("wm", 2, 41)):
        left = lab == left_code
        right = lab == right_code
        overlap = int(np.logical_and(left, right).sum())
        lc = _centroid_ijk(left)
        rc = _centroid_ijk(right)
        sep = None if lc is None or rc is None else float(abs(lc[0] - rc[0]))
        separated = bool(overlap == 0 and sep is not None and sep >= 8.0)
        out[f"{name}_voxel_overlap"] = overlap
        out[f"left_{name}_centroid_ijk"] = lc
        out[f"right_{name}_centroid_ijk"] = rc
        out[f"{name}_centroid_separation_x"] = sep
        out[f"{name}_masks_spatially_separated"] = separated
        if lc is not None and rc is not None:
            out[f"{name}_left_centroid_x_lt_right"] = bool(lc[0] < rc[0])
    return out


def _centroid_ijk(mask: np.ndarray) -> list[float] | None:
    if not np.any(mask):
        return None
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    return [float(x) for x in coords.mean(axis=0)]


def _tumor_traces_on_top(fig: Any) -> None:
    """Brain + translucent anatomy first; fully opaque tumor last (in front)."""
    brain: list[Any] = []
    anatomy: list[Any] = []
    tumor: list[Any] = []
    for tr in fig.data:
        name = getattr(tr, "name", "") or ""
        if name in TUMOR_TRACE_NAMES:
            tr.opacity = 1.0
            tumor.append(tr)
        elif name == MESH_LABELS["brain"]:
            brain.append(tr)
        else:
            anatomy.append(tr)
    fig.data = tuple(brain + anatomy + tumor)


def _subprocess_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    # Do not leak the main pipeline venv / PYTHONPATH=src into SynthSeg.
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _patch_meta(meta_path: Path, **fields: Any) -> None:
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    meta.update(fields)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _write_combined_mesh(
    out_dir: Path,
    *,
    t1_path: Path,
    pred: np.ndarray,
    labels: np.ndarray,
    hide_et: bool,
    case_id: str,
    postprocess: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Write full (translucent) and cutaway (opaque) anatomy HTML files.

    ``mesh_3d.html`` stays the existing full-brain overlay for backward
    compatibility. ``mesh_3d_cutaway.html`` is the sagittal half at opacity 1.0.
    """
    wt = pred > 0
    if labels.shape != pred.shape:
        logger.warning(
            "Anatomy shape %s != pred %s — skipping anatomy traces",
            labels.shape,
            pred.shape,
        )
        return 0, []

    structures = plan_anatomy_structures(labels, wt)
    n_full = _write_anatomy_figure(
        out_dir / "mesh_3d.html",
        t1_path=t1_path,
        pred=pred,
        labels=labels,
        hide_et=hide_et,
        case_id=case_id,
        postprocess=postprocess,
        structures=structures,
        view="full",
    )
    if n_full == 0:
        return 0, structures
    try:
        _write_anatomy_figure(
            out_dir / "mesh_3d_cutaway.html",
            t1_path=t1_path,
            pred=pred,
            labels=labels,
            hide_et=hide_et,
            case_id=case_id,
            postprocess=postprocess,
            structures=structures,
            view="cutaway",
        )
    except Exception:  # noqa: BLE001 — full overlay already written
        logger.exception("Cutaway anatomy mesh failed for %s", case_id)
    return n_full, structures


def _write_anatomy_figure(
    out_path: Path,
    *,
    t1_path: Path,
    pred: np.ndarray,
    labels: np.ndarray,
    hide_et: bool,
    case_id: str,
    postprocess: str,
    structures: list[dict[str, Any]],
    view: str = "full",
) -> int:
    view = normalize_mesh_view(view)
    cutaway = view == "cutaway"
    midline: int | None = None
    keep_low_x: bool | None = None
    pred_m = pred
    labels_m = labels
    sag_kw: dict[str, Any] = {}
    if cutaway:
        midline, keep_low_x = choose_sagittal_keep_low_x(pred > 0)
        pred_m = clip_array_to_sagittal_half(
            pred, keep_low_x=keep_low_x, midline=midline
        )
        labels_m = clip_array_to_sagittal_half(
            labels, keep_low_x=keep_low_x, midline=midline
        )
        sag_kw = {
            "sagittal_keep_low_x": keep_low_x,
            "sagittal_midline": midline,
        }

    wt = pred_m > 0
    tc = np.isin(pred_m, (1, 3))
    et = pred_m == 3
    tumor = {
        "whole_tumor": wt,
        "tumor_core": tc,
        "enhancing_tumor": et if not hide_et else np.zeros_like(wt, dtype=bool),
    }
    # Hover volumes are the full (unclipped) lesion so the legend still
    # reports true tumor size, not the half shown in cutaway.
    wt_full = pred > 0
    volumes: dict[str, float] = {
        "whole_tumor": float(wt_full.sum()) / 1000.0,
        "tumor_core": float(np.isin(pred, (1, 3)).sum()) / 1000.0,
        "enhancing_tumor": float((pred == 3).sum()) / 1000.0,
    }
    meshes: dict[str, Any] = {}
    for key, mask in tumor.items():
        if not np.any(mask):
            continue
        mesh = binary_mask_to_trimesh(
            mask, SPACING, smooth=True, target_faces=35_000, **sag_kw
        )
        if mesh is None:
            continue
        meshes[key] = mesh

    meshes_for_fig: dict[str, Any] = dict(meshes)
    if not cutaway:
        t1 = nib.load(str(t1_path)).get_fdata(dtype=np.float32)
        include = {k: v for k, v in tumor.items() if np.any(v)}
        brain = brain_surface_mesh(t1, SPACING, include_masks=include)
        meshes_for_fig = {"brain": brain, **meshes}

    extra_traces: list[tuple[Any, str, str, float, float]] = []
    for spec, row in zip(ANAT_SPEC, structures):
        _key, codes, color, opacity, title = spec
        if not row["rendered"]:
            continue
        mask = np.isin(labels_m, codes) & np.logical_not(wt)
        mesh = binary_mask_to_trimesh(
            mask, SPACING, smooth=True, target_faces=45_000, **sag_kw
        )
        if mesh is None:
            if not cutaway:
                row["rendered"] = False
                row["reason"] = "mesh extraction failed, not rendered"
            continue
        vol = float((np.isin(labels, codes) & np.logical_not(pred > 0)).sum()) / 1000.0
        extra_traces.append(
            (
                mesh,
                title,
                color,
                CUTAWAY_OPACITY if cutaway else opacity,
                vol,
            )
        )

    if not extra_traces:
        return 0

    suffix = " · sagittal cutaway" if cutaway else ""
    fig = build_labeled_mesh_figure(
        meshes_for_fig,
        subject_id=case_id,
        checkpoint="v3 + SynthSeg",
        postprocess=f"{postprocess} + WT-masked anatomy",
        volumes_ml=volumes,
        title=f"{case_id} · tumor + experimental anatomy ({postprocess}){suffix}",
        height=640,
        view=view,
        cutaway_keep_low_x=keep_low_x,
    )
    for mesh, title, color, opacity, vol in extra_traces:
        fig.add_trace(
            mesh3d_from_trimesh(
                mesh,
                name=title,
                color=color,
                opacity=opacity,
                hovertemplate=f"<b>{title}</b><br>volume = {vol:.2f} mL<extra></extra>",
                showlegend=True,
                flatshading=False,
            )
        )
    _tumor_traces_on_top(fig)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return len(extra_traces)
