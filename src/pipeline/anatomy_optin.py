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
    binary_mask_to_trimesh,
    brain_surface_mesh,
    build_labeled_mesh_figure,
    mesh3d_from_trimesh,
)

logger = logging.getLogger(__name__)

SYNTHSEG_PYTHON = PROJECT_ROOT / ".venv-synthseg" / "bin" / "python"
SYNTHSEG_SCRIPT = (
    PROJECT_ROOT / "experiments" / "feasibility_parcellation" / "run_synthseg_fullbrain.py"
)
ANATOMY_TIMEOUT_S = 60
# Cortex + WM voxel floor; synthetic/non-brain volumes typically fail this.
MIN_CORTEX_WM_VOXELS = 5_000
SPACING = (1.0, 1.0, 1.0)

ANAT_SPEC = [
    ("left_cortex", (3,), "#E89AB8", 0.42, "Left cerebral cortex"),
    ("right_cortex", (42,), "#9B59B6", 0.42, "Right cerebral cortex"),
    ("left_wm", (2,), "#F6E7C1", 0.38, "Left cerebral white matter"),
    ("right_wm", (41,), "#C4A882", 0.38, "Right cerebral white matter"),
    ("cerebellum", (7, 8, 46, 47), "#7A8450", 0.72, "Cerebellum"),
    ("brainstem", (16,), "#6B4C9A", 0.92, "Brainstem"),
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
    n_traces = _write_combined_mesh(
        out_dir,
        t1_path=t1_path,
        pred=pred,
        labels=labels,
        hide_et=hide_et,
        case_id=str(meta.get("case_id") or out_dir.name),
        postprocess=str(meta.get("postprocess") or "lcc"),
    )
    if n_traces == 0:
        return (
            "unavailable",
            "SynthSeg labels did not produce usable anatomy meshes",
        )
    _patch_meta(
        meta_path,
        anatomy_status="ok",
        anatomy_reason=f"cortex+WM voxels={cortex_wm}",
    )
    return "ok", f"cortex+WM voxels={cortex_wm}"


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
) -> int:
    wt = pred > 0
    tc = np.isin(pred, (1, 3))
    et = pred == 3
    tumor = {
        "whole_tumor": wt,
        "tumor_core": tc,
        "enhancing_tumor": et if not hide_et else np.zeros_like(wt, dtype=bool),
    }
    meshes: dict[str, Any] = {}
    volumes: dict[str, float] = {}
    for key, mask in tumor.items():
        if not np.any(mask):
            continue
        mesh = binary_mask_to_trimesh(mask, SPACING, smooth=True, target_faces=35_000)
        if mesh is None:
            continue
        meshes[key] = mesh
        volumes[key] = float(mask.sum()) / 1000.0

    t1 = nib.load(str(t1_path)).get_fdata(dtype=np.float32)
    include = {k: v for k, v in tumor.items() if np.any(v)}
    brain = brain_surface_mesh(t1, SPACING, include_masks=include)
    meshes_for_fig = {"brain": brain, **meshes}

    extra_traces = []
    if labels.shape != pred.shape:
        logger.warning(
            "Anatomy shape %s != pred %s — skipping anatomy traces",
            labels.shape,
            pred.shape,
        )
        return 0
    for _key, codes, color, opacity, title in ANAT_SPEC:
        mask = np.isin(labels, codes) & np.logical_not(wt)
        if not np.any(mask):
            continue
        mesh = binary_mask_to_trimesh(mask, SPACING, smooth=True, target_faces=45_000)
        if mesh is None:
            continue
        vol = float(mask.sum()) / 1000.0
        extra_traces.append((mesh, title, color, opacity, vol))

    if not extra_traces:
        return 0

    fig = build_labeled_mesh_figure(
        meshes_for_fig,
        subject_id=case_id,
        checkpoint="v3 + SynthSeg",
        postprocess=f"{postprocess} + WT-masked anatomy",
        volumes_ml=volumes,
        title=f"{case_id} · tumor + experimental anatomy ({postprocess})",
        height=640,
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
    fig.write_html(str(out_dir / "mesh_3d.html"), include_plotlyjs="cdn", full_html=True)
    return len(extra_traces)
