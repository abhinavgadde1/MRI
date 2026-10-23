#!/usr/bin/env python3
"""Standalone Plotly HTML: v3 tumor meshes + selected SynthSeg anatomy.

Uses the main pipeline venv (trimesh / plotly). Not wired into API/CLI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

from config import PROJECT_ROOT
from visualization.plotly_3d import (
    binary_mask_to_trimesh,
    build_labeled_mesh_figure,
    mesh3d_from_trimesh,
)

ROOT = PROJECT_ROOT
SID = "BraTS20_Training_001"
OUT = ROOT / "outputs" / "feasibility_parcellation"
SPACING = (1.0, 1.0, 1.0)

# Distinct from WT green / TC blue / ET red / brain beige
ANAT_SPEC = [
    ("left_cortex", (3,), "#E89AB8", 0.42, "Left cerebral cortex"),
    ("right_cortex", (42,), "#9B59B6", 0.42, "Right cerebral cortex"),
    ("left_wm", (2,), "#F6E7C1", 0.38, "Left cerebral white matter"),
    ("right_wm", (41,), "#C4A882", 0.38, "Right cerebral white matter"),
    ("cerebellum", (7, 8, 46, 47), "#7A8450", 0.72, "Cerebellum"),
    ("brainstem", (16,), "#6B4C9A", 0.92, "Brainstem"),
]


def _volume_ml(mask: np.ndarray) -> float:
    return float(mask.astype(bool).sum()) / 1000.0


def main() -> int:
    t0 = time.perf_counter()
    labels = np.asanyarray(
        nib.load(str(OUT / f"{SID}_synthseg_fullfov.nii.gz")).dataobj
    )
    tumor = {}
    for short in ("WT", "TC", "ET"):
        p = OUT / f"{SID}_v3_{short}.nii.gz"
        tumor[short] = np.asanyarray(nib.load(str(p)).dataobj) > 0

    meshes: dict[str, trimesh.Trimesh | None] = {}
    volumes: dict[str, float] = {}
    extra_traces = []

    # Tumor (existing convention)
    for short, key in (("WT", "whole_tumor"), ("TC", "tumor_core"), ("ET", "enhancing_tumor")):
        m = binary_mask_to_trimesh(tumor[short], SPACING, smooth=True, target_faces=35_000)
        meshes[key] = m
        volumes[key] = _volume_ml(tumor[short])
        volumes[short] = volumes[key]

    anat_stats = []
    for key, codes, color, opacity, title in ANAT_SPEC:
        mask = np.zeros(labels.shape, dtype=bool)
        for c in codes:
            mask |= labels == c
        # Keep tumor hole (already zeroed in nifti); also exclude any leftover overlap
        mask &= ~tumor["WT"]
        vol = _volume_ml(mask)
        mesh = (
            binary_mask_to_trimesh(mask, SPACING, smooth=True, target_faces=45_000)
            if np.any(mask)
            else None
        )
        anat_stats.append(
            {
                "key": key,
                "title": title,
                "voxels": int(mask.sum()),
                "volume_ml": round(vol, 2),
                "faces": int(len(mesh.faces)) if mesh is not None else 0,
            }
        )
        extra_traces.append((mesh, title, color, opacity, vol))

    mesh_s = time.perf_counter() - t0

    # Build figure from tumor helpers, then append anatomy traces.
    fig = build_labeled_mesh_figure(
        meshes,
        subject_id=SID,
        checkpoint="brats_scale_full + SynthSeg",
        postprocess="lcc + WT-masked anatomy",
        volumes_ml=volumes,
        title=(
            f"{SID} · prototype: v3 tumor (WT/TC/ET) + SynthSeg anatomy "
            "(full-FOV letterbox, WT masked out of CSF/tissue)"
        ),
        height=640,
    )
    for mesh, title, color, opacity, vol in extra_traces:
        if mesh is None:
            continue
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
    html_path = OUT / f"{SID}_tumor_anatomy_prototype.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    write_s = time.perf_counter() - t0 - mesh_s
    report = {
        "html": str(html_path),
        "mesh_plus_html_seconds": round(mesh_s + write_s, 2),
        "mesh_seconds": round(mesh_s, 2),
        "anatomy": anat_stats,
        "tumor_ml": {k: round(_volume_ml(tumor[k]), 2) for k in ("WT", "TC", "ET")},
    }
    (OUT / f"{SID}_prototype_render.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
