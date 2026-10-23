#!/usr/bin/env python3
"""Regenerate labeled Plotly mesh HTML for paper + real-patient demo (no retrain)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from preprocessing.registration import REGISTERED_NAMES  # noqa: E402
from segmentation.evaluate import load_model_from_checkpoint  # noqa: E402
from visualization.plotly_3d import (  # noqa: E402
    binary_mask_to_trimesh,
    brain_surface_mesh,
    write_labeled_mesh_html,
)


def _load_demo():
    path = ROOT / "scripts" / "real_patients_demo.py"
    spec = importlib.util.spec_from_file_location("real_patients_demo", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SPACING = (1.0, 1.0, 1.0)
CKPT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
CKPT_LABEL = "checkpoints/brats_scale_full/best_model.pt (v3)"
OUT = ROOT / "outputs" / "real_patients_demo"
PAPER_FIG = ROOT / "results" / "paper" / "figures"
SHORT = {"ET": "enhancing_tumor", "TC": "tumor_core", "WT": "whole_tumor"}


def _regions_from_label(label: np.ndarray) -> dict[str, np.ndarray]:
    """Inverse of nested BraTS-style labels 0–3 used by ``regions_to_label_map``."""
    lab = np.asarray(label)
    et = lab == 3
    tc = (lab == 1) | (lab == 3)
    wt = (lab == 1) | (lab == 2) | (lab == 3)
    return {
        "enhancing_tumor": et,
        "tumor_core": tc,
        "whole_tumor": wt,
    }


def regen_patient_case(case_dir: Path, patients_root: Path) -> Path:
    sid = case_dir.name
    study = patients_root / sid
    reg = study / "04_registered_1mm"
    t1 = nib.load(str(reg / REGISTERED_NAMES["T1"])).get_fdata(dtype=np.float32)
    morph = pd.read_csv(case_dir / "morphometrics.csv")
    volumes = {
        str(r.region): float(r.volume_ml)
        for r in morph.itertuples()
        if float(r.volume_ml) > 0
    }
    pred_path = case_dir / "pred_label_lcc.nii.gz"
    regions: dict[str, np.ndarray] = {}
    if pred_path.is_file():
        regions = _regions_from_label(
            nib.load(str(pred_path)).get_fdata().astype(np.uint8)
        )
    meshes: dict[str, trimesh.Trimesh | None] = {
        "brain": brain_surface_mesh(
            t1, SPACING, include_masks={k: v for k, v in regions.items() if np.any(v)}
        ),
    }
    for short, region in SHORT.items():
        if region in regions and np.any(regions[region]):
            meshes[region] = binary_mask_to_trimesh(regions[region], SPACING, step_size=1)
            continue
        obj = case_dir / f"{short}.obj"
        if obj.is_file():
            meshes[region] = trimesh.load(str(obj), force="mesh")
    out = case_dir / "mesh_3d.html"
    write_labeled_mesh_html(
        out,
        meshes,
        subject_id=sid,
        checkpoint=CKPT_LABEL,
        postprocess="lcc",
        volumes_ml=volumes,
    )
    return out


def regen_brats_paper_mesh(demo, device: torch.device) -> Path:
    extremes = pd.read_csv(
        ROOT / "results" / "paper" / "tables" / "fig_a_wt_extremes_case_ids.csv"
    )
    best_id = extremes.loc[extremes["rank"] == "best", "subject_id"].iloc[0]
    app = load_config()
    brats_root = Path(app.paths.brats_nifti)
    model = load_model_from_checkpoint(CKPT, device)
    model.float().eval()
    flair, regions, _gt, dice = demo.infer_brats_regions(
        model, device, best_id, brats_root
    )
    out = PAPER_FIG / f"fig_mesh_wt_{best_id}.html"
    demo.write_brats_mesh_html(
        flair=flair,
        regions=regions,
        subject_id=best_id,
        out_html=out,
        dice=dice,
    )
    wt_mesh = binary_mask_to_trimesh(regions["whole_tumor"], SPACING, step_size=1)
    if wt_mesh is not None:
        demo.mesh_static_png(
            wt_mesh,
            PAPER_FIG / f"fig_mesh_wt_{best_id}.png",
            title=f"WT mesh · {best_id} · v3-lcc (best held-out Dice)",
        )
    print(f"Paper mesh case: {best_id} (WT Dice={dice:.3f})")
    return out


def main() -> int:
    demo = _load_demo()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    app = load_config()
    patients_root = Path(app.paths.processed) / "real_patients"
    cases = sorted((OUT / "cases").glob("*_MR_Study1"))
    for case_dir in cases:
        if not (case_dir / "morphometrics.csv").is_file():
            continue
        out = regen_patient_case(case_dir, patients_root)
        print(f"Wrote {out}")
    paper = regen_brats_paper_mesh(demo, device)
    print(f"Wrote {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
