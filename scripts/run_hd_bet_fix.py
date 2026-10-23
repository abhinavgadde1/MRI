#!/usr/bin/env python3
"""Re-skull-strip real-patient T1s with HD-BET; write 2D overlays + 3D meshes."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.registration import REGISTERED_NAMES  # noqa: E402
from preprocessing.skull_strip import (  # noqa: E402
    default_hd_bet_device,
    hd_bet_status,
    skull_strip,
)
from visualization.plotly_3d import (  # noqa: E402
    MESH_COLORS,
    _default_camera,
    binary_mask_to_trimesh,
    brain_surface_mesh,
    mesh3d_from_trimesh,
    mesh_surface_to_volume_ratio,
    write_labeled_mesh_html,
)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("run_hd_bet_fix")

OUT = ROOT / "outputs" / "hd_bet_fix"
CASES = [
    "26070613014390354814_MR_Study1",
    "26070614152390454835_MR_Study1",
    "26070605394269334502_MR_Study1",
]


def mid_axial_overlay(
    vol_path: Path, mask_path: Path, out_png: Path, title: str
) -> dict[str, int]:
    vol = nib.load(str(vol_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata() > 0
    z = vol.shape[2] // 2
    img = vol[:, :, z].T
    m = mask[:, :, z].T
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    vmax = float(np.percentile(img[img > 0], 99)) if np.any(img > 0) else 1.0
    ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=max(vmax, 1e-6))
    ax.contour(m.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    ax.imshow(
        np.ma.masked_where(~m, m.astype(float)),
        cmap="autumn",
        alpha=0.25,
        origin="lower",
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"mask_voxels": int(mask.sum()), "slice": int(z)}


def regions_from_pred(pred_path: Path | None) -> dict[str, np.ndarray]:
    if pred_path is None or not pred_path.is_file():
        return {}
    lab = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    d = {
        "enhancing_tumor": lab == 3,
        "tumor_core": (lab == 1) | (lab == 3),
        "whole_tumor": (lab == 1) | (lab == 2) | (lab == 3),
    }
    return {k: v for k, v in d.items() if np.any(v)}


def find_pred(sid: str) -> Path | None:
    cands = [
        ROOT / "outputs/api_jobs/f3f0613ef54a/out/pred_label_lcc.nii.gz",
        ROOT / "outputs/api_jobs/27520445d9a3/out/pred_label_lcc.nii.gz",
        ROOT / "outputs/real_patients_demo/cases" / sid / "pred_label_lcc.nii.gz",
    ]
    for c in cands:
        if "130143" in sid or "real_patients_demo" in str(c):
            if c.is_file():
                return c
    for c in cands[2:]:
        if c.is_file():
            return c
    return None


def main() -> int:
    status = hd_bet_status()
    print("HD-BET status:", json.dumps(status, indent=2))
    print("device:", default_hd_bet_device())
    if not status["available"]:
        print("HD-BET not available; aborting", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for sid in CASES:
        print(f"\n======== {sid} ========")
        study = ROOT / "data/processed/real_patients" / sid
        bias_t1 = study / "02_bias" / "T1.nii.gz"
        reg_t1 = study / "04_registered_1mm" / REGISTERED_NAMES["T1"]
        if not bias_t1.is_file():
            bias_t1 = (
                ROOT
                / "outputs/api_jobs/f3f0613ef54a/out/preprocess/02_bias/T1.nii.gz"
            )
            reg_t1 = (
                ROOT
                / "outputs/api_jobs/f3f0613ef54a/out/preprocess/04_registered_1mm"
                / REGISTERED_NAMES["T1"]
            )
        case_out = OUT / sid
        case_out.mkdir(parents=True, exist_ok=True)

        strip_out = case_out / "T1_hdbet.nii.gz"
        res = skull_strip(bias_t1, strip_out, prefer_hd_bet=True, mode="fast")
        print("pipeline method:", res.method)
        if res.method != "hd-bet":
            raise RuntimeError(f"{sid}: expected hd-bet, got {res.method}")

        ov = mid_axial_overlay(
            bias_t1,
            res.mask_path,
            case_out / "mask_overlay_axial.png",
            f"{sid}\nHD-BET mask mid-axial",
        )
        print("overlay", ov)

        reg_strip = case_out / "T1_1mm_hdbet.nii.gz"
        res_reg = skull_strip(reg_t1, reg_strip, prefer_hd_bet=True, mode="fast")
        print("registered strip method:", res_reg.method)
        if res_reg.method != "hd-bet":
            raise RuntimeError(f"{sid}: registered strip got {res_reg.method}")

        tumor = regions_from_pred(find_pred(sid))
        vol = nib.load(str(res_reg.stripped_path)).get_fdata(dtype=np.float32)
        brain = brain_surface_mesh(
            vol, (1.0, 1.0, 1.0), include_masks=tumor or None, mask_source="clean"
        )
        if brain is None:
            raise RuntimeError(f"{sid}: brain mesh failed")
        sav = mesh_surface_to_volume_ratio(brain)
        print(f"mesh SA/V={sav:.5f} faces={len(brain.faces)} extents={brain.extents}")

        meshes = {"brain": brain}
        for region, arr in tumor.items():
            meshes[region] = binary_mask_to_trimesh(arr, (1.0, 1.0, 1.0), step_size=1)
        html = case_out / "mesh_3d.html"
        write_labeled_mesh_html(
            html,
            meshes,
            subject_id=sid,
            checkpoint="hd-bet",
            postprocess="lcc",
            volumes_ml={},
        )
        brain.export(str(case_out / "brain.obj"))

        try:
            import plotly.graph_objects as go

            fig = go.Figure(
                mesh3d_from_trimesh(
                    brain,
                    name="Brain",
                    color=MESH_COLORS["brain"],
                    opacity=1.0,
                    flatshading=False,
                )
            )
            fig.update_layout(
                title=f"{sid} HD-BET brain SA/V={sav:.4f}",
                scene=dict(aspectmode="data", camera=_default_camera([brain])),
                height=700,
                width=900,
            )
            fig.write_image(str(case_out / "brain_preview.png"), scale=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("preview skip: %s", exc)

        summary.append(
            {
                "case": sid,
                "method": res.method,
                "reg_method": res_reg.method,
                "mask_voxels": ov["mask_voxels"],
                "mesh_sav": sav,
                "faces": len(brain.faces),
                "overlay": str(case_out / "mask_overlay_axial.png"),
                "mesh_html": str(html),
            }
        )

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nDONE")
    for row in summary:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
