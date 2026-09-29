#!/usr/bin/env python3
"""Real-patient demo + paper overlays (v3-lcc), no retraining.

Confirm T1c/stand-in counts from ``batch_summary_patients.csv`` + on-disk
bit-identity of T1 vs T1c. Then segment 4 real patients (all stand-in when no
genuine T1c exists) and the BraTS held-out median/best WT cases.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import trimesh
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from config import PROJECT_ROOT, load_config
from preprocessing.registration import REGISTERED_NAMES
from reconstruction.measurements import measure_from_mask
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS
from segmentation.postprocess import apply_channel_postprocess, lcc_then_fill
from segmentation.pseudo_label import (
    get_patient_inference_transforms,
    registered_dir_to_datadict,
    regions_to_label_map,
)
from segmentation.train import post_transforms
from visualization.plotly_3d import mesh3d_from_trimesh

ROOT = PROJECT_ROOT
OUT = ROOT / "outputs" / "real_patients_demo"
PAPER_FIG = ROOT / "results" / "paper" / "figures"
CKPT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
SPACING = (1.0, 1.0, 1.0)
REGION_COLORS = {
    "enhancing_tumor": "#C44E52",
    "tumor_core": "#4C72B0",
    "whole_tumor": "#55A868",
}
REGION_SHORT = {
    "enhancing_tumor": "ET",
    "tumor_core": "TC",
    "whole_tumor": "WT",
}


def confirm_t1c_counts() -> dict[str, Any]:
    batch = pd.read_csv(ROOT / "data" / "processed" / "batch_summary_patients.csv")
    ok = batch[batch["status"] == "success"].copy()
    failed = batch[batch["status"] == "failed"]
    n_t1c_in_log = int(
        ok["modalities_found"].fillna("").str.contains(r"\bT1c\b", regex=True).sum()
    )
    identical = 0
    different = 0
    for _, row in ok.iterrows():
        reg = Path(row["output_dir"]) / "04_registered_1mm"
        t1 = reg / REGISTERED_NAMES["T1"]
        t1c = reg / REGISTERED_NAMES["T1c"]
        if not (t1.is_file() and t1c.is_file()):
            continue
        if t1.read_bytes() == t1c.read_bytes():
            identical += 1
        else:
            different += 1
    summary = {
        "n_total_in_log": int(len(batch)),
        "n_success": int(len(ok)),
        "n_failed": int(len(failed)),
        "n_genuine_t1c_in_modalities_found": n_t1c_in_log,
        "n_t1_standin_modalities_found": int(len(ok) - n_t1c_in_log),
        "n_t1c_bit_identical_to_t1": identical,
        "n_t1c_differs_from_t1": different,
        "modalities_found_success_unique": sorted(
            ok["modalities_found"].fillna("").unique().tolist()
        ),
        "failed_study_ids": failed["study_id"].tolist(),
        "source": "data/processed/batch_summary_patients.csv + on-disk T1/T1c byte compare",
    }
    return summary


def pick_four_patients() -> list[dict[str, Any]]:
    """No genuine T1c exists — pick 4 size-quartile stand-in studies."""
    batch = pd.read_csv(ROOT / "data" / "processed" / "batch_summary_patients.csv")
    ok = batch[batch["status"] == "success"]
    ranked: list[tuple[str, int]] = []
    for _, row in ok.iterrows():
        reg = Path(row["output_dir"]) / "04_registered_1mm"
        if not all((reg / REGISTERED_NAMES[m]).is_file() for m in REGISTERED_NAMES):
            continue
        size = (reg / REGISTERED_NAMES["T1"]).stat().st_size
        ranked.append((row["study_id"], size))
    ranked.sort(key=lambda x: x[1])
    idxs = [0, len(ranked) // 3, 2 * len(ranked) // 3, len(ranked) - 1]
    picks = []
    for i in idxs:
        sid, _ = ranked[i]
        picks.append(
            {
                "study_id": sid,
                "t1c_status": "T1_standin",
                "t1c_note": "No genuine T1c in cohort; T1 bit-identical to T1c file",
            }
        )
    return picks


def binary_to_mesh(binary: np.ndarray, spacing: tuple[float, float, float]) -> trimesh.Trimesh | None:
    if not np.any(binary):
        return None
    # Downsample very large masks for Plotly responsiveness.
    step = 1
    if binary.sum() > 250_000:
        step = 2
    try:
        verts, faces, normals, _ = marching_cubes(
            binary.astype(np.float32),
            level=0.5,
            spacing=spacing,
            step_size=step,
        )
    except (ValueError, RuntimeError):
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)


def save_slice_overlay(
    image_3d: np.ndarray,
    regions: dict[str, np.ndarray],
    out_path: Path,
    *,
    title: str,
) -> Path:
    wt = regions.get("whole_tumor")
    if wt is not None and np.any(wt):
        cx, cy, cz = [int(np.round(c)) for c in ndi.center_of_mass(wt)]
    else:
        cx, cy, cz = [s // 2 for s in image_3d.shape]

    def _plane(vol: np.ndarray, axis: int, index: int) -> np.ndarray:
        if axis == 0:
            return vol[index]
        if axis == 1:
            return vol[:, index]
        return vol[:, :, index]

    def _norm(sl: np.ndarray) -> np.ndarray:
        sl = sl.astype(float)
        p2, p98 = np.percentile(sl, (2, 98))
        if p98 <= p2:
            return np.zeros_like(sl)
        return np.clip((sl - p2) / (p98 - p2), 0, 1)

    titles = ("axial (z)", "coronal (y)", "sagittal (x)")
    axes_idx = (2, 1, 0)
    indices = (
        int(np.clip(cz, 0, image_3d.shape[2] - 1)),
        int(np.clip(cy, 0, image_3d.shape[1] - 1)),
        int(np.clip(cx, 0, image_3d.shape[0] - 1)),
    )
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ttl, axis, idx in zip(axs, titles, axes_idx, indices):
        base = _norm(_plane(image_3d, axis, idx))
        rgb = np.stack([base, base, base], axis=-1)
        for region, color in (
            ("whole_tumor", np.array([0.1, 0.75, 0.3])),
            ("tumor_core", np.array([0.15, 0.45, 0.95])),
            ("enhancing_tumor", np.array([0.95, 0.2, 0.2])),
        ):
            m = regions.get(region)
            if m is None or not np.any(m):
                continue
            sl = _plane(m.astype(bool), axis, idx)
            rgb[sl] = rgb[sl] * 0.35 + color * 0.65
        ax.imshow(np.rot90(rgb), origin="lower")
        ax.set_title(f"{ttl} @{idx}")
        ax.axis("off")
    fig.suptitle(title + "\ngreen=WT  blue=TC  red=ET", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_brats_gt_pred_overlay(
    image_3d: np.ndarray,
    gt_wt: np.ndarray,
    pred_lcc: np.ndarray,
    pred_raw: np.ndarray,
    out_path: Path,
    *,
    subject_id: str,
    dice: float,
) -> Path:
    """Same visual language as existing worst-case figures (GT / LCC / removed)."""
    from validation.lcc_followup import _overlay_gt_lcc_removed
    from segmentation.postprocess import fragmentation_stats

    removed = pred_raw & ~pred_lcc
    gt_stats = fragmentation_stats(gt_wt, SPACING)
    _overlay_gt_lcc_removed(
        image_3d,
        gt_wt,
        pred_lcc,
        removed,
        out_path,
        subject_id=subject_id,
        delta_dice=float(dice),
        gt_n_cc=int(gt_stats["n_components"]),
        gt_frac_out=float(gt_stats.get("frac_outside_largest") or 0.0),
    )
    return out_path


def plotly_regions_figure(meshes: dict[str, trimesh.Trimesh]) -> go.Figure:
    fig = go.Figure()
    for region, mesh in meshes.items():
        if mesh is None or len(mesh.vertices) == 0:
            continue
        fig.add_trace(
            mesh3d_from_trimesh(
                mesh,
                name=REGION_SHORT[region],
                color=REGION_COLORS[region],
                opacity=0.55 if region == "whole_tumor" else 0.85,
            )
        )
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h"),
        height=520,
    )
    return fig


def mesh_static_png(mesh: trimesh.Trimesh, out_path: Path, title: str) -> Path:
    """Matplotlib 3D render for paper (no kaleido required)."""
    v = mesh.vertices
    f = mesh.faces
    fig = plt.figure(figsize=(6, 5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    # Subsample faces if huge
    if len(f) > 40_000:
        rng = np.random.default_rng(42)
        f = f[rng.choice(len(f), size=40_000, replace=False)]
    ax.plot_trisurf(
        v[:, 0],
        v[:, 1],
        v[:, 2],
        triangles=f,
        color="#55A868",
        alpha=0.85,
        linewidth=0.0,
        edgecolor="none",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    try:
        ax.set_box_aspect(
            (
                float(np.ptp(v[:, 0]) or 1),
                float(np.ptp(v[:, 1]) or 1),
                float(np.ptp(v[:, 2]) or 1),
            )
        )
    except Exception:  # noqa: BLE001
        pass
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def infer_patient(
    model: torch.nn.Module,
    device: torch.device,
    study_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    reg = study_dir / "04_registered_1mm"
    record = registered_dir_to_datadict(reg, study_dir.name)
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(device, dtype=torch.float32)
    logits = sliding_window_inference(
        image, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
    )
    post_pred, _ = post_transforms()
    pred = post_pred(decollate_batch(logits)[0]).detach().cpu().numpy()
    pred = apply_channel_postprocess(pred.astype(np.float32), mode="lcc")
    regions = {r: pred[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    # FLAIR channel for display if available else T1 (channel order FLAIR,T1,T1c,T2)
    flair = data["image"][0].detach().cpu().numpy()
    t1 = data["image"][1].detach().cpu().numpy()
    return regions, flair, t1


@torch.no_grad()
def infer_brats_wt(
    model: torch.nn.Module,
    device: torch.device,
    subject_id: str,
    brats_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rec = subject_to_datadict(brats_root / subject_id)
    batch = next(
        iter(DataLoader(Dataset([rec], transform=get_val_transforms()), batch_size=1))
    )
    gt_wt = batch["label"].squeeze(0).squeeze(0).detach().cpu().numpy() != 0
    img = batch["image"].to(device, dtype=torch.float32)
    logits = sliding_window_inference(
        img, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
    )
    pred = (torch.sigmoid(logits) > 0.5).squeeze(0).detach().cpu().numpy()
    pred_raw = pred[2].astype(bool)
    pred_lcc = lcc_then_fill(pred_raw)
    inter = int((pred_lcc & gt_wt).sum())
    dice = (
        2.0 * inter / (pred_lcc.sum() + gt_wt.sum())
        if (pred_lcc.sum() + gt_wt.sum())
        else 0.0
    )
    flair = batch["image"][0, 0].detach().cpu().numpy()
    return flair, gt_wt, pred_raw, pred_lcc, float(dice)


def process_patient_case(
    model: torch.nn.Module,
    device: torch.device,
    pick: dict[str, Any],
    patients_root: Path,
) -> dict[str, Any]:
    sid = pick["study_id"]
    study_dir = patients_root / sid
    case_dir = OUT / "cases" / sid
    case_dir.mkdir(parents=True, exist_ok=True)

    regions, flair, t1 = infer_patient(model, device, study_dir)
    # Save label NIfTI for reproducibility
    label = regions_to_label_map(
        regions["enhancing_tumor"], regions["tumor_core"], regions["whole_tumor"]
    )
    ref = nib.load(str(study_dir / "04_registered_1mm" / REGISTERED_NAMES["T1"]))
    nib.save(nib.Nifti1Image(label.astype(np.uint8), ref.affine, ref.header), str(case_dir / "pred_label_lcc.nii.gz"))

    metrics_rows = []
    meshes: dict[str, trimesh.Trimesh] = {}
    for region in BRATS_REGIONS:
        binary = regions[region]
        short = REGION_SHORT[region]
        if not np.any(binary):
            metrics_rows.append(
                {
                    "region": short,
                    "n_voxels": 0,
                    "volume_ml": 0.0,
                    "sphericity": float("nan"),
                    "pca_a_mm": float("nan"),
                    "pca_b_mm": float("nan"),
                    "pca_c_mm": float("nan"),
                    "mesh_faces": 0,
                }
            )
            continue
        meas = measure_from_mask(binary, spacing_mm=SPACING, label=None)
        mesh = binary_to_mesh(binary, SPACING)
        if mesh is not None:
            mesh_path = case_dir / f"{short}.obj"
            mesh.export(str(mesh_path))
            meshes[region] = mesh
        metrics_rows.append(
            {
                "region": short,
                "n_voxels": int(binary.sum()),
                "volume_ml": float(meas.volume_cm3),
                "sphericity": float(meas.sphericity),
                "pca_a_mm": float(meas.principal_axis_lengths_mm[0]),
                "pca_b_mm": float(meas.principal_axis_lengths_mm[1]),
                "pca_c_mm": float(meas.principal_axis_lengths_mm[2]),
                "mesh_faces": int(len(mesh.faces)) if mesh is not None else 0,
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(case_dir / "morphometrics.csv", index=False)

    slice_path = case_dir / "slice_overlay.png"
    save_slice_overlay(
        flair if flair.std() > 0 else t1,
        regions,
        slice_path,
        title=f"{sid} · v3-lcc · {pick['t1c_status']}",
    )

    fig3d = plotly_regions_figure(meshes)
    fig3d.update_layout(title=f"{sid} · predicted meshes (v3-lcc)")
    html3d = case_dir / "mesh_3d.html"
    fig3d.write_html(str(html3d), include_plotlyjs="cdn", full_html=True)

    meta = {
        **pick,
        "case_dir": str(case_dir.relative_to(OUT)),
        "slice_overlay": str(slice_path.relative_to(OUT)),
        "mesh_html": str(html3d.relative_to(OUT)),
        "metrics": metrics.to_dict(orient="records"),
        "regions_nonzero": [REGION_SHORT[r] for r in BRATS_REGIONS if np.any(regions[r])],
    }
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_report_html(
    t1c_summary: dict[str, Any],
    cases: list[dict[str, Any]],
    heldout_note: str,
) -> Path:
    lim = f"""
    <div class="limitations">
      <h2>Limitations (read first)</h2>
      <ul>
        <li><b>No ground truth</b> on real patients — this report is <b>qualitative only</b>.</li>
        <li><b>Primary model = v3</b> (289-train BraTS scale-up, LCC). Held-out Dice
          ET 0.633 / TC 0.716 / WT 0.871. Ablation: v1→v2→v3 (see paper Table 2).</li>
        <li><b>LCC largest-component failure</b> remains possible (v3: 1/60 held-out,
          case 259): a larger false-positive blob can replace the true tumor (Dice→0).</li>
        <li><b>T1c:</b> processing log shows <b>{t1c_summary['n_genuine_t1c_in_modalities_found']}</b> genuine T1c
          among {t1c_summary['n_success']} successes; on-disk T1c is bit-identical to T1 for
          <b>{t1c_summary['n_t1c_bit_identical_to_t1']}</b> studies (stand-in).</li>
      </ul>
    </div>
    """
    pages = []
    for i, c in enumerate(cases, 1):
        flag = (
            '<span class="badge standin">T1 stand-in (no genuine T1c)</span>'
            if c["t1c_status"] == "T1_standin"
            else '<span class="badge genuine">Genuine T1c</span>'
        )
        rows = "".join(
            f"<tr><td>{m['region']}</td><td>{m['volume_ml']:.2f}</td>"
            f"<td>{m['sphericity']:.3f}</td>"
            f"<td>{m['pca_a_mm']:.1f} / {m['pca_b_mm']:.1f} / {m['pca_c_mm']:.1f}</td>"
            f"<td>{m['n_voxels']}</td></tr>"
            for m in c["metrics"]
        )
        pages.append(
            f"""
            <section class="case" id="case-{i}">
              <h2>Case {i}: {c['study_id']} {flag}</h2>
              <p class="note">{c['t1c_note']}. Nonzero regions: {', '.join(c['regions_nonzero']) or 'none'}.</p>
              <div class="grid">
                <div>
                  <h3>3-plane slice overlay</h3>
                  <img src="{c['slice_overlay']}" alt="slices" style="max-width:100%;"/>
                </div>
                <div>
                  <h3>Volumes &amp; morphometrics</h3>
                  <table>
                    <thead><tr><th>Region</th><th>Vol (mL)</th><th>Sphericity</th>
                    <th>PCA A/B/C (mm)</th><th>Voxels</th></tr></thead>
                    <tbody>{rows}</tbody>
                  </table>
                </div>
              </div>
              <h3>Plotly 3D meshes</h3>
              <iframe src="{c['mesh_html']}" style="width:100%;height:540px;border:1px solid #ddd;"></iframe>
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Real patients demo — v3-lcc</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 1.5rem auto; max-width: 1100px; color: #222; line-height: 1.45; }}
  .limitations {{ background: #fff8e6; border: 1px solid #e6d5a8; padding: 1rem 1.25rem;
                  border-radius: 8px; margin-bottom: 1.5rem; }}
  .badge {{ display:inline-block; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.85rem; }}
  .standin {{ background:#fde2e1; color:#8a1f11; }}
  .genuine {{ background:#e1f5e3; color:#1b5e20; }}
  .case {{ border-top: 2px solid #ddd; padding-top: 1rem; margin-top: 2rem; }}
  .grid {{ display:grid; grid-template-columns: 1.2fr 0.8fr; gap: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  .note {{ color: #555; font-size: 0.95rem; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
  <h1>Real-patient qualitative demo (v3-lcc)</h1>
  <p>Checkpoint: <code>checkpoints/brats_scale_full/best_model.pt</code> · postprocess: <code>lcc</code></p>
  {lim}
  <p>{heldout_note}</p>
  {"".join(pages)}
</body>
</html>
"""
    out = OUT / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    t1c = confirm_t1c_counts()
    (OUT / "t1c_standin_counts.json").write_text(json.dumps(t1c, indent=2), encoding="utf-8")
    print("=== T1c / stand-in counts ===")
    print(json.dumps(t1c, indent=2))

    picks = pick_four_patients()
    extremes = pd.read_csv(ROOT / "results" / "paper" / "tables" / "fig_a_wt_extremes_case_ids.csv")
    median_id = extremes.loc[extremes["rank"] == "median", "subject_id"].iloc[0]
    best_id = extremes.loc[extremes["rank"] == "best", "subject_id"].iloc[0]
    print("Patients:", [p["study_id"] for p in picks])
    print("BraTS median/best:", median_id, best_id)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = load_model_from_checkpoint(CKPT, device)
    model.float().eval()
    app = load_config()
    patients_root = Path(app.paths.processed) / "real_patients"
    brats_root = Path(app.paths.brats_nifti)

    case_metas = []
    for pick in picks:
        print(f"Processing patient {pick['study_id']} ...")
        case_metas.append(process_patient_case(model, device, pick, patients_root))

    # BraTS median + best overlays + mesh for paper
    paper_mesh_case = best_id
    for sid, tag in ((median_id, "median"), (best_id, "best")):
        print(f"BraTS overlay {tag} {sid} ...")
        flair, gt_wt, pred_raw, pred_lcc, dice = infer_brats_wt(
            model, device, sid, brats_root
        )
        out_png = PAPER_FIG / f"fig_seg_wt_{tag}_{sid}.png"
        save_brats_gt_pred_overlay(
            flair, gt_wt, pred_lcc, pred_raw, out_png, subject_id=sid, dice=dice
        )
        print(f"  wrote {out_png} dice={dice:.3f}")
        if sid == paper_mesh_case and np.any(pred_lcc):
            mesh = binary_to_mesh(pred_lcc, SPACING)
            if mesh is not None:
                mesh_png = PAPER_FIG / f"fig_mesh_wt_{sid}.png"
                mesh_static_png(
                    mesh,
                    mesh_png,
                    title=f"WT mesh · {sid} · v3-lcc (best held-out Dice)",
                )
                # Also keep interactive HTML next to paper figs
                fig = go.Figure(
                    data=[
                        mesh3d_from_trimesh(
                            mesh, name="WT", color="#55A868", opacity=0.85
                        )
                    ]
                )
                fig.update_layout(
                    title=f"WT mesh {sid} (v3-lcc)",
                    scene=dict(aspectmode="data"),
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                fig.write_html(
                    str(PAPER_FIG / f"fig_mesh_wt_{sid}.html"),
                    include_plotlyjs="cdn",
                    full_html=True,
                )
                print(f"  wrote mesh {mesh_png}")

    # Update extremes CSV overlay flags
    extremes = extremes.copy()
    extremes.loc[extremes["rank"] == "median", "overlay_available"] = True
    extremes.loc[extremes["rank"] == "best", "overlay_available"] = True
    extremes.to_csv(
        ROOT / "results" / "paper" / "tables" / "fig_a_wt_extremes_case_ids.csv",
        index=False,
    )

    heldout_note = (
        f"BraTS held-out overlays also written for median ({median_id}) and "
        f"best ({best_id}) under results/paper/figures/."
    )
    report = build_report_html(t1c, case_metas, heldout_note)
    print(f"\nReport → {report}")
    print("\n=== Output listing ===")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ROOT)}")
    for p in sorted(PAPER_FIG.glob("fig_seg_wt_*")):
        print(f"  {p.relative_to(ROOT)}")
    for p in sorted(PAPER_FIG.glob("fig_mesh_wt_*")):
        print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
