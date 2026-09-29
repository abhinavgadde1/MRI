"""Single-case end-to-end: ingest → segment → reconstruct → validate → report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import trimesh
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi

from config import PROJECT_ROOT
from preprocessing.registration import REGISTERED_NAMES, run_nifti_preprocessing, run_patient_preprocessing
from reconstruction.measurements import measure_from_mask
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS, build_brats_model
from segmentation.postprocess import apply_channel_postprocess
from segmentation.pseudo_label import (
    get_patient_inference_transforms,
    registered_dir_to_datadict,
    regions_to_label_map,
)
from segmentation.train import post_transforms
from validation.ellipsoid import diameters_from_binary, ellipsoid_volume_cm3, voxel_volume_ml_from_binary

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
SPACING = (1.0, 1.0, 1.0)
REGION_SHORT = {
    "enhancing_tumor": "ET",
    "tumor_core": "TC",
    "whole_tumor": "WT",
}


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _looks_like_dicom_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".dcm", ""}:
            # Heuristic: presence of .dcm or many files without nifti
            if p.suffix.lower() == ".dcm":
                return True
    # Nested DICOM folders often lack extensions — check for no NIfTI but many files
    niftis = list(path.rglob("*.nii*"))
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(niftis) == 0 and len(files) >= 10


def _find_registered_dir(path: Path) -> Path | None:
    cand = path / "04_registered_1mm"
    if cand.is_dir() and all((cand / REGISTERED_NAMES[m]).is_file() for m in REGISTERED_NAMES):
        return cand
    if all((path / REGISTERED_NAMES[m]).is_file() for m in REGISTERED_NAMES):
        return path
    return None


def _write_synthetic_registered(out_reg: Path) -> Path:
    """Create a tiny synthetic 4-modality registered folder (no BraTS/DICOM needed)."""
    out_reg.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    shape = (64, 64, 48)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    tumor = ((xx - 32) ** 2 + (yy - 32) ** 2 + (zz - 24) ** 2) < 10**2
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    for name in REGISTERED_NAMES.values():
        vol = rng.normal(0.4, 0.05, size=shape).astype(np.float32)
        vol[tumor] += 0.8
        nib.save(nib.Nifti1Image(vol, affine), str(out_reg / name))
    return out_reg


def _load_or_init_model(
    checkpoint: Path,
    device: torch.device,
    *,
    allow_random_weights: bool,
) -> torch.nn.Module:
    if checkpoint.is_file():
        logger.info("Loading checkpoint %s", checkpoint)
        model = load_model_from_checkpoint(checkpoint, device)
        model.float().eval()
        return model
    if not allow_random_weights:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "Place v3 weights there, pass --checkpoint, or use --allow-random-weights / --synthetic."
        )
    logger.warning("Checkpoint missing — using randomly initialized SegResNet (demo only)")
    model = build_brats_model("segresnet").to(device)
    model.float().eval()
    return model


@torch.no_grad()
def _segment_registered(
    model: torch.nn.Module,
    device: torch.device,
    registered_dir: Path,
    case_id: str,
    *,
    postprocess: str = "lcc",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    record = registered_dir_to_datadict(registered_dir, case_id)
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(device, dtype=torch.float32)
    logits = sliding_window_inference(
        image, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
    )
    from monai.data import decollate_batch

    post_pred, _ = post_transforms()
    pred = post_pred(decollate_batch(logits)[0]).detach().cpu().numpy()
    if postprocess != "raw":
        pred = apply_channel_postprocess(pred.astype(np.float32), mode=postprocess)  # type: ignore[arg-type]
    regions = {r: pred[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    display = data["image"][0].detach().cpu().numpy()
    return regions, display


def _binary_to_mesh(binary: np.ndarray, spacing: tuple[float, float, float]) -> trimesh.Trimesh | None:
    from skimage.measure import marching_cubes

    if not np.any(binary):
        return None
    try:
        verts, faces, *_ = marching_cubes(
            binary.astype(np.float32), level=0.5, spacing=spacing
        )
    except (ValueError, RuntimeError):
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _save_overlay(
    image_3d: np.ndarray,
    regions: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
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

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ttl, axis, idx in zip(
        axs,
        ("axial (z)", "coronal (y)", "sagittal (x)"),
        (2, 1, 0),
        (
            int(np.clip(cz, 0, image_3d.shape[2] - 1)),
            int(np.clip(cy, 0, image_3d.shape[1] - 1)),
            int(np.clip(cx, 0, image_3d.shape[0] - 1)),
        ),
    ):
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
    fig.suptitle(title + "\ngreen=WT  blue=TC  red=ET")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_report(
    out_dir: Path,
    *,
    case_id: str,
    checkpoint: Path,
    postprocess: str,
    metrics: pd.DataFrame,
    ellipsoid: dict[str, Any],
    notes: list[str],
) -> Path:
    rows = "".join(
        f"<tr><td>{r.region}</td><td>{r.volume_ml:.2f}</td>"
        f"<td>{r.sphericity:.3f}</td><td>{int(r.mesh_faces)}</td></tr>"
        for r in metrics.itertuples()
    )
    note_html = "".join(f"<li>{n}</li>" for n in notes)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Pipeline report — {case_id}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 1.5rem auto; max-width: 960px; color: #222; line-height: 1.45; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
 th {{ background: #f4f4f4; }}
 .meta {{ background: #f7f9fc; padding: 1rem; border-radius: 8px; }}
 img {{ max-width: 100%; }}
</style></head><body>
<h1>Single-case pipeline report</h1>
<div class="meta">
  <p><b>Case:</b> {case_id}</p>
  <p><b>Checkpoint:</b> <code>{checkpoint}</code></p>
  <p><b>Postprocess:</b> {postprocess}</p>
</div>
<h2>Morphometrics (predicted, LCC)</h2>
<table><tr><th>Region</th><th>Volume (mL)</th><th>Sphericity</th><th>Mesh faces</th></tr>
{rows}</table>
<h2>Ellipsoid validation (WT)</h2>
<pre>{json.dumps(ellipsoid, indent=2)}</pre>
<h2>Slice overlay</h2>
<p><img src="slice_overlay.png" alt="overlay"/></p>
<h2>Notes</h2>
<ul>{note_html}</ul>
<p class="meta">See also <code>results/paper/LIMITATIONS.md</code>.</p>
</body></html>
"""
    report = out_dir / "report.html"
    report.write_text(html, encoding="utf-8")
    return report


def run_case(
    *,
    input_dir: Path | None,
    output_dir: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    postprocess: str = "lcc",
    device: str | None = None,
    allow_random_weights: bool = False,
    synthetic: bool = False,
) -> Path:
    """Run ingest → segment → reconstruct → validate → report for one case."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = _device(device)
    notes: list[str] = []

    if synthetic:
        case_id = "synthetic_demo"
        reg = _write_synthetic_registered(output_dir / "04_registered_1mm")
        notes.append("Synthetic 64×64×48 multi-modal volume (no DICOM/BraTS required).")
        allow_random_weights = True
    else:
        assert input_dir is not None
        input_dir = Path(input_dir)
        case_id = input_dir.name
        reg = _find_registered_dir(input_dir)
        if reg is None and _looks_like_dicom_dir(input_dir):
            logger.info("Ingesting DICOM study %s", input_dir)
            prep_out = output_dir / "preprocess"
            result = run_patient_preprocessing(
                input_dir,
                prep_out,
                study_name=case_id,
                use_hd_bet=False,
            )
            reg = _find_registered_dir(prep_out)
            if reg is None and result.registered_1mm:
                # Symlink/copy into expected layout if needed
                reg = prep_out / "04_registered_1mm"
            if reg is None:
                raise RuntimeError(f"Preprocessing did not produce registered NIfTIs under {prep_out}")
            notes.append(f"Ingested DICOM from {input_dir}.")
        elif reg is None:
            # Try treating input as modality NIfTI map via run_nifti_preprocessing
            modality_paths = {}
            for key, fname in REGISTERED_NAMES.items():
                for cand in (input_dir / fname, input_dir / f"{key}.nii.gz", input_dir / f"{key}.nii"):
                    if cand.is_file():
                        modality_paths[key] = cand
                        break
            if len(modality_paths) < 2:
                raise FileNotFoundError(
                    f"Could not find DICOM series or registered NIfTIs under {input_dir}"
                )
            logger.info("Running NIfTI preprocessing on %s", input_dir)
            prep_out = output_dir / "preprocess"
            run_nifti_preprocessing(
                modality_paths,
                prep_out,
                study_name=case_id,
                use_hd_bet=False,
            )
            reg = _find_registered_dir(prep_out)
            if reg is None:
                raise RuntimeError(f"NIfTI preprocessing failed to write registered volumes in {prep_out}")
            notes.append(f"Preprocessed NIfTI folder {input_dir}.")
        else:
            notes.append(f"Using existing registered NIfTIs at {reg}.")

    model = _load_or_init_model(
        Path(checkpoint), device_t, allow_random_weights=allow_random_weights
    )
    if not Path(checkpoint).is_file():
        notes.append("Used randomly initialized weights (demo/smoke only — not for clinical use).")

    regions, display = _segment_registered(
        model, device_t, reg, case_id, postprocess=postprocess
    )

    # Save prediction
    label = regions_to_label_map(
        regions["enhancing_tumor"], regions["tumor_core"], regions["whole_tumor"]
    )
    ref = nib.load(str(reg / REGISTERED_NAMES["T1"]))
    nib.save(
        nib.Nifti1Image(label.astype(np.uint8), ref.affine, ref.header),
        str(output_dir / "pred_label_lcc.nii.gz"),
    )

    # Reconstruct + morphometrics
    rows: list[dict[str, Any]] = []
    for region in BRATS_REGIONS:
        binary = regions[region]
        short = REGION_SHORT[region]
        if not np.any(binary):
            rows.append(
                {
                    "region": short,
                    "volume_ml": 0.0,
                    "sphericity": float("nan"),
                    "mesh_faces": 0,
                }
            )
            continue
        meas = measure_from_mask(binary, spacing_mm=SPACING, label=None)
        mesh = _binary_to_mesh(binary, SPACING)
        if mesh is not None:
            mesh.export(str(output_dir / f"{short}.obj"))
        rows.append(
            {
                "region": short,
                "volume_ml": float(meas.volume_cm3),
                "sphericity": float(meas.sphericity),
                "mesh_faces": int(len(mesh.faces)) if mesh is not None else 0,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "morphometrics.csv", index=False)

    # Ellipsoid validate WT
    wt = regions["whole_tumor"]
    ellipsoid: dict[str, Any] = {"n_wt_voxels": int(wt.sum())}
    if np.any(wt):
        voxel_ml = voxel_volume_ml_from_binary(wt, SPACING)
        diameters, _ = diameters_from_binary(wt, SPACING, method="radiologist")
        ellipsoid.update(
            {
                "voxel_volume_ml": float(voxel_ml),
                "radiologist_diameters_mm": [float(x) for x in diameters],
                "abc_over_2_ml": float(
                    ellipsoid_volume_cm3(diameters, formula="abc_over_2")
                ),
                "pi_over_6_ml": float(
                    ellipsoid_volume_cm3(diameters, formula="pi_over_6")
                ),
            }
        )
    (output_dir / "ellipsoid_wt.json").write_text(json.dumps(ellipsoid, indent=2))

    _save_overlay(
        display,
        regions,
        output_dir / "slice_overlay.png",
        title=f"{case_id} · {postprocess}",
    )

    meta = {
        "case_id": case_id,
        "checkpoint": str(checkpoint),
        "postprocess": postprocess,
        "registered_dir": str(reg),
        "device": str(device_t),
        "notes": notes,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = _write_report(
        output_dir,
        case_id=case_id,
        checkpoint=Path(checkpoint),
        postprocess=postprocess,
        metrics=metrics,
        ellipsoid=ellipsoid,
        notes=notes,
    )
    logger.info("Wrote report %s", report)
    return report
