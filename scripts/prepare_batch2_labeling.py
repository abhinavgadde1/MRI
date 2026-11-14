#!/usr/bin/env python3
"""Build ITK-SNAP manual-correction packages for native-T1c batch2 studies.

Writes only under outputs/batch2_labeling/ (reads from
data/processed/real_patients_batch2/). Does not train, touch checkpoints, or
modify data/processed/real_patients/.
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi

from preprocessing.h5_to_nifti import MODALITY_NAMES
from preprocessing.registration import REGISTERED_NAMES
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS
from segmentation.postprocess import apply_channel_postprocess, enforce_brats_nesting
from segmentation.pseudo_label import get_patient_inference_transforms
from segmentation.train import post_transforms

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "real_patients_batch2"
OUT = ROOT / "outputs" / "batch2_labeling"
CKPT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
FORBIDDEN = ROOT / "data" / "processed" / "real_patients"

NATIVE_IDS = [
    "batch2_01",
    "batch2_02",
    "batch2_03",
    "batch2_05",
    "batch2_06",
    "batch2_07",
    "batch2_08",
    "batch2_10",
    "batch2_12",
    "batch2_13",
]

REG_COPY = {
    "T1": ("T1.nii.gz", REGISTERED_NAMES["T1"]),
    "T1c": ("T1c.nii.gz", REGISTERED_NAMES["T1c"]),
    "T2": ("T2.nii.gz", REGISTERED_NAMES["T2"]),
    "FLAIR": ("FLAIR.nii.gz", REGISTERED_NAMES["FLAIR"]),
}

# MODALITY_NAMES = (FLAIR, T1, T1c, T2) — must match v3 training.
# A prior bug used (T1c, T1, T2, FLAIR) and produced stale v3_draft_label.nii.gz.
ROI = (96, 96, 96)
VOXEL_ML = 0.001


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def brats_label_124(et: np.ndarray, tc: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Classic BraTS labels: 1=NCR/NET, 2=ED, 4=ET (user/ITK-SNAP convention)."""
    et_b = et.astype(bool)
    tc_b = tc.astype(bool)
    wt_b = wt.astype(bool)
    label = np.zeros(wt_b.shape, dtype=np.uint8)
    label[wt_b & ~tc_b] = 2
    label[tc_b & ~et_b] = 1
    label[et_b] = 4
    return label


@torch.no_grad()
def infer_lcc_label(model, dev: torch.device, study_dir: Path) -> tuple[np.ndarray, dict]:
    reg = study_dir / "04_registered_1mm"
    record = {
        "study_id": study_dir.name,
        "image": [str(reg / REGISTERED_NAMES[m]) for m in MODALITY_NAMES],
    }
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(dev, dtype=torch.float32)
    logits = sliding_window_inference(
        image, roi_size=ROI, sw_batch_size=1, predictor=model
    )
    post_pred, _ = post_transforms()
    pred_raw = (
        post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(np.float32)
    )
    pred = apply_channel_postprocess(pred_raw.copy(), mode="lcc")
    pred = enforce_brats_nesting(pred)
    regions = {r: pred[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    label = brats_label_124(
        regions["enhancing_tumor"],
        regions["tumor_core"],
        regions["whole_tumor"],
    )
    vols = {
        "WT_ml": round(float(regions["whole_tumor"].sum()) * VOXEL_ML, 3),
        "TC_ml": round(float(regions["tumor_core"].sum()) * VOXEL_ML, 3),
        "ET_ml": round(float(regions["enhancing_tumor"].sum()) * VOXEL_ML, 3),
    }
    return label, vols


def copy_registered(study_dir: Path, out_dir: Path) -> dict[str, Path]:
    reg = study_dir / "04_registered_1mm"
    paths = {}
    for key, (dst_name, src_name) in REG_COPY.items():
        src = reg / src_name
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = out_dir / dst_name
        shutil.copy2(src, dst)
        paths[key] = dst
    return paths


def copy_native_resolution(study_dir: Path, out_dir: Path, selection: dict) -> dict:
    """Copy pre-resample DICOM→NIfTI series for T2/FLAIR (and T1/T1c for reference)."""
    native = {}
    for mod in ("T1", "T1c", "T2", "FLAIR"):
        info = selection.get(mod) or {}
        src = info.get("nifti_path")
        if not src:
            continue
        src_p = Path(src)
        if not src_p.is_file():
            # try relative under study
            alt = study_dir / "01_dicom_nifti" / study_dir.name / mod
            if alt.is_dir():
                desc = (info.get("series_description") or "").replace(" ", "_")
                cands = list(alt.glob("*.nii.gz"))
                # prefer filename containing safe desc tokens
                pick = None
                for c in cands:
                    if desc and desc[:12] in c.name:
                        pick = c
                        break
                if pick is None and cands:
                    # largest by matching n_slices via sidecar
                    best = None
                    for c in cands:
                        side = c.with_suffix("").with_suffix(".json")
                        n = 0
                        if side.is_file():
                            n = int(json.loads(side.read_text()).get("number_of_instances") or 0)
                        if best is None or n > best[0]:
                            best = (n, c)
                    pick = best[1] if best else None
                src_p = pick if pick else src_p
        if not src_p.is_file():
            continue
        dst = out_dir / f"{mod}_native.nii.gz"
        shutil.copy2(src_p, dst)
        side_src = src_p.with_name(src_p.name.replace(".nii.gz", ".json"))
        if side_src.is_file():
            shutil.copy2(side_src, out_dir / f"{mod}_native.json")
        native[mod] = {
            "path": str(dst.relative_to(out_dir)),
            "source": str(src_p),
            "series_description": info.get("series_description"),
            "n_slices": info.get("n_slices"),
            "slice_thickness_mm": info.get("slice_thickness_mm"),
        }
    return native


def write_readme(
    out_dir: Path,
    study_id: str,
    selection: dict,
    vols: dict,
    native: dict,
) -> None:
    lines = [
        f"# {study_id} — manual correction package (DRAFT, not truth)\n",
        "\n",
        "All `T1.nii.gz` / `T1c.nii.gz` / `T2.nii.gz` / `FLAIR.nii.gz` / "
        "`v3_draft_label.nii.gz` share the **registered 1 mm grid** "
        "(same affine/shape as pipeline T1c). Corrected masks must stay on this grid.\n",
        "\n",
        "## Label values (classic BraTS)\n",
        "- `0` background\n",
        "- `1` necrotic / non-enhancing core (NCR/NET)\n",
        "- `2` peritumoral edema (ED)\n",
        "- `4` enhancing tumor (ET)\n",
        "\n",
        "`v3_draft_label.nii.gz` is a **v3+LCC draft for correction**, not ground truth. "
        "Where the draft is clearly wrong, redraw from scratch rather than editing the draft.\n",
        "\n",
        "## Channels (series used by the pipeline)\n",
        "\n",
        "| channel | SeriesDescription | native n_slices | native thickness_mm |\n",
        "|---|---|---:|---:|\n",
    ]
    for mod in ("T1", "T1c", "T2", "FLAIR"):
        info = selection.get(mod) or {}
        lines.append(
            f"| {mod} | {info.get('series_description')} | "
            f"{info.get('n_slices')} | {info.get('slice_thickness_mm')} |\n"
        )
    lines.append("\n## v3+LCC draft volumes (mL)\n\n")
    lines.append(f"- WT: **{vols['WT_ml']}** mL\n")
    lines.append(f"- TC: **{vols['TC_ml']}** mL\n")
    lines.append(f"- ET: **{vols['ET_ml']}** mL\n")
    lines.append("\n## Native-resolution copies (uninterpolated)\n\n")
    for mod, info in native.items():
        lines.append(
            f"- `{mod}_native.nii.gz` ← `{info.get('series_description')}` "
            f"({info.get('n_slices')} slices @ {info.get('slice_thickness_mm')} mm)\n"
        )
    lines.append(
        "\nUse native T2/FLAIR to check findings; edit labels only on the 1 mm registered grid.\n"
    )
    (out_dir / "README.md").write_text("".join(lines), encoding="utf-8")


def write_itksnap_workspace(out_dir: Path, study_id: str) -> Path:
    """Minimal ITK-SNAP 3.x/4.x workspace XML (T1c main, overlays, draft seg)."""
    # Relative paths so the folder is portable.
    files = {
        "main": "T1c.nii.gz",
        "overlays": ["FLAIR.nii.gz", "T2.nii.gz", "T1.nii.gz"],
        "seg": "v3_draft_label.nii.gz",
    }
    root = ET.Element("ITKSnap")
    root.set("version", "2.0")
    project = ET.SubElement(root, "project")
    layers = ET.SubElement(project, "layers")

    main = ET.SubElement(layers, "layer")
    main.set("type", "main")
    ET.SubElement(main, "filename").text = files["main"]
    ET.SubElement(main, "nickname").text = "T1c"

    for nick, fname in zip(("FLAIR", "T2", "T1"), files["overlays"]):
        lay = ET.SubElement(layers, "layer")
        lay.set("type", "overlay")
        ET.SubElement(lay, "filename").text = fname
        ET.SubElement(lay, "nickname").text = nick
        ET.SubElement(lay, "opacity").text = "0.35"

    seg = ET.SubElement(layers, "layer")
    seg.set("type", "segmentation")
    ET.SubElement(seg, "filename").text = files["seg"]
    ET.SubElement(seg, "nickname").text = "v3_draft"

    # Pretty XML
    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    # drop XML declaration quirks from minidom empty lines
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    text = "\n".join(lines) + "\n"
    path = out_dir / f"{study_id}.itksnap"
    path.write_text(text, encoding="utf-8")

    cmd = (
        f'itksnap -g "{out_dir / "T1c.nii.gz"}" '
        f'-s "{out_dir / "v3_draft_label.nii.gz"}" '
        f'-o "{out_dir / "FLAIR.nii.gz"}" '
        f'"{out_dir / "T2.nii.gz"}" '
        f'"{out_dir / "T1.nii.gz"}"'
    )
    (out_dir / "open_itksnap.sh").write_text(
        "#!/bin/zsh\n"
        "# Open this study in ITK-SNAP (requires `itksnap` on PATH).\n"
        "# Prefer the .itksnap workspace if your ITK-SNAP version loads it;\n"
        "# otherwise run this script.\n"
        f'cd "{out_dir}"\n'
        f"{cmd}\n",
        encoding="utf-8",
    )
    (out_dir / "open_itksnap.sh").chmod(0o755)
    (out_dir / "OPEN.txt").write_text(
        "ITK-SNAP open command (from repo root or any cwd):\n\n"
        f"{cmd}\n\n"
        f"Or: open workspace {path.name} from ITK-SNAP File → Open Workspace "
        "(if supported by your build).\n",
        encoding="utf-8",
    )
    return path


def _norm_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(float)
    p2, p98 = np.percentile(sl[np.isfinite(sl)], (2, 98)) if np.any(np.isfinite(sl)) else (0, 1)
    if p98 <= p2:
        return np.zeros_like(sl)
    return np.clip((sl - p2) / (p98 - p2), 0, 1)


def _overlay_contour(rgb: np.ndarray, label_sl: np.ndarray) -> np.ndarray:
    """Draw draft contours: 1=red-ish, 2=green, 4=yellow."""
    out = rgb.copy()
    for val, color in (
        (2, np.array([0.15, 0.9, 0.25])),
        (1, np.array([0.95, 0.25, 0.2])),
        (4, np.array([1.0, 0.95, 0.15])),
    ):
        m = label_sl == val
        if not np.any(m):
            continue
        # contour = mask XOR eroded
        er = ndi.binary_erosion(m, iterations=1)
        edge = m & ~er
        if not np.any(edge):
            edge = m
        out[edge] = color
    return out


def save_review_montage(
    out_dir: Path,
    t1c: np.ndarray,
    flair: np.ndarray,
    t2: np.ndarray,
    label: np.ndarray,
    *,
    study_id: str,
    wt_ml: float,
) -> Path:
    wt = label > 0
    if wt_ml < 1.0 or not np.any(wt):
        # Full-brain: every 4th axial slice of FLAIR and T1c
        zs = list(range(0, flair.shape[2], 4))
        if not zs:
            zs = [flair.shape[2] // 2]
        n = len(zs)
        fig, axs = plt.subplots(2, n, figsize=(max(2.0 * n, 8), 5.5), dpi=120)
        if n == 1:
            axs = np.array([[axs[0]], [axs[1]]])
        for i, z in enumerate(zs):
            for row, vol, name in (
                (0, flair, "FLAIR"),
                (1, t1c, "T1c"),
            ):
                sl = _norm_slice(vol[:, :, z])
                rgb = np.stack([sl, sl, sl], axis=-1)
                rgb = _overlay_contour(rgb, label[:, :, z])
                axs[row, i].imshow(np.rot90(rgb), origin="lower")
                axs[row, i].set_axis_off()
                if i == 0:
                    axs[row, i].set_ylabel(name, fontsize=8)
                axs[row, i].set_title(f"z={z}", fontsize=7)
        fig.suptitle(
            f"{study_id}  full-brain montage (WT={wt_ml} mL < 1 — check for missed lesion)",
            fontsize=10,
        )
    else:
        zs_present = np.where(wt.any(axis=(0, 1)))[0]
        z0, z1 = int(zs_present.min()), int(zs_present.max())
        if z1 <= z0:
            picks = [z0]
        else:
            picks = [
                int(round(z))
                for z in np.linspace(z0, z1, num=6)
            ]
            # unique preserve order
            seen = set()
            uniq = []
            for z in picks:
                if z not in seen:
                    seen.add(z)
                    uniq.append(z)
            picks = uniq
            while len(picks) < 6:
                picks.append(picks[-1])
            picks = picks[:6]
        fig, axs = plt.subplots(3, 6, figsize=(14, 7), dpi=120)
        for i, z in enumerate(picks):
            for row, vol, name in (
                (0, t1c, "T1c"),
                (1, flair, "FLAIR"),
                (2, t2, "T2"),
            ):
                sl = _norm_slice(vol[:, :, z])
                rgb = np.stack([sl, sl, sl], axis=-1)
                rgb = _overlay_contour(rgb, label[:, :, z])
                axs[row, i].imshow(np.rot90(rgb), origin="lower")
                axs[row, i].set_axis_off()
                if i == 0:
                    axs[row, i].set_ylabel(name, fontsize=9)
                if row == 0:
                    axs[row, i].set_title(f"z={z}", fontsize=8)
        fig.suptitle(
            f"{study_id}  v3 draft contours on T1c / FLAIR / T2  (WT={wt_ml} mL)  "
            "1=NCR 2=ED 4=ET",
            fontsize=10,
        )
    fig.tight_layout()
    path = out_dir / "review_montage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_instructions(rows: list[dict]) -> None:
    lines = [
        "# Batch2 native-T1c labeling instructions (ITK-SNAP)\n\n",
        "These packages are for **manual correction** of v3+LCC drafts on the "
        "10 studies with verified native post-contrast T1c. "
        "Do **not** treat `v3_draft_label.nii.gz` as truth.\n\n",
        "## Label values (classic BraTS)\n\n",
        "| value | meaning |\n|---:|---|\n",
        "| 0 | background |\n",
        "| 1 | necrotic / non-enhancing core (NCR/NET) |\n",
        "| 2 | peritumoral edema (ED) |\n",
        "| 4 | enhancing tumor (ET) |\n\n",
        "## Grid / coordinates\n\n",
        "Edit labels **only** on the registered 1 mm volumes "
        "(`T1c.nii.gz` / `T1.nii.gz` / `T2.nii.gz` / `FLAIR.nii.gz` / "
        "`v3_draft_label.nii.gz`). They share one voxel grid and affine so "
        "corrected masks can drop into training later. "
        "`*_native.nii.gz` files are uninterpolated references (often ~5 mm "
        "T2/FLAIR) — use them to check findings, not as the edit target.\n\n",
        "## ITK-SNAP steps\n\n",
        "1. Open the study folder `outputs/batch2_labeling/<study_id>/`.\n",
        "2. Preferred: File → Open Workspace → `<study_id>.itksnap` "
        "(if your ITK-SNAP build supports workspaces).\n",
        "3. Otherwise run `./open_itksnap.sh` in that folder, or the command in "
        "`OPEN.txt` (main image = T1c, overlays = FLAIR/T2/T1, segmentation = draft).\n",
        "4. Paint labels with values **1 / 2 / 4** only (active label selector).\n",
        "5. Save the segmentation as `corrected_label.nii.gz` in the same folder "
        "(keep the same grid as `T1c.nii.gz`).\n",
        "6. Check `review_montage.png` first; for WT &lt; 1 mL the montage is "
        "full-brain so a missed lesion is easier to spot.\n\n",
        "## Anchoring warning\n\n",
        "If the draft is clearly wrong (scattered speckles, wrong lobe, empty "
        "where tumor is obvious on FLAIR/T1c), **redraw from scratch** instead of "
        "nibbling the draft. Anchoring on v3 mistakes wastes time and biases labels.\n\n",
        "## Study checklist\n\n",
        "| study_id | v3 WT (mL) | v3 TC (mL) | v3 ET (mL) | montage | reviewed |\n",
        "|---|---:|---:|---:|---|---|\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['study_id']} | {r['WT_ml']} | {r['TC_ml']} | {r['ET_ml']} | "
            f"`{r['study_id']}/review_montage.png` | [ ] |\n"
        )
    lines.append(
        "\n## Paths\n\n"
        "- Packages: `outputs/batch2_labeling/<study_id>/`\n"
        "- Source registered volumes: `data/processed/real_patients_batch2/"
        "<study_id>/04_registered_1mm/` (read-only for labeling)\n"
    )
    (OUT / "INSTRUCTIONS.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    assert FORBIDDEN.resolve() not in OUT.resolve().parents
    assert CKPT.is_file(), f"missing checkpoint {CKPT}"
    OUT.mkdir(parents=True, exist_ok=True)

    # Snapshot before volumes from prior preview if useful
    prev_csv = ROOT / "outputs" / "batch2_native_t1c" / "after_native_preview_summary.csv"
    prev = pd.read_csv(prev_csv) if prev_csv.is_file() else pd.DataFrame()

    dev = device()
    model = load_model_from_checkpoint(CKPT, dev)
    model.eval()
    print(f"device={dev} out={OUT}", flush=True)

    rows = []
    for sid in NATIVE_IDS:
        study = PROC / sid
        if not study.is_dir():
            raise FileNotFoundError(study)
        out_dir = OUT / sid
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        print(f"==== {sid} ====", flush=True)
        selection = {}
        sel_path = study / "series_selection.json"
        if sel_path.is_file():
            selection = json.loads(sel_path.read_text(encoding="utf-8"))

        paths = copy_registered(study, out_dir)
        native = copy_native_resolution(study, out_dir, selection)

        label, vols = infer_lcc_label(model, dev, study)
        ref = nib.load(str(paths["T1c"]))
        # Ensure same shape as T1c grid
        if label.shape != ref.shape:
            raise RuntimeError(
                f"{sid}: label shape {label.shape} != T1c {ref.shape}"
            )
        nib.save(
            nib.Nifti1Image(label.astype(np.uint8), ref.affine, ref.header),
            str(out_dir / "v3_draft_label.nii.gz"),
        )

        write_readme(out_dir, sid, selection, vols, native)
        write_itksnap_workspace(out_dir, sid)

        t1c = nib.load(str(paths["T1c"])).get_fdata()
        flair = nib.load(str(paths["FLAIR"])).get_fdata()
        t2 = nib.load(str(paths["T2"])).get_fdata()
        save_review_montage(
            out_dir,
            t1c,
            flair,
            t2,
            label,
            study_id=sid,
            wt_ml=float(vols["WT_ml"]),
        )

        # Prefer freshly inferred volumes; fall back to previous preview CSV
        if prev is not None and len(prev):
            hit = prev[prev.study_id == sid]
            if len(hit) and abs(float(hit.iloc[0]["WT_ml"]) - vols["WT_ml"]) > 5:
                print(
                    f"  note: WT now {vols['WT_ml']} vs prior preview "
                    f"{hit.iloc[0]['WT_ml']}",
                    flush=True,
                )

        meta = {
            "study_id": sid,
            "native_t1c": True,
            "volumes_ml": vols,
            "label_convention": "BraTS classic 1=NCR, 2=ED, 4=ET",
            "grid": "registered_1mm_same_as_T1c",
            "channels": {
                m: {
                    "series_description": (selection.get(m) or {}).get(
                        "series_description"
                    ),
                    "n_slices": (selection.get(m) or {}).get("n_slices"),
                    "slice_thickness_mm": (selection.get(m) or {}).get(
                        "slice_thickness_mm"
                    ),
                }
                for m in ("T1", "T1c", "T2", "FLAIR")
            },
            "native_files": native,
            "draft": "v3_draft_label.nii.gz",
        }
        (out_dir / "package_meta.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
        )
        rows.append({"study_id": sid, **vols})
        print(
            f"  packaged WT={vols['WT_ml']} TC={vols['TC_ml']} ET={vols['ET_ml']}",
            flush=True,
        )
        if dev.type == "mps":
            torch.mps.empty_cache()

    write_instructions(rows)
    pd.DataFrame(rows).to_csv(OUT / "study_volumes.csv", index=False)
    print(f"Wrote {OUT / 'INSTRUCTIONS.md'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
