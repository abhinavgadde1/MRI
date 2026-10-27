#!/usr/bin/env python3
"""Full-FOV SynthSeg (isolated .venv-synthseg).

Default inference is overlapping native-resolution 192³ patches (logit average
in overlap), then labels are written at the T1 grid and zeroed inside v3 WT.

``--mode letterbox`` keeps the older full-FOV downsample-to-192³ path.

Standalone prototype defaults to BraTS20_Training_001. The API opt-in path
passes ``--t1``, ``--wt``, ``--out`` (and optional ``--json`` / ``--mode``).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import nibabel as nib
import numpy as np
import torch
from dipy.nn.torch.synthseg import SynthSeg
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
SID = "BraTS20_Training_001"
T1_PATH = ROOT / "data" / "processed" / "brats_nifti" / SID / "T1.nii.gz"
OUT = ROOT / "outputs" / "feasibility_parcellation"
WT_PATH = OUT / f"{SID}_v3_WT.nii.gz"


def letterbox_to_cube(vol: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Downsample the *full* volume so the longest axis fits ``size``, then pad.

    Unlike DIPY ``transform_img(..., final_size=192)`` this does not crop.
    """
    shape = np.array(vol.shape, dtype=float)
    scale = float(size) / float(shape.max())
    new_shape = np.maximum(1, np.round(shape * scale).astype(int))
    zooms = new_shape / np.array(vol.shape)
    zoomed = ndi.zoom(vol, zooms, order=3)
    cube = np.zeros((size, size, size), dtype=np.float32)
    starts = [(size - s) // 2 for s in zoomed.shape]
    sl = tuple(slice(st, st + s) for st, s in zip(starts, zoomed.shape))
    cube[sl] = zoomed.astype(np.float32)
    return cube, {
        "size": size,
        "orig_shape": list(vol.shape),
        "new_shape": [int(x) for x in zoomed.shape],
        "starts": [int(x) for x in starts],
        "scale": scale,
    }


def unletterbox(labels_cube: np.ndarray, meta: dict) -> np.ndarray:
    starts = meta["starts"]
    new_shape = meta["new_shape"]
    orig = tuple(meta["orig_shape"])
    sl = tuple(slice(st, st + s) for st, s in zip(starts, new_shape))
    cropped = labels_cube[sl]
    zooms = np.array(orig) / np.array(cropped.shape)
    back = ndi.zoom(cropped, zooms, order=0)
    # Fix off-by-one from rounding
    out = np.zeros(orig, dtype=np.int32)
    sl_out = tuple(slice(0, min(a, b)) for a, b in zip(orig, back.shape))
    sl_src = tuple(slice(0, min(a, b)) for a, b in zip(orig, back.shape))
    out[sl_out] = np.round(back[sl_src]).astype(np.int32)
    return out


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), 0, 1).astype(np.float32)


def pad_to_min(vol: np.ndarray, min_size: int) -> tuple[np.ndarray, list[int]]:
    """Pad each axis to at least ``min_size`` (even split). Returns padded vol + pad_before."""
    pads = []
    slices_unpad = []
    for s in vol.shape:
        need = max(0, min_size - s)
        before = need // 2
        after = need - before
        pads.append((before, after))
        slices_unpad.append(slice(before, before + s))
    if any(a + b > 0 for a, b in pads):
        vol = np.pad(vol, pads, mode="constant")
    return vol, [int(p[0]) for p in pads]


def overlapping_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def run_patches(
    model,
    device,
    t1: np.ndarray,
    patch: int = 192,
    stride: int = 48,
) -> tuple[np.ndarray, float, str]:
    """Native-resolution overlapping 192³ patches, logit average in overlap."""
    padded, pad_before = pad_to_min(t1, patch)
    acc = None
    nmap = None
    starts_x = overlapping_starts(padded.shape[0], patch, stride)
    starts_y = overlapping_starts(padded.shape[1], patch, stride)
    starts_z = overlapping_starts(padded.shape[2], patch, stride)
    coords = [(x, y, z) for x in starts_x for y in starts_y for z in starts_z]
    print(f"  {len(coords)} patches covering {padded.shape} (orig {t1.shape})")
    t0 = time.perf_counter()
    for i, (x0, y0, z0) in enumerate(coords, start=1):
        sl = (
            slice(x0, x0 + patch),
            slice(y0, y0 + patch),
            slice(z0, z0 + patch),
        )
        cube = normalize_volume(padded[sl].astype(np.float32))
        xt = torch.tensor(cube[None, None], dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits = model.model(xt)
            pred = logits.detach().cpu().numpy()[0]  # (C, 192, 192, 192)
        if acc is None:
            acc = np.zeros((pred.shape[0],) + padded.shape, dtype=np.float32)
            nmap = np.zeros(padded.shape, dtype=np.float32)
        acc[:, sl[0], sl[1], sl[2]] += pred
        nmap[sl] += 1.0
        print(f"    patch {i}/{len(coords)} @ {(x0, y0, z0)}")
        if device.type == "mps":
            torch.mps.empty_cache()
    elapsed = time.perf_counter() - t0
    nmap = np.maximum(nmap, 1.0)
    acc /= nmap[None]
    idx = acc.argmax(0).astype(np.int32)
    labels_pad = model.labels_segmentation[idx].astype(np.int32)
    sl_orig = tuple(
        slice(b, b + s) for b, s in zip(pad_before, t1.shape)
    )
    labels = labels_pad[sl_orig]
    return labels, elapsed, f"overlapping_{patch}^3_stride{stride}_n{len(coords)}"


def run_size(model, device, t1: np.ndarray, size: int) -> tuple[np.ndarray, float, str]:
    cube, meta = letterbox_to_cube(t1, size)
    cube = normalize_volume(cube)
    x = torch.tensor(cube[None, None], dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model.model(x)
        pred = logits.detach().cpu().numpy()[0]
    elapsed = time.perf_counter() - t0
    labels_cube = model.labels_segmentation[pred.argmax(0).astype(np.int32)].astype(
        np.int32
    )
    labels = unletterbox(labels_cube, meta)
    return labels, elapsed, f"letterbox_full_fov_to_{size}^3"


def count_labels(labels: np.ndarray, label_dict: dict) -> list[dict]:
    rows = []
    for lab in sorted(np.unique(labels).astype(int).tolist()):
        n = int((labels == lab).sum())
        rows.append(
            {
                "label": int(lab),
                "name": label_dict.get(int(lab), f"unknown_{lab}"),
                "voxels": n,
            }
        )
    return rows


def _resample_mask_to(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(mask.shape) == tuple(target_shape):
        return mask.astype(bool)
    zooms = np.array(target_shape, dtype=float) / np.array(mask.shape, dtype=float)
    return ndi.zoom(mask.astype(np.float32), zooms, order=0) > 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t1", type=Path, default=T1_PATH, help="T1 NIfTI (native or 1 mm)")
    p.add_argument("--wt", type=Path, default=WT_PATH, help="Whole-tumor mask NIfTI")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Labeled anatomy NIfTI (default: prototype SID_synthseg_fullfov.nii.gz)",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        dest="json_path",
        help="Sidecar JSON with counts / timing",
    )
    p.add_argument(
        "--mode",
        choices=("patches", "letterbox"),
        default="patches",
        help="patches = native-res overlapping 192³ (default); "
        "letterbox = downsample full FOV to one 192³ cube",
    )
    return p.parse_args()


def _sidecar_json(nifti: Path) -> Path:
    name = nifti.name
    if name.endswith(".nii.gz"):
        return nifti.with_name(name[: -len(".nii.gz")] + ".json")
    return nifti.with_suffix(".json")


def main() -> int:
    args = parse_args()
    t1_path = args.t1
    wt_path = args.wt
    out_nii = args.out or (OUT / f"{SID}_synthseg_fullfov.nii.gz")
    json_path = args.json_path or _sidecar_json(out_nii)

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(t1_path))
    t1 = np.asanyarray(img.dataobj, dtype=np.float32)
    print(f"T1 {t1.shape} from {t1_path}")

    t_init0 = time.perf_counter()
    ss = SynthSeg(verbose=False, use_cuda=False)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        ss.model = ss.model.to(device)
        ss.device = device
    else:
        device = torch.device("cpu")
    init_s = time.perf_counter() - t_init0
    print(f"init {init_s:.1f}s device={device}")

    notes = [
        "224^3 single cube: MPS OOM / logits collapse to background.",
        "Default: overlapping native 192^3 patches (stride 48) to keep full FOV resolution.",
        "letterbox 192^3 remains available via --mode letterbox (faster, blurrier cortex/WM).",
    ]
    if device.type == "mps":
        torch.mps.empty_cache()
    if args.mode == "letterbox":
        labels, infer_s, approach = run_size(ss, device, t1, 192)
        print(f"  ok 192^3 letterbox in {infer_s:.1f}s")
    else:
        labels, infer_s, approach = run_patches(ss, device, t1, patch=192, stride=48)
        print(f"  ok {approach} in {infer_s:.1f}s")
    chosen_size = 192
    notes.append(f"{approach} succeeded in {infer_s:.1f}s")

    raw_counts = count_labels(labels, ss.label_dict)
    print("=== before WT mask ===")
    for r in raw_counts:
        print(f"  {r['label']:3d}  {r['name']:40s}  {r['voxels']}")

    wt = None
    if wt_path.is_file():
        wt = np.asanyarray(nib.load(str(wt_path)).dataobj) > 0
        if wt.shape != labels.shape:
            notes.append(
                f"WT shape {tuple(wt.shape)} vs labels {tuple(labels.shape)} — resampling mask"
            )
            wt = _resample_mask_to(wt, labels.shape)
    masked = labels.copy()
    n_cleared = 0
    if wt is not None:
        n_cleared = int(((wt) & (masked != 0)).sum())
        masked[wt] = 0
        notes.append(f"cleared {n_cleared} SynthSeg voxels inside v3 WT")

    masked_counts = count_labels(masked, ss.label_dict)
    print("=== after WT mask ===")
    for r in masked_counts:
        print(f"  {r['label']:3d}  {r['name']:40s}  {r['voxels']}")

    key_names = (
        "left lateral ventricle",
        "right lateral ventricle",
        "3rd ventricle",
        "4th ventricle",
        "brain-stem",
        "left cerebellum cortex",
        "right cerebellum cortex",
        "left thalamus",
        "right thalamus",
        "left hippocampus",
        "right hippocampus",
    )
    name_to_vox = {r["name"]: r["voxels"] for r in masked_counts}
    print("=== structures of interest (after mask) ===")
    for name in key_names:
        print(f"  {name:40s}  {name_to_vox.get(name, 0)}")

    nib.save(nib.Nifti1Image(masked.astype(np.int16), img.affine, img.header), str(out_nii))
    # Raw (unmasked) copy only for the standalone prototype default output.
    if args.out is None:
        raw_nii = OUT / f"{SID}_synthseg_fullfov_raw.nii.gz"
        nib.save(nib.Nifti1Image(labels.astype(np.int16), img.affine, img.header), str(raw_nii))

    meta = {
        "approach": approach,
        "cube_size": chosen_size,
        "device": str(device),
        "init_seconds": round(init_s, 2),
        "infer_seconds": round(float(infer_s), 2),
        "notes": notes,
        "t1": str(t1_path),
        "wt": str(wt_path),
        "wt_voxels_cleared": n_cleared,
        "counts_before_wt_mask": raw_counts,
        "counts_after_wt_mask": masked_counts,
        "output_nifti": str(out_nii),
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("approach", "cube_size", "infer_seconds", "wt_voxels_cleared")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
