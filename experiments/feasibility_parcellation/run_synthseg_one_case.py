#!/usr/bin/env python3
"""Standalone SynthSeg feasibility run — not wired into the MRI pipeline.

Uses an isolated venv (.venv-synthseg). DIPY's PyTorch port of Billot SynthSeg.
Force CPU as requested.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.nn.torch.synthseg import SynthSeg

ROOT = Path(__file__).resolve().parents[2]
T1_PATH = ROOT / "data" / "processed" / "brats_nifti" / "BraTS20_Training_001" / "T1.nii.gz"
OUT = ROOT / "outputs" / "feasibility_parcellation"
OUT.mkdir(parents=True, exist_ok=True)


def overlay_slice(t1: np.ndarray, labels: np.ndarray, z: int, out_png: Path) -> None:
    sl_t1 = np.rot90(t1[:, :, z])
    sl_lab = np.rot90(labels[:, :, z])
    p2, p98 = np.percentile(sl_t1[sl_t1 > 0], (2, 98)) if np.any(sl_t1 > 0) else (0, 1)
    base = np.clip((sl_t1 - p2) / max(p98 - p2, 1e-6), 0, 1)
    rng = np.random.default_rng(0)
    n = int(sl_lab.max()) + 1
    cmap = rng.random((n, 3))
    cmap[0] = 0
    rgb = cmap[np.clip(sl_lab.astype(int), 0, n - 1)]
    mix = np.stack([base, base, base], -1)
    mask = sl_lab > 0
    mix[mask] = 0.35 * mix[mask] + 0.65 * rgb[mask]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(base, cmap="gray", origin="lower")
    axs[0].set_title("T1")
    axs[1].imshow(sl_lab, cmap="nipy_spectral", origin="lower", interpolation="nearest")
    axs[1].set_title("SynthSeg labels")
    axs[2].imshow(mix, origin="lower")
    axs[2].set_title("overlay")
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"BraTS20_Training_001 T1  z={z}  CPU SynthSeg (DIPY)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    img = nib.load(str(T1_PATH))
    t1 = np.asanyarray(img.dataobj, dtype=np.float32)
    aff = img.affine
    print(f"input {T1_PATH} shape={t1.shape} dtype={t1.dtype}")

    t_init0 = time.perf_counter()
    model = SynthSeg(verbose=True, use_cuda=False)
    t_init = time.perf_counter() - t_init0
    print(f"model init (weights already cached): {t_init:.1f}s device={model.device}")

    # DIPY's SynthSeg.predict() hit ~60 GB on this Mac (autograd + 5-D gaussian
    # + TTA). CPU 192³ conv3d then used ~19 GB and ~16 s CPU / 2 min wall
    # (swapping). Feasibility path that actually finishes: same weights, 192³,
    # torch.inference_mode, Apple MPS if present else CPU.
    import torch
    from dipy.nn.utils import normalize, recover_img, transform_img

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.model = model.model.to(device)
        model.device = device
        print("moved network to mps")
    else:
        device = torch.device("cpu")
        print("MPS unavailable — staying on CPU")

    t0 = time.perf_counter()
    t_img, params = transform_img(
        t1, aff, target_voxsize=(1.0, 1.0, 1.0), final_size=(192, 192, 192), order=3
    )
    min_v, max_v = np.percentile(t_img, (0.5, 99.5))
    t_img = normalize(t_img, min_v=min_v, max_v=max_v, new_min=0, new_max=1)
    x = torch.tensor(t_img[None, None], dtype=torch.float32, device=device)
    print(f"network input {tuple(x.shape)} after resample {t_img.shape}")
    with torch.inference_mode():
        logits = model.model(x)
        pred192 = logits.detach().cpu().numpy()[0]
    labels192 = model.labels_segmentation[pred192.argmax(0).astype(np.int32)].astype(
        np.int32
    )
    labels = np.round(recover_img(labels192, params, order=0)).astype(np.int32)
    wall = time.perf_counter() - t0
    label_dict = model.label_dict
    print(f"predict wall-clock: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"label volume shape={labels.shape} unique={sorted(np.unique(labels).astype(int).tolist())}")

    present = []
    for lab in sorted(np.unique(labels).astype(int).tolist()):
        n = int((labels == lab).sum())
        name = label_dict.get(lab, label_dict.get(int(lab), f"unknown_{lab}"))
        present.append({"label": int(lab), "name": name, "voxels": n})
        print(f"  {lab:3d}  {name:40s}  {n} voxels")

    out_nii = OUT / "BraTS20_Training_001_synthseg.nii.gz"
    nib.save(nib.Nifti1Image(labels.astype(np.int16), aff, img.header), str(out_nii))
    z = int(np.argmax([(labels[:, :, i] > 0).sum() for i in range(labels.shape[2])]))
    png = OUT / "BraTS20_Training_001_synthseg_slice.png"
    overlay_slice(t1, labels, z, png)

    meta = {
        "backend": "dipy.nn.torch.synthseg.SynthSeg (PyTorch port of Billot SynthSeg)",
        "isolated_venv": ".venv-synthseg (Python 3.11, not the pipeline .venv)",
        "device": str(model.device),
        "predict_notes": (
            "inference_mode + 192^3, no TTA/gaussian. "
            "DIPY predict() ~60GB; CPU conv3d ~19GB and too slow (swap). "
            "This run uses MPS when available."
        ),
        "input": str(T1_PATH),
        "shape": list(t1.shape),
        "init_seconds": round(t_init, 2),
        "predict_seconds": round(wall, 2),
        "weights_path": str(Path.home() / ".dipy" / "synthseg" / "synthseg_model_weights.pth"),
        "weights_bytes": (Path.home() / ".dipy" / "synthseg" / "synthseg_model_weights.pth").stat().st_size,
        "output_nifti": str(out_nii),
        "output_png": str(png),
        "slice_z": z,
        "regions_present": present,
        "label_dict_full": {str(k): v for k, v in label_dict.items()},
    }
    (OUT / "run_report.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_nii}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
