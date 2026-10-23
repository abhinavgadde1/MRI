#!/usr/bin/env python3
"""Save v3-lcc ET/TC/WT NIfTIs for BraTS20_Training_001 (main pipeline venv)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference

from config import PROJECT_ROOT, load_config
from device import get_device
from segmentation.dataset import get_val_transforms, subject_to_datadict
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS
from segmentation.postprocess import apply_channel_postprocess, enforce_brats_nesting
from segmentation.train import post_transforms

ROOT = PROJECT_ROOT
SID = "BraTS20_Training_001"
CKPT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
OUT = ROOT / "outputs" / "feasibility_parcellation"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    brats = Path(load_config().paths.brats_nifti)
    t1_img = nib.load(str(brats / SID / "T1.nii.gz"))
    device = get_device()
    t0 = time.perf_counter()
    model = load_model_from_checkpoint(CKPT, device)
    model.float().eval()
    rec = subject_to_datadict(brats / SID)
    batch = next(
        iter(DataLoader(Dataset([rec], transform=get_val_transforms()), batch_size=1))
    )
    image = batch["image"].to(device, dtype=torch.float32)
    post_pred, _ = post_transforms()
    with torch.inference_mode():
        logits = sliding_window_inference(
            image, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
        )
    pred = post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(np.float32)
    pred = apply_channel_postprocess(pred, mode="lcc")
    pred = enforce_brats_nesting(pred)
    elapsed = time.perf_counter() - t0
    regions = {r: pred[i].astype(bool) for i, r in enumerate(BRATS_REGIONS)}
    counts = {k: int(v.sum()) for k, v in regions.items()}
    for name, mask in regions.items():
        short = {"enhancing_tumor": "ET", "tumor_core": "TC", "whole_tumor": "WT"}[name]
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), t1_img.affine, t1_img.header),
            str(OUT / f"{SID}_v3_{short}.nii.gz"),
        )
    meta = {
        "subject_id": SID,
        "checkpoint": str(CKPT),
        "device": str(device),
        "seconds": round(elapsed, 2),
        "shape": list(regions["whole_tumor"].shape),
        "voxel_counts": counts,
    }
    (OUT / f"{SID}_v3_regions.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
