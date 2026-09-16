"""MONAI segmentation inference."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, LoadImage, Orientation, ScaleIntensity, Spacing

from segmentation.model import build_brats_model


def run_inference(
    image_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    device: str | None = None,
) -> Path:
    """Run sliding-window inference and write an argmax label map.

    Parameters
    ----------
    image_path:
        Preprocessed multi-channel or single-channel NIfTI.
    checkpoint_path:
        Training checkpoint containing ``model_state``.
    output_path:
        Destination label NIfTI.
    roi_size:
        Sliding-window patch size.
    device:
        Torch device string; defaults to CUDA when available.

    Returns
    -------
    Path
        Path to the predicted segmentation.
    """
    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device_t)
    model_name = ckpt.get("model_name", "segresnet") if isinstance(ckpt, dict) else "segresnet"
    model = build_brats_model(model_name).to(device_t)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensity(),
            EnsureType(),
        ]
    )
    image = transforms(str(image_path)).unsqueeze(0).to(device_t)

    with torch.no_grad():
        logits = sliding_window_inference(image, roi_size, sw_batch_size=1, predictor=model)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    ref = nib.load(str(image_path))
    nib.save(nib.Nifti1Image(pred, ref.affine, ref.header), str(output_path))
    return output_path
