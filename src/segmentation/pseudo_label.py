"""Generate pseudo-label tumor masks for preprocessed clinical MRI studies."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
)

from config import load_config
from preprocessing.h5_to_nifti import MODALITY_NAMES
from preprocessing.registration import REGISTERED_NAMES
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS
from segmentation.train import post_transforms

logger = logging.getLogger(__name__)

PSEUDO_LABEL_FILENAME = "pseudo_tumor_label.nii.gz"
PSEUDO_WT_FILENAME = "pseudo_tumor_WT.nii.gz"

__all__ = [
    "REGISTERED_NAMES",
    "PSEUDO_LABEL_FILENAME",
    "PSEUDO_WT_FILENAME",
    "PseudoLabelResult",
    "discover_preprocessed_studies",
    "registered_dir_to_datadict",
    "get_patient_inference_transforms",
    "regions_to_label_map",
    "pseudo_label_study",
    "pseudo_label_all_patients",
]


@dataclass
class PseudoLabelResult:
    study_id: str
    status: str
    label_path: str = ""
    wt_path: str = ""
    reference_scan: str = ""
    error: str = ""


def discover_preprocessed_studies(processed_patients_root: str | Path) -> list[Path]:
    """Find studies with a complete set of 1 mm registered modalities."""
    root = Path(processed_patients_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Processed patients root not found: {root}")

    studies: list[Path] = []
    for study_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if study_dir.name == "corrected":
            continue
        reg_dir = study_dir / "04_registered_1mm"
        if not reg_dir.is_dir():
            logger.debug("Skipping %s: no 04_registered_1mm/", study_dir.name)
            continue
        missing = [
            name
            for modality, name in REGISTERED_NAMES.items()
            if not (reg_dir / name).is_file()
        ]
        if missing:
            logger.warning(
                "Skipping %s: missing registered volumes %s",
                study_dir.name,
                missing,
            )
            continue
        studies.append(study_dir)
    logger.info("Discovered %d preprocessed patient studies under %s", len(studies), root)
    return studies


def registered_dir_to_datadict(reg_dir: str | Path, study_id: str) -> dict[str, Any]:
    """Build a MONAI dict with four registered modalities stacked as channels."""
    reg_dir = Path(reg_dir)
    return {
        "study_id": study_id,
        "image": [str(reg_dir / REGISTERED_NAMES[modality]) for modality in MODALITY_NAMES],
        "reference": str(reg_dir / REGISTERED_NAMES["T1"]),
    }


def get_patient_inference_transforms() -> Compose:
    """Deterministic transforms for already-registered 1 mm patient volumes."""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )


def regions_to_label_map(
    et: np.ndarray,
    tc: np.ndarray,
    wt: np.ndarray,
) -> np.ndarray:
    """Convert nested ET/TC/WT binary masks to BraTS-style labels 0–3.

    Labels match the BraTS convention used elsewhere in this project:
    0 = background, 1 = necrotic core, 2 = edema, 3 = enhancing tumor.
    """
    et_b = et.astype(bool)
    tc_b = tc.astype(bool)
    wt_b = wt.astype(bool)
    label = np.zeros(wt_b.shape, dtype=np.uint8)
    label[wt_b & ~tc_b] = 2
    label[tc_b & ~et_b] = 1
    label[et_b] = 3
    return label


@torch.no_grad()
def pseudo_label_study(
    study_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    device: str | None = None,
    overwrite: bool = False,
) -> PseudoLabelResult:
    """Run BraTS-pretrained inference on one preprocessed study."""
    study_dir = Path(study_dir)
    study_id = study_dir.name
    reg_dir = study_dir / "04_registered_1mm"
    label_path = reg_dir / PSEUDO_LABEL_FILENAME
    wt_path = reg_dir / PSEUDO_WT_FILENAME

    if label_path.is_file() and not overwrite:
        logger.info("Skipping %s: pseudo label already exists", study_id)
        return PseudoLabelResult(
            study_id=study_id,
            status="cached",
            label_path=str(label_path),
            wt_path=str(wt_path) if wt_path.is_file() else "",
            reference_scan=str(reg_dir / REGISTERED_NAMES["T1"]),
        )

    try:
        record = registered_dir_to_datadict(reg_dir, study_id)
        data = get_patient_inference_transforms()(record)

        device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = load_model_from_checkpoint(checkpoint_path, device_t)
        post_pred, _ = post_transforms()

        image = data["image"].unsqueeze(0).to(device_t)
        with torch.autocast(device_type=device_t.type, enabled=device_t.type == "cuda"):
            outputs = sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
            )

        pred_regions = post_pred(decollate_batch(outputs)[0])
        pred_np = pred_regions.detach().cpu().numpy()
        if pred_np.ndim != 4 or pred_np.shape[0] != len(BRATS_REGIONS):
            raise ValueError(f"Expected (3, H, W, D) regions, got {pred_np.shape}")

        et, tc, wt = (pred_np[i].astype(bool) for i in range(3))
        label_map = regions_to_label_map(et, tc, wt)

        ref = nib.load(record["reference"])
        affine = ref.affine
        header = ref.header.copy()
        header.set_data_dtype(np.uint8)

        nib.save(nib.Nifti1Image(label_map, affine, header), str(label_path))
        nib.save(
            nib.Nifti1Image(wt.astype(np.uint8), affine, header),
            str(wt_path),
        )

        meta = {
            "study_id": study_id,
            "checkpoint": str(checkpoint_path),
            "label_path": str(label_path),
            "wt_path": str(wt_path),
            "reference_scan": record["reference"],
            "modalities": list(MODALITY_NAMES),
            "label_encoding": {
                "0": "background",
                "1": "necrotic_core",
                "2": "edema",
                "3": "enhancing_tumor",
            },
            "viewer_note": (
                "Load T1_1mm.nii.gz and pseudo_tumor_label.nii.gz together "
                "in ITK-SNAP or 3D Slicer."
            ),
        }
        sidecar = reg_dir / "pseudo_label_meta.json"
        sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        logger.info("Wrote pseudo labels for %s -> %s", study_id, label_path)
        return PseudoLabelResult(
            study_id=study_id,
            status="success",
            label_path=str(label_path),
            wt_path=str(wt_path),
            reference_scan=record["reference"],
        )
    except Exception as exc:
        logger.exception("Pseudo-labeling failed for %s: %s", study_id, exc)
        return PseudoLabelResult(
            study_id=study_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def pseudo_label_all_patients(
    checkpoint_path: str | Path,
    processed_patients_root: str | Path | None = None,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    device: str | None = None,
    overwrite: bool = False,
    summary_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Pseudo-label every preprocessed real-patient study under ``processed_patients_root``."""
    if processed_patients_root is None:
        processed_patients_root = load_config().paths.processed / "real_patients"

    studies = discover_preprocessed_studies(processed_patients_root)
    results: list[PseudoLabelResult] = []
    for index, study_dir in enumerate(studies, start=1):
        logger.info("[%d/%d] Pseudo-labeling %s", index, len(studies), study_dir.name)
        results.append(
            pseudo_label_study(
                study_dir,
                checkpoint_path,
                roi_size=roi_size,
                device=device,
                overwrite=overwrite,
            )
        )

    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = (
        Path(summary_csv)
        if summary_csv is not None
        else Path(processed_patients_root) / "pseudo_label_summary.csv"
    )
    df.to_csv(csv_path, index=False)

    n_ok = int((df["status"].isin(["success", "cached"])).sum()) if len(df) else 0
    n_fail = int((df["status"] == "failed").sum()) if len(df) else 0
    logger.info(
        "Pseudo-labeling complete: ok/cached=%d failed=%d | summary=%s",
        n_ok,
        n_fail,
        csv_path,
    )
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate pseudo tumor labels for preprocessed clinical studies."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="BraTS-pretrained checkpoint (default: checkpoints/brats_pretrain/best_model.pt)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--patients-root",
        type=Path,
        default=None,
        help="Root of preprocessed patient studies (default: data/processed/real_patients)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    checkpoint = args.checkpoint or (app_cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt")
    patients_root = args.patients_root or (app_cfg.paths.processed / "real_patients")

    pseudo_label_all_patients(
        checkpoint,
        patients_root,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
