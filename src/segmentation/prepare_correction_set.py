"""Stage manually correctable patient cases from pseudo-labels (ITK-SNAP / Slicer)."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from config import load_config
from preprocessing.registration import REGISTERED_NAMES
from segmentation.pseudo_label import (
    PSEUDO_LABEL_FILENAME,
    discover_preprocessed_studies,
    pseudo_label_study,
)

logger = logging.getLogger(__name__)

CORRECTED_LABEL_NAME = "corrected_tumor_label.nii.gz"
README_NAME = "CORRECTION_README.txt"

README_TEXT = """Manual correction instructions (step 5)
=====================================

1. Open ITK-SNAP or 3D Slicer.
2. Load as main image: 04_registered_1mm/T1_1mm.nii.gz
   (optional overlays: FLAIR_registered_1mm.nii.gz, T2_registered_1mm.nii.gz)
3. Load as segmentation: 04_registered_1mm/corrected_tumor_label.nii.gz
4. Edit labels (BraTS-style):
     0 = background
     1 = necrotic / non-enhancing core
     2 = edema
     3 = enhancing tumor
5. Save over corrected_tumor_label.nii.gz (same filename / folder).

Notes
-----
- The starter mask is a MODEL PSEUDO-LABEL, not a clinical gold standard.
- Many studies had no contrast T1c; T1 was copied to T1c_registered_1mm.nii.gz
  so the 4-channel network can run — document this limitation in the paper.
- Fine-tuning needs ~8+ corrected cases (3–5 held out as test).
"""


def fill_missing_t1c_from_t1(patients_root: str | Path) -> int:
    """Copy T1_1mm → T1c_registered_1mm when contrast T1c is absent."""
    root = Path(patients_root)
    n = 0
    t1_name = REGISTERED_NAMES["T1"]
    t1c_name = REGISTERED_NAMES["T1c"]
    for study in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "corrected"):
        reg = study / "04_registered_1mm"
        t1 = reg / t1_name
        t1c = reg / t1c_name
        if t1.is_file() and not t1c.is_file():
            shutil.copy2(t1, t1c)
            n += 1
            logger.info("Filled T1c from T1 for %s", study.name)
    return n


def stage_corrected_cases(
    patients_root: str | Path,
    corrected_root: str | Path,
    *,
    study_ids: list[str] | None = None,
    n_cases: int = 10,
) -> list[Path]:
    """Copy registered scans + pseudo label into corrected/ as editable starters."""
    patients_root = Path(patients_root)
    corrected_root = Path(corrected_root)
    corrected_root.mkdir(parents=True, exist_ok=True)

    if study_ids is None:
        studies = discover_preprocessed_studies(patients_root)[:n_cases]
    else:
        studies = [patients_root / sid for sid in study_ids]

    staged: list[Path] = []
    for study in studies:
        reg = study / "04_registered_1mm"
        pseudo = reg / PSEUDO_LABEL_FILENAME
        if not pseudo.is_file():
            logger.warning("Skip %s: no pseudo label yet", study.name)
            continue

        dest_study = corrected_root / study.name
        dest_reg = dest_study / "04_registered_1mm"
        dest_reg.mkdir(parents=True, exist_ok=True)

        for name in REGISTERED_NAMES.values():
            src = reg / name
            if src.is_file():
                shutil.copy2(src, dest_reg / name)

        shutil.copy2(pseudo, dest_reg / CORRECTED_LABEL_NAME)
        # Keep original pseudo for reference
        shutil.copy2(pseudo, dest_reg / PSEUDO_LABEL_FILENAME)
        (dest_study / README_NAME).write_text(README_TEXT, encoding="utf-8")
        staged.append(dest_study)
        logger.info("Staged correction case -> %s", dest_study)

    return staged


def prepare_correction_set(
    checkpoint: str | Path,
    patients_root: str | Path,
    corrected_root: str | Path,
    *,
    n_cases: int = 10,
    device: str | None = None,
) -> list[Path]:
    """Fill T1c, pseudo-label a subset, stage corrected/ folders for manual edit."""
    n_filled = fill_missing_t1c_from_t1(patients_root)
    logger.info("Filled T1c from T1 on %d studies", n_filled)

    studies = discover_preprocessed_studies(patients_root)[:n_cases]
    if len(studies) < n_cases:
        logger.warning(
            "Only %d complete studies available (wanted %d)",
            len(studies),
            n_cases,
        )
    if not studies:
        raise RuntimeError(
            "No complete registered studies found after T1c fill. "
            "Check data/processed/real_patients/*/04_registered_1mm/"
        )

    for study in studies:
        result = pseudo_label_study(study, checkpoint, device=device)
        logger.info("Pseudo-label %s: %s", result.study_id, result.status)
        if result.status == "failed":
            logger.error("Failed %s: %s", result.study_id, result.error)

    return stage_corrected_cases(
        patients_root,
        corrected_root,
        study_ids=[s.name for s in studies],
        n_cases=n_cases,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare corrected/ staging set for manual ITK-SNAP editing (step 5).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--patients-root", type=Path, default=None)
    parser.add_argument("--corrected-root", type=Path, default=None)
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    checkpoint = args.checkpoint or (
        app_cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt"
    )
    patients_root = args.patients_root or (app_cfg.paths.processed / "real_patients")
    corrected_root = args.corrected_root or (patients_root / "corrected")

    staged = prepare_correction_set(
        checkpoint,
        patients_root,
        corrected_root,
        n_cases=args.n_cases,
    )
    print(f"Staged {len(staged)} cases under {corrected_root}")
    print("Open each CORRECTION_README.txt and edit corrected_tumor_label.nii.gz in ITK-SNAP/Slicer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
