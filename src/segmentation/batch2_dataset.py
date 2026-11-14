"""Batch2 real-patient cohort dataset for fine-tuning / CV.

Discovers ``outputs/batch2_labeling/<study_id>/corrected_label.nii.gz`` (or
``v3_draft_label.nii.gz`` in dry-run). Channel order matches v3 exactly
(FLAIR, T1, T1c, T2). Labels use classic BraTS values ``1/2/4`` and convert to
nested regions via :func:`segmentation.model.brats_label_to_regions`.

**Reference masks, not clinical ground truth.** Batch2 corrected labels are
single-annotator edits that started from a v3 draft.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from config import PROJECT_ROOT
from preprocessing.h5_to_nifti import MODALITY_NAMES
from segmentation.dataset import IMAGE_KEY, LABEL_KEY

logger = logging.getLogger(__name__)

DEFAULT_BATCH2_LABELING_ROOT = PROJECT_ROOT / "outputs" / "batch2_labeling"
CORRECTED_LABEL_NAME = "corrected_label.nii.gz"
DRAFT_LABEL_NAME = "v3_draft_label.nii.gz"
NO_CLEAR_LESION_PHRASE = "no clear lesion"
REFERENCE_MASK_DISCLAIMER = (
    "Reference masks (single-annotator corrections from a v3 draft), "
    "not clinical ground truth."
)

# Short names matching v3 prediction channel order.
REGION_SHORT: tuple[str, ...] = ("ET", "TC", "WT")


@dataclass
class Batch2Study:
    """One batch2 labeling package with a usable reference mask."""

    study_id: str
    study_dir: Path
    label_path: Path
    patient_id: str
    image_paths: dict[str, Path]
    skipped: bool = False
    skip_reason: str | None = None
    label_is_draft: bool = False


@dataclass
class DiscoveryResult:
    accepted: list[Batch2Study] = field(default_factory=list)
    skipped: list[Batch2Study] = field(default_factory=list)


def classic_brats_label_to_regions_numpy(label: np.ndarray) -> np.ndarray:
    """Convert BraTS labels to ``(3, *spatial)`` ET/TC/WT (numpy).

    Accepts project ``{0,1,2,3}`` or classic ``{0,1,2,4}``. Raises if both
    enhancing codes 3 and 4 appear (same policy as
    :func:`segmentation.model.brats_label_to_regions`).
    """
    lbl = np.asarray(label)
    if lbl.ndim == 4 and lbl.shape[0] == 1:
        lbl = lbl[0]
    has3 = bool(np.any(lbl == 3))
    has4 = bool(np.any(lbl == 4))
    if has3 and has4:
        raise ValueError(
            "Label volume mixes enhancing codes 3 and 4. Use a single "
            "convention per volume: project {0,1,2,3} or classic BraTS "
            "{0,1,2,4}."
        )
    necrotic = lbl == 1
    edema = lbl == 2
    enhancing = (lbl == 3) | (lbl == 4)
    et = enhancing
    tc = necrotic | enhancing
    wt = necrotic | edema | enhancing
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def _read_readme_text(study_dir: Path) -> str:
    for name in ("README.md", "README.txt", "README"):
        path = study_dir / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _readme_says_no_clear_lesion(study_dir: Path) -> bool:
    text = _read_readme_text(study_dir).lower()
    return NO_CLEAR_LESION_PHRASE in text


def _grids_match(a: nib.Nifti1Image, b: nib.Nifti1Image, *, atol: float = 1e-3) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a.affine, b.affine, atol=atol, rtol=0.0))


def _label_is_empty(label_img: nib.Nifti1Image) -> bool:
    data = np.asanyarray(label_img.dataobj)
    return int(np.count_nonzero(data)) == 0


def read_patient_id(study_dir: Path, study_id: str) -> tuple[str, bool]:
    """Return ``(patient_id, missing)``.

    Prefers DICOM metadata JSON next to native volumes. If missing, treats the
    study as its own patient and warns.
    """
    candidates = [
        study_dir / "T1c_native.json",
        study_dir / "T1_native.json",
        study_dir / "FLAIR_native.json",
        study_dir / "T2_native.json",
        study_dir / "package_meta.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = meta.get("patient_id")
        if pid is not None and str(pid).strip():
            return str(pid).strip(), False
    warnings.warn(
        f"{study_id}: patient_id missing in DICOM metadata JSON; "
        "treating study as its own patient",
        UserWarning,
        stacklevel=2,
    )
    logger.warning(
        "%s: patient_id missing — treating study as its own patient", study_id
    )
    return study_id, True


def _modality_paths(study_dir: Path) -> dict[str, Path] | None:
    paths: dict[str, Path] = {}
    for name in MODALITY_NAMES:
        path = study_dir / f"{name}.nii.gz"
        if not path.is_file():
            return None
        paths[name] = path
    return paths


def discover_batch2_studies(
    labeling_root: str | Path | None = None,
    *,
    use_draft_labels: bool = False,
    label_name: str | None = None,
) -> DiscoveryResult:
    """Discover batch2 studies with usable reference (or draft) masks.

    Skips a study when:
    - README mentions ``no clear lesion``
    - the label volume is empty
    - the label grid does not match ``T1c.nii.gz``
    - required modality NIfTIs are missing
    """
    root = Path(labeling_root) if labeling_root is not None else DEFAULT_BATCH2_LABELING_ROOT
    if not root.is_dir():
        raise NotADirectoryError(f"batch2 labeling root not found: {root}")

    fname = label_name or (DRAFT_LABEL_NAME if use_draft_labels else CORRECTED_LABEL_NAME)
    result = DiscoveryResult()

    for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        study_id = study_dir.name
        label_path = study_dir / fname
        if not label_path.is_file():
            continue

        image_paths = _modality_paths(study_dir)
        patient_id, _missing = read_patient_id(study_dir, study_id)
        base = Batch2Study(
            study_id=study_id,
            study_dir=study_dir,
            label_path=label_path,
            patient_id=patient_id,
            image_paths=image_paths or {},
            label_is_draft=use_draft_labels or fname == DRAFT_LABEL_NAME,
        )

        if image_paths is None:
            base.skipped = True
            base.skip_reason = "missing modality NIfTI(s) (need FLAIR/T1/T1c/T2.nii.gz)"
            logger.info("Skip %s: %s", study_id, base.skip_reason)
            result.skipped.append(base)
            continue

        if _readme_says_no_clear_lesion(study_dir):
            base.skipped = True
            base.skip_reason = f'README contains "{NO_CLEAR_LESION_PHRASE}"'
            logger.info("Skip %s: %s", study_id, base.skip_reason)
            result.skipped.append(base)
            continue

        try:
            label_img = nib.load(str(label_path))
            t1c_img = nib.load(str(image_paths["T1c"]))
        except Exception as exc:  # noqa: BLE001 — surface load failures as skips
            base.skipped = True
            base.skip_reason = f"failed to load NIfTI: {exc}"
            logger.info("Skip %s: %s", study_id, base.skip_reason)
            result.skipped.append(base)
            continue

        if not _grids_match(label_img, t1c_img):
            base.skipped = True
            base.skip_reason = (
                f"label grid mismatch vs T1c "
                f"(label shape={tuple(label_img.shape)} affine≠T1c "
                f"shape={tuple(t1c_img.shape)})"
            )
            logger.info("Skip %s: %s", study_id, base.skip_reason)
            result.skipped.append(base)
            continue

        if _label_is_empty(label_img):
            base.skipped = True
            base.skip_reason = "empty label (no foreground voxels)"
            logger.info("Skip %s: %s", study_id, base.skip_reason)
            result.skipped.append(base)
            continue

        result.accepted.append(base)
        logger.info(
            "Accept %s (patient_id=%s, label=%s)",
            study_id,
            patient_id,
            label_path.name,
        )

    logger.info(
        "batch2 discovery: accepted=%d skipped=%d under %s (label=%s)",
        len(result.accepted),
        len(result.skipped),
        root,
        fname,
    )
    return result


def study_to_datadict(study: Batch2Study) -> dict[str, Any]:
    """MONAI-style record; channel order matches v3 (FLAIR, T1, T1c, T2)."""
    return {
        "subject_id": study.study_id,
        "patient_id": study.patient_id,
        "study_id": study.study_id,
        # Do NOT name this label_*: MONAI Spacingd treats keys starting with
        # "{key}_" as MetaTensor side-channels and would call .update() on a bool.
        "is_draft_label": study.label_is_draft,
        IMAGE_KEY: [str(study.image_paths[m]) for m in MODALITY_NAMES],
        LABEL_KEY: str(study.label_path),
    }


def build_batch2_file_list(
    labeling_root: str | Path | None = None,
    *,
    use_draft_labels: bool = False,
    label_name: str | None = None,
) -> list[dict[str, Any]]:
    """Accepted studies as MONAI data dicts."""
    discovered = discover_batch2_studies(
        labeling_root,
        use_draft_labels=use_draft_labels,
        label_name=label_name,
    )
    return [study_to_datadict(s) for s in discovered.accepted]


def group_studies_by_patient(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group MONAI records by ``patient_id`` (falls back to ``subject_id``)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        pid = str(rec.get("patient_id") or rec.get("subject_id") or "")
        if not pid:
            raise ValueError(f"Record missing patient_id/subject_id: {rec}")
        groups.setdefault(pid, []).append(rec)
    return groups


def make_patient_folds(
    records: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    max_lopo_patients: int = 12,
    n_folds: int = 5,
    balance_negatives: bool = False,
) -> list[dict[str, Any]]:
    """Patient-grouped CV folds: LOPO if ``n_patients <= max_lopo_patients``, else k-fold.

    When ``balance_negatives=True``, positive and negative *patients* are
    stratified separately into folds so each fold gets a balanced share of
    negatives (for ``splits/batch2_pos_neg`` manifests).

    Asserts that no patient appears in both train and held-out of the same fold.
    """
    groups = group_studies_by_patient(records)
    patient_ids = sorted(groups.keys())
    n_patients = len(patient_ids)
    if n_patients < 2:
        raise ValueError(
            f"Need at least 2 patients for CV, got {n_patients}"
        )

    rng = np.random.default_rng(seed)

    def _is_neg_patient(pid: str) -> bool:
        return any(bool(r.get("is_negative")) for r in groups[pid])

    folds: list[dict[str, Any]] = []
    if balance_negatives and any(_is_neg_patient(p) for p in patient_ids):
        pos_pids = [p for p in patient_ids if not _is_neg_patient(p)]
        neg_pids = [p for p in patient_ids if _is_neg_patient(p)]
        rng.shuffle(pos_pids)
        rng.shuffle(neg_pids)
        k = min(n_folds, max(len(pos_pids), 2), max(len(neg_pids), 1) + len(pos_pids))
        k = min(n_folds, max(2, min(len(pos_pids) + len(neg_pids), n_folds)))
        pos_buckets: list[list[str]] = [[] for _ in range(k)]
        neg_buckets: list[list[str]] = [[] for _ in range(k)]
        for i, pid in enumerate(pos_pids):
            pos_buckets[i % k].append(pid)
        for i, pid in enumerate(neg_pids):
            neg_buckets[i % k].append(pid)
        for i in range(k):
            held_pids = pos_buckets[i] + neg_buckets[i]
            train_pids = [
                p
                for j in range(k)
                if j != i
                for p in (pos_buckets[j] + neg_buckets[j])
            ]
            train = [r for p in train_pids for r in groups[p]]
            held = [r for p in held_pids for r in groups[p]]
            _assert_no_patient_leakage(train, held)
            n_neg_held = sum(1 for r in held if r.get("is_negative"))
            folds.append(
                {
                    "fold": i,
                    "mode": f"{k}-fold-balanced-neg",
                    "train_patient_ids": train_pids,
                    "held_patient_ids": held_pids,
                    "train": train,
                    "held": held,
                    "n_negatives_held": n_neg_held,
                }
            )
        return folds

    order = list(patient_ids)
    rng.shuffle(order)

    if n_patients <= max_lopo_patients:
        for i, held_pid in enumerate(order):
            train_pids = [p for p in order if p != held_pid]
            train = [r for p in train_pids for r in groups[p]]
            held = list(groups[held_pid])
            _assert_no_patient_leakage(train, held)
            folds.append(
                {
                    "fold": i,
                    "mode": "lopo",
                    "train_patient_ids": train_pids,
                    "held_patient_ids": [held_pid],
                    "train": train,
                    "held": held,
                }
            )
    else:
        k = min(n_folds, n_patients)
        # Round-robin assignment into k folds.
        buckets: list[list[str]] = [[] for _ in range(k)]
        for i, pid in enumerate(order):
            buckets[i % k].append(pid)
        for i in range(k):
            held_pids = buckets[i]
            train_pids = [p for j, bucket in enumerate(buckets) if j != i for p in bucket]
            train = [r for p in train_pids for r in groups[p]]
            held = [r for p in held_pids for r in groups[p]]
            _assert_no_patient_leakage(train, held)
            folds.append(
                {
                    "fold": i,
                    "mode": f"{k}-fold",
                    "train_patient_ids": train_pids,
                    "held_patient_ids": held_pids,
                    "train": train,
                    "held": held,
                }
            )
    return folds


def _assert_no_patient_leakage(
    train: Sequence[dict[str, Any]],
    held: Sequence[dict[str, Any]],
) -> None:
    train_pids = {str(r.get("patient_id") or r["subject_id"]) for r in train}
    held_pids = {str(r.get("patient_id") or r["subject_id"]) for r in held}
    overlap = train_pids & held_pids
    if overlap:
        raise AssertionError(
            f"Patient leakage across folds: {sorted(overlap)}"
        )


POS_NEG_SPLIT_DIR = PROJECT_ROOT / "splits" / "batch2_pos_neg"
EMPTY_LABEL_DIR = PROJECT_ROOT / "outputs" / "batch2_pos_neg" / "empty_labels"
NEG_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients"
BATCH2_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients_batch2"


def write_empty_label_like(reference_nifti: Path, out_path: Path) -> Path:
    """Write an all-zero int16 label on the reference grid (for negatives)."""
    ref = nib.load(str(reference_nifti))
    zeros = np.zeros(ref.shape, dtype=np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(zeros, ref.affine, ref.header), str(out_path))
    return out_path


def build_pos_neg_file_list(
    split_dir: str | Path | None = None,
    *,
    use_draft_labels_for_positives: bool = True,
    include_unused_negatives: bool = False,
) -> list[dict[str, Any]]:
    """Build MONAI records from ``splits/batch2_pos_neg`` manifests.

    Positives use labeling-package volumes + corrected (or draft) labels.
    Negatives use ``data/processed/real_patients/<id>/04_registered_1mm`` with an
    **empty** label written under ``outputs/batch2_pos_neg/empty_labels/``
    (does not modify ``real_patients/``).

    Per-scan tumor-free status lives in ``splits/batch2_pos_neg/manifest.csv``;
    empty masks encode a negative class for training plumbing only.
    """
    from preprocessing.registration import REGISTERED_NAMES

    split = Path(split_dir) if split_dir is not None else POS_NEG_SPLIT_DIR
    pos_ids = [
        ln.strip()
        for ln in (split / "positives.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    neg_ids = [
        ln.strip()
        for ln in (split / "negatives_selected.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if include_unused_negatives:
        neg_ids = neg_ids + [
            ln.strip()
            for ln in (split / "negatives_unused.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    records: list[dict[str, Any]] = []
    # Positives via labeling packages when present
    label_name = DRAFT_LABEL_NAME if use_draft_labels_for_positives else CORRECTED_LABEL_NAME
    for sid in pos_ids:
        pkg = DEFAULT_BATCH2_LABELING_ROOT / sid
        if pkg.is_dir() and (pkg / label_name).is_file():
            study = Batch2Study(
                study_id=sid,
                study_dir=pkg,
                label_path=pkg / label_name,
                patient_id=read_patient_id(pkg, sid)[0],
                image_paths=_modality_paths(pkg) or {},
                label_is_draft=use_draft_labels_for_positives,
            )
            if not study.image_paths:
                continue
            rec = study_to_datadict(study)
            rec["is_negative"] = False
            records.append(rec)
            continue
        # Fallback: registered batch2 volumes + empty/draft elsewhere
        reg = BATCH2_PROCESSED_ROOT / sid / "04_registered_1mm"
        if not reg.is_dir():
            continue
        img = {m: reg / REGISTERED_NAMES[m] for m in MODALITY_NAMES}
        if not all(p.is_file() for p in img.values()):
            continue
        empty = EMPTY_LABEL_DIR / f"{sid}_empty_label.nii.gz"
        write_empty_label_like(img["T1c"], empty)
        records.append(
            {
                "subject_id": sid,
                "patient_id": sid,
                "study_id": sid,
                "is_draft_label": True,
                "is_negative": False,
                IMAGE_KEY: [str(img[m]) for m in MODALITY_NAMES],
                LABEL_KEY: str(empty),
            }
        )

    for sid in neg_ids:
        reg = NEG_PROCESSED_ROOT / sid / "04_registered_1mm"
        img = {m: reg / REGISTERED_NAMES[m] for m in MODALITY_NAMES}
        if not all(p.is_file() for p in img.values()):
            logger.warning("Skip negative %s: missing registered modalities", sid)
            continue
        empty = EMPTY_LABEL_DIR / f"{sid}_empty_label.nii.gz"
        write_empty_label_like(img["T1c"], empty)
        # patient_id from folder prefix
        pid = sid.split("_")[0]
        records.append(
            {
                "subject_id": sid,
                "patient_id": pid,
                "study_id": sid,
                "is_draft_label": False,
                "is_negative": True,
                IMAGE_KEY: [str(img[m]) for m in MODALITY_NAMES],
                LABEL_KEY: str(empty),
            }
        )
    return records


def assert_no_patient_in_two_folds(folds: Sequence[dict[str, Any]]) -> None:
    """For each fold, assert train/held patient sets are disjoint."""
    for fold in folds:
        _assert_no_patient_leakage(fold["train"], fold["held"])
