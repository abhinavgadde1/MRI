"""QC checks for batch2 ``corrected_label.nii.gz`` packages.

Read-only with respect to labels. Corrected masks are **reference masks**
(single-annotator corrections from a v3 draft), not clinical ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from config import PROJECT_ROOT
from validation.ellipsoid import voxel_volume_ml_from_binary

logger = logging.getLogger(__name__)

DEFAULT_LABELING_ROOT = PROJECT_ROOT / "outputs" / "batch2_labeling"
CORRECTED_NAME = "corrected_label.nii.gz"
DRAFT_NAME = "v3_draft_label.nii.gz"
T1C_NAME = "T1c.nii.gz"
ALLOWED_LABELS = frozenset({0, 1, 2, 4})
TINY_ISLAND_ML = 0.1
REFERENCE_DISCLAIMER = (
    "Reference masks (single-annotator corrections from a v3 draft), "
    "not clinical ground truth."
)


def _rel_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass
class StudyQCResult:
    study_id: str
    status: str  # labeled | not_labeled | invalid
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    wt_ml: float | None = None
    tc_ml: float | None = None
    et_ml: float | None = None
    wt_n_components: int | None = None
    voxel_counts: dict[str, int] = field(default_factory=dict)
    dice_vs_draft: dict[str, float] = field(default_factory=dict)
    volume_delta_ml: dict[str, float] = field(default_factory=dict)
    identical_to_draft: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_nifti(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def _grids_match(a: nib.Nifti1Image, b: nib.Nifti1Image, *, atol: float = 1e-5) -> bool:
    return a.shape == b.shape and bool(np.allclose(a.affine, b.affine, atol=atol, rtol=0.0))


def _spacing_mm(img: nib.Nifti1Image) -> tuple[float, float, float]:
    zooms = img.header.get_zooms()[:3]
    return tuple(float(z) for z in zooms)  # type: ignore[return-value]


def _regions_from_classic(label: np.ndarray) -> dict[str, np.ndarray]:
    """ET/TC/WT from classic BraTS {0,1,2,4} (also accepts 3 alone)."""
    lbl = np.asarray(label)
    if np.any(lbl == 3) and np.any(lbl == 4):
        raise ValueError("mixes enhancing codes 3 and 4")
    necrotic = lbl == 1
    edema = lbl == 2
    enhancing = (lbl == 3) | (lbl == 4)
    return {
        "ET": enhancing,
        "TC": necrotic | enhancing,
        "WT": necrotic | edema | enhancing,
    }


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    denom = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def _n_components(binary: np.ndarray) -> int:
    labeled, n = ndi.label(np.asarray(binary, dtype=bool))
    return int(n)


def _component_volumes_ml(
    binary: np.ndarray, spacing_mm: tuple[float, float, float]
) -> list[float]:
    mask = np.asarray(binary, dtype=bool)
    if not np.any(mask):
        return []
    labeled, n = ndi.label(mask)
    vols: list[float] = []
    voxel_ml = float(np.prod(spacing_mm) / 1000.0)
    for i in range(1, n + 1):
        vols.append(float(np.count_nonzero(labeled == i) * voxel_ml))
    return vols


def _touches_border(binary: np.ndarray) -> bool:
    m = np.asarray(binary, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {m.shape}")
    return bool(
        np.any(m[0, :, :])
        or np.any(m[-1, :, :])
        or np.any(m[:, 0, :])
        or np.any(m[:, -1, :])
        or np.any(m[:, :, 0])
        or np.any(m[:, :, -1])
    )


def find_brain_mask(
    study_dir: Path,
    *,
    target_shape: tuple[int, ...],
    target_affine: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Locate an HD-BET (or similar) brain mask; resample if grid differs.

    Search order: package-local, then ``data/processed/real_patients_batch2``.
    Returns ``(mask_bool_or_None, note)``.
    """
    study_id = study_dir.name
    candidates = [
        study_dir / "T1c_brainmask.nii.gz",
        study_dir / "brain_mask.nii.gz",
        study_dir / "brainmask.nii.gz",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "real_patients_batch2"
        / study_id
        / "03_skull"
        / "T1c_brainmask.nii.gz",
        PROJECT_ROOT
        / "data"
        / "processed"
        / "real_patients_batch2"
        / study_id
        / "04_registered_1mm"
        / "T1c_brainmask.nii.gz",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        img = _load_nifti(path)
        data = np.asanyarray(img.dataobj)
        if img.shape == target_shape and np.allclose(
            img.affine, target_affine, atol=1e-3, rtol=0.0
        ):
            return data > 0, f"used {_rel_or_str(path)}"
        # Resample nearest to label grid (voxel mapping via affines).
        try:
            # i_tgt = (inv(tgt_aff) @ src_aff) @ i_src  →  i_src = inv(M) @ i_tgt
            transform = np.linalg.inv(target_affine) @ img.affine
            inv = np.linalg.inv(transform)
            resampled = ndi.affine_transform(
                (data > 0).astype(np.float32),
                matrix=inv[:3, :3],
                offset=inv[:3, 3],
                output_shape=target_shape,
                order=0,
            )
            return resampled > 0.5, (
                f"resampled {_rel_or_str(path)} → label grid"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to resample brain mask %s: %s", path, exc)
            continue
    return None, "no brain mask available"


def check_corrected_label(
    study_dir: Path,
    *,
    tiny_island_ml: float = TINY_ISLAND_ML,
) -> StudyQCResult:
    """Run QC on one labeling package. Never writes the label file."""
    study_dir = Path(study_dir)
    study_id = study_dir.name
    corrected = study_dir / CORRECTED_NAME
    if not corrected.is_file():
        return StudyQCResult(study_id=study_id, status="not_labeled")

    result = StudyQCResult(study_id=study_id, status="labeled")
    t1c_path = study_dir / T1C_NAME
    if not t1c_path.is_file():
        result.status = "invalid"
        result.reasons.append(f"missing {T1C_NAME}")
        return result

    try:
        label_img = _load_nifti(corrected)
        t1c_img = _load_nifti(t1c_path)
    except Exception as exc:  # noqa: BLE001
        result.status = "invalid"
        result.reasons.append(f"failed to load NIfTI: {exc}")
        return result

    if not _grids_match(label_img, t1c_img):
        result.status = "invalid"
        result.reasons.append(
            f"grid mismatch vs T1c (label shape={tuple(label_img.shape)} "
            f"affine≠T1c shape={tuple(t1c_img.shape)})"
        )
        return result

    data = np.asanyarray(label_img.dataobj)
    # Integerize for value checks (ITK-SNAP may write float).
    labs = np.rint(data).astype(np.int16)
    uniq = sorted(int(x) for x in np.unique(labs))
    bad = [v for v in uniq if v not in ALLOWED_LABELS]
    if bad:
        result.status = "invalid"
        result.reasons.append(f"disallowed label values {bad} (allowed {sorted(ALLOWED_LABELS)})")
        return result

    if int(np.count_nonzero(labs)) == 0:
        result.status = "invalid"
        result.reasons.append("empty label (no foreground)")
        return result

    spacing = _spacing_mm(label_img)
    try:
        regions = _regions_from_classic(labs)
    except ValueError as exc:
        result.status = "invalid"
        result.reasons.append(str(exc))
        return result

    result.wt_ml = voxel_volume_ml_from_binary(regions["WT"], spacing)
    result.tc_ml = voxel_volume_ml_from_binary(regions["TC"], spacing)
    result.et_ml = voxel_volume_ml_from_binary(regions["ET"], spacing)
    result.wt_n_components = _n_components(regions["WT"])
    result.voxel_counts = {
        str(v): int(np.count_nonzero(labs == v)) for v in (0, 1, 2, 4) if v in uniq or v == 0
    }
    for v in uniq:
        result.voxel_counts[str(v)] = int(np.count_nonzero(labs == v))

    # Nestedness: ET ⊆ TC ⊆ WT is automatic for classic labels if we derive
    # regions correctly; still flag volume order mistakes and raw nesting
    # via region volumes / set logic.
    if result.wt_ml + 1e-9 < result.tc_ml:
        result.flags.append(
            f"WT ({result.wt_ml:.3f} mL) < TC ({result.tc_ml:.3f} mL) — nesting broken"
        )
    if result.tc_ml + 1e-9 < result.et_ml:
        result.flags.append(
            f"TC ({result.tc_ml:.3f} mL) < ET ({result.et_ml:.3f} mL) — ET not inside TC"
        )
    # Explicit voxel nesting (should hold for 1/2/4 encoding).
    if np.any(regions["ET"] & ~regions["TC"]):
        result.flags.append("ET voxels outside TC")
    if np.any(regions["TC"] & ~regions["WT"]):
        result.flags.append("TC voxels outside WT")

    if _touches_border(regions["WT"]):
        result.flags.append("WT touches image border")

    island_vols = _component_volumes_ml(regions["WT"], spacing)
    tiny = [v for v in island_vols if v < tiny_island_ml and v > 0]
    if tiny:
        result.flags.append(
            f"{len(tiny)} WT connected component(s) < {tiny_island_ml} mL "
            f"(smallest={min(tiny):.4f} mL)"
        )

    brain, brain_note = find_brain_mask(
        study_dir, target_shape=labs.shape, target_affine=label_img.affine
    )
    if brain is None:
        result.warnings.append(f"outside-brain check skipped ({brain_note})")
    else:
        outside = regions["WT"] & ~brain
        if np.any(outside):
            out_ml = voxel_volume_ml_from_binary(outside, spacing)
            result.flags.append(
                f"WT has {out_ml:.3f} mL outside brain mask ({brain_note})"
            )
        else:
            result.warnings.append(f"WT inside brain mask ({brain_note})")

    draft_path = study_dir / DRAFT_NAME
    if draft_path.is_file():
        try:
            draft_img = _load_nifti(draft_path)
            if not _grids_match(draft_img, label_img):
                result.warnings.append("v3 draft grid ≠ corrected; skip draft comparison")
            else:
                draft = np.rint(np.asanyarray(draft_img.dataobj)).astype(np.int16)
                result.identical_to_draft = bool(np.array_equal(labs, draft))
                if result.identical_to_draft:
                    result.warnings.append(
                        "corrected_label identical to v3_draft_label "
                        "(may be unedited)"
                    )
                try:
                    draft_regions = _regions_from_classic(draft)
                except ValueError:
                    # Drafts use classic 1/2/4; if somehow mixed, skip dice.
                    draft_regions = None
                if draft_regions is not None:
                    for name in ("ET", "TC", "WT"):
                        result.dice_vs_draft[name] = _dice(
                            regions[name], draft_regions[name]
                        )
                        d_ml = voxel_volume_ml_from_binary(draft_regions[name], spacing)
                        c_ml = voxel_volume_ml_from_binary(regions[name], spacing)
                        result.volume_delta_ml[name] = c_ml - d_ml
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"draft comparison failed: {exc}")
    else:
        result.warnings.append(f"missing {DRAFT_NAME}; skip draft comparison")

    if result.flags and result.status == "labeled":
        # Flags are likely mistakes but file is still structurally valid.
        pass
    return result


def discover_study_dirs(labeling_root: Path | None = None) -> list[Path]:
    root = Path(labeling_root) if labeling_root else DEFAULT_LABELING_ROOT
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "T1c.nii.gz").is_file()
    )


def check_all_studies(
    labeling_root: Path | None = None,
) -> list[StudyQCResult]:
    return [check_corrected_label(d) for d in discover_study_dirs(labeling_root)]


def format_progress_table(results: list[StudyQCResult]) -> str:
    lines = [
        "# Batch2 labeling progress",
        "",
        f"**{REFERENCE_DISCLAIMER}**",
        "",
        "Generated by `scripts/check_corrected_labels.py` (read-only on labels).",
        "",
        "| study_id | status | WT_mL | TC_mL | ET_mL | WT_CCs | vs draft | flags / reasons |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        def _f(x: float | None) -> str:
            return f"{x:.3f}" if x is not None else "—"

        vs = "—"
        if r.identical_to_draft is True:
            vs = "identical (warn)"
        elif r.dice_vs_draft:
            vs = (
                f"Dice WT={r.dice_vs_draft.get('WT', float('nan')):.3f} "
                f"ΔWT={r.volume_delta_ml.get('WT', float('nan')):+.3f}mL"
            )
        notes = "; ".join(r.reasons + r.flags + r.warnings) or "—"
        lines.append(
            f"| {r.study_id} | {r.status} | {_f(r.wt_ml)} | {_f(r.tc_ml)} | "
            f"{_f(r.et_ml)} | {r.wt_n_components if r.wt_n_components is not None else '—'} | "
            f"{vs} | {notes} |"
        )

    n_lab = sum(1 for r in results if r.status == "labeled")
    n_not = sum(1 for r in results if r.status == "not_labeled")
    n_inv = sum(1 for r in results if r.status == "invalid")
    n_flagged = sum(1 for r in results if r.flags)
    lines.extend(
        [
            "",
            f"**Totals:** labeled={n_lab}, not_labeled={n_not}, invalid={n_inv}, "
            f"flagged_likely_mistakes={n_flagged} / {len(results)} studies.",
            "",
        ]
    )
    return "\n".join(lines)


def write_progress_md(
    results: list[StudyQCResult],
    output_path: Path | None = None,
) -> Path:
    path = Path(output_path) if output_path else (
        DEFAULT_LABELING_ROOT / "LABELING_PROGRESS.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_progress_table(results), encoding="utf-8")
    return path
