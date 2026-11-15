"""Fetch / cache MNI template + Harvard-Oxford atlases; optional JHU tracts.

Public atlas files only — never patient scans. Large downloads live under
``atlases/`` (gitignored) or the nilearn data directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ATLAS_ROOT = PROJECT_ROOT / "atlases"
MNI_T1_NAME = "MNI152_T1_1mm.nii.gz"
HO_CORT_NAME = "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
HO_SUB_NAME = "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
HO_CORT_LABELS_NAME = "HarvardOxford-cort-labels.json"
HO_SUB_LABELS_NAME = "HarvardOxford-sub-labels.json"
JHU_PROB_NAME = "jhu/JHU-ICBM-tracts-prob-2mm.nii.gz"

# JHU 4D volume indices (0-based), matching mri_brats docs/ATLAS_SETUP.md.
JHU_TRACT_VOLUMES: dict[str, tuple[int, ...]] = {
    "corticospinal_tract_L": (2,),
    "corticospinal_tract_R": (3,),
    "optic_radiation_L": (),  # optional standalone files
    "optic_radiation_R": (),
}

ATLAS_LICENSES: dict[str, str] = {
    "MNI152 / ICBM152": (
        "ICBM 152 Nonlinear atlases (McGill / ICBM). Distributed via nilearn "
        "(fetch_icbm152_2009 / load_mni152_template). See ICBM / nilearn docs "
        "for redistribution terms; academic research use."
    ),
    "Harvard-Oxford": (
        "Harvard-Oxford cortical and subcortical structural atlases as "
        "distributed with FSL and fetched by nilearn "
        "(fetch_atlas_harvard_oxford, maxprob thr25 1mm). FSL atlas license; "
        "cite Makris / Desikan / Frazier / Harvard Center for Morphometric Analysis."
    ),
    "JHU ICBM tracts": (
        "JHU white-matter tractography atlas (Mori lab / Johns Hopkins) as "
        "distributed with FSL (JHU-ICBM-tracts-prob-2mm). Optional in this "
        "pipeline; used only when files are present under atlases/jhu/. "
        "Cite Hua / Mori / Wakana / FSL."
    ),
}

# Map Harvard-Oxford cortical region names → coarse lobes.
_LOBE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "frontal",
        (
            "frontal",
            "precentral",
            "paracingulate",
            "juxtapositional",
            "subcallosal",
            "operculum",  # frontal opercular handled below with care
        ),
    ),
    ("insula", ("insular",)),
    (
        "parietal",
        (
            "parietal",
            "postcentral",
            "supramarginal",
            "angular",
            "precuneous",
            "precuneus",
        ),
    ),
    (
        "temporal",
        (
            "temporal",
            "heschl",
            "planum",
            "parahippocampal",
            "fusiform",  # temporal fusiform; occipital fusiform overridden below
        ),
    ),
    (
        "occipital",
        (
            "occipital",
            "intracalcarine",
            "supracalcarine",
            "cuneal",
            "lingual",
        ),
    ),
    (
        "cingulate",
        ("cingulate",),
    ),
)


@dataclass(frozen=True)
class AtlasBundle:
    """Paths + label tables for location reporting."""

    mni_t1: Path
    ho_cortical: Path
    ho_subcortical: Path
    cortical_labels: dict[int, str]
    subcortical_labels: dict[int, str]
    jhu_prob: Path | None
    optic_radiation_L: Path | None
    optic_radiation_R: Path | None
    licenses: dict[str, str]


def lobe_for_cortical_label(name: str) -> str | None:
    """Return coarse lobe name for a Harvard-Oxford cortical label, or None."""
    lower = name.lower().strip()
    if not lower or lower == "background":
        return None
    # More specific overrides first.
    if "insular" in lower:
        return "insula"
    if "occipital fusiform" in lower or "temporal occipital fusiform" in lower:
        return "occipital" if "occipital fusiform" in lower else "temporal"
    if "temporooccipital" in lower or "temporo-occipital" in lower:
        return "temporal"
    if "frontal opercular" in lower:
        return "frontal"
    if "central opercular" in lower:
        return "frontal"
    if "parietal opercular" in lower:
        return "parietal"
    if "cingulate" in lower:
        return "cingulate"
    for lobe, keys in _LOBE_RULES:
        if any(k in lower for k in keys):
            # Avoid mapping "Frontal Opercular" twice; already handled.
            if lobe == "frontal" and "opercul" in lower and "frontal" not in lower:
                continue
            return lobe
    return "other"


def cortical_label_to_lobe_map(labels: dict[int, str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for idx, name in labels.items():
        if idx == 0:
            continue
        lobe = lobe_for_cortical_label(name)
        if lobe is not None:
            out[int(idx)] = lobe
    return out


def _write_label_json(path: Path, labels: list[str] | tuple[str, ...]) -> dict[int, str]:
    mapping = {i: str(name) for i, name in enumerate(labels)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def _require_nilearn():
    try:
        from nilearn import datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "nilearn is required for location reports. "
            "Install with: pip install 'nilearn>=0.10'"
        ) from exc


def ensure_mni_and_harvard_oxford(
    atlas_root: Path | None = None,
    *,
    force: bool = False,
) -> AtlasBundle:
    """Download (via nilearn) and cache MNI T1 + HO maxprob thr25 1 mm atlases.

    HO volumes are resampled onto the MNI152 1 mm template grid so registration
    and atlas warps share one reference space.
    """
    _require_nilearn()
    from nilearn import datasets

    root = Path(atlas_root) if atlas_root is not None else ATLAS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    (root / "jhu").mkdir(parents=True, exist_ok=True)

    mni_path = root / MNI_T1_NAME
    cort_path = root / HO_CORT_NAME
    sub_path = root / HO_SUB_NAME
    cort_labels_path = root / HO_CORT_LABELS_NAME
    sub_labels_path = root / HO_SUB_LABELS_NAME

    if force or not mni_path.is_file():
        template = datasets.load_mni152_template(resolution=1)
        nib.save(template, str(mni_path))
        logger.info("Cached MNI152 1 mm template → %s", mni_path)

    mni_img = nib.load(str(mni_path))

    if force or not cort_path.is_file() or not cort_labels_path.is_file():
        ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
        cort_img = ho_cort.maps
        if not isinstance(cort_img, nib.Nifti1Image):
            cort_img = nib.load(str(cort_img))
        cort_res = resample_from_to(cort_img, mni_img, order=0)
        nib.save(cort_res, str(cort_path))
        cort_labels = _write_label_json(cort_labels_path, list(ho_cort.labels))
        logger.info("Cached Harvard-Oxford cortical atlas → %s", cort_path)
    else:
        cort_labels = {
            int(k): v for k, v in json.loads(cort_labels_path.read_text()).items()
        }

    if force or not sub_path.is_file() or not sub_labels_path.is_file():
        ho_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
        sub_img = ho_sub.maps
        if not isinstance(sub_img, nib.Nifti1Image):
            sub_img = nib.load(str(sub_img))
        sub_res = resample_from_to(sub_img, mni_img, order=0)
        nib.save(sub_res, str(sub_path))
        sub_labels = _write_label_json(sub_labels_path, list(ho_sub.labels))
        logger.info("Cached Harvard-Oxford subcortical atlas → %s", sub_path)
    else:
        sub_labels = {
            int(k): v for k, v in json.loads(sub_labels_path.read_text()).items()
        }

    jhu = root / JHU_PROB_NAME
    or_l = root / "jhu" / "optic_radiation_L.nii.gz"
    or_r = root / "jhu" / "optic_radiation_R.nii.gz"

    return AtlasBundle(
        mni_t1=mni_path,
        ho_cortical=cort_path,
        ho_subcortical=sub_path,
        cortical_labels=cort_labels,
        subcortical_labels=sub_labels,
        jhu_prob=jhu if jhu.is_file() else None,
        optic_radiation_L=or_l if or_l.is_file() else None,
        optic_radiation_R=or_r if or_r.is_file() else None,
        licenses=dict(ATLAS_LICENSES),
    )


def try_copy_jhu_from(source: Path, atlas_root: Path | None = None) -> Path | None:
    """Optionally copy a local JHU 4D probability atlas into ``atlases/jhu/``."""
    root = Path(atlas_root) if atlas_root is not None else ATLAS_ROOT
    dest = root / JHU_PROB_NAME
    source = Path(source)
    if not source.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or source.stat().st_size != dest.stat().st_size:
        shutil.copy2(source, dest)
    return dest


def jhu_tract_mask_mni(
    bundle: AtlasBundle,
    *,
    prob_threshold: float = 0.25,
) -> dict[str, np.ndarray]:
    """Build MNI-space boolean masks for CST (+ optic radiation if present).

    Returns arrays on the MNI template grid. Empty dict when JHU is absent.
    """
    out: dict[str, np.ndarray] = {}
    mni_img = nib.load(str(bundle.mni_t1))
    target_shape = mni_img.shape[:3]

    if bundle.jhu_prob is not None and bundle.jhu_prob.is_file():
        jhu_img = nib.load(str(bundle.jhu_prob))
        data = np.asanyarray(jhu_img.dataobj)
        if data.ndim != 4:
            raise ValueError(f"Expected 4D JHU atlas, got shape {data.shape}")
        # Percent (0–100) → probability when peak > 1.5 (mri_brats convention).
        scale = 100.0 if float(np.nanmax(data)) > 1.5 else 1.0
        for name, vols in JHU_TRACT_VOLUMES.items():
            if not vols:
                continue
            vol = np.max(np.stack([data[..., int(v)] for v in vols], axis=0), axis=0)
            mask = (vol.astype(np.float32) / scale) >= float(prob_threshold)
            mask_img = nib.Nifti1Image(mask.astype(np.uint8), jhu_img.affine)
            res = resample_from_to(mask_img, mni_img, order=0)
            out[name] = np.asanyarray(res.dataobj).astype(bool)

    for key, path in (
        ("optic_radiation_L", bundle.optic_radiation_L),
        ("optic_radiation_R", bundle.optic_radiation_R),
    ):
        if path is None or not path.is_file():
            continue
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj).astype(np.float32)
        if float(np.nanmax(arr)) > 1.5:
            arr = arr / 100.0
        mask_img = nib.Nifti1Image((arr >= float(prob_threshold)).astype(np.uint8), img.affine)
        res = resample_from_to(mask_img, mni_img, order=0)
        out[key] = np.asanyarray(res.dataobj).astype(bool)

    # Sanity: all masks match MNI shape.
    for name, mask in out.items():
        if mask.shape != target_shape:
            raise ValueError(f"{name} shape {mask.shape} != MNI {target_shape}")
    return out


def license_notes(bundle: AtlasBundle | None = None) -> dict[str, str]:
    if bundle is not None:
        notes = dict(bundle.licenses)
        if bundle.jhu_prob is None:
            notes["JHU ICBM tracts"] = (
                notes["JHU ICBM tracts"]
                + " (not present on this machine — tract distances skipped)."
            )
        return notes
    return dict(ATLAS_LICENSES)
