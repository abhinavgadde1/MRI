"""Sagittal cutaway: clip volumes before marching cubes, opaque half-brain."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from pipeline import anatomy_optin
from visualization.plotly_3d import (
    binary_mask_to_trimesh,
    choose_sagittal_keep_low_x,
    clip_array_to_sagittal_half,
    normalize_mesh_view,
    sagittal_midline_index,
    validate_tumor_inside_brain,
)


def _centered_ellipsoid(shape=(48, 40, 36), frac=0.38) -> np.ndarray:
    xx, yy, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    c = np.array(shape, dtype=float) / 2.0
    r = np.array(shape, dtype=float) * float(frac)
    return (
        ((xx - c[0]) / r[0]) ** 2
        + ((yy - c[1]) / r[1]) ** 2
        + ((zz - c[2]) / r[2]) ** 2
    ) <= 1.0


def test_normalize_mesh_view():
    assert normalize_mesh_view("full") == "full"
    assert normalize_mesh_view("CUTAWAY") == "cutaway"
    assert normalize_mesh_view(None) == "full"
    assert normalize_mesh_view("nope") == "full"


def test_cutaway_keeps_half_with_more_tumor():
    shape = (40, 20, 20)
    mid = sagittal_midline_index(shape)
    off_right = np.zeros(shape, dtype=bool)
    off_right[mid + 4 : mid + 12, 4:16, 4:16] = True
    _, keep_low = choose_sagittal_keep_low_x(off_right)
    assert keep_low is False

    off_left = np.zeros(shape, dtype=bool)
    off_left[2:10, 4:16, 4:16] = True
    _, keep_low = choose_sagittal_keep_low_x(off_left)
    assert keep_low is True

    spanning = np.zeros(shape, dtype=bool)
    spanning[mid - 2 : mid + 10, 4:16, 4:16] = True  # more voxels on high-X
    _, keep_low = choose_sagittal_keep_low_x(spanning)
    assert keep_low is False


def test_cutaway_removes_roughly_half_mesh_volume():
    """Centered ellipsoid: clipped voxels and mesh extent are ~half."""
    mask = _centered_ellipsoid()
    midline, keep_low = choose_sagittal_keep_low_x(mask)
    clipped = clip_array_to_sagittal_half(
        mask, keep_low_x=keep_low, midline=midline
    )
    ratio = float(clipped.sum()) / float(mask.sum())
    assert 0.40 <= ratio <= 0.60, f"voxel ratio {ratio:.3f} not ~half"

    full = binary_mask_to_trimesh(
        mask, (1.0, 1.0, 1.0), smooth=False, target_faces=0
    )
    cut = binary_mask_to_trimesh(
        mask,
        (1.0, 1.0, 1.0),
        smooth=False,
        target_faces=0,
        sagittal_keep_low_x=keep_low,
        sagittal_midline=midline,
    )
    assert full is not None and cut is not None
    assert len(cut.faces) > 0
    full_x = float(full.bounds[1, 0] - full.bounds[0, 0])
    cut_x = float(cut.bounds[1, 0] - cut.bounds[0, 0])
    assert cut_x < 0.70 * full_x
    # Vertices stay on the kept side of the midline (mm = voxel index × 1 mm)
    # plus a 1-voxel MC pad slack.
    if keep_low:
        assert float(cut.vertices[:, 0].max()) <= midline + 1.5
    else:
        assert float(cut.vertices[:, 0].min()) >= midline - 1.5
    if full.is_watertight and cut.is_watertight:
        vol_ratio = abs(float(cut.volume)) / max(abs(float(full.volume)), 1e-6)
        assert 0.35 <= vol_ratio <= 0.65, f"mesh volume ratio {vol_ratio:.3f}"


def test_cutaway_does_not_bypass_tumor_inside_brain():
    """Containment check still runs on the unclipped volumes."""
    mask = _centered_ellipsoid()
    tumor = np.zeros_like(mask)
    c = np.array(mask.shape) // 2
    tumor[c[0] - 3 : c[0] + 3, c[1] - 3 : c[1] + 3, c[2] - 3 : c[2] + 3] = True
    tumor &= mask
    fracs = validate_tumor_inside_brain(mask, {"whole_tumor": tumor})
    assert fracs["whole_tumor"] == 0.0
    # Clipping is a rendering step; it does not replace this check.
    mid, keep = choose_sagittal_keep_low_x(tumor)
    clipped_brain = clip_array_to_sagittal_half(mask, keep_low_x=keep, midline=mid)
    clipped_tumor = clip_array_to_sagittal_half(tumor, keep_low_x=keep, midline=mid)
    fracs_cut = validate_tumor_inside_brain(
        clipped_brain, {"whole_tumor": clipped_tumor}
    )
    assert fracs_cut["whole_tumor"] == 0.0


def _synthetic_anatomy_volumes(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    shape = (40, 32, 28)
    t1 = np.zeros(shape, dtype=np.float32)
    t1[2:38, 2:30, 2:26] = 80.0
    labels = np.zeros(shape, dtype=np.int16)
    labels[2:16, 3:29, 3:25] = 3
    labels[24:38, 3:29, 3:25] = 42
    labels[4:14, 8:24, 6:20] = 2
    labels[26:36, 8:24, 6:20] = 41
    labels[16:24, 2:12, 6:20] = 7
    labels[18:22, 20:30, 8:18] = 16
    pred = np.zeros(shape, dtype=np.int16)
    # Tumor mostly on the low-X (left) half so cutaway keeps that side.
    pred[6:14, 10:20, 8:18] = 2
    pred[8:12, 12:18, 10:16] = 1
    pred[9:11, 13:16, 11:15] = 3
    t1_path = tmp_path / "T1_1mm.nii.gz"
    nib.save(nib.Nifti1Image(t1, np.eye(4)), str(t1_path))
    return t1_path, pred, labels


def test_cutaway_renders_synthetic_anatomy_without_error(tmp_path: Path):
    t1_path, pred, labels = _synthetic_anatomy_volumes(tmp_path)
    n, structures = anatomy_optin._write_combined_mesh(
        tmp_path,
        t1_path=t1_path,
        pred=pred,
        labels=labels,
        hide_et=False,
        case_id="synth_cutaway",
        postprocess="lcc",
    )
    assert n >= 1
    full_html = tmp_path / "mesh_3d.html"
    cut_html = tmp_path / "mesh_3d_cutaway.html"
    assert full_html.is_file() and cut_html.is_file()
    full_txt = full_html.read_text(encoding="utf-8")
    cut_txt = cut_html.read_text(encoding="utf-8")
    assert "Left cerebral cortex" in full_txt
    assert "Left cerebral cortex" in cut_txt
    assert "Whole Tumor" in cut_txt
    assert "sagittal cutaway" in cut_txt.lower()
    assert "Brain surface" in full_txt
    # Gray shell is dropped in cutaway so it cannot cap the cut face.
    assert "Brain surface" not in cut_txt
    rendered = [s for s in structures if s["rendered"]]
    assert rendered
