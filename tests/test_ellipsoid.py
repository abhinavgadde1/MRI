"""Tests for clinical ellipsoid volume (A × B × C / 2 and π/6 × ABC)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from validation.ellipsoid import (
    clinical_ellipsoid_from_diameters,
    clinical_ellipsoid_volume_cm3,
    compare_to_ellipsoid,
    diameters_from_binary,
    fit_ellipsoid,
    geometric_ellipsoid_volume_cm3,
    mesh_voxel_relative_error,
    principal_axis_bounding_diameters,
    voxel_volume_ml_from_binary,
)
from validation.ellipsoid_batch import checkpoint_tag, icc_2_1


def test_clinical_formula_abc_over_2():
    # A,B,C in mm → V = ABC/2 / 1000 cm³
    assert clinical_ellipsoid_volume_cm3((40.0, 30.0, 20.0)) == pytest.approx(
        40 * 30 * 20 / 2 / 1000
    )


def test_geometric_formula_pi_over_6():
    assert geometric_ellipsoid_volume_cm3((40.0, 30.0, 20.0)) == pytest.approx(
        np.pi / 6.0 * 40 * 30 * 20 / 1000
    )


def test_manual_diameters_sorted():
    ell = clinical_ellipsoid_from_diameters((20.0, 40.0, 30.0))
    assert ell.diameters_mm == (40.0, 30.0, 20.0)
    assert ell.source == "manual"
    assert ell.volume_cm3 == pytest.approx(12.0)


def test_principal_axis_extents_of_box():
    # Axis-aligned box 10×6×4 mm (voxel centers on a grid)
    xs, ys, zs = np.meshgrid(
        np.linspace(0, 10, 11),
        np.linspace(0, 6, 7),
        np.linspace(0, 4, 5),
        indexing="ij",
    )
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    diameters, center, _ = principal_axis_bounding_diameters(pts)
    assert diameters[0] == pytest.approx(10.0, abs=0.01)
    assert diameters[1] == pytest.approx(6.0, abs=0.01)
    assert diameters[2] == pytest.approx(4.0, abs=0.01)
    assert center[0] == pytest.approx(5.0, abs=0.01)


def test_fit_from_mask_nifti(tmp_path: Path):
    data = np.zeros((30, 20, 16), dtype=np.uint8)
    # Cuboid 20×10×8 voxels at 1 mm → diameters ~19, 9, 7 (voxel-index extents)
    data[5:25, 5:15, 4:12] = 1
    path = tmp_path / "mask.nii.gz"
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(img, str(path))

    ell = fit_ellipsoid(path, label=1)
    assert ell.source == "principal_axes"
    assert ell.diameters_mm[0] >= ell.diameters_mm[1] >= ell.diameters_mm[2]
    assert ell.volume_cm3 == pytest.approx(
        clinical_ellipsoid_volume_cm3(ell.diameters_mm)
    )

    comparison = compare_to_ellipsoid(path, label=1)
    assert comparison.ellipsoid_volume_ml == pytest.approx(ell.volume_ml)
    assert comparison.mask_volume_ml > 0


def test_fit_manual_overrides_mask():
    ell = fit_ellipsoid(diameters_mm=(50.0, 40.0, 30.0))
    assert ell.source == "manual"
    assert ell.volume_cm3 == pytest.approx(30.0)


def test_rejects_nonpositive_diameters():
    with pytest.raises(ValueError):
        clinical_ellipsoid_volume_cm3((10.0, 0.0, 5.0))


def _synthetic_ellipsoid_mask(
    *,
    radii_mm: tuple[float, float, float] = (20.0, 15.0, 10.0),
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, float]:
    """Voxelize an ellipsoid; return mask and true geometric volume (mL)."""
    rx, ry, rz = radii_mm
    sx, sy, sz = spacing_mm
    # Pad so the ellipsoid is fully inside the grid.
    nx = int(np.ceil(2 * rx / sx)) + 6
    ny = int(np.ceil(2 * ry / sy)) + 6
    nz = int(np.ceil(2 * rz / sz)) + 6
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0
    xs = (np.arange(nx) - cx) * sx
    ys = (np.arange(ny) - cy) * sy
    zs = (np.arange(nz) - cz) * sz
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    mask = (xx / rx) ** 2 + (yy / ry) ** 2 + (zz / rz) ** 2 <= 1.0
    true_ml = (4.0 / 3.0) * np.pi * rx * ry * rz / 1000.0
    return mask, float(true_ml)


def test_pi_over_6_recovers_true_ellipsoid_volume_within_3pct():
    """π/6 × ABC on AABB diameters recovers the analytic ellipsoid volume."""
    mask, true_ml = _synthetic_ellipsoid_mask(radii_mm=(24.0, 18.0, 12.0))
    spacing = (1.0, 1.0, 1.0)
    diameters, _ = diameters_from_binary(mask, spacing, method="aabb")
    est = geometric_ellipsoid_volume_cm3(diameters)
    rel_err = abs(est - true_ml) / true_ml
    assert rel_err < 0.03, f"rel_err={rel_err:.4f} est={est:.3f} true={true_ml:.3f}"


def test_pca_pi_over_6_also_within_3pct_on_sphere():
    """Sphere is rotationally invariant — PCA extents ≈ AABB extents."""
    mask, true_ml = _synthetic_ellipsoid_mask(radii_mm=(16.0, 16.0, 16.0))
    spacing = (1.0, 1.0, 1.0)
    diameters, _ = diameters_from_binary(mask, spacing, method="pca")
    est = geometric_ellipsoid_volume_cm3(diameters)
    assert abs(est - true_ml) / true_ml < 0.03


def test_mesh_voxel_agree_within_2pct_on_ellipsoid():
    mask, _true = _synthetic_ellipsoid_mask(radii_mm=(18.0, 14.0, 10.0))
    spacing = (1.0, 1.0, 1.0)
    err = mesh_voxel_relative_error(mask, spacing)
    assert err < 0.02, f"mesh/voxel rel_err={err:.4f}"
    assert voxel_volume_ml_from_binary(mask, spacing) > 0


def test_radiologist_diameters_positive():
    mask, _ = _synthetic_ellipsoid_mask(radii_mm=(20.0, 12.0, 8.0))
    diameters, _ = diameters_from_binary(mask, (1.0, 1.0, 1.0), method="radiologist")
    assert all(d > 0 for d in diameters)
    assert diameters[0] >= diameters[1] >= diameters[2]


def test_checkpoint_tag_uses_parent_for_best_model():
    assert checkpoint_tag("checkpoints/brats_pretrain/best_model.pt") == "brats_pretrain"
    assert checkpoint_tag("checkpoints/other/model.pt") == "other"
    assert checkpoint_tag("checkpoints/custom_weights.pt") == "custom_weights"


def test_icc_perfect_agreement():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert icc_2_1(x, x) == pytest.approx(1.0)


def test_brats_region_volumes_nest_et_subset_tc_subset_wt():
    """ET ⊆ TC ⊆ WT and voxel counts are consistent with multi-channel defs."""
    from validation.ellipsoid_gt_regions import (
        assert_region_nesting,
        brats_label_to_region_masks,
    )

    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:6, 2:6, 2:6] = 2  # edema → WT only
    label[3:5, 3:5, 3:5] = 1  # necrotic → TC+WT
    label[3:4, 3:4, 3:4] = 3  # enhancing → ET+TC+WT

    assert_region_nesting(label)
    masks = brats_label_to_region_masks(label)
    assert int(masks["enhancing_tumor"].sum()) == 1
    assert int(masks["tumor_core"].sum()) == 8  # 2³ necrotic/ET box
    assert int(masks["whole_tumor"].sum()) == 64  # 4³ edema box
    assert np.all(masks["enhancing_tumor"] <= masks["tumor_core"])
    assert np.all(masks["tumor_core"] <= masks["whole_tumor"])

    # Empty label: all empty subsets still nest.
    assert_region_nesting(np.zeros((3, 3, 3), dtype=np.uint8))
