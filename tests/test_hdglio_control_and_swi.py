"""Tests for HD-GLIO control-case selection and SWI dark-focus heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from segmentation.batch2_swi import swi_dark_focus_inside_contour
from segmentation.hdglio_control import has_identity_affine, select_heldout_control_cases
import nibabel as nib


def test_select_heldout_includes_small_median_large():
    cases = select_heldout_control_cases(n=5, seed=42)
    assert len(cases) == 5
    roles = {c.role for c in cases}
    assert {"small", "median", "large", "random"} <= roles
    by_role = {c.role: c for c in cases}
    assert by_role["small"].wt_ml <= by_role["median"].wt_ml <= by_role["large"].wt_ml
    # deterministic
    again = select_heldout_control_cases(n=5, seed=42)
    assert [c.study_id for c in cases] == [c.study_id for c in again]


def test_identity_affine():
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    assert has_identity_affine(img)
    aff = np.eye(4)
    aff[0, 0] = -1
    assert not has_identity_affine(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), aff))


def test_swi_dark_focus_yes():
    swi = np.ones((20, 20, 8), dtype=np.float32) * 100
    contour = np.zeros_like(swi, dtype=bool)
    contour[8:12, 8:12, 3:5] = True
    swi[contour] = 10  # dark inside
    brain = swi > 0
    out = swi_dark_focus_inside_contour(swi, contour, brain)
    assert out["label"] == "yes"


def test_swi_dark_focus_no():
    swi = np.ones((20, 20, 8), dtype=np.float32) * 100
    contour = np.zeros_like(swi, dtype=bool)
    contour[8:12, 8:12, 3:5] = True
    out = swi_dark_focus_inside_contour(swi, contour)
    assert out["label"] == "no"


def test_swi_dark_focus_empty_not_assessable():
    swi = np.ones((8, 8, 4), dtype=np.float32)
    contour = np.zeros_like(swi, dtype=bool)
    out = swi_dark_focus_inside_contour(swi, contour)
    assert out["label"] == "not_assessable"
