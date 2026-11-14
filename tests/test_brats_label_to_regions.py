"""Regression tests for ``brats_label_to_regions`` (project vs classic BraTS)."""

from __future__ import annotations

import pytest
import torch

from segmentation.model import brats_label_to_regions


def _vol_project() -> torch.Tensor:
    """Tiny volume using project convention {0,1,2,3}."""
    lab = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    lab[0, 0, 1, 1, 1] = 1  # necrotic
    lab[0, 0, 1, 1, 2] = 2  # edema
    lab[0, 0, 2, 2, 2] = 3  # enhancing (project)
    return lab


def _vol_classic_from_project(project: torch.Tensor) -> torch.Tensor:
    """Remap enhancing 3 → 4; geometry otherwise identical."""
    out = project.clone()
    out[out == 3] = 4
    return out


def test_project_convention_0123():
    lab = _vol_project()
    regions = brats_label_to_regions(lab)
    assert regions.shape == (1, 3, 4, 4, 4)
    # necrotic → TC+WT, not ET
    assert regions[0, 0, 1, 1, 1] == 0.0
    assert regions[0, 1, 1, 1, 1] == 1.0
    assert regions[0, 2, 1, 1, 1] == 1.0
    # edema → WT only
    assert regions[0, 0, 1, 1, 2] == 0.0
    assert regions[0, 1, 1, 1, 2] == 0.0
    assert regions[0, 2, 1, 1, 2] == 1.0
    # enhancing 3 → ET+TC+WT
    assert torch.equal(regions[0, :, 2, 2, 2], torch.tensor([1.0, 1.0, 1.0]))


def test_classic_convention_0124():
    lab = _vol_classic_from_project(_vol_project())
    assert set(lab.unique().tolist()) == {0, 1, 2, 4}
    regions = brats_label_to_regions(lab)
    assert torch.equal(regions[0, :, 2, 2, 2], torch.tensor([1.0, 1.0, 1.0]))
    assert regions[0, 2, 1, 1, 2] == 1.0  # edema in WT


def test_project_and_classic_yield_identical_regions():
    project = _vol_project()
    classic = _vol_classic_from_project(project)
    r_p = brats_label_to_regions(project)
    r_c = brats_label_to_regions(classic)
    assert torch.equal(r_p, r_c)


def test_mixed_3_and_4_raises():
    """Mixed enhancing codes in one volume → clear error (encoding bug)."""
    lab = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
    lab[0, 0, 0, 0, 0] = 3
    lab[0, 0, 1, 1, 1] = 4
    with pytest.raises(ValueError, match="mixes enhancing codes 3 and 4"):
        brats_label_to_regions(lab)
