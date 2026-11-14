"""Tests for batch2 dataset discovery, domain augs, folds, and checkpoint guards."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from segmentation.batch2_augment import (
    DomainAugmentFlags,
    mean_abs_slice_diff,
    thick_slice_simulate,
)
from segmentation.batch2_dataset import (
    classic_brats_label_to_regions_numpy,
    discover_batch2_studies,
    group_studies_by_patient,
    make_patient_folds,
    study_to_datadict,
)
from segmentation.batch2_finetune import (
    assert_safe_checkpoint_output_dir,
    sample_replay_mix,
)
from segmentation.model import brats_label_to_regions


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aff = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(data.astype(np.float32), aff), str(path))


def _make_study(
    root: Path,
    study_id: str,
    *,
    patient_id: str | None = "P1",
    label: np.ndarray | None = None,
    label_name: str = "corrected_label.nii.gz",
    readme: str = "# ok\n",
    mismatch_grid: bool = False,
    empty_label: bool = False,
) -> Path:
    study = root / study_id
    study.mkdir(parents=True, exist_ok=True)
    shape = (16, 16, 12)
    aff = np.eye(4)
    for mod in ("FLAIR", "T1", "T1c", "T2"):
        _write_nifti(study / f"{mod}.nii.gz", np.random.randn(*shape).astype(np.float32), aff)
    if empty_label:
        lab = np.zeros(shape, dtype=np.float32)
    elif label is not None:
        lab = label
    else:
        lab = np.zeros(shape, dtype=np.float32)
        lab[4:8, 4:8, 4:8] = 4
        lab[5:7, 5:7, 5:7] = 1
        lab[3:9, 3:9, 3:4] = 2
    lab_aff = np.eye(4) * (1.1 if mismatch_grid else 1.0)
    if mismatch_grid:
        lab_aff[0, 0] = 1.5
    _write_nifti(study / label_name, lab, lab_aff if mismatch_grid else aff)
    (study / "README.md").write_text(readme, encoding="utf-8")
    meta = {"patient_id": patient_id} if patient_id is not None else {}
    (study / "T1c_native.json").write_text(json.dumps(meta), encoding="utf-8")
    return study


def test_skip_no_clear_lesion(tmp_path: Path):
    _make_study(tmp_path, "s1", readme="This case has no clear lesion on review.\n")
    _make_study(tmp_path, "s2", patient_id="P2")
    result = discover_batch2_studies(tmp_path)
    assert len(result.accepted) == 1
    assert result.accepted[0].study_id == "s2"
    assert any("no clear lesion" in (s.skip_reason or "") for s in result.skipped)


def test_skip_empty_label(tmp_path: Path):
    _make_study(tmp_path, "empty", empty_label=True)
    result = discover_batch2_studies(tmp_path)
    assert len(result.accepted) == 0
    assert "empty" in (result.skipped[0].skip_reason or "")


def test_skip_grid_mismatch(tmp_path: Path):
    _make_study(tmp_path, "badgrid", mismatch_grid=True)
    result = discover_batch2_studies(tmp_path)
    assert len(result.accepted) == 0
    assert "mismatch" in (result.skipped[0].skip_reason or "").lower()


def test_label_to_region_conversion_classic_brats():
    label = np.zeros((4, 4, 4), dtype=np.int16)
    label[1, 1, 1] = 4  # ET
    label[2, 2, 2] = 1  # necrotic → TC+WT
    label[3, 3, 3] = 2  # edema → WT
    regions = classic_brats_label_to_regions_numpy(label)
    assert regions.shape == (3, 4, 4, 4)
    # ET
    assert regions[0, 1, 1, 1] == 1
    assert regions[0, 2, 2, 2] == 0
    # TC includes necrotic + ET
    assert regions[1, 1, 1, 1] == 1
    assert regions[1, 2, 2, 2] == 1
    assert regions[1, 3, 3, 3] == 0
    # WT includes all
    assert regions[2, 1, 1, 1] == 1
    assert regions[2, 2, 2, 2] == 1
    assert regions[2, 3, 3, 3] == 1

    # Torch path accepts classic 4 as well.
    t = torch.as_tensor(label).unsqueeze(0).unsqueeze(0)
    tr = brats_label_to_regions(t)
    assert torch.allclose(tr, torch.as_tensor(regions).unsqueeze(0))


def test_patient_grouping_no_leakage(tmp_path: Path):
    _make_study(tmp_path, "a1", patient_id="PA")
    _make_study(tmp_path, "a2", patient_id="PA")
    _make_study(tmp_path, "b1", patient_id="PB")
    _make_study(tmp_path, "c1", patient_id="PC")
    discovered = discover_batch2_studies(tmp_path)
    records = [study_to_datadict(s) for s in discovered.accepted]
    groups = group_studies_by_patient(records)
    assert set(groups) == {"PA", "PB", "PC"}
    assert len(groups["PA"]) == 2

    folds = make_patient_folds(records, seed=0, max_lopo_patients=12)
    assert folds[0]["mode"] == "lopo"
    for fold in folds:
        train_pids = {r["patient_id"] for r in fold["train"]}
        held_pids = {r["patient_id"] for r in fold["held"]}
        assert train_pids.isdisjoint(held_pids)
        # Both studies of PA stay together.
        if "PA" in held_pids:
            assert {r["subject_id"] for r in fold["held"]} == {"a1", "a2"}
            assert "a1" not in {r["subject_id"] for r in fold["train"]}


def test_patient_id_missing_warns(tmp_path: Path):
    _make_study(tmp_path, "solo", patient_id=None)
    with pytest.warns(UserWarning, match="patient_id missing"):
        result = discover_batch2_studies(tmp_path)
    assert result.accepted[0].patient_id == "solo"


def test_thick_slice_shape_and_smoothness():
    rng = np.random.default_rng(0)
    # High-frequency along z → thick-slice should increase smoothness.
    vol = rng.normal(0, 1, size=(4, 32, 32, 48)).astype(np.float32)
    image = torch.from_numpy(vol)
    before = mean_abs_slice_diff(image, axis=-1)
    out, _ = thick_slice_simulate(image, factor=4, label=None)
    assert out.shape == image.shape
    after = mean_abs_slice_diff(out, axis=-1)
    assert after < before


def test_five_fold_when_many_patients():
    records = [
        {"subject_id": f"s{i}", "patient_id": f"P{i}"} for i in range(15)
    ]
    folds = make_patient_folds(records, seed=1, max_lopo_patients=12, n_folds=5)
    assert len(folds) == 5
    assert folds[0]["mode"] == "5-fold"
    # Every patient appears in exactly one held set.
    held_counts: dict[str, int] = {}
    for fold in folds:
        for pid in fold["held_patient_ids"]:
            held_counts[pid] = held_counts.get(pid, 0) + 1
        train_pids = set(fold["train_patient_ids"])
        held_pids = set(fold["held_patient_ids"])
        assert train_pids.isdisjoint(held_pids)
    assert all(c == 1 for c in held_counts.values())
    assert len(held_counts) == 15


def test_guard_refuses_brats_scale_full(tmp_path: Path):
    from config import PROJECT_ROOT

    protected = PROJECT_ROOT / "checkpoints" / "brats_scale_full"
    with pytest.raises(ValueError, match="protected"):
        assert_safe_checkpoint_output_dir(protected)
    with pytest.raises(ValueError, match="protected"):
        assert_safe_checkpoint_output_dir(protected / "nested")
    # Safe path is allowed.
    safe = assert_safe_checkpoint_output_dir(tmp_path / "batch2_finetune_final")
    assert safe == (tmp_path / "batch2_finetune_final").resolve()


def test_sample_replay_mix_ratio():
    b2 = [{"subject_id": f"b{i}"} for i in range(4)]
    brats = [{"subject_id": f"r{i}"} for i in range(20)]
    mixed = sample_replay_mix(
        b2, brats, ratio_batch2=1, ratio_brats=1, seed=0, epoch=0
    )
    n_b2 = sum(1 for r in mixed if r.get("source") == "batch2")
    n_br = sum(1 for r in mixed if r.get("source") == "brats_replay")
    assert n_b2 == 4
    assert n_br == 4


def test_domain_aug_flags_log_names():
    flags = DomainAugmentFlags(thick_slice=True, intensity=False, modality_dropout=True)
    names = flags.enabled_names()
    assert any("thick_slice" in n for n in names)
    assert any("modality_dropout" in n for n in names)
    assert not any("intensity" in n for n in names)
