"""Synthetic tests for optional tumor location reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from location import LOCATION_REPORT_ENABLED_DEFAULT
from location.atlases import ATLAS_LICENSES, lobe_for_cortical_label
from location.metrics import (
    DEEP_LABEL,
    hemisphere_fractions,
    lobe_occupancy,
    plain_language_summary,
    region_location_block,
)
from location.qc import (
    BRAIN_DICE_GOOD,
    COVERAGE_GOOD,
    SURFACE_P95_GOOD_MM,
    VENTRICLE_DICE_GOOD,
    VENTRICLE_DICE_GOOD_CSF,
    dice_bool,
    estimate_patient_ventricle_mask,
    flag_registration_quality,
    fov_coverage_fraction,
    gate_message,
    surface_distance_mm,
)
from location.register import (
    BRAIN_DICE_MIN,
    build_registration_cost_mask,
    estimate_brain_mask,
    _brain_fov_mask,
)
from location.report import (
    LCC_FRACTION_MIN,
    assess_location_plausibility,
    largest_component_fraction,
)


def test_location_report_off_by_default():
    assert LOCATION_REPORT_ENABLED_DEFAULT is False


def test_lobe_mapping_harvard_oxford_names():
    assert lobe_for_cortical_label("Frontal Pole") == "frontal"
    assert lobe_for_cortical_label("Superior Parietal Lobule") == "parietal"
    assert lobe_for_cortical_label("Middle Temporal Gyrus, posterior division") == "temporal"
    assert lobe_for_cortical_label("Occipital Pole") == "occipital"
    assert lobe_for_cortical_label("Insular Cortex") == "insula"
    assert lobe_for_cortical_label("Background") is None


def test_blob_in_known_lobe_returns_that_lobe():
    """Synthetic: parietal-labeled voxels + blob → dominant parietal."""
    shape = (40, 40, 40)
    cortical = np.zeros(shape, dtype=np.int16)
    labels = {
        0: "Background",
        1: "Frontal Pole",
        18: "Superior Parietal Lobule",
        8: "Temporal Pole",
    }
    cortical[5:35, 5:35, 20:35] = 18  # parietal slab
    cortical[5:35, 5:35, 5:15] = 1  # frontal slab

    mask = np.zeros(shape, dtype=bool)
    mask[15:25, 15:25, 24:32] = True

    result = lobe_occupancy(mask, cortical, labels, spacing_mm=(1.0, 1.0, 1.0))
    assert result["dominant_lobe"] == "parietal"
    assert result["dominant_pct"] >= 90.0
    assert "other" not in result["lobes_nearest"]
    assert result["dominant_lobe"] != "other"


def test_deep_wm_blob_near_known_lobe_returns_that_lobe():
    """Deep white-matter blob within 15 mm of parietal → nearest parietal."""
    shape = (50, 50, 50)
    cortical = np.zeros(shape, dtype=np.int16)
    labels = {
        0: "Background",
        18: "Superior Parietal Lobule",
        1: "Frontal Pole",
    }
    # Parietal cortex slab at z=40..45
    cortical[10:40, 10:40, 40:46] = 18
    # Tumor entirely in WM below cortex (z=28..32), ~8–12 mm away at 1 mm spacing
    mask = np.zeros(shape, dtype=bool)
    mask[20:30, 20:30, 28:33] = True
    assert not np.any(cortical[mask] > 0)

    result = lobe_occupancy(
        mask, cortical, labels, spacing_mm=(1.0, 1.0, 1.0), nearest_max_mm=15.0
    )
    assert result["dominant_lobe"] == "parietal"
    assert result["dominant_pct"] >= 90.0
    assert result["unlabeled_pct"] == 100.0
    assert result["lobes_cortical_only"] == {}
    assert "parietal" in result["lobes_nearest"]
    assert "other" not in (result["dominant_lobe"] or "")


def test_far_deep_blob_is_deep_central():
    """Beyond 15 mm from any cortex → deep/central, never 'other'."""
    shape = (60, 60, 60)
    cortical = np.zeros(shape, dtype=np.int16)
    labels = {0: "Background", 18: "Superior Parietal Lobule"}
    cortical[5:15, 5:15, 5:15] = 18
    mask = np.zeros(shape, dtype=bool)
    mask[45:50, 45:50, 45:50] = True  # far corner

    result = lobe_occupancy(
        mask, cortical, labels, spacing_mm=(1.0, 1.0, 1.0), nearest_max_mm=15.0
    )
    assert result["dominant_lobe"] == DEEP_LABEL
    assert "other" not in result["lobes_nearest"]


def test_registration_cost_mask_excludes_dilated_tumor():
    """Tumor voxels (and 5 mm neighbourhood) are excluded from the cost mask."""
    t1 = np.ones((40, 40, 40), dtype=np.float32)
    tumor = np.zeros((40, 40, 40), dtype=bool)
    tumor[18:22, 18:22, 18:22] = True
    cost = build_registration_cost_mask(
        t1, tumor, spacing_mm=(1.0, 1.0, 1.0), dilate_mm=5.0
    )
    assert not np.any(cost[tumor]), "tumor voxels must be excluded from cost"
    # Edge of tumor at z=18; z=14 is 4 mm away → excluded; z=5 is far → kept.
    assert not cost[18, 18, 14]
    assert cost[5, 5, 5]


def test_estimate_brain_mask_hole_fill_and_lcc():
    """Noise speckles dropped; holes inside brain filled."""
    t1 = np.zeros((30, 30, 30), dtype=np.float32)
    t1[5:25, 5:25, 5:25] = 1.0
    t1[10:12, 10:12, 10:12] = 0.0  # hole
    t1[0, 0, 0] = 5.0  # noise speck
    brain = estimate_brain_mask(t1)
    assert brain[10, 10, 10], "interior hole should be filled"
    assert not brain[0, 0, 0], "isolated speck must not survive LCC"
    assert brain[15, 15, 15]


def test_brain_fov_dice_not_full_grid():
    """Dilated-brain FOV must not use the full rectangular grid."""
    brain = np.zeros((40, 40, 40), dtype=bool)
    brain[10:30, 10:30, 10:30] = True
    fov = _brain_fov_mask(brain, dilate_vox=2)
    assert fov.sum() < brain.size
    assert fov.sum() > brain.sum()
    template = brain.copy()
    assert dice_bool(brain & fov, template & fov) == 1.0
    assert BRAIN_DICE_MIN == BRAIN_DICE_GOOD == 0.85


def test_independent_qc_metrics_and_good_rule():
    """Ventricle Dice, surface distance, coverage, and good/caution/unreliable."""
    brain = np.zeros((40, 40, 40), dtype=bool)
    brain[5:35, 5:35, 5:35] = True
    vent = np.zeros_like(brain)
    vent[18:22, 18:22, 15:25] = True
    t1 = np.ones((40, 40, 40), dtype=np.float32)
    t1[vent] = 0.1  # dark CSF
    est, src = estimate_patient_ventricle_mask(t1, brain)
    assert src.startswith("t1")
    assert est.sum() > 0

    # SynthSeg labels preferred
    synth = np.zeros((40, 40, 40), dtype=np.int16)
    synth[18:22, 18:22, 15:25] = 4
    est2, src2 = estimate_patient_ventricle_mask(t1, brain, synthseg=synth)
    assert src2 == "synthseg"
    assert dice_bool(est2, vent) == 1.0

    mean_d, p95 = surface_distance_mm(brain, brain, (1.0, 1.0, 1.0))
    assert mean_d < 1.0
    assert p95 < 2.0

    fov = np.zeros_like(brain)
    fov[5:35, 5:35, 5:30] = True  # missing superior slab
    cov = fov_coverage_fraction(brain, fov)
    assert 0.7 < cov < 1.0

    good_q, _ = flag_registration_quality(
        brain_dice=0.90,
        ventricle_dice=0.70,
        surface_p95_mm=5.0,
        coverage=0.95,
        stage="affine",
        ventricle_source="synthseg",
    )
    assert good_q == "good"
    assert VENTRICLE_DICE_GOOD == 0.60
    assert VENTRICLE_DICE_GOOD_CSF == 0.30
    assert SURFACE_P95_GOOD_MM == 8.0
    assert COVERAGE_GOOD == 0.90

    # CSF-mode: 0.40 clears the 0.30 CSF gate
    good_csf, _ = flag_registration_quality(
        brain_dice=0.90,
        ventricle_dice=0.40,
        surface_p95_mm=5.0,
        coverage=0.95,
        stage="affine",
        ventricle_source="t1_t2_csf_near_atlas",
    )
    assert good_csf == "good"

    caution_q, note = flag_registration_quality(
        brain_dice=0.90,
        ventricle_dice=0.20,
        surface_p95_mm=5.0,
        coverage=0.95,
        stage="affine",
        ventricle_source="t1_csf_near_atlas",
    )
    assert caution_q == "caution"
    assert "ventricle" in note.lower()

    bad_q, _ = flag_registration_quality(
        brain_dice=0.50,
        ventricle_dice=0.70,
        surface_p95_mm=5.0,
        coverage=0.95,
        stage="affine",
        ventricle_source="synthseg",
    )
    assert bad_q == "registration unreliable"


def test_plausibility_gate_wording_never_says_no_tumor_detected():
    """Below floor or LCC < 50% → gated with required wording."""
    spacing = (1.0, 1.0, 1.0)
    small = np.zeros((64, 64, 64), dtype=bool)
    small[20:30, 20:30, 20:25] = True  # 0.5 mL
    gate = assess_location_plausibility(small, spacing_mm=spacing)
    assert gate["gated"] is True
    msg = gate["message"]
    assert "no tumor detected" not in msg.lower()
    assert "BraTS size floor" in msg
    assert "location not reported" in msg
    assert "small lesion or false positive, unverified" in msg
    assert "0.50 mL" in msg or "0.5" in msg
    assert "model volume" in msg

    frag = np.zeros((100, 100, 100), dtype=bool)
    frag[5:25, 5:25, 5:30] = True
    frag[40:60, 40:60, 5:30] = True
    frag[70:90, 70:90, 5:35] = True
    assert frag.sum() * 0.001 > 20.0
    assert largest_component_fraction(frag) < LCC_FRACTION_MIN
    gate2 = assess_location_plausibility(frag, spacing_mm=spacing)
    assert gate2["gated"] is True
    assert gate2["reason"] == "fragmented"
    assert "no tumor detected" not in gate2["message"].lower()

    ok = np.zeros((80, 80, 80), dtype=bool)
    ok[20:50, 20:50, 20:50] = True
    gate3 = assess_location_plausibility(ok, spacing_mm=spacing)
    assert gate3["gated"] is False

    # Helper matches floor formatting
    assert "20.9 mL" in gate_message(14.2, 20.935) or "20.9" in gate_message(
        14.2, 20.935
    )


def test_run_location_report_gated_writes_statement(tmp_path: Path):
    """Gated cases write location.json + report text without registration."""
    from location.report import run_location_report
    import nibabel as nib

    shape = (32, 32, 32)
    t1 = np.zeros(shape, dtype=np.float32)
    t1[5:27, 5:27, 5:27] = 1.0
    affine = np.eye(4)
    t1_path = tmp_path / "T1.nii.gz"
    nib.save(nib.Nifti1Image(t1, affine), str(t1_path))
    (tmp_path / "report.html").write_text(
        "<html><body><h1>x</h1></body></html>", encoding="utf-8"
    )
    wt = np.zeros(shape, dtype=bool)
    wt[10:12, 10:12, 10:12] = True
    out = run_location_report(
        out_dir=tmp_path,
        t1_path=t1_path,
        regions={"whole_tumor": wt},
        case_id="gate_case",
        affine=affine,
        enabled=True,
    )
    assert out is not None
    assert out["plausibility_gated"] is True
    assert "no tumor detected" not in out["location_statement"].lower()
    assert "BraTS size floor" in out["location_statement"]
    loc = json.loads((tmp_path / "location.json").read_text(encoding="utf-8"))
    assert loc["plausibility_gated"] is True
    assert "no tumor detected" not in loc["location_statement"].lower()
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Location statement withheld" in html
    assert "no tumor detected" not in html.lower()
    assert not (tmp_path / "registration").exists()


def test_hemisphere_right_sided_blob():
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float64)
    affine[0, 3] = -20
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[25:35, 10:30, 10:30] = True
    hemi = hemisphere_fractions(mask, affine)
    assert hemi["label"] in {"right", "bilateral"}
    assert hemi["right_pct"] >= hemi["left_pct"]


def test_plain_language_summary_and_unreliable_registration():
    summary = plain_language_summary(
        hemisphere={"label": "right", "left_pct": 5.0, "right_pct": 95.0},
        lobes={"dominant_lobe": "parietal", "dominant_pct": 90.0},
        distances={
            "ventricles_mm": 1.0,
            "corticospinal_tract_mm": float("nan"),
            "optic_radiation_mm": float("nan"),
            "midline_mm": 12.0,
        },
        registration_quality="good",
    )
    assert "right" in summary
    assert "parietal" in summary
    assert "ventricle" in summary.lower()

    bad = plain_language_summary(
        hemisphere={"label": "right"},
        lobes={"dominant_lobe": "parietal", "dominant_pct": 90.0},
        distances={},
        registration_quality="registration unreliable",
    )
    assert "unreliable" in bad.lower() or "not claimed" in bad.lower()


def test_region_location_block_top_structures():
    shape = (30, 30, 30)
    mask = np.zeros(shape, dtype=bool)
    mask[10:20, 10:20, 10:20] = True
    cortical = np.zeros(shape, dtype=np.int16)
    cortical[10:20, 10:20, 10:20] = 18
    subcortical = np.zeros(shape, dtype=np.int16)
    subcortical[10:15, 10:15, 10:15] = 3
    labels_c = {0: "Background", 18: "Superior Parietal Lobule"}
    labels_s = {0: "Background", 3: "Left Lateral Ventricle"}
    affine = np.eye(4)
    block = region_location_block(
        mask,
        affine=affine,
        cortical=cortical,
        cortical_labels=labels_c,
        subcortical=subcortical,
        subcortical_labels=labels_s,
        ventricle_mask=subcortical == 3,
        cst_mask=None,
        optic_mask=None,
        spacing_mm=(1.0, 1.0, 1.0),
        registration_quality="good",
    )
    assert block["dominant_lobe"] == "parietal"
    assert block["distances_mm"]["ventricles_mm"] == 0.0
    assert "lobes_nearest" in block
    assert "lobes_cortical_only" in block
    assert "summary" in block


def test_atlas_licenses_documented():
    assert "Harvard-Oxford" in ATLAS_LICENSES
    assert "MNI152 / ICBM152" in ATLAS_LICENSES
    assert any("JHU" in k for k in ATLAS_LICENSES)


def test_run_location_report_disabled_returns_none(tmp_path: Path):
    from location.report import run_location_report

    out = run_location_report(
        out_dir=tmp_path,
        t1_path=tmp_path / "missing.nii.gz",
        regions={},
        case_id="x",
        enabled=False,
    )
    assert out is None


def test_inject_location_into_report(tmp_path: Path):
    from location.report import inject_location_into_report

    html = tmp_path / "report.html"
    html.write_text("<html><body><h1>Report</h1></body></html>", encoding="utf-8")
    payload = {
        "registration": {
            "mi_before": -0.5,
            "mi_after": -0.8,
            "brain_dice": 0.91,
            "quality": "good",
            "quality_note": "ok",
        },
        "jhu": "JHU atlas not present — CST / optic-radiation distances skipped.",
        "regions": {
            "whole_tumor": {
                "summary": "right parietal, 90%, abuts lateral ventricle",
                "hemisphere": {"label": "right", "left_pct": 5.0, "right_pct": 95.0},
                "lobes": {"parietal": {"pct": 90.0, "ml": 12.0}},
                "lobes_nearest": {"parietal": {"pct": 90.0, "ml": 12.0}},
                "lobes_cortical_only": {"parietal": {"pct": 100.0, "ml": 8.0}},
                "top_structures": [
                    {"name": "Superior Parietal Lobule", "ml": 10.0, "pct": 80.0}
                ],
                "distances_mm": {
                    "ventricles_mm": 1.0,
                    "corticospinal_tract_mm": float("nan"),
                    "optic_radiation_mm": float("nan"),
                    "midline_mm": 10.0,
                },
            }
        },
        "atlas_licenses": {"Harvard-Oxford": "FSL / nilearn"},
    }
    inject_location_into_report(html, payload)
    text = html.read_text(encoding="utf-8")
    assert "Location" in text
    assert "right parietal" in text
    assert "Nearest-lobe" in text


def test_setup_jhu_atlas_instructions_when_missing(tmp_path: Path, monkeypatch):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "setup_jhu_atlas.py"
    spec = importlib.util.spec_from_file_location("setup_jhu_atlas", path)
    assert spec and spec.loader
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    monkeypatch.delenv("FSLDIR", raising=False)
    monkeypatch.setattr(setup, "DEFAULT_DEST", tmp_path / "jhu.nii.gz")
    rc = setup.main([])
    assert rc == 1
