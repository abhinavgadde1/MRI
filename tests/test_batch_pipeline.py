"""Tests for batch preprocessing discovery, BraTS H5 conversion, and CSV summary."""

from pathlib import Path
from unittest.mock import patch

import SimpleITK as sitk

from preprocessing.batch_pipeline import (
    StudySummary,
    convert_brats_h5_volume,
    discover_brats_subjects,
    discover_patient_studies,
    run_batch_patients,
    write_summary_csv,
)


def test_discover_patient_studies():
    studies = discover_patient_studies("MRI DATA")
    assert len(studies) == 100
    assert all(p.is_dir() for p in studies)


def test_discover_brats_h5_subjects():
    subjects = discover_brats_subjects("archive/BraTS2020_training_data/content/data")
    assert len(subjects) >= 369
    assert any(s["kind"] == "h5" for s in subjects)


def test_convert_brats_h5_volume(tmp_path: Path):
    paths = convert_brats_h5_volume(
        "archive/BraTS2020_training_data/content/data",
        volume_id=100,
        output_dir=tmp_path / "nifti",
    )
    assert set(paths) == {"FLAIR", "T1", "T1c", "T2"}
    img = sitk.ReadImage(str(paths["T1"]))
    assert img.GetSize()[0] == 240 and img.GetSize()[1] == 240
    assert img.GetSize()[2] > 1


def test_write_summary_csv(tmp_path: Path):
    rows = [
        StudySummary(
            dataset="real_patients",
            study_id="a",
            status="success",
            modalities_found="T1,T2",
            modalities_processed="T1,T2",
        ),
        StudySummary(
            dataset="real_patients",
            study_id="b",
            status="failed",
            modalities_found="FLAIR",
            error="RuntimeError: boom",
        ),
    ]
    csv_path = write_summary_csv(rows, tmp_path / "summary.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "success" in text and "failed" in text and "modalities_found" in text


def test_batch_patients_skips_failures(tmp_path: Path):
    studies = discover_patient_studies("MRI DATA")[:2]

    def _fake_run(study_dir, output_dir, **kwargs):
        if study_dir.name == studies[0].name:
            raise RuntimeError("simulated failure")
        from preprocessing.registration import PatientPreprocessResult

        result = PatientPreprocessResult(
            study_name=study_dir.name,
            output_dir=Path(output_dir),
            registered_1mm={"T2": Path(output_dir) / "T2.nii.gz"},
        )
        return result

    with patch(
        "preprocessing.batch_pipeline.run_patient_preprocessing", side_effect=_fake_run
    ):
        with patch(
            "preprocessing.batch_pipeline.discover_patient_studies", return_value=studies
        ):
            with patch(
                "preprocessing.batch_pipeline.probe_patient_modalities",
                return_value=["T2", "FLAIR"],
            ):
                batch = run_batch_patients(
                    "MRI DATA",
                    tmp_path / "out",
                    summary_csv=tmp_path / "sum.csv",
                )

    assert batch.n_failed == 1
    assert batch.n_success == 1
    assert batch.summary_csv is not None and batch.summary_csv.is_file()
