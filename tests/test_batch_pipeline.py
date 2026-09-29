"""Tests for batch preprocessing discovery helpers."""

from pathlib import Path
from unittest.mock import patch

from preprocessing.batch_pipeline import (
    StudySummary,
    discover_brats_subjects,
    discover_patient_studies,
    run_batch_patients,
    write_summary_csv,
)


def test_discover_patient_studies():
    studies = discover_patient_studies("MRI DATA")
    assert len(studies) >= 1
    assert all(p.is_dir() for p in studies)
    assert any(p.name.endswith("_MR_Study1") or "Study" in p.name for p in studies)


def test_discover_brats_h5_subjects():
    subjects = discover_brats_subjects(
        "archive/BraTS2020_training_data/content/data"
    )
    assert len(subjects) >= 1
    assert any(s["kind"] == "h5" for s in subjects)
    assert all("study_id" in s and "path" in s for s in subjects)


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
    studies = [
        tmp_path / "study_ok",
        tmp_path / "study_fail",
    ]
    for s in studies:
        s.mkdir()

    def _fake_run(study_dir, output_dir, **kwargs):
        if study_dir.name == "study_fail":
            raise RuntimeError("simulated failure")
        from preprocessing.registration import PatientPreprocessResult

        return PatientPreprocessResult(
            study_name=study_dir.name,
            output_dir=Path(output_dir),
            registered_1mm={"T2": Path(output_dir) / "T2.nii.gz"},
        )

    with patch(
        "preprocessing.batch_pipeline.run_patient_preprocessing", side_effect=_fake_run
    ):
        with patch(
            "preprocessing.batch_pipeline.discover_patient_studies",
            return_value=studies,
        ):
            with patch(
                "preprocessing.batch_pipeline.probe_patient_modalities",
                return_value=["T2", "FLAIR"],
            ):
                batch = run_batch_patients(
                    tmp_path,
                    tmp_path / "out",
                    summary_csv=tmp_path / "sum.csv",
                )

    assert batch.n_failed == 1
    assert batch.n_success == 1
    assert batch.summary_csv is not None and batch.summary_csv.is_file()
