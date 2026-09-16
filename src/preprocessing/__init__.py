"""MRI preprocessing: DICOM conversion, bias correction, skull stripping, registration."""

from preprocessing.dicom_to_nifti import convert_dicom_series
from preprocessing.dicom_loader import (
    ConvertedSeries,
    SeriesMeta,
    discover_dicom_files,
    group_by_series,
    identify_modality,
    load_study_to_nifti,
)
from preprocessing.bias_correction import correct_bias_field
from preprocessing.skull_strip import SkullStripResult, hd_bet_available, skull_strip
from preprocessing.registration import (
    PatientPreprocessResult,
    register_modalities_to_reference,
    register_to_reference,
    resample_isotropic,
    rigid_register,
    run_nifti_preprocessing,
    run_patient_preprocessing,
)
from preprocessing.batch_pipeline import (
    run_all_batches,
    run_batch_brats,
    run_batch_patients,
    write_summary_csv,
)
from preprocessing.h5_to_nifti import convert_brats_h5_directory, group_slices_by_patient

__all__ = [
    "convert_dicom_series",
    "ConvertedSeries",
    "SeriesMeta",
    "discover_dicom_files",
    "group_by_series",
    "identify_modality",
    "load_study_to_nifti",
    "correct_bias_field",
    "SkullStripResult",
    "hd_bet_available",
    "skull_strip",
    "PatientPreprocessResult",
    "register_to_reference",
    "rigid_register",
    "resample_isotropic",
    "register_modalities_to_reference",
    "run_nifti_preprocessing",
    "run_patient_preprocessing",
    "run_batch_patients",
    "run_batch_brats",
    "run_all_batches",
    "write_summary_csv",
    "convert_brats_h5_directory",
    "group_slices_by_patient",
]
