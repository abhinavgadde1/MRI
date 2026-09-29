"""MRI preprocessing: DICOM conversion, bias correction, skull stripping, registration."""

from preprocessing.bias_correction import correct_bias_field, n4_bias_correct
from preprocessing.batch_pipeline import (
    run_all_batches,
    run_batch_brats,
    run_batch_patients,
    write_summary_csv,
)
from preprocessing.dicom_loader import (
    ConvertedSeries,
    SeriesMeta,
    discover_dicom_files,
    group_by_series,
    identify_modality,
    load_study_to_nifti,
)
from preprocessing.dicom_to_nifti import convert_dicom_series
from preprocessing.h5_to_nifti import (
    MODALITY_NAMES,
    convert_brats_h5_directory,
    group_slices_by_patient,
)
from preprocessing.registration import (
    REGISTERED_NAMES,
    PatientPreprocessResult,
    register_modalities_to_reference,
    register_to_reference,
    resample_isotropic,
    rigid_register,
    run_nifti_preprocessing,
    run_patient_preprocessing,
)
from preprocessing.skull_strip import SkullStripResult, hd_bet_available, skull_strip

__all__ = [
    "MODALITY_NAMES",
    "REGISTERED_NAMES",
    "ConvertedSeries",
    "PatientPreprocessResult",
    "SeriesMeta",
    "SkullStripResult",
    "convert_brats_h5_directory",
    "convert_dicom_series",
    "correct_bias_field",
    "discover_dicom_files",
    "group_by_series",
    "group_slices_by_patient",
    "hd_bet_available",
    "identify_modality",
    "load_study_to_nifti",
    "n4_bias_correct",
    "register_modalities_to_reference",
    "register_to_reference",
    "resample_isotropic",
    "rigid_register",
    "run_all_batches",
    "run_batch_brats",
    "run_batch_patients",
    "run_nifti_preprocessing",
    "run_patient_preprocessing",
    "skull_strip",
    "write_summary_csv",
]
