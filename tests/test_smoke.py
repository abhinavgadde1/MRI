"""Smoke tests for preprocessing package imports."""

import preprocessing
from preprocessing import (
    REGISTERED_NAMES,
    convert_dicom_series,
    identify_modality,
    n4_bias_correct,
    run_patient_preprocessing,
    skull_strip,
)
from preprocessing.h5_to_nifti import MODALITY_NAMES, parse_slice_filename


def test_package_exports():
    assert callable(convert_dicom_series)
    assert callable(n4_bias_correct)
    assert callable(skull_strip)
    assert callable(run_patient_preprocessing)
    assert callable(identify_modality)
    assert "T1" in REGISTERED_NAMES
    assert MODALITY_NAMES[0] == "FLAIR"
    assert parse_slice_filename.__name__ == "parse_slice_filename"
    assert hasattr(preprocessing, "__all__")
