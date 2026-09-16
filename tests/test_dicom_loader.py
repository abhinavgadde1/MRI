"""Unit tests for DICOM modality identification and sidecar naming."""

from pathlib import Path

from preprocessing.dicom_loader import _sidecar_path_for, identify_modality


def test_identify_modality_rules():
    assert identify_modality("eT2W_FLAIR SPIR CLEAR", "eT2W_FLAIR SPIR CLEAR") == "FLAIR"
    assert identify_modality("T2W_TSE", "T2W_TSE") == "T2"
    assert identify_modality("eT1W_SE", "eT1W_SE") == "T1"
    assert identify_modality("T1W GD", "T1W POST CONTRAST") == "T1c"
    assert identify_modality("MIP - VEN_3D_PCA", "MIP - VEN_3D_PCA") == "other"


def test_sidecar_path_for_nii_gz():
    nifti = Path("/tmp/out/T1/ser001_eT1W_SE.nii.gz")
    assert _sidecar_path_for(nifti) == Path("/tmp/out/T1/ser001_eT1W_SE.json")
