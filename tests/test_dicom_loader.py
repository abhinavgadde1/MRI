"""Unit tests for DICOM modality identification and sidecar naming."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from preprocessing.dicom_loader import (
    _sidecar_path_for,
    group_by_series,
    identify_modality,
)


def test_identify_modality_rules():
    assert identify_modality("eT2W_FLAIR SPIR CLEAR", "eT2W_FLAIR SPIR CLEAR") == "FLAIR"
    assert identify_modality("T2W_TSE", "T2W_TSE") == "T2"
    assert identify_modality("eT1W_SE", "eT1W_SE") == "T1"
    assert identify_modality("T1W GD", "T1W POST CONTRAST") == "T1c"
    assert identify_modality("MIP - VEN_3D_PCA", "MIP - VEN_3D_PCA") == "other"


def test_t1c_before_t1():
    assert identify_modality("T1 CE", None) == "T1c"
    assert identify_modality("MPRAGE post", "post contrast") == "T1c"


def test_sidecar_path_for_nii_gz():
    nifti = Path("/tmp/out/T1/ser001_eT1W_SE.nii.gz")
    assert _sidecar_path_for(nifti) == Path("/tmp/out/T1/ser001_eT1W_SE.json")


def test_group_by_series_with_mocks():
    paths = [Path(f"/fake/a{i}.dcm") for i in range(3)]

    def _fake_read(path, **_kwargs):
        ds = MagicMock()
        # Two files share UID-A, one is UID-B
        name = Path(path).name
        ds.SeriesInstanceUID = "UID-A" if name in {"a0.dcm", "a1.dcm"} else "UID-B"
        return ds

    with patch("preprocessing.dicom_loader.pydicom.dcmread", side_effect=_fake_read):
        groups = group_by_series(paths)

    assert set(groups) == {"UID-A", "UID-B"}
    assert len(groups["UID-A"]) == 2
    assert len(groups["UID-B"]) == 1
