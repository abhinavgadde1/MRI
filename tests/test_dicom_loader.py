"""Unit tests for DICOM modality identification and sidecar naming."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from preprocessing.dicom_loader import (
    ConvertedSeries,
    SeriesMeta,
    _sidecar_path_for,
    group_by_series,
    identify_modality,
)
from preprocessing.registration import (
    _select_primary_series,
    _series_preference_key,
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


# ---------------------------------------------------------------------------
# Batch2 inventory strings (outputs/batch2_inventory/series_inventory.csv)
# ---------------------------------------------------------------------------

# All 10 clinical *_GD post-contrast T1 names seen in batch2.
_BATCH2_GD_T1C = [
    "T1W_3D_TFE_GD",
    "3D_T1W_TFE_GD",
    "V3D_T1W_TFE_GD LT",
    "V3D_T1W_TFE_GD COR",
    "V3D_T1W_TFE_GD RT",
]


@pytest.mark.parametrize("desc", _BATCH2_GD_T1C)
def test_batch2_gd_is_t1c(desc: str):
    """Underscore before GD must not prevent T1c (old \\bgd\\b failed here)."""
    assert identify_modality(desc, desc) == "T1c"


@pytest.mark.parametrize(
    "desc",
    [
        "T1W_FFE",
        "T1W_SE",
        "eT1W_SE",
        "eT1W_IR_2mm CLEAR",
        "3D Ax BRAVO",
        "Ax T1 FSE",
    ],
)
def test_batch2_precontrast_t1(desc: str):
    assert identify_modality(desc, desc) == "T1"


@pytest.mark.parametrize(
    "desc",
    [
        "s3DI_MC",
        "Vs3DI_MC",
        "MIP - s3DI_MC_3MIN",
        "MIP - s3DI_MC_F",
        "MIP - s3DI_MC_HR",
        "Vs3DI_MC_F",
        "Vs3DI_MC_HR",
    ],
)
def test_angio_s3di_not_t1c(desc: str):
    """Former false-positive T1c (bare trailing C / 3DI angio)."""
    assert identify_modality(desc, desc) == "other"


@pytest.mark.parametrize(
    "desc,expected",
    [
        ("esT2W_FLAIR", "FLAIR"),
        ("sT2W_FLAIR", "FLAIR"),
        ("AX T2 FLAIR", "FLAIR"),
        ("eT2W_TSE SENSE", "T2"),
        ("T2W_TSE", "T2"),
        ("Ax T2 FSE FS", "T2"),
        ("SWIp_fast", "other"),
        ("DWI", "other"),
        ("Reg - DWI", "other"),
    ],
)
def test_batch2_other_modalities(desc: str, expected: str):
    assert identify_modality(desc, desc) == expected


@pytest.mark.parametrize(
    "desc",
    [
        "T1+C",
        "T1_C",
        "T1W_Gad",
        "T1W post",
        "T1W_POST",
    ],
)
def test_t1c_alternate_contrast_spellings(desc: str):
    assert identify_modality(desc, None) == "T1c"


# ---------------------------------------------------------------------------
# Earlier 99-study cohort strings (anonymizer inventory + hemorrhage extras)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "desc,expected",
    [
        ("eT2W_FLAIR SPIR CLEAR", "FLAIR"),
        ("eT2W_TSE SENSE", "T2"),
        ("T1W_FFE", "T1"),
        ("T1W_SE", "T1"),
        ("eT1W_SE", "T1"),
        ("T1W_FFE CE", "T1c"),
        ("T1W_SPIR CE", "T1c"),
        ("T1W_SPIR_CE", "T1c"),
        ("T1W_CE_MT", "T1c"),
        ("eT1W_SPIR CE", "T1c"),
        ("T1W_ SPIR CE", "T1c"),
        ("MIP - VEN_3D_PCA", "other"),
        ("VEN_3D_PCA", "other"),
        ("SWIp_fast", "other"),
        ("SURVEY", "other"),
        ("T1W_FLAIR_MV", "FLAIR"),  # FLAIR wins over T1 token
    ],
)
def test_earlier_cohort_regressions(desc: str, expected: str):
    assert identify_modality(desc, desc) == expected


def _fake_converted(
    label: str,
    *,
    n: int,
    thickness: float | None,
    desc: str,
    series_number: int = 1,
) -> ConvertedSeries:
    meta = SeriesMeta(
        series_instance_uid=f"uid-{desc}-{n}",
        series_number=series_number,
        series_description=desc,
        protocol_name=desc,
        modality_dicom="MR",
        modality_label=label,  # type: ignore[arg-type]
        pixel_spacing=[1.0, 1.0],
        slice_thickness=thickness,
        spacing_between_slices=None,
        image_orientation_patient=None,
        image_position_patient=None,
        rows=256,
        columns=256,
        number_of_instances=n,
    )
    return ConvertedSeries(
        nifti_path=Path(f"/tmp/{label}/{desc}.nii.gz"),
        sidecar_path=Path(f"/tmp/{label}/{desc}.json"),
        meta=meta,
    )


def test_select_primary_prefers_more_then_thinner_slices():
    """When two T1c candidates compete, more slices win; thickness breaks ties."""
    thick_many = _fake_converted(
        "T1c", n=250, thickness=1.2, desc="T1W_3D_TFE_GD", series_number=10
    )
    thin_few = _fake_converted(
        "T1c", n=110, thickness=1.0, desc="3D_T1W_TFE_GD", series_number=11
    )
    pre_t1 = _fake_converted("T1", n=25, thickness=5.0, desc="T1W_FFE", series_number=5)
    paths, meta = _select_primary_series([thick_many, thin_few, pre_t1])
    assert set(paths) == {"T1", "T1c"}
    assert paths["T1c"] == thick_many.nifti_path  # 250 > 110
    assert "more slices" in meta["T1c"]["reason"]
    assert meta["T1"]["series_description"] == "T1W_FFE"


def test_select_primary_thickness_tiebreak():
    a = _fake_converted("T2", n=25, thickness=5.0, desc="T2_thick", series_number=1)
    b = _fake_converted("T2", n=25, thickness=3.0, desc="T2_thin", series_number=2)
    paths, meta = _select_primary_series([a, b])
    assert paths["T2"] == b.nifti_path
    assert meta["T2"]["slice_thickness_mm"] == 3.0


def test_series_preference_key_ordering():
    thick = _fake_converted("T1", n=100, thickness=5.0, desc="a")
    thin = _fake_converted("T1", n=100, thickness=1.0, desc="b")
    more = _fake_converted("T1", n=200, thickness=5.0, desc="c")
    assert _series_preference_key(more) > _series_preference_key(thin)
    assert _series_preference_key(thin) > _series_preference_key(thick)
