"""API tests: synthetic ZIP through /api/analyze (no BraTS / real MRI)."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import PROJECT_ROOT


@pytest.fixture(scope="module")
def synthetic_zip(tmp_path_factory) -> Path:
    """Build a zip of four registered NIfTIs via make_synthetic_case helpers."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_synthetic_case",
        PROJECT_ROOT / "scripts" / "make_synthetic_case.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    root = tmp_path_factory.mktemp("synth")
    case_dir = mod.write_synthetic_case(root / "case")
    zip_path = root / "case.zip"
    mod.zip_case(case_dir, zip_path)
    assert zip_path.is_file() and zip_path.stat().st_size > 0
    with zipfile.ZipFile(zip_path) as zf:
        assert len(zf.namelist()) == 4
    return zip_path


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Isolated jobs root; import app after patching JOBS_ROOT."""
    import api.app as api_app

    jobs = tmp_path / "api_jobs"
    jobs.mkdir()
    monkeypatch.setattr(api_app, "JOBS_ROOT", jobs)
    # Fresh job table per test
    with api_app._jobs_lock:
        api_app._jobs.clear()
    return TestClient(api_app.app)


def test_analyze_rejects_bad_extension(client: TestClient):
    res = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", b"not imaging", "text/plain")},
    )
    assert res.status_code == 400
    body = res.json()
    assert "error" in body
    assert "Unsupported" in body["error"] or "not" in body["error"].lower()


def test_analyze_rejects_corrupt_zip(client: TestClient):
    res = client.post(
        "/api/analyze",
        files={"file": ("bad.zip", b"this is not a zip", "application/zip")},
    )
    assert res.status_code == 400
    assert "error" in res.json()


def test_analyze_rejects_single_nifti(client: TestClient, synthetic_zip: Path):
    # Pull one nifti bytes from the synthetic zip
    with zipfile.ZipFile(synthetic_zip) as zf:
        name = zf.namelist()[0]
        data = zf.read(name)
    res = client.post(
        "/api/analyze",
        files={"file": (name, data, "application/gzip")},
    )
    assert res.status_code == 400
    assert "single NIfTI" in res.json()["error"] or "four modalities" in res.json()["error"]


def test_analyze_rejects_invalid_mesh_view(
    client: TestClient, synthetic_zip: Path, monkeypatch
):
    import api.app as api_app

    monkeypatch.setattr(api_app, "_run_job", lambda *args, **kwargs: None)
    with synthetic_zip.open("rb") as fh:
        res = client.post(
            "/api/analyze",
            params={"postprocess": "lcc", "anatomy": True, "view": "oblique"},
            files={"file": ("synthetic_case.zip", fh, "application/zip")},
        )
    assert res.status_code == 400
    assert "view" in res.json()["error"].lower()


def test_analyze_accepts_cutaway_view(
    client: TestClient, synthetic_zip: Path, monkeypatch
):
    import api.app as api_app

    monkeypatch.setattr(api_app, "_run_job", lambda *args, **kwargs: None)
    with synthetic_zip.open("rb") as fh:
        res = client.post(
            "/api/analyze",
            params={"postprocess": "lcc", "anatomy": True, "view": "cutaway"},
            files={"file": ("synthetic_case.zip", fh, "application/zip")},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body.get("mesh_view") == "cutaway"
    assert body.get("anatomy_requested") is True


def test_analyze_synthetic_end_to_end(client: TestClient, synthetic_zip: Path):
    """Upload synthetic case, poll until done, check report/mesh/volumes."""
    with synthetic_zip.open("rb") as fh:
        res = client.post(
            "/api/analyze",
            params={"postprocess": "lcc"},
            files={"file": ("synthetic_case.zip", fh, "application/zip")},
        )
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    assert job_id

    # TestClient runs BackgroundTasks after the request returns when using
    # the default transport — poll with a timeout for safety.
    deadline = time.time() + 180
    status = None
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}")
        assert st.status_code == 200
        status = st.json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.5)

    assert status == "done", client.get(f"/api/jobs/{job_id}").json()
    done_body = client.get(f"/api/jobs/{job_id}").json()
    assert done_body.get("anatomy_requested") is False
    assert done_body.get("anatomy_status") == "skipped"

    report = client.get(f"/api/jobs/{job_id}/report")
    assert report.status_code == 200
    assert "text/html" in report.headers.get("content-type", "")
    assert b"<html" in report.content.lower() or b"<!DOCTYPE" in report.content

    mesh = client.get(f"/api/jobs/{job_id}/mesh")
    assert mesh.status_code == 200
    assert b"Brain surface" in mesh.content or b"Whole Tumor" in mesh.content
    # No cutaway file on tumor-only jobs: fall back to the full mesh.
    cut = client.get(f"/api/jobs/{job_id}/mesh", params={"view": "cutaway"})
    assert cut.status_code == 200
    bad = client.get(f"/api/jobs/{job_id}/mesh", params={"view": "oblique"})
    assert bad.status_code == 400

    vols = client.get(f"/api/jobs/{job_id}/volumes")
    assert vols.status_code == 200
    payload = vols.json()
    assert "regions" in payload
    assert isinstance(payload["regions"], list)
    assert len(payload["regions"]) >= 1
    assert "ellipsoid_wt" in payload


def test_index_serves_ui(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Analyze" in res.content
    assert b"/api/analyze" in res.content
    assert b"standinBanner" in res.content
    assert b"T1c stand-in" in res.content
    assert b"plausibilityBanner" in res.content
    assert b"Show brain anatomy" in res.content
    assert b"~2" in res.content and b"3 min" in res.content
    assert b"anatomy=true" in res.content
    assert b"view=cutaway" in res.content or b"view=${preferredView}" in res.content
    assert b"Sagittal cutaway" in res.content
    assert b"Anatomical labels may be incomplete" in res.content
    assert b"ventricles, small nuclei" in res.content


def test_analyze_anatomy_true_synthetic_completes(client: TestClient, synthetic_zip: Path):
    """Opt-in anatomy must not fail the job. Synthetic data should typically
    fall back to tumor-only (anatomy_status=unavailable).
    """
    with synthetic_zip.open("rb") as fh:
        res = client.post(
            "/api/analyze",
            params={"postprocess": "lcc", "anatomy": True},
            files={"file": ("synthetic_case.zip", fh, "application/zip")},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    job_id = body["job_id"]
    assert body.get("anatomy_requested") is True

    deadline = time.time() + 420
    payload = None
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}")
        assert st.status_code == 200
        payload = st.json()
        if payload["status"] in ("done", "error"):
            break
        time.sleep(0.5)

    assert payload is not None
    assert payload["status"] == "done", payload
    assert payload.get("anatomy_requested") is True
    assert payload["anatomy_status"] in ("ok", "unavailable")

    mesh = client.get(f"/api/jobs/{job_id}/mesh")
    assert mesh.status_code == 200
    assert b"Whole Tumor" in mesh.content or b"whole_tumor" in mesh.content.lower()

    if payload["anatomy_status"] == "unavailable":
        assert payload.get("anatomy_reason")
        # Fallback: tumor-only HTML, no anatomy structure names required
        assert b"<html" in mesh.content.lower() or b"<!DOCTYPE" in mesh.content
    else:
        assert b"Left cerebral cortex" in mesh.content or b"Cerebellum" in mesh.content


def _poll_job(client: TestClient, job_id: str, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    payload = None
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}")
        assert st.status_code == 200
        payload = st.json()
        if payload["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert payload is not None
    assert payload["status"] == "done", payload
    return payload


def _patch_segment_wt_voxels(monkeypatch, wt_voxels: int) -> None:
    import numpy as np
    import pipeline.run_case as rc

    monkeypatch.setattr(rc, "_load_or_init_model", lambda *a, **k: object())

    def fake_seg(_model, _device, registered_dir, _case_id, postprocess="lcc"):
        import nibabel as nib
        from preprocessing.registration import REGISTERED_NAMES

        t1 = nib.load(str(Path(registered_dir) / REGISTERED_NAMES["T1"]))
        shape = t1.shape
        wt = np.zeros(shape, dtype=bool)
        wt.ravel()[: int(wt_voxels)] = True
        empty = np.zeros(shape, dtype=bool)
        return (
            {
                "enhancing_tumor": empty,
                "tumor_core": empty.copy(),
                "whole_tumor": wt,
            },
            np.zeros(shape, dtype=np.float32),
        )

    monkeypatch.setattr(rc, "_segment_registered", fake_seg)


def test_api_tiny_wt_flags_low_in_job_and_volumes(
    client: TestClient, synthetic_zip: Path, monkeypatch
):
    _patch_segment_wt_voxels(monkeypatch, 1_000)
    with synthetic_zip.open("rb") as fh:
        res = client.post(
            "/api/analyze",
            params={"postprocess": "lcc"},
            files={"file": ("synthetic_case.zip", fh, "application/zip")},
        )
    assert res.status_code == 200, res.text
    payload = _poll_job(client, res.json()["job_id"])
    assert payload.get("volume_plausibility") == "low"
    assert "segmentation artifact" in (payload.get("volume_plausibility_message") or "")

    report = client.get(f"/api/jobs/{payload['job_id']}/report")
    assert report.status_code == 200
    assert b"volume-plausibility-banner" in report.content
    assert b"segmentation artifact" in report.content

    vols = client.get(f"/api/jobs/{payload['job_id']}/volumes")
    assert vols.status_code == 200
    body = vols.json()
    assert body["volume_plausibility"] == "low"
    assert "below the smallest confirmed tumor" in body["volume_plausibility_message"]


def test_api_plausible_wt_has_no_note(
    client: TestClient, synthetic_zip: Path, monkeypatch
):
    _patch_segment_wt_voxels(monkeypatch, 50_000)
    with synthetic_zip.open("rb") as fh:
        res = client.post(
            "/api/analyze",
            params={"postprocess": "lcc"},
            files={"file": ("synthetic_case.zip", fh, "application/zip")},
        )
    assert res.status_code == 200, res.text
    payload = _poll_job(client, res.json()["job_id"])
    assert payload.get("volume_plausibility") == "plausible"
    assert payload.get("volume_plausibility_message") in (None, "")

    report = client.get(f"/api/jobs/{payload['job_id']}/report")
    assert report.status_code == 200
    assert b"volume-plausibility-banner" not in report.content
    assert b"segmentation artifact" not in report.content

    vols = client.get(f"/api/jobs/{payload['job_id']}/volumes")
    assert vols.status_code == 200
    body = vols.json()
    assert body["volume_plausibility"] == "plausible"
    assert body.get("volume_plausibility_message") in (None, "")
