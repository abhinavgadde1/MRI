"""FastAPI web service wrapping ``pipeline.run_case`` (same path as CLI).

Job state is an in-memory dict — fine for single-user local use; it resets
when the process restarts (limitation, not a bug).
"""

from __future__ import annotations

# Background workers cannot use macOS GUI matplotlib backends.
import matplotlib

matplotlib.use("Agg")

import logging
import shutil
import threading
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import PROJECT_ROOT
from pipeline.anatomy_optin import try_add_anatomy_mesh
from pipeline.run_case import DEFAULT_CHECKPOINT, run_case
from preprocessing.registration import REGISTERED_NAMES
from visualization.plotly_3d import MESH_VIEWS, normalize_mesh_view

logger = logging.getLogger(__name__)

APP_ROOT = PROJECT_ROOT
JOBS_ROOT = APP_ROOT / "outputs" / "api_jobs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

JobStatus = Literal["queued", "running", "done", "error"]

# In-memory job tracker (resets on process restart).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

ALLOWED_UPLOAD_SUFFIXES = {".zip", ".nii", ".gz"}  # .nii.gz handled via name
ALLOWED_POSTPROCESS = {"raw", "lcc", "lcc_min"}
ALLOWED_MESH_VIEWS = set(MESH_VIEWS)

app = FastAPI(
    title="MRI pipeline API",
    description=(
        "Upload a DICOM/NIfTI study and run the same end-to-end pipeline as "
        "`python -m pipeline.cli run`. Job state is in-memory (lost on restart)."
    ),
    version="0.1.0",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_paths(job_id: str) -> tuple[Path, Path]:
    base = JOBS_ROOT / job_id
    return base / "input", base / "out"


def _get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return job


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(fields)
        _jobs[job_id]["updated_at"] = _utcnow()


def _is_nifti_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _looks_like_zip(filename: str, content_type: str | None) -> bool:
    lower = (filename or "").lower()
    if lower.endswith(".zip"):
        return True
    ct = (content_type or "").lower()
    # Avoid matching "gzip" (contains the substring "zip").
    return ct in {
        "application/zip",
        "application/x-zip-compressed",
        "multipart/x-zip",
    }


def _safe_extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip, rejecting path-traversal entries."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                name = info.filename
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"Unsafe zip entry: {name}")
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise ValueError("Corrupt or invalid ZIP file.") from exc


def _format_job_error(exc: BaseException) -> tuple[str, str]:
    """Return (human_message, full_traceback). Unwrap MONAI transform wrappers."""
    tb = traceback.format_exc()
    root: BaseException = exc
    while root.__cause__ is not None:
        root = root.__cause__
    # Prefer the deepest cause when MONAI wraps FileNotFoundError etc.
    if root is not exc and str(root):
        msg = f"{type(root).__name__}: {root}"
    else:
        msg = f"{type(exc).__name__}: {exc}"
    # Soften the opaque MONAI wrapper string when we have a better cause
    if "applying transform" in str(exc) and root is not exc:
        msg = (
            f"{type(root).__name__}: {root} "
            f"(underlying cause of MONAI LoadImaged / transform failure)"
        )
    return msg, tb


def _iter_content_children(root: Path) -> list[Path]:
    """Children of ``root``, ignoring macOS zip junk and hidden files."""
    return [
        p
        for p in root.iterdir()
        if p.name not in {"__MACOSX", ".DS_Store"} and not p.name.startswith("._")
    ]


def _unwrap_single_subdir(root: Path) -> Path:
    """If root contains exactly one real subdirectory (and no files), return that child."""
    entries = _iter_content_children(root)
    files = [p for p in entries if p.is_file()]
    dirs = [p for p in entries if p.is_dir()]
    if not files and len(dirs) == 1:
        return dirs[0]
    return root


def _find_pipeline_input(staged: Path) -> Path:
    """Locate a path suitable for ``run_case(input_dir=...)``."""
    staged = _unwrap_single_subdir(Path(staged))

    # Already-registered 4-modality folder (or nested 04_registered_1mm)
    cand = staged / "04_registered_1mm"
    if cand.is_dir() and all((cand / n).is_file() for n in REGISTERED_NAMES.values()):
        return cand
    if all((staged / n).is_file() for n in REGISTERED_NAMES.values()):
        return staged

    # Prefer a nested study folder that actually holds DICOMs (zip often has
    # study_name/ + __MACOSX/; unwrapping above handles the common case).
    dcm_here = list(staged.glob("*.dcm")) + list(staged.glob("**/*.dcm"))
    if dcm_here:
        # If DICOMs live only under one child study dir, point there
        child_dirs = [p for p in _iter_content_children(staged) if p.is_dir()]
        if child_dirs:
            with_dcm = [d for d in child_dirs if any(d.rglob("*.dcm"))]
            if len(with_dcm) == 1:
                return with_dcm[0]
        return staged

    # Any NIfTIs present → pass the folder (run_case will preprocess or error)
    niftis = list(staged.rglob("*.nii")) + list(staged.rglob("*.nii.gz"))
    # Ignore AppleDouble / junk under __MACOSX
    niftis = [p for p in niftis if "__MACOSX" not in p.parts]
    if niftis:
        return niftis[0].parent

    files = [p for p in staged.rglob("*") if p.is_file() and "__MACOSX" not in p.parts]
    if len(files) >= 10:
        return staged

    raise ValueError(
        "Upload did not contain a usable DICOM series or NIfTI set. "
        "Expected a ZIP of DICOM files, a ZIP/folder with the four registered "
        f"modalities ({', '.join(REGISTERED_NAMES.values())}), or .nii/.nii.gz files."
    )


def stage_upload(upload: UploadFile, input_dir: Path) -> Path:
    """Save upload under ``input_dir`` and return the path for ``run_case``."""
    input_dir.mkdir(parents=True, exist_ok=True)
    filename = upload.filename or "upload.bin"
    suffix = Path(filename).suffix.lower()

    if not filename or filename in (".", ".."):
        raise ValueError("Missing upload filename.")

    # Single NIfTI
    if _is_nifti_name(filename) and not _looks_like_zip(filename, upload.content_type):
        dest = input_dir / Path(filename).name
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        # One modality alone is not enough for the BraTS 4-channel model
        raise ValueError(
            "A single NIfTI is not enough — the pipeline needs four modalities "
            f"({', '.join(REGISTERED_NAMES.values())}) or a DICOM study ZIP. "
            "Use scripts/make_synthetic_case.py to build a valid smoke ZIP."
        )

    if _looks_like_zip(filename, upload.content_type) or suffix == ".zip":
        zip_path = input_dir / "upload.zip"
        with zip_path.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        extract_dir = input_dir / "extracted"
        _safe_extract_zip(zip_path, extract_dir)
        return _find_pipeline_input(extract_dir)

    raise ValueError(
        f"Unsupported file type: {filename!r}. "
        "Upload a .zip (DICOM or NIfTI set) or use the synthetic case ZIP."
    )


def _run_job(
    job_id: str,
    input_path: Path,
    out_dir: Path,
    checkpoint: Path,
    postprocess: str,
    want_anatomy: bool = False,
) -> None:
    _update_job(job_id, status="running")
    allow_random = not checkpoint.is_file()
    try:
        report = run_case(
            input_dir=input_path,
            output_dir=out_dir,
            checkpoint=checkpoint,
            postprocess=postprocess,
            allow_random_weights=allow_random,
            synthetic=False,
        )
        anatomy_status = "skipped"
        anatomy_reason = None
        if want_anatomy:
            _update_job(job_id, anatomy_status="running")
            anatomy_status, anatomy_reason = try_add_anatomy_mesh(out_dir)
            if anatomy_status != "ok":
                logger.warning(
                    "Job %s anatomy unavailable: %s", job_id, anatomy_reason
                )
        _update_job(
            job_id,
            status="done",
            report_path=str(report),
            error=None,
            anatomy_status=anatomy_status,
            anatomy_reason=anatomy_reason,
        )
    except Exception as exc:  # noqa: BLE001 — surface as job error for UI
        logger.exception("Job %s failed", job_id)
        msg, tb = _format_job_error(exc)
        _update_job(job_id, status="error", error=msg, error_traceback=tb)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="UI index.html missing")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP of DICOM/NIfTI or synthetic case"),
    checkpoint: str | None = Query(
        None, description="Checkpoint path (default: v3 brats_scale_full)"
    ),
    postprocess: str = Query("lcc", description="raw | lcc | lcc_min"),
    anatomy: bool = Query(
        False,
        description=(
            "Opt-in SynthSeg anatomical overlay after the normal pipeline "
            "(~2–3 min extra, native-resolution patches; requires .venv-synthseg). "
            "Failures fall back to the tumor-only mesh."
        ),
    ),
    view: str = Query(
        "full",
        description=(
            "Anatomy mesh view: 'full' (translucent whole brain, default) or "
            "'cutaway' (opaque sagittal half). Both HTML files are written when "
            "anatomy succeeds; this selects the default iframe."
        ),
    ),
) -> JSONResponse:
    if postprocess not in ALLOWED_POSTPROCESS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid postprocess={postprocess!r}; "
                f"choose one of {sorted(ALLOWED_POSTPROCESS)}"
            },
        )
    view_norm = (view or "full").strip().lower()
    if view_norm not in ALLOWED_MESH_VIEWS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid view={view!r}; "
                f"choose one of {sorted(ALLOWED_MESH_VIEWS)}"
            },
        )

    ckpt = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if not ckpt.is_absolute():
        ckpt = (APP_ROOT / ckpt).resolve()

    job_id = uuid.uuid4().hex[:12]
    input_dir, out_dir = _job_paths(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline_input = stage_upload(file, input_dir)
    except ValueError as exc:
        # Clean partial staging
        shutil.rmtree(JOBS_ROOT / job_id, ignore_errors=True)
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(JOBS_ROOT / job_id, ignore_errors=True)
        logger.exception("Staging failed")
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not stage upload: {exc}"},
        )

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "error": None,
            "error_traceback": None,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "checkpoint": str(ckpt),
            "postprocess": postprocess,
            "input_path": str(pipeline_input),
            "out_dir": str(out_dir),
            "report_path": None,
            "anatomy_requested": bool(anatomy),
            "anatomy_status": "queued" if anatomy else "skipped",
            "anatomy_reason": None,
            "mesh_view": view_norm,
        }

    background_tasks.add_task(
        _run_job, job_id, pipeline_input, out_dir, ckpt, postprocess, bool(anatomy)
    )
    return JSONResponse(
        {
            "job_id": job_id,
            "anatomy_requested": bool(anatomy),
            "mesh_view": view_norm,
        }
    )


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    payload = {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "error_traceback": job.get("error_traceback"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "checkpoint": job.get("checkpoint"),
        "postprocess": job.get("postprocess"),
        "anatomy_requested": bool(job.get("anatomy_requested")),
        "anatomy_status": job.get("anatomy_status") or "skipped",
        "anatomy_reason": job.get("anatomy_reason"),
        "mesh_view": job.get("mesh_view") or "full",
    }
    if job["status"] == "done":
        meta_path = Path(job["out_dir"]) / "meta.json"
        if meta_path.is_file():
            import json

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            payload["t1c_is_standin"] = bool(meta.get("t1c_is_standin"))
            payload["et_hidden"] = bool(meta.get("et_hidden"))
            payload["volume_plausibility"] = meta.get("volume_plausibility")
            payload["volume_plausibility_message"] = meta.get(
                "volume_plausibility_message"
            )
            if meta.get("anatomy_structures") is not None:
                payload["anatomy_structures"] = meta.get("anatomy_structures")
            if meta.get("anatomy_laterality") is not None:
                payload["anatomy_laterality"] = meta.get("anatomy_laterality")
            if meta.get("anatomy_infer_seconds") is not None:
                payload["anatomy_infer_seconds"] = meta.get("anatomy_infer_seconds")
            if meta.get("anatomy_views") is not None:
                payload["anatomy_views"] = meta.get("anatomy_views")
            payload["has_cutaway_mesh"] = (
                Path(job["out_dir"]) / "mesh_3d_cutaway.html"
            ).is_file()
    return payload


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")
    path = Path(job["out_dir"]) / "report.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report.html not found for job")
    return FileResponse(path, media_type="text/html")


@app.get("/api/jobs/{job_id}/mesh")
def job_mesh(
    job_id: str,
    view: str = Query(
        "full",
        description="Mesh HTML: 'full' (default) or 'cutaway' sagittal half.",
    ),
) -> FileResponse:
    job = _get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")
    view_norm = normalize_mesh_view(view)
    if (view or "full").strip().lower() not in ALLOWED_MESH_VIEWS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid view={view!r}; choose one of {sorted(ALLOWED_MESH_VIEWS)}",
        )
    out = Path(job["out_dir"])
    path = out / "mesh_3d.html"
    if view_norm == "cutaway":
        cut = out / "mesh_3d_cutaway.html"
        if cut.is_file():
            path = cut
    if not path.is_file():
        raise HTTPException(status_code=404, detail="mesh_3d.html not found for job")
    return FileResponse(path, media_type="text/html")


@app.get("/api/jobs/{job_id}/volumes")
def job_volumes(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")
    out = Path(job["out_dir"])
    morph_path = out / "morphometrics.csv"
    ell_path = out / "ellipsoid_wt.json"
    meta_path = out / "meta.json"
    if not morph_path.is_file():
        raise HTTPException(status_code=404, detail="morphometrics.csv missing")

    import json

    import pandas as pd

    morph = pd.read_csv(morph_path)
    regions = morph.to_dict(orient="records")
    ellipsoid = {}
    if ell_path.is_file():
        ellipsoid = json.loads(ell_path.read_text(encoding="utf-8"))
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "job_id": job_id,
        "regions": regions,
        "ellipsoid_wt": ellipsoid,
        "meta": meta,
        "volume_plausibility": meta.get("volume_plausibility"),
        "volume_plausibility_message": meta.get("volume_plausibility_message"),
    }


# Serve static assets (CSS/JS if added later)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
