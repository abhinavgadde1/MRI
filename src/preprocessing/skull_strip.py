"""Skull stripping via HD-BET with a SimpleITK morphological fallback."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import SimpleITK as sitk

logger = logging.getLogger(__name__)

SkullStripMethod = Literal["hd-bet", "simpleitk_otsu"]


@dataclass(frozen=True)
class SkullStripResult:
    """Paths and backend used for a skull-stripping run."""

    stripped_path: Path
    mask_path: Path
    method: SkullStripMethod


def hd_bet_available() -> bool:
    """Return True if HD-BET can be imported or its CLI is on PATH."""
    try:
        import HD_BET  # noqa: F401

        return True
    except ImportError:
        pass
    return _hd_bet_cli() is not None


def hd_bet_status() -> dict[str, object]:
    """Diagnostic info for why HD-BET may be unavailable."""
    py_ok = False
    py_err: str | None = None
    py_ver: str | None = None
    try:
        import HD_BET
        import importlib.metadata as im

        py_ok = True
        try:
            py_ver = im.version("HD_BET")
        except Exception:  # noqa: BLE001
            py_ver = getattr(HD_BET, "__version__", None)
    except ImportError as exc:
        py_err = str(exc)
    cli = _hd_bet_cli()
    params = Path.home() / "hd-bet_params" / "release_2.0.0"
    return {
        "available": bool(py_ok or cli),
        "python_package": py_ok,
        "python_version": py_ver,
        "python_import_error": py_err,
        "cli_path": str(cli) if cli else None,
        "params_dir": str(params),
        "params_present": params.is_dir() and any(params.iterdir()),
        "install_hint": (
            "pip install hd-bet  # HD-BET>=2.0; downloads weights to "
            "~/hd-bet_params/release_2.0.0 on first run"
        ),
    }


def default_hd_bet_device() -> str:
    """Prefer CUDA, then Apple MPS, else CPU.

    Override with env ``MRI_HD_BET_DEVICE=cpu|mps|cuda`` (``cpu`` runs cooler/slower).
    """
    import os

    override = os.environ.get("MRI_HD_BET_DEVICE", "").strip().lower()
    if override in {"cpu", "mps", "cuda"}:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def default_hd_bet_mode() -> str:
    """``fast`` (no TTA) by default on Apple MPS to limit heat; ``accurate`` on CUDA.

    Override with env ``MRI_HD_BET_MODE=fast|accurate``.
    """
    import os

    override = os.environ.get("MRI_HD_BET_MODE", "").strip().lower()
    if override in {"fast", "accurate"}:
        return override
    # TTA on MPS is a major heat/source of fan spin for multi-modality studies.
    if default_hd_bet_device() == "mps":
        return "fast"
    return "accurate"


def skull_strip(
    input_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    *,
    prefer_hd_bet: bool = True,
    device: str | None = None,
    mode: str | None = None,
    closing_radius: int = 3,
    opening_radius: int = 2,
) -> SkullStripResult:
    """Skull-strip a brain MRI NIfTI volume.

    Prefers HD-BET when installed. Falls back to SimpleITK Otsu + morphology.
    Writes ``*_skullstrip_method.json`` next to the mask.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {input_path}")

    mask_path = (
        Path(mask_path) if mask_path is not None else _default_mask_path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    device = device or default_hd_bet_device()
    mode = mode or default_hd_bet_mode()

    if prefer_hd_bet and hd_bet_available():
        try:
            _run_hd_bet(
                input_path, output_path, mask_path, device=device, mode=mode
            )
            logger.info(
                "Skull stripping used HD-BET for %s -> %s (mask=%s, device=%s)",
                input_path,
                output_path,
                mask_path,
                device,
            )
            result = SkullStripResult(
                stripped_path=output_path, mask_path=mask_path, method="hd-bet"
            )
            _write_method_sidecar(mask_path, result.method)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HD-BET failed for %s (%s: %s); falling back to SimpleITK Otsu",
                input_path,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
    elif prefer_hd_bet:
        status = hd_bet_status()
        logger.warning(
            "HD-BET is not installed; using SimpleITK Otsu fallback for %s "
            "(python_package=%s, cli=%s). Install hint: %s",
            input_path,
            status["python_package"],
            status["cli_path"],
            status["install_hint"],
        )

    _run_simpleitk_otsu(
        input_path,
        output_path,
        mask_path,
        closing_radius=closing_radius,
        opening_radius=opening_radius,
    )
    logger.info(
        "Skull stripping used SimpleITK Otsu+morphology for %s -> %s (mask=%s)",
        input_path,
        output_path,
        mask_path,
    )
    result = SkullStripResult(
        stripped_path=output_path, mask_path=mask_path, method="simpleitk_otsu"
    )
    _write_method_sidecar(mask_path, result.method)
    return result


def read_strip_method(mask_or_strip_dir: str | Path) -> SkullStripMethod | None:
    """Read method sidecar next to a brainmask, or the first one in a strip dir."""
    path = Path(mask_or_strip_dir)
    if path.is_dir():
        sidecars = sorted(path.glob("*_skullstrip_method.json"))
        if not sidecars:
            return None
        path = sidecars[0]
    elif path.name.endswith("_brainmask.nii.gz"):
        path = path.with_name(
            path.name.replace("_brainmask.nii.gz", "_skullstrip_method.json")
        )
    elif not path.name.endswith("_skullstrip_method.json"):
        stem = path.name
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        path = path.with_name(f"{stem}_skullstrip_method.json")
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("method")
        if method in ("hd-bet", "simpleitk_otsu"):
            return method  # type: ignore[return-value]
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _hd_bet_cli() -> str | None:
    """Locate hd-bet, including beside the venv interpreter (not resolved Frameworks)."""
    found = shutil.which("hd-bet")
    if found:
        return found
    exe = Path(sys.executable)
    for base in (exe.parent, exe.resolve().parent):
        sibling = base / "hd-bet"
        if sibling.is_file():
            return str(sibling)
    return None


def _write_method_sidecar(mask_path: Path, method: SkullStripMethod) -> None:
    name = mask_path.name
    if name.endswith("_brainmask.nii.gz"):
        side = mask_path.with_name(
            name.replace("_brainmask.nii.gz", "_skullstrip_method.json")
        )
    else:
        side = mask_path.with_suffix("").with_suffix(".skullstrip_method.json")
    side.write_text(
        json.dumps({"method": method, "mask_path": str(mask_path)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _default_mask_path(output_path: Path) -> Path:
    name = output_path.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        stem = name[: -len(".nii")]
    else:
        stem = output_path.stem
    return output_path.with_name(f"{stem}_brainmask.nii.gz")


def _run_hd_bet(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    *,
    device: str,
    mode: str,
) -> None:
    """Run HD-BET via CLI (preferred), v2 API, or legacy v1 API."""
    use_tta = mode != "fast"
    errors: list[str] = []

    cli = _hd_bet_cli()
    if cli is not None:
        try:
            _run_hd_bet_cli(
                input_path, output_path, mask_path, device=device, use_tta=use_tta
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"CLI: {type(exc).__name__}: {exc}")
            logger.warning("HD-BET CLI failed (%s); trying Python API", exc)

    try:
        import HD_BET.hd_bet_prediction  # noqa: F401
    except ImportError:
        pass
    else:
        try:
            _run_hd_bet_v2_api(
                input_path, output_path, mask_path, device=device, use_tta=use_tta
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"v2 API: {type(exc).__name__}: {exc}")
            logger.warning("HD-BET v2 API failed (%s); trying legacy API", exc)

    try:
        from HD_BET.run import run_hd_bet
    except ImportError:
        run_hd_bet = None  # type: ignore[assignment]
    if run_hd_bet is not None:
        try:
            run_hd_bet(
                str(input_path),
                str(output_path),
                mode=mode,
                device=device,
                postprocess=True,
                do_tta=use_tta,
            )
        except TypeError:
            run_hd_bet(str(input_path), str(output_path))
        _ensure_mask_alongside(input_path, output_path, mask_path)
        return

    detail = "; ".join(errors) if errors else "no backend available"
    raise RuntimeError(f"HD-BET could not run ({detail})")


def _run_hd_bet_v2_api(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    *,
    device: str,
    use_tta: bool,
) -> None:
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

    maybe_download_parameters()
    if device == "cpu":
        use_tta = False
    predictor = get_hdbet_predictor(
        use_tta=use_tta, device=torch.device(device), verbose=False
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdbet_predict(
        str(input_path.resolve()),
        str(output_path.resolve()),
        predictor,
        keep_brain_mask=True,
        compute_brain_extracted_image=True,
    )
    if not output_path.is_file():
        raise FileNotFoundError(f"HD-BET produced no stripped volume at {output_path}")
    bet_mask = output_path.with_name(
        output_path.name[: -len(".nii.gz")] + "_bet.nii.gz"
        if output_path.name.endswith(".nii.gz")
        else output_path.stem + "_bet.nii.gz"
    )
    if bet_mask.is_file():
        shutil.copy2(bet_mask, mask_path)
        try:
            bet_mask.unlink()
        except OSError:
            pass
    else:
        _ensure_mask_alongside(input_path, output_path, mask_path)


def _run_hd_bet_cli(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    *,
    device: str,
    use_tta: bool,
) -> None:
    cli = _hd_bet_cli()
    if cli is None:
        raise RuntimeError("HD-BET CLI unavailable")

    with tempfile.TemporaryDirectory(prefix="hd_bet_") as tmp:
        tmp_out = Path(tmp) / output_path.name
        cmd = [
            cli,
            "-i",
            str(input_path),
            "-o",
            str(tmp_out),
            "-device",
            str(device),
            "--save_bet_mask",
        ]
        if not use_tta or device == "cpu":
            cmd.append("--disable_tta")
        subprocess.run(cmd, check=True)

        produced = tmp_out if tmp_out.is_file() else None
        if produced is None:
            candidates = sorted(Path(tmp).glob("*.nii*"))
            non_mask = [
                p
                for p in candidates
                if "_bet" not in p.name.lower() and "mask" not in p.name.lower()
            ]
            produced = (non_mask or candidates)[0] if candidates else None
        if produced is None:
            raise FileNotFoundError("HD-BET CLI produced no NIfTI output")

        shutil.copy2(produced, output_path)
        mask_candidates = sorted(Path(tmp).glob("*_bet.nii.gz")) + sorted(
            Path(tmp).glob("*mask*.nii*")
        )
        if mask_candidates:
            shutil.copy2(mask_candidates[0], mask_path)
        else:
            _mask_from_stripped(input_path, output_path, mask_path)


def _ensure_mask_alongside(
    input_path: Path, output_path: Path, mask_path: Path
) -> None:
    stem = output_path.name
    if stem.endswith(".nii.gz"):
        base = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        base = stem[: -len(".nii")]
    else:
        base = output_path.stem

    candidates = [
        output_path.with_name(f"{base}_bet.nii.gz"),
        output_path.with_name(f"{base}_mask.nii.gz"),
        output_path.with_name(f"{base}_mask.nii"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            if candidate.resolve() != mask_path.resolve():
                shutil.copy2(candidate, mask_path)
            return

    _mask_from_stripped(input_path, output_path, mask_path)


def _mask_from_stripped(
    input_path: Path, stripped_path: Path, mask_path: Path
) -> None:
    stripped = sitk.ReadImage(str(stripped_path))
    mask = sitk.Cast(stripped != 0, sitk.sitkUInt8)
    original = sitk.ReadImage(str(input_path))
    mask.CopyInformation(original)
    sitk.WriteImage(mask, str(mask_path))


def _run_simpleitk_otsu(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    *,
    closing_radius: int,
    opening_radius: int,
) -> None:
    """Otsu threshold + morphological clean-up + largest connected component."""
    original = sitk.ReadImage(str(input_path))
    image = sitk.Cast(original, sitk.sitkFloat32)
    image.CopyInformation(original)

    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    close_r = [max(1, int(closing_radius))] * image.GetDimension()
    open_r = [max(1, int(opening_radius))] * image.GetDimension()

    mask = sitk.BinaryMorphologicalClosing(mask, close_r)
    mask = sitk.BinaryMorphologicalOpening(mask, open_r)
    mask = sitk.BinaryFillhole(mask)

    components = sitk.ConnectedComponent(mask)
    sorted_labels = sitk.RelabelComponent(components, sortByObjectSize=True)
    mask = sitk.Cast(sorted_labels == 1, sitk.sitkUInt8)
    mask.CopyInformation(original)

    mask = sitk.BinaryMorphologicalClosing(mask, close_r)
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.BinaryDilate(mask, [1] * image.GetDimension())
    mask = sitk.BinaryFillhole(mask)
    mask.CopyInformation(original)

    stripped = sitk.Mask(original, mask)
    stripped.CopyInformation(original)

    sitk.WriteImage(stripped, str(output_path))
    sitk.WriteImage(mask, str(mask_path))
