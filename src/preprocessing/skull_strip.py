"""Skull stripping via HD-BET with a SimpleITK morphological fallback."""

from __future__ import annotations

import logging
import shutil
import subprocess
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
    return shutil.which("hd-bet") is not None


def skull_strip(
    input_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    *,
    prefer_hd_bet: bool = True,
    device: str = "cpu",
    mode: str = "accurate",
    closing_radius: int = 2,
) -> SkullStripResult:
    """Skull-strip a brain MRI NIfTI volume.

    Prefers HD-BET when installed. If HD-BET is unavailable (or fails), falls
    back to SimpleITK Otsu thresholding plus morphological clean-up. The method
    used is logged for each call.

    Parameters
    ----------
    input_path:
        Input brain MRI NIfTI.
    output_path:
        Destination for the skull-stripped volume.
    mask_path:
        Optional brain-mask destination. Defaults to ``*_brainmask.nii.gz``
        next to ``output_path``.
    prefer_hd_bet:
        When True (default), try HD-BET first.
    device:
        HD-BET device string (``\"cpu\"`` or ``\"cuda\"`` / GPU index).
    mode:
        HD-BET mode (``\"accurate\"`` or ``\"fast\"``).
    closing_radius:
        Morphological closing radius (voxels) for the SimpleITK fallback.

    Returns
    -------
    SkullStripResult
        Stripped volume path, mask path, and method name.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {input_path}")

    mask_path = Path(mask_path) if mask_path is not None else _default_mask_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_hd_bet and hd_bet_available():
        try:
            _run_hd_bet(input_path, output_path, mask_path, device=device, mode=mode)
            logger.info(
                "Skull stripping used HD-BET for %s -> %s (mask=%s)",
                input_path,
                output_path,
                mask_path,
            )
            return SkullStripResult(
                stripped_path=output_path, mask_path=mask_path, method="hd-bet"
            )
        except Exception as exc:
            logger.warning(
                "HD-BET failed for %s (%s); falling back to SimpleITK Otsu",
                input_path,
                exc,
            )
    elif prefer_hd_bet:
        logger.warning(
            "HD-BET is not installed; using SimpleITK Otsu fallback for %s",
            input_path,
        )

    _run_simpleitk_otsu(
        input_path,
        output_path,
        mask_path,
        closing_radius=closing_radius,
    )
    logger.info(
        "Skull stripping used SimpleITK Otsu+morphology for %s -> %s (mask=%s)",
        input_path,
        output_path,
        mask_path,
    )
    return SkullStripResult(
        stripped_path=output_path, mask_path=mask_path, method="simpleitk_otsu"
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
    """Run HD-BET via Python API, then CLI if needed."""
    # Newer HD-BET exposes run_hd_bet; older builds differ slightly.
    try:
        from HD_BET.run import run_hd_bet

        run_hd_bet(
            str(input_path),
            str(output_path),
            mode=mode,
            device=device,
            postprocess=True,
            do_tta=mode == "accurate",
        )
        _ensure_mask_alongside(input_path, output_path, mask_path)
        return
    except ImportError:
        pass
    except TypeError:
        # Signature mismatch across HD-BET versions — try positional-only form.
        try:
            from HD_BET.run import run_hd_bet

            run_hd_bet(str(input_path), str(output_path))
            _ensure_mask_alongside(input_path, output_path, mask_path)
            return
        except Exception:
            pass

    cli = shutil.which("hd-bet")
    if cli is None:
        raise RuntimeError("HD-BET Python API and CLI are both unavailable")

    with tempfile.TemporaryDirectory(prefix="hd_bet_") as tmp:
        tmp_out = Path(tmp) / output_path.name
        cmd = [cli, "-i", str(input_path), "-o", str(tmp_out), "-device", str(device)]
        if mode:
            cmd.extend(["-mode", mode])
        subprocess.run(cmd, check=True)

        produced = tmp_out if tmp_out.is_file() else None
        if produced is None:
            candidates = sorted(Path(tmp).glob("*.nii*"))
            # Prefer non-mask volumes.
            non_mask = [p for p in candidates if "mask" not in p.name.lower()]
            produced = (non_mask or candidates)[0] if candidates else None
        if produced is None:
            raise FileNotFoundError("HD-BET CLI produced no NIfTI output")

        shutil.copy2(produced, output_path)
        mask_candidates = sorted(Path(tmp).glob("*mask*.nii*"))
        if mask_candidates:
            shutil.copy2(mask_candidates[0], mask_path)
        else:
            _mask_from_stripped(input_path, output_path, mask_path)


def _ensure_mask_alongside(input_path: Path, output_path: Path, mask_path: Path) -> None:
    """Locate an HD-BET mask or derive one from the stripped volume."""
    # HD-BET commonly writes <stem>_mask.nii.gz next to the stripped output.
    stem = output_path.name
    if stem.endswith(".nii.gz"):
        base = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        base = stem[: -len(".nii")]
    else:
        base = output_path.stem

    candidates = [
        output_path.with_name(f"{base}_mask.nii.gz"),
        output_path.with_name(f"{base}_mask.nii"),
        output_path.parent / f"{base}_mask.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            if candidate.resolve() != mask_path.resolve():
                shutil.copy2(candidate, mask_path)
            return

    _mask_from_stripped(input_path, output_path, mask_path)


def _mask_from_stripped(input_path: Path, stripped_path: Path, mask_path: Path) -> None:
    """Build a binary mask where the stripped volume is non-zero."""
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
) -> None:
    """Otsu threshold + morphological clean-up + largest connected component."""
    original = sitk.ReadImage(str(input_path))
    image = sitk.Cast(original, sitk.sitkFloat32)
    image.CopyInformation(original)

    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    radius = [int(closing_radius)] * image.GetDimension()
    mask = sitk.BinaryMorphologicalClosing(mask, radius)
    mask = sitk.BinaryFillhole(mask)

    # Keep the largest foreground component (brain).
    components = sitk.ConnectedComponent(mask)
    sorted_labels = sitk.RelabelComponent(components, sortByObjectSize=True)
    mask = sitk.Equal(sorted_labels, 1)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(original)

    # Light opening to trim thin skull fragments, then dilate slightly.
    open_radius = [max(1, closing_radius - 1)] * image.GetDimension()
    mask = sitk.BinaryMorphologicalOpening(mask, open_radius)
    mask = sitk.BinaryDilate(mask, [1] * image.GetDimension())
    mask = sitk.BinaryFillhole(mask)
    mask.CopyInformation(original)

    stripped = sitk.Mask(original, mask)
    stripped.CopyInformation(original)

    sitk.WriteImage(stripped, str(output_path))
    sitk.WriteImage(mask, str(mask_path))
