"""TractSeg integration helpers."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path


def tractseg_available() -> bool:
    """Return True if the TractSeg package or CLI appears installed."""
    if importlib.util.find_spec("tractseg") is not None:
        return True
    return shutil.which("TractSeg") is not None


def run_tractseg(
    peaks_or_dwi_dir: str | Path,
    output_dir: str | Path,
    *,
    bundle: str | None = None,
    python_executable: str = "TractSeg",
) -> Path:
    """Run TractSeg CLI on peak / CSD inputs.

    Parameters
    ----------
    peaks_or_dwi_dir:
        Directory or file accepted by the installed TractSeg entrypoint.
    output_dir:
        Destination for TractSeg outputs (bundle masks / tractograms).
    bundle:
        Optional single-bundle name; when omitted, TractSeg default set is used.
    python_executable:
        TractSeg console-script name on PATH.

    Returns
    -------
    Path
        Output directory containing TractSeg results.

    Raises
    ------
    RuntimeError
        If TractSeg is not installed or not on ``PATH``.
    """
    peaks_or_dwi_dir = Path(peaks_or_dwi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(python_executable) is None and importlib.util.find_spec("tractseg") is None:
        raise RuntimeError(
            "TractSeg is not installed. Install it (e.g. `pip install TractSeg`) "
            "and ensure the `TractSeg` console script is on PATH."
        )

    if shutil.which(python_executable) is None:
        raise RuntimeError(
            f"TractSeg package is importable but executable {python_executable!r} "
            "was not found on PATH."
        )

    cmd = [python_executable, "-i", str(peaks_or_dwi_dir), "-o", str(output_dir)]
    if bundle:
        cmd.extend(["--bundle", bundle])
    subprocess.run(cmd, check=True)
    return output_dir
