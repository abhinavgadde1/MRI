"""TractSeg integration helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


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

    Notes
    -----
    Requires a working TractSeg installation (listed in ``requirements.txt``).
    This wrapper shells out to the CLI so GPU / model weights follow TractSeg's
    own setup conventions.
    """
    peaks_or_dwi_dir = Path(peaks_or_dwi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(python_executable) is None:
        raise RuntimeError(
            "TractSeg executable not found on PATH. Install TractSeg and ensure "
            "its console script is available."
        )

    cmd = [python_executable, "-i", str(peaks_or_dwi_dir), "-o", str(output_dir)]
    if bundle:
        cmd.extend(["--bundle", bundle])
    subprocess.run(cmd, check=True)
    return output_dir
