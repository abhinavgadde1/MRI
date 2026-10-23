#!/usr/bin/env python3
"""Write a tiny synthetic 4-modality NIfTI case (and optional zip) for API/CLI smoke tests.

No BraTS / DICOM required. Filenames match ``REGISTERED_NAMES`` so
``pipeline.run_case`` treats the folder as already-registered 1 mm data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.registration import REGISTERED_NAMES  # noqa: E402


def write_synthetic_case(out_dir: Path, *, shape=(64, 64, 48), seed: int = 42) -> Path:
    """Write four registered 1 mm NIfTIs under ``out_dir``; return ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    tumor = ((xx - shape[2] // 2) ** 2 + (yy - shape[1] // 2) ** 2 + (zz - shape[0] // 2) ** 2) < 10**2
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    for fname in REGISTERED_NAMES.values():
        vol = rng.normal(0.4, 0.05, size=shape).astype(np.float32)
        vol[tumor] += 0.8
        nib.save(nib.Nifti1Image(vol, affine), str(out_dir / fname))
    return out_dir


def zip_case(case_dir: Path, zip_path: Path) -> Path:
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(Path(case_dir).iterdir()):
            if p.is_file():
                zf.write(p, arcname=p.name)
    return zip_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "synthetic_case",
        help="Directory for the four NIfTI files",
    )
    p.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Optional zip path (default: <out>.zip next to --out)",
    )
    args = p.parse_args()
    case_dir = write_synthetic_case(args.out)
    zip_path = args.zip or Path(str(args.out) + ".zip")
    # If --out is a directory, default zip beside it
    if args.zip is None:
        zip_path = args.out.parent / f"{args.out.name}.zip"
    zip_case(case_dir, zip_path)
    print(f"Wrote case → {case_dir}")
    print(f"Wrote zip  → {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
