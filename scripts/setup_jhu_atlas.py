#!/usr/bin/env python3
"""Copy the JHU tract probability atlas from a local FSL install (optional).

Does **not** download or commit atlas files. If ``$FSLDIR`` is set and the
standard JHU 4D volume exists, copies it into ``atlases/jhu/`` (gitignored).
Otherwise prints exact manual instructions and exits non-zero.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "atlases" / "jhu" / "JHU-ICBM-tracts-prob-2mm.nii.gz"
REL_FSL = Path("data") / "atlases" / "JHU" / "JHU-ICBM-tracts-prob-2mm.nii.gz"

INSTRUCTIONS = """
JHU white-matter tract atlas not found.

This pipeline never downloads atlas files. To enable CST / optic-radiation
distance reporting:

1. Install FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
   and ensure FSLDIR is exported, e.g.:
     export FSLDIR=/usr/local/fsl
     source $FSLDIR/etc/fslconf/fsl.sh

2. Confirm the file exists:
     ls "$FSLDIR/data/atlases/JHU/JHU-ICBM-tracts-prob-2mm.nii.gz"

3. Re-run:
     python scripts/setup_jhu_atlas.py

   Or copy manually (do not commit):
     mkdir -p atlases/jhu
     cp "$FSLDIR/data/atlases/JHU/JHU-ICBM-tracts-prob-2mm.nii.gz" \\
        atlases/jhu/JHU-ICBM-tracts-prob-2mm.nii.gz

Optional optic-radiation probability maps (also not downloaded):
     atlases/jhu/optic_radiation_L.nii.gz
     atlases/jhu/optic_radiation_R.nii.gz

See sravanidasari4488/mri_brats docs/ATLAS_SETUP.md for volume indices.
""".strip()


def find_fsl_jhu(fsldir: str | None = None) -> Path | None:
    root = Path(fsldir or os.environ.get("FSLDIR", "") or "")
    if not root.is_dir():
        return None
    cand = root / REL_FSL
    return cand if cand.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination path (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--fsldir",
        type=str,
        default=None,
        help="Override $FSLDIR",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing destination file",
    )
    args = parser.parse_args(argv)

    src = find_fsl_jhu(args.fsldir)
    if src is None:
        print(INSTRUCTIONS, file=sys.stderr)
        return 1

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not args.force:
        print(f"Already present: {dest}")
        print(f"(source would be {src}; pass --force to overwrite)")
        return 0

    shutil.copy2(src, dest)
    print(f"Copied {src} → {dest}")
    print("Atlas files are gitignored; do not commit them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
