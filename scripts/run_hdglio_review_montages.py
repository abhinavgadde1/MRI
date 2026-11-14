#!/usr/bin/env python3
"""Build HD-GLIO vs v3 review montages (main .venv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.hdglio_review import save_review_montage  # noqa: E402

OUT = ROOT / "outputs" / "batch2_second_opinion"
BATCH2 = ROOT / "data" / "processed" / "real_patients_batch2"
LABELING = ROOT / "outputs" / "batch2_labeling"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--study", nargs="*", default=None)
    args = p.parse_args(argv)
    studies = sorted(
        d.name
        for d in OUT.iterdir()
        if d.is_dir() and (d / "hdglio_draft_label.nii.gz").is_file()
    )
    if args.study:
        studies = [s for s in studies if s in set(args.study)]
    for sid in studies:
        t1c = LABELING / sid / "T1c.nii.gz"
        if not t1c.is_file():
            t1c = BATCH2 / sid / "04_registered_1mm" / "T1c_registered_1mm.nii.gz"
        flair = LABELING / sid / "FLAIR.nii.gz"
        if not flair.is_file():
            flair = BATCH2 / sid / "04_registered_1mm" / "FLAIR_registered_1mm.nii.gz"
        v3 = LABELING / sid / "v3_draft_label.nii.gz"
        hd = OUT / sid / "hdglio_draft_label.nii.gz"
        swi = BATCH2 / sid / "04_registered_1mm" / "SWI_registered_1mm.nii.gz"
        out_png = OUT / sid / "review_montage.png"
        meta = save_review_montage(
            sid,
            t1c,
            flair,
            v3 if v3.is_file() else None,
            hd,
            out_png,
            swi_path=swi if swi.is_file() else None,
        )
        print(sid, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
