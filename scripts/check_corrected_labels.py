#!/usr/bin/env python3
"""Validate batch2 corrected_label.nii.gz packages and refresh LABELING_PROGRESS.md.

Read-only with respect to label NIfTIs. Prints a progress table and writes
``outputs/batch2_labeling/LABELING_PROGRESS.md``.

Corrected masks are reference masks (single-annotator corrections from a v3
draft), not clinical ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.batch2_label_qc import (  # noqa: E402
    REFERENCE_DISCLAIMER,
    check_all_studies,
    format_progress_table,
    write_progress_md,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "QC batch2 corrected labels (read-only). "
            f"{REFERENCE_DISCLAIMER}"
        )
    )
    parser.add_argument(
        "--labeling-root",
        type=Path,
        default=ROOT / "outputs" / "batch2_labeling",
    )
    parser.add_argument(
        "--progress-md",
        type=Path,
        default=None,
        help="Default: <labeling-root>/LABELING_PROGRESS.md",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional per-study JSON dump",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    results = check_all_studies(args.labeling_root)
    table = format_progress_table(results)
    print(table)

    progress_path = args.progress_md or (args.labeling_root / "LABELING_PROGRESS.md")
    written = write_progress_md(results, progress_path)
    print(f"\nWrote {written}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "disclaimer": REFERENCE_DISCLAIMER,
            "studies": [r.to_dict() for r in results],
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")

    n_invalid = sum(1 for r in results if r.status == "invalid")
    return 1 if n_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
