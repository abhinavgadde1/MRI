#!/usr/bin/env python3
"""Run batch2 scan QC → outputs/batch2_second_opinion/scan_qc.{csv,md}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.batch2_scan_qc import run_scan_qc, write_scan_qc_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch2 scan QC (read-only).")
    p.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "batch2_second_opinion")
    p.add_argument("--no-overlays", action="store_true")
    args = p.parse_args(argv)
    results = run_scan_qc(out_root=args.out_root, write_overlays=not args.no_overlays)
    csv_path, md_path = write_scan_qc_reports(results, args.out_root)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    for r in results:
        print(f"  {r.study_id}: {r.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
