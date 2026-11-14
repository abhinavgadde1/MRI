#!/usr/bin/env python3
"""Apply radiologist_reads.csv to pos/neg manifests; suggest replacements (no selection change)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.batch2_radiologist_status import (  # noqa: E402
    RAD_CSV,
    STATUS_PHRASE,
    STATUS_REPORT,
    STATUS_UNVERIFIED,
    STATUS_VERBAL,
    apply_radiologist_status_to_manifest,
    list_unverified_selected,
    status_counts,
    suggest_confirmed_replacements,
    write_status_reports,
)


def _patch_reports(counts: dict) -> None:
    report_dir = ROOT / "outputs" / "batch2_pos_neg"
    # README
    readme = report_dir / "README.md"
    readme.write_text(
        "# Batch2 positives ↔ tumor-free negatives\n\n"
        "Manifest / baseline only — **no training**. Checkpoints, `splits/heldout.txt`, "
        "`data/processed/real_patients/` image files, and existing labels were not modified.\n\n"
        "- Matching report: `MATCHING_REPORT.md`\n"
        "- Radiologist status: `RADIOLOGIST_STATUS.md`\n"
        "- Baseline v3+LCC: `BASELINE.md`\n"
        "- Confirmed-only FP table: `BASELINE_CONFIRMED.md`\n"
        "- Splits: `splits/batch2_pos_neg/`\n\n"
        "Per-scan tumor-free status (from `radiologist_reads.csv` only):\n"
        f"- `{STATUS_REPORT}`: {counts.get(STATUS_REPORT, 0)}\n"
        f"- `{STATUS_VERBAL}`: {counts.get(STATUS_VERBAL, 0)}\n"
        f"- `{STATUS_UNVERIFIED}`: {counts.get(STATUS_UNVERIFIED, 0)}\n"
        f"\nPhrase map: {STATUS_PHRASE}\n",
        encoding="utf-8",
    )
    # MATCHING_REPORT: replace blanket claim wording
    mr = report_dir / "MATCHING_REPORT.md"
    if mr.is_file():
        text = mr.read_text(encoding="utf-8")
        old = (
            "tumor-free status is the user's claim and is not verified in the data "
            "(no radiology report / diagnosis / tumor label found in-repo)"
        )
        new = (
            "per-scan tumor-free status is recorded in `manifest.csv` "
            f"(`{STATUS_REPORT}` / `{STATUS_VERBAL}` / `{STATUS_UNVERIFIED}`); "
            "see `RADIOLOGIST_STATUS.md` for counts"
        )
        text = text.replace(old, new)
        text = text.replace(
            "claimed tumor-free negatives",
            "negatives with per-scan tumor-free status",
        )
        text = text.replace(
            "Tumor-free status remains an **unverified claim**.",
            "Tumor-free status is per-scan (radiologist report / verbal / claimed_unverified).",
        )
        text = text.replace(
            "tumor-free status remains an **unverified claim**",
            "tumor-free status is per-scan in the manifest",
        )
        if "## Radiologist status counts" not in text:
            text = text.rstrip() + (
                "\n\n## Radiologist status counts\n\n"
                f"- `{STATUS_REPORT}`: {counts.get(STATUS_REPORT, 0)}\n"
                f"- `{STATUS_VERBAL}`: {counts.get(STATUS_VERBAL, 0)}\n"
                f"- `{STATUS_UNVERIFIED}`: {counts.get(STATUS_UNVERIFIED, 0)}\n"
            )
        mr.write_text(text, encoding="utf-8")


def main() -> int:
    if not RAD_CSV.is_file():
        print(f"ERROR: missing {RAD_CSV}", file=sys.stderr)
        return 2
    df = apply_radiologist_status_to_manifest()
    counts = status_counts(df)
    print("Status counts (negatives):")
    for k in (STATUS_REPORT, STATUS_VERBAL, STATUS_UNVERIFIED):
        print(f"  {k}: {counts.get(k, 0)}")
    unver = list_unverified_selected(df)
    print(f"Selected still {STATUS_UNVERIFIED}: {len(unver)}")
    for s in unver:
        print(f"  - {s}")
    suggestions = suggest_confirmed_replacements(df, unver)
    write_status_reports(df, suggestions)
    _patch_reports(counts)
    print(f"Wrote {ROOT / 'outputs' / 'batch2_pos_neg' / 'RADIOLOGIST_STATUS.md'}")
    if unver and not any(s.get("suggested_replacement") for s in suggestions):
        print(
            "No radiologist-confirmed unused replacements available "
            "(CSV has no confirmed reads yet). Selection unchanged."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
