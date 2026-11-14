#!/usr/bin/env python3
"""Build matched batch2 pos/neg manifests (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.batch2_pos_neg_manifest import (  # noqa: E402
    NEG_ROOT,
    TUMOR_FREE_CLAIM,
    collect_negatives,
    collect_positives,
    compare_groups,
    select_matched_negatives,
    write_manifests,
)


def main() -> int:
    if not NEG_ROOT.is_dir() or not any(NEG_ROOT.iterdir()):
        print(
            f"ERROR: could not find tumor-free scans at {NEG_ROOT}. "
            "Please provide the path to your 100 tumor-free scans.",
            file=sys.stderr,
        )
        return 2
    print(f"Negatives root: {NEG_ROOT}")
    print(f"NOTE: {TUMOR_FREE_CLAIM}")
    positives = collect_positives()
    print(f"Positives: {len(positives)}")
    negatives = collect_negatives(positives)
    print(f"Negatives after overlap exclusion: {len(negatives)}")
    selected, unused, deviation = select_matched_negatives(
        positives, negatives, n=20, seed=42
    )
    print(f"Selected: {len(selected)}; unused: {len(unused)}")
    print("Deviation:", deviation.get("t1c_note"))
    compare_df, warnings_out = compare_groups(positives, selected)
    print(compare_df.to_string(index=False))
    for w in warnings_out:
        print(w)
    paths = write_manifests(
        positives,
        selected,
        unused,
        deviation=deviation,
        compare_df=compare_df,
        warnings_out=warnings_out,
    )
    for k, p in paths.items():
        print(f"Wrote {k}: {p}")
    corrected = [p.study_id for p in positives if p.corrected_label_exists]
    print(
        "Positives with corrected_label:",
        ", ".join(corrected) if corrected else "(none yet)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
