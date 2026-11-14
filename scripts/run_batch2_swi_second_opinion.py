#!/usr/bin/env python3
"""Convert/register SWI for native-T1c batch2 studies; rebuild montages; update SUMMARY."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.batch2_swi import (  # noqa: E402
    assess_dark_focus_study,
    convert_and_register_swi,
    native_t1c_ids,
)
from segmentation.hdglio_review import save_review_montage  # noqa: E402

OUT = ROOT / "outputs" / "batch2_second_opinion"
LABELING = ROOT / "outputs" / "batch2_labeling"
BATCH2 = ROOT / "data" / "processed" / "real_patients_batch2"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--study", nargs="*", default=None)
    p.add_argument("--skip-convert", action="store_true")
    args = p.parse_args(argv)

    ids = args.study or native_t1c_ids()
    convert_rows = []
    dark_rows = []

    for sid in ids:
        print(f"=== {sid} SWI ===", flush=True)
        reg = OUT / sid / "swi" / "SWI_registered_to_t1c.nii.gz"
        if args.skip_convert and reg.is_file():
            meta_path = OUT / sid / "swi" / "swi_convert_meta.json"
            info = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {
                "study_id": sid, "notes": "reused", "ncc_to_t1c": None, "chosen": "magnitude"
            }
        else:
            info = convert_and_register_swi(sid).__dict__
        convert_rows.append(info)
        print(
            f"  chosen={info.get('chosen')} ncc={info.get('ncc_to_t1c')} "
            f"mag={info.get('n_magnitude')} phase={info.get('n_phase')}",
            flush=True,
        )

        # Rebuild montage with SWI panel (new filename + replace review_montage.png)
        t1c = LABELING / sid / "T1c.nii.gz"
        if not t1c.is_file():
            t1c = BATCH2 / sid / "04_registered_1mm" / "T1c_registered_1mm.nii.gz"
        flair = LABELING / sid / "FLAIR.nii.gz"
        if not flair.is_file():
            flair = BATCH2 / sid / "04_registered_1mm" / "FLAIR_registered_1mm.nii.gz"
        v3 = LABELING / sid / "v3_draft_label.nii.gz"
        hd = OUT / sid / "hdglio_draft_label.nii.gz"
        swi = OUT / sid / "swi" / "SWI_registered_to_t1c.nii.gz"
        if hd.is_file():
            for name in ("review_montage.png", "review_montage_swi.png"):
                save_review_montage(
                    sid,
                    t1c,
                    flair,
                    v3 if v3.is_file() else None,
                    hd,
                    OUT / sid / name,
                    swi_path=swi if swi.is_file() else None,
                )
            print(f"  montage rebuilt", flush=True)

        dark = assess_dark_focus_study(sid)
        dark_rows.append(dark)
        print(f"  dark_focus={dark['swi_dark_focus_inside_either_contour']}", flush=True)

    (OUT / "swi_convert_summary.json").write_text(
        json.dumps(convert_rows, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "swi_dark_focus.json").write_text(
        json.dumps(dark_rows, indent=2) + "\n", encoding="utf-8"
    )
    _update_readme()
    _update_summary(dark_rows, convert_rows)
    return 0


def _update_readme() -> None:
    readme = OUT / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    block = (
        "\n## SWI review note\n\n"
        "**Dark SWI signal within a contour is a flag for human review and is not a diagnosis.**\n"
        "SWI magnitude (Philips `SW_M`) was converted under `<study_id>/swi/` and rigid-registered "
        "to T1c. Montages include an SWI panel with v3 (green) and HD-GLIO (red) contours.\n"
    )
    if "Dark SWI signal within a contour" not in text:
        text = text.rstrip() + "\n" + block
    else:
        # ensure the stronger wording is present
        if "not a diagnosis" not in text:
            text = text.rstrip() + "\n" + block
    readme.write_text(text, encoding="utf-8")


def _update_summary(dark_rows: list[dict], convert_rows: list[dict]) -> None:
    summary = OUT / "SUMMARY.md"
    text = summary.read_text(encoding="utf-8")
    dark_map = {d["study_id"]: d["swi_dark_focus_inside_either_contour"] for d in dark_rows}
    ncc_map = {c["study_id"]: c.get("ncc_to_t1c") for c in convert_rows}
    section = [
        "",
        "## SWI second-opinion check (native-T1c)",
        "",
        "Philips `SWIp_fast` series contain interleaved **magnitude** (`SW_M`) and **phase** (`SW_P`). "
        "We converted **magnitude only**, rigid-registered to T1c (Mattes MI, same as other sequences). "
        "Dark-focus column is a **heuristic** (lowest 5% SWI inside either contour vs surrounding brain); "
        "**not a diagnosis**.",
        "",
        "| study | SWI chosen | NCC→T1c | SWI dark focus inside either contour |",
        "|---|---|---:|---|",
    ]
    for sid in sorted(dark_map):
        ncc = ncc_map.get(sid)
        ncc_s = f"{ncc:.3f}" if isinstance(ncc, (int, float)) and ncc == ncc else "—"
        chosen = next((c.get("chosen") for c in convert_rows if c["study_id"] == sid), "—")
        section.append(
            f"| {sid} | {chosen} | {ncc_s} | {dark_map[sid]} |"
        )
    section.append("")
    marker = "## SWI second-opinion check"
    if marker in text:
        # replace existing section through next ## or EOF
        pre, rest = text.split(marker, 1)
        # drop until next top-level ## after first line of rest
        lines = rest.splitlines()
        idx = 1
        while idx < len(lines):
            if lines[idx].startswith("## ") and not lines[idx].startswith("## SWI"):
                break
            idx += 1
        text = pre.rstrip() + "\n" + "\n".join(section) + "\n" + "\n".join(lines[idx:])
    else:
        text = text.rstrip() + "\n" + "\n".join(section) + "\n"
    # refresh SWI note in intro
    text = text.replace(
        "SWI panels were omitted where SWI was not converted to a registered 1 mm NIfTI "
        "(present in DICOM inventory for 18/19 studies, but not under `01_dicom_nifti`).",
        "SWI magnitude was converted and registered for the 10 native-T1c studies "
        "(see SWI section below). Dark SWI inside a contour is a review flag, not a diagnosis.",
    )
    summary.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
