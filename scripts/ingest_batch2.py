#!/usr/bin/env python3
"""Ingest PROJECT/1..20 usable studies into data/processed/real_patients_batch2/.

Read-only on PROJECT/; does not touch data/processed/real_patients/.
Requires HD-BET (aborts if Otsu would be used).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

from preprocessing.registration import run_patient_preprocessing
from preprocessing.skull_strip import hd_bet_available
from segmentation.prepare_correction_set import fill_missing_t1c_from_t1

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "PROJECT"
OUT = ROOT / "data" / "processed" / "real_patients_batch2"
# Explicit guard: never write into the original cohort folder.
FORBIDDEN = ROOT / "data" / "processed" / "real_patients"


def mask_fraction(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = nib.load(str(path)).get_fdata()
    return float(np.mean(data > 0))


def main() -> None:
    assert OUT.resolve() != FORBIDDEN.resolve()
    assert FORBIDDEN.resolve() not in OUT.resolve().parents
    OUT.mkdir(parents=True, exist_ok=True)
    if not hd_bet_available():
        raise SystemExit("HD-BET is not available; aborting so we do not write Otsu masks")

    usable = [i for i in range(1, 21) if i != 19]
    summary: list[dict] = []

    for idx, i in enumerate(usable, start=1):
        sid = f"batch2_{i:02d}"
        src = PROJECT / str(i)
        dst = OUT / sid
        print(f"\n==== [{idx}/{len(usable)}] {sid} from {src} -> {dst} ====", flush=True)
        t0 = time.perf_counter()
        row: dict = {
            "study_id": sid,
            "src": str(src),
            "output_dir": str(dst),
            "status": "failed",
        }
        reg_dir = dst / "04_registered_1mm"
        already = all(
            (reg_dir / name).is_file()
            for name in ("T1_1mm.nii.gz", "T2_registered_1mm.nii.gz", "FLAIR_registered_1mm.nii.gz")
        )
        if already:
            print(f"SKIP {sid}: registered outputs already present", flush=True)
            # Still verify HD-BET sidecars / fractions below by reading disk.
            methods = {}
            strip_dir = dst / "03_skull"
            fracs: dict[str, float | None] = {}
            method_files: dict[str, dict] = {}
            for mod in ("T1", "T1c", "T2", "FLAIR"):
                mask = strip_dir / f"{mod}_brainmask.nii.gz"
                fracs[mod] = mask_fraction(mask)
                side = strip_dir / f"{mod}_skullstrip_method.json"
                if side.is_file():
                    method_files[mod] = json.loads(side.read_text(encoding="utf-8"))
                    methods[mod] = method_files[mod].get("method")
            otsu = [m for m, v in methods.items() if v and "otsu" in str(v).lower()]
            if otsu:
                row["error"] = f"Otsu fallback found on existing outputs: {methods}"
            elif not methods:
                row["error"] = "Existing registered outputs lack skullstrip method sidecars"
            else:
                row.update(
                    {
                        "status": "success",
                        "modalities_processed": ",".join(
                            sorted(
                                m
                                for m in ("T1", "T1c", "T2", "FLAIR")
                                if (reg_dir / f"{m}_1mm.nii.gz").is_file()
                                or (reg_dir / f"{m}_registered_1mm.nii.gz").is_file()
                            )
                        ),
                        "skull_strip_methods": methods,
                        "mask_fractions": fracs,
                        "method_sidecars": method_files,
                        "duration_sec": 0.0,
                        "skipped_existing": True,
                    }
                )
            summary.append(row)
            (OUT / "ingest_summary.json").write_text(
                json.dumps(summary, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            continue
        try:
            result = run_patient_preprocessing(
                src,
                dst,
                study_name=sid,
                use_hd_bet=True,
            )
            methods = dict(getattr(result, "skull_strip_methods", {}) or {})
            strip_dir = dst / "03_skull"
            fracs: dict[str, float | None] = {}
            method_files: dict[str, dict] = {}
            for mod in ("T1", "T1c", "T2", "FLAIR"):
                mask = strip_dir / f"{mod}_brainmask.nii.gz"
                fracs[mod] = mask_fraction(mask)
                side = strip_dir / f"{mod}_skullstrip_method.json"
                if side.is_file():
                    method_files[mod] = json.loads(side.read_text(encoding="utf-8"))
                    methods[mod] = method_files[mod].get("method", methods.get(mod))
            otsu = [m for m, v in methods.items() if v and "otsu" in str(v).lower()]
            if otsu:
                raise RuntimeError(f"Otsu fallback used for {otsu}: {methods}")
            missing_hd = [
                m
                for m, v in methods.items()
                if v and "hd" not in str(v).lower() and str(v) != "pre_stripped"
            ]
            if missing_hd:
                raise RuntimeError(f"Non-HD-BET methods for {missing_hd}: {methods}")
            row.update(
                {
                    "status": "success",
                    "modalities_processed": ",".join(sorted(result.registered_1mm.keys())),
                    "skull_strip_methods": methods,
                    "mask_fractions": fracs,
                    "method_sidecars": method_files,
                    "duration_sec": round(time.perf_counter() - t0, 2),
                }
            )
            print(
                f"SUCCESS {sid} methods={methods} fracs={fracs} in {row['duration_sec']}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["duration_sec"] = round(time.perf_counter() - t0, 2)
            print(f"FAILED {sid}: {row['error']}\n{traceback.format_exc()}", flush=True)
        summary.append(row)
        (OUT / "ingest_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    n_filled = fill_missing_t1c_from_t1(OUT)
    print(f"Filled missing T1c stand-ins: {n_filled}", flush=True)
    payload = {"cases": summary, "t1c_standins_filled": n_filled}
    (OUT / "ingest_summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("INGEST DONE", flush=True)


if __name__ == "__main__":
    main()
