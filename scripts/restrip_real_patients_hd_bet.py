#!/usr/bin/env python3
"""Re-skull-strip every real-patient study with HD-BET and re-run 1 mm registration.

Overwrites ``03_skull/`` and ``04_registered_1mm/`` under
``data/processed/real_patients/``. Does not re-run N4, DICOM conversion, or the
false-positive audit. Writes a before/after mask-fraction table.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.registration import (  # noqa: E402
    REGISTERED_NAMES,
    register_modalities_to_reference,
)
from preprocessing.skull_strip import (  # noqa: E402
    hd_bet_available,
    hd_bet_status,
    skull_strip,
    write_study_strip_summary,
)

logger = logging.getLogger("restrip_hd_bet")

PATIENTS = ROOT / "data" / "processed" / "real_patients"
OUT = ROOT / "outputs" / "hd_bet_restrip"
DEMO_CASES = [
    "26070314071267604026_MR_Study1",
    "26072008063136939243_MR_Study1",
    "26072110054761049720_MR_Study1",
    "26072209432761090085_MR_Study1",
]
SKIP_DIRS = {"corrected"}
MODALITIES = ("T1", "T1c", "T2", "FLAIR")


def mask_fraction(path: Path) -> float | None:
    if not path.is_file():
        return None
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    n = int(arr.size)
    if n == 0:
        return None
    return float((arr > 0).sum()) / float(n)


def discover_studies(root: Path) -> list[Path]:
    studies = [
        p
        for p in sorted(root.iterdir())
        if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_DIRS
    ]
    demo = [p for p in studies if p.name in DEMO_CASES]
    rest = [p for p in studies if p.name not in DEMO_CASES]
    # Demo cases first so meshes / UI can proceed before the full cohort finishes.
    demo_ordered = [root / sid for sid in DEMO_CASES if (root / sid).is_dir()]
    return demo_ordered + rest


def snapshot_old_fractions(studies: list[Path], dest: Path) -> dict[str, dict]:
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    data: dict[str, dict] = {}
    for study in studies:
        row: dict[str, float | None] = {}
        for mod in MODALITIES:
            mask = study / "03_skull" / f"{mod}_brainmask.nii.gz"
            row[mod] = mask_fraction(mask)
        data[study.name] = row
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def flag_row(row: dict) -> list[str]:
    flags: list[str] = []
    methods = row.get("methods") or {}
    if any(m != "hd-bet" for m in methods.values()):
        flags.append("fallback")
    if row.get("error"):
        flags.append("error")
    t1_new = row.get("t1_new_frac")
    t1_old = row.get("t1_old_frac")
    if t1_new is None:
        flags.append("missing_t1_mask")
    else:
        if t1_new < 0.04:
            flags.append("too_small")
        if t1_new > 0.50:
            flags.append("too_large")
        if t1_old and t1_old > 0:
            ratio = t1_new / t1_old
            if ratio < 0.5 or ratio > 4.0:
                flags.append(f"wild_change_ratio={ratio:.2f}")
    if not row.get("registered"):
        flags.append("registration_skipped")
    return flags


def already_done(study: Path) -> bool:
    summary = study / "03_skull" / "skullstrip_methods.json"
    t1_reg = study / "04_registered_1mm" / REGISTERED_NAMES["T1"]
    if not summary.is_file() or not t1_reg.is_file():
        return False
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    methods = data.get("methods") or {}
    return bool(methods) and all(v == "hd-bet" for v in methods.values())


def update_manifest(
    study: Path,
    methods: dict[str, str],
    reasons: dict[str, str],
    registered: dict[str, Path],
) -> None:
    path = study / "pipeline_manifest.json"
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    data["study_name"] = study.name
    data["output_dir"] = str(study)
    data["skull_strip_methods"] = methods
    data["skull_strip_reasons"] = reasons
    data["skull_stripped"] = {
        mod: str(study / "03_skull" / f"{mod}.nii.gz") for mod in methods
    }
    if registered:
        data["registered_1mm"] = {k: str(v) for k, v in registered.items()}
        tfm_dir = study / "04_registered_1mm" / "transforms"
        if tfm_dir.is_dir():
            data["transforms"] = {p.stem: str(p) for p in sorted(tfm_dir.glob("*.tfm"))}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def refresh_t1c_standin(reg_dir: Path, had_native_t1c: bool) -> None:
    """Keep T1c in lockstep with T1 when the study has no native contrast T1c."""
    if had_native_t1c:
        return
    t1 = reg_dir / REGISTERED_NAMES["T1"]
    t1c = reg_dir / REGISTERED_NAMES["T1c"]
    if t1.is_file():
        shutil.copy2(t1, t1c)


def process_study(study: Path) -> dict:
    bias_dir = study / "02_bias"
    strip_dir = study / "03_skull"
    reg_dir = study / "04_registered_1mm"
    strip_dir.mkdir(parents=True, exist_ok=True)

    bias_mods = {
        mod: bias_dir / f"{mod}.nii.gz"
        for mod in MODALITIES
        if (bias_dir / f"{mod}.nii.gz").is_file()
    }
    if not bias_mods:
        return {
            "study": study.name,
            "error": "no 02_bias NIfTIs",
            "methods": {},
            "registered": False,
        }

    methods: dict[str, str] = {}
    reasons: dict[str, str] = {}
    stripped: dict[str, Path] = {}
    new_frac: dict[str, float | None] = {}

    for mod, src in bias_mods.items():
        out = strip_dir / f"{mod}.nii.gz"
        mask = strip_dir / f"{mod}_brainmask.nii.gz"
        result = skull_strip(src, out, mask_path=mask, prefer_hd_bet=True)
        methods[mod] = result.method
        reasons[mod] = result.reason
        stripped[mod] = result.stripped_path
        new_frac[mod] = mask_fraction(result.mask_path)
        logger.info("%s %s -> %s (%s)", study.name, mod, result.method, result.reason)

    write_study_strip_summary(
        strip_dir, study_name=study.name, methods=methods, reasons=reasons
    )

    registered_paths: dict[str, Path] = {}
    registered = False
    error = None
    if "T1" in stripped:
        try:
            registered_paths = register_modalities_to_reference(stripped, reg_dir)
            refresh_t1c_standin(reg_dir, had_native_t1c="T1c" in stripped)
            t1c = reg_dir / REGISTERED_NAMES["T1c"]
            if t1c.is_file():
                registered_paths.setdefault("T1c", t1c)
            registered = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Registration failed for %s", study.name)
    else:
        error = "no T1 in 02_bias; skipped registration"

    update_manifest(study, methods, reasons, registered_paths)
    return {
        "study": study.name,
        "methods": methods,
        "reasons": reasons,
        "new_frac": new_frac,
        "registered": registered,
        "error": error,
    }


def write_sanity_table(rows: list[dict], old: dict[str, dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "study,t1_old_frac,t1_new_frac,t2_old_frac,t2_new_frac,"
        "flair_old_frac,flair_new_frac,methods,flags"
    ]
    for row in rows:
        sid = row["study"]
        old_row = old.get(sid) or {}
        methods = row.get("methods") or {}
        new_frac = row.get("new_frac") or {}
        enriched = {
            **row,
            "t1_old_frac": old_row.get("T1"),
            "t1_new_frac": new_frac.get("T1"),
            "methods": methods,
        }
        flags = flag_row(enriched)
        row["flags"] = flags
        row["t1_old_frac"] = old_row.get("T1")
        row["t1_new_frac"] = new_frac.get("T1")

        def fmt(val: object) -> str:
            if val is None:
                return ""
            if isinstance(val, float):
                return f"{val:.6f}"
            return str(val)

        method_s = ";".join(f"{k}={v}" for k, v in methods.items())
        lines.append(
            ",".join(
                [
                    sid,
                    fmt(old_row.get("T1")),
                    fmt(new_frac.get("T1")),
                    fmt(old_row.get("T2")),
                    fmt(new_frac.get("T2")),
                    fmt(old_row.get("FLAIR")),
                    fmt(new_frac.get("FLAIR")),
                    method_s.replace(",", "|"),
                    "|".join(flags),
                ]
            )
        )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PATIENTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip studies whose 03_skull summary already records hd-bet for all modalities",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)

    status = hd_bet_status()
    (args.out / "hd_bet_status.json").write_text(
        json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("HD-BET status: %s", json.dumps(status, default=str))
    if not hd_bet_available():
        logger.error("HD-BET is not available; aborting so we do not rewrite Otsu masks")
        return 1

    studies = discover_studies(args.root)
    if args.limit is not None:
        studies = studies[: args.limit]
    logger.info("Studies to consider: %d", len(studies))

    old_path = args.out / "old_mask_fractions.json"
    old = snapshot_old_fractions(studies, old_path)

    results: list[dict] = []
    jsonl = args.out / "progress.jsonl"
    if jsonl.is_file() and args.resume:
        # Keep prior rows for studies we skip.
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
    else:
        jsonl.write_text("", encoding="utf-8")

    done_ids = {r["study"] for r in results if not r.get("error")}
    n_ok = 0
    n_fail = 0
    for i, study in enumerate(studies, start=1):
        if args.resume and (study.name in done_ids or already_done(study)):
            logger.info("[%d/%d] skip (already hd-bet) %s", i, len(studies), study.name)
            if study.name not in done_ids:
                summary = study / "03_skull" / "skullstrip_methods.json"
                methods = {}
                reasons = {}
                if summary.is_file():
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    methods = data.get("methods") or {}
                    reasons = data.get("reasons") or {}
                new_frac = {
                    mod: mask_fraction(study / "03_skull" / f"{mod}_brainmask.nii.gz")
                    for mod in MODALITIES
                }
                row = {
                    "study": study.name,
                    "methods": methods,
                    "reasons": reasons,
                    "new_frac": new_frac,
                    "registered": (study / "04_registered_1mm" / REGISTERED_NAMES["T1"]).is_file(),
                    "error": None,
                    "skipped": True,
                }
                results.append(row)
                with jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            continue
        logger.info("[%d/%d] %s", i, len(studies), study.name)
        row = process_study(study)
        results.append(row)
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if row.get("error"):
            n_fail += 1
        else:
            n_ok += 1

    # Deduplicate if resume re-appended.
    uniq: dict[str, dict] = {}
    for row in results:
        uniq[row["study"]] = row
    rows = [uniq[s.name] for s in studies if s.name in uniq]

    write_sanity_table(rows, old, args.out / "sanity_table.csv")
    flagged = [r for r in rows if r.get("flags")]
    elapsed = time.perf_counter() - started
    summary = {
        "n_studies": len(studies),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_flagged": len(flagged),
        "flagged_studies": [
            {"study": r["study"], "flags": r.get("flags"), "error": r.get("error")}
            for r in flagged
        ],
        "elapsed_sec": round(elapsed, 1),
        "hd_bet_status": status,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info(
        "Done n=%d ok=%d fail=%d flagged=%d elapsed=%.1fs",
        len(studies),
        n_ok,
        n_fail,
        len(flagged),
        elapsed,
    )
    for r in flagged:
        logger.warning("FLAG %s %s %s", r["study"], r.get("flags"), r.get("error"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
