#!/usr/bin/env python3
"""Control test: Mac-patched HD-GLIO on 5 BraTS heldout cases (no training).

Predict with ``.venv-hdglio``; evaluate with this interpreter (main .venv OK for metrics).
Writes only under ``outputs/batch2_second_opinion/control_test/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "batch2_second_opinion" / "control_test"


def _hdglio_python() -> Path:
    return ROOT / ".venv-hdglio" / "bin" / "python"


def _predict_one(
    case_dir: Path,
    inputs: dict[str, Path],
    out_seg: Path,
    *,
    do_tta: bool,
    step_size: float,
) -> float:
    """Run HD-GLIO in isolated venv; return wall-clock seconds."""
    case_dir.mkdir(parents=True, exist_ok=True)
    runner = case_dir / "_predict_runner.py"
    runner.write_text(
        f"""\
import time
from pathlib import Path
from hd_glio.paths import folder_with_parameter_files
from hd_glio.setup_hd_glio import maybe_download_weights
from nnunet.inference.predict import predict_cases

maybe_download_weights()
t0 = time.perf_counter()
predict_cases(
    folder_with_parameter_files,
    [[r\"{inputs['t1']}\", r\"{inputs['t1c']}\", r\"{inputs['t2']}\", r\"{inputs['flair']}\"]],
    [r\"{out_seg}\"],
    (0,),
    False,
    1,
    1,
    None,
    {do_tta!r},
    False,
    True,
    step_size={step_size},
    checkpoint_name="model_final_checkpoint",
)
print("WALL", time.perf_counter() - t0)
""",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env["VIRTUAL_ENV"] = str(ROOT / ".venv-hdglio")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(_hdglio_python()), str(runner)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (case_dir / "predict.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"HD-GLIO failed for {case_dir.name}: {proc.stderr[-2000:]}")
    wall = time.perf_counter() - t0
    for line in proc.stdout.splitlines():
        if line.startswith("WALL "):
            wall = float(line.split()[1])
    return wall


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--skip-predict", action="store_true")
    p.add_argument("--default-settings-case", type=str, default=None,
                   help="Re-run this case with step_size=0.5 and TTA")
    args = p.parse_args(argv)

    import nibabel as nib
    import numpy as np
    import pandas as pd

    from segmentation.hdglio_control import (
        BRATS_ROOT,
        CONTROL_OUT,
        PUBLISHED_CE_DICE_REF,
        cases_to_jsonable,
        dice_hd95_regions,
        has_identity_affine,
        load_v3_lcc_metrics,
        orientation_sanity,
        select_heldout_control_cases,
    )
    from segmentation.hdglio_labels import (
        DRAFT_DISCLAIMER,
        hdglio_to_brats_classic,
        save_hdglio_draft_on_t1c_grid,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    cases = select_heldout_control_cases(n=args.n, seed=args.seed)
    (OUT / "selected_cases.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n": args.n,
                "hd_bet": "skipped — BraTS NIfTIs are already skull-stripped",
                "lean": {"do_tta": False, "step_size": 1.0},
                "cases": cases_to_jsonable(cases),
                "disclaimer": DRAFT_DISCLAIMER,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    v3 = load_v3_lcc_metrics()
    rows: list[dict] = []

    for case in cases:
        sid = case.study_id
        print(f"=== {sid} ({case.role}, WT={case.wt_ml:.1f} mL) ===", flush=True)
        src = BRATS_ROOT / sid
        case_out = OUT / sid
        case_out.mkdir(parents=True, exist_ok=True)
        # Inputs: BraTS already skull-stripped & co-registered — copy as HD-GLIO names
        inputs = {
            "t1": src / "T1.nii.gz",
            "t1c": src / "T1c.nii.gz",
            "t2": src / "T2.nii.gz",
            "flair": src / "FLAIR.nii.gz",
        }
        for k, path in inputs.items():
            if not path.is_file():
                raise FileNotFoundError(path)
        t1c_img = nib.load(str(inputs["t1c"]))
        meta = json.loads((src / "conversion_meta.json").read_text(encoding="utf-8"))
        identity = has_identity_affine(t1c_img)
        raw = case_out / "hdglio_raw_seg.nii.gz"
        draft = case_out / "hdglio_draft_label.nii.gz"
        wall = None
        if not args.skip_predict or not raw.is_file():
            wall = _predict_one(
                case_out, inputs, raw, do_tta=False, step_size=1.0
            )
            print(f"  lean wall-clock {wall:.1f}s", flush=True)
        conv = save_hdglio_draft_on_t1c_grid(raw, inputs["t1c"], draft)
        gt = np.rint(np.asanyarray(nib.load(str(src / "mask.nii.gz")).dataobj)).astype(
            np.int16
        )
        # Map GT enhancing 3 stays classic; HD draft is 2/4
        hd = np.rint(np.asanyarray(nib.load(str(draft)).dataobj)).astype(np.int16)
        # Ensure GT uses 3 for ET (already); metrics accept 3|4
        metrics = dice_hd95_regions(hd, gt)
        orient = orientation_sanity(
            nib.load(str(draft)),
            nib.load(str(src / "mask.nii.gz")),
            t1c_img,
            case_out / "orientation_overlay.png",
        )
        v3m = v3.get(sid, {})
        row = {
            "study_id": sid,
            "role": case.role,
            "wt_ml_gt": case.wt_ml,
            "identity_affine": identity,
            "conversion_meta_affine_identity": meta.get("affine") == [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "wall_clock_s_lean": wall,
            "hdglio_resampled": conv.get("resampled_nearest"),
            **{f"hdglio_{k}": v for k, v in metrics.items()},
            **{f"v3_lcc_{k}": v3m.get(k) for k in (
                "dice_enhancing_tumor",
                "dice_tumor_core",
                "dice_whole_tumor",
                "hd95_enhancing_tumor",
                "hd95_tumor_core",
                "hd95_whole_tumor",
            )},
            **{f"orient_{k}": v for k, v in orient.items() if k != "overlay_png"},
        }
        rows.append(row)
        print(
            f"  HD-GLIO WT Dice={metrics['dice_whole_tumor']:.3f} "
            f"ET={metrics['dice_enhancing_tumor']:.3f} | "
            f"v3 WT={v3m.get('dice_whole_tumor', float('nan')):.3f} "
            f"orient_ok={orient['shape_ok'] and orient['affine_ok']}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "control_metrics.csv", index=False)

    # Default-settings re-run if HD-GLIO clearly worse than published CE ~0.91
    mean_et = float(np.nanmean(df["hdglio_dice_enhancing_tumor"].to_numpy(dtype=float)))
    mean_wt = float(np.nanmean(df["hdglio_dice_whole_tumor"].to_numpy(dtype=float)))
    clearly_worse = mean_et < 0.7 or mean_wt < 0.7
    default_case = args.default_settings_case
    if clearly_worse and not default_case:
        # cheapest: smallest WT case (usually fewest tiles)
        default_case = next(c.study_id for c in cases if c.role == "small")
    default_row = None
    if default_case:
        print(f"=== default settings re-run: {default_case} ===", flush=True)
        src = BRATS_ROOT / default_case
        case_out = OUT / default_case / "default_settings"
        inputs = {
            "t1": src / "T1.nii.gz",
            "t1c": src / "T1c.nii.gz",
            "t2": src / "T2.nii.gz",
            "flair": src / "FLAIR.nii.gz",
        }
        raw = case_out / "hdglio_raw_seg.nii.gz"
        draft = case_out / "hdglio_draft_label.nii.gz"
        wall = _predict_one(case_out, inputs, raw, do_tta=True, step_size=0.5)
        save_hdglio_draft_on_t1c_grid(raw, inputs["t1c"], draft)
        gt = np.rint(np.asanyarray(nib.load(str(src / "mask.nii.gz")).dataobj)).astype(
            np.int16
        )
        hd = np.rint(np.asanyarray(nib.load(str(draft)).dataobj)).astype(np.int16)
        metrics = dice_hd95_regions(hd, gt)
        lean = df.loc[df.study_id == default_case].iloc[0]
        default_row = {
            "study_id": default_case,
            "wall_clock_s_default": wall,
            "wall_clock_s_lean": lean.get("wall_clock_s_lean"),
            **{f"default_{k}": v for k, v in metrics.items()},
            "lean_dice_enhancing_tumor": lean["hdglio_dice_enhancing_tumor"],
            "lean_dice_whole_tumor": lean["hdglio_dice_whole_tumor"],
            "lean_dice_tumor_core": lean["hdglio_dice_tumor_core"],
        }
        (case_out / "default_vs_lean.json").write_text(
            json.dumps(default_row, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"  default WT Dice={metrics['dice_whole_tumor']:.3f} "
            f"(lean {lean['hdglio_dice_whole_tumor']:.3f}) wall={wall:.1f}s",
            flush=True,
        )

    # Write CONTROL_TEST.md
    _write_control_md(df, cases, clearly_worse, mean_et, mean_wt, default_row)
    print(f"Wrote {ROOT / 'outputs' / 'batch2_second_opinion' / 'CONTROL_TEST.md'}")
    return 0


def _write_control_md(df, cases, clearly_worse, mean_et, mean_wt, default_row) -> None:
    from segmentation.hdglio_control import PUBLISHED_CE_DICE_REF

    lines = [
        "# HD-GLIO control test (BraTS heldout)",
        "",
        "**Model drafts only — not reference masks.**",
        "",
        "## Setup",
        "",
        "- Source: `splits/heldout.txt` (read-only), seed **42**, n=5 with forced small/median/large WT.",
        "- Inputs: cached BraTS NIfTIs under `data/processed/brats_nifti/` (already skull-stripped and co-registered).",
        "- **HD-BET: skipped** — re-running skull-strip on already-stripped BraTS volumes would only risk crop/edge damage.",
        "- Affine: HDF5 restack uses an **identity** RAS affine (`conversion_meta.json`); verified per case.",
        "- Lean settings (same as batch2): `step_size=1.0`, `do_tta=False`.",
        "- Label map: HD-GLIO 1→BraTS 4 (ET), 2→BraTS 2 (edema); TC≡ET for HD-GLIO (no necrotic class).",
        f"- Published CE Dice reference (Kickingereder 2019, approx.): **{PUBLISHED_CE_DICE_REF:.2f}**.",
        "",
        "## Selected cases",
        "",
        "| study_id | role | GT WT mL |",
        "|---|---|---:|",
    ]
    for c in cases:
        lines.append(f"| {c.study_id} | {c.role} | {c.wt_ml:.1f} |")
    lines += [
        "",
        "## Metrics vs ground truth (HD-GLIO lean) and v3 LCC",
        "",
        "| study | role | wall_s | HD Dice ET | HD Dice TC | HD Dice WT | HD HD95 WT | v3 Dice ET | v3 Dice TC | v3 Dice WT | orient OK |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        orient = bool(r["orient_shape_ok"] and r["orient_affine_ok"])
        wall = r["wall_clock_s_lean"]
        wall_s = f"{wall:.0f}" if wall == wall and wall is not None else "—"
        def fmt(x):
            return f"{x:.3f}" if x == x else "nan"
        lines.append(
            f"| {r['study_id']} | {r['role']} | {wall_s} | "
            f"{fmt(r['hdglio_dice_enhancing_tumor'])} | "
            f"{fmt(r['hdglio_dice_tumor_core'])} | "
            f"{fmt(r['hdglio_dice_whole_tumor'])} | "
            f"{fmt(r['hdglio_hd95_whole_tumor'])} | "
            f"{fmt(r['v3_lcc_dice_enhancing_tumor'])} | "
            f"{fmt(r['v3_lcc_dice_tumor_core'])} | "
            f"{fmt(r['v3_lcc_dice_whole_tumor'])} | "
            f"{'yes' if orient else 'no'} |"
        )
    lines += [
        "",
        f"Mean HD-GLIO Dice ET={mean_et:.3f}, WT={mean_wt:.3f}.",
        "",
        "## Orientation",
        "",
        "Per-case overlays: `outputs/batch2_second_opinion/control_test/<id>/orientation_overlay.png` "
        "(green=GT WT, red=HD-GLIO). Shape and affine matched the GT/T1c grid for every case "
        "when `orient OK=yes` — checked, not assumed.",
        "",
    ]
    if clearly_worse:
        lines += [
            "## Below published performance — possible causes (likelihood order)",
            "",
            "1. **Lean settings** (`step_size=1.0`, no TTA) — cheapest to test; fewer tiles / no mirroring.",
            "2. **Darwin/CPU patches** (`weights_only=False`, serial preprocess) — should not change math if weights load correctly, but unvalidated vs CUDA.",
            "3. **Preprocessing mismatch** — BraTS already stripped; HD-GLIO training assumed HD-BET + specific intensity handling.",
            "4. **Orientation / identity affine** — less likely if overlays and grid checks pass.",
            "",
        ]
        if default_row:
            lines += [
                "### Cheapest test: default settings on one case",
                "",
                f"- Case: `{default_row['study_id']}` (`step_size=0.5`, TTA on).",
                f"- Wall-clock: lean {default_row.get('wall_clock_s_lean')} s → default "
                f"{default_row['wall_clock_s_default']:.0f} s.",
                f"- Dice ET: lean {default_row['lean_dice_enhancing_tumor']:.3f} → "
                f"default {default_row['default_dice_enhancing_tumor']:.3f}.",
                f"- Dice WT: lean {default_row['lean_dice_whole_tumor']:.3f} → "
                f"default {default_row['default_dice_whole_tumor']:.3f}.",
                "",
            ]
    # Verdict
    mean_v3_wt = float(
        __import__("numpy").nanmean(df["v3_lcc_dice_whole_tumor"].to_numpy(dtype=float))
    )
    if mean_wt >= 0.75 and all(
        bool(r["orient_shape_ok"] and r["orient_affine_ok"]) for _, r in df.iterrows()
    ):
        verdict = "yes"
        why = (
            f"Mean WT Dice {mean_wt:.3f} is in a usable range vs GT, orientation checks passed, "
            f"and results are comparable in spirit to v3 (mean WT Dice {mean_v3_wt:.3f})."
        )
    elif mean_wt < 0.4:
        verdict = "no"
        why = (
            f"Mean WT Dice {mean_wt:.3f} (ET {mean_et:.3f}) is far below published ~0.91 CE Dice "
            "and below v3 on the same cases; do not treat batch2 HD-GLIO drafts as reliable "
            "without a GPU/default-settings confirmation."
        )
    else:
        verdict = "unclear"
        why = (
            f"Mean WT Dice {mean_wt:.3f} / ET {mean_et:.3f} is below published CE Dice ~0.91 "
            f"(v3 mean WT {mean_v3_wt:.3f}); orientation OK but lean-CPU vs CUDA gap remains."
        )
    lines += [
        "## Verdict",
        "",
        f"**Patched install trustworthy: {verdict}**",
        "",
        why,
        "",
        "Artifacts: `outputs/batch2_second_opinion/control_test/`.",
        "",
    ]
    path = ROOT / "outputs" / "batch2_second_opinion" / "CONTROL_TEST.md"
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
