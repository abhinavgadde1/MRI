#!/usr/bin/env python3
"""Run HD-GLIO second-opinion segmentation for batch2 (isolated .venv-hdglio).

Invoke with the *hdglio* interpreter only::

    unset VIRTUAL_ENV PYTHONPATH
    .venv-hdglio/bin/python scripts/run_hdglio_second_opinion.py

Writes under ``outputs/batch2_second_opinion/<study_id>/`` only.
Does not touch v3 drafts, corrected_label, checkpoints, or heldout splits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Source path for helpers; avoid importing package __init__ (pulls MONAI).
sys.path.insert(0, str(ROOT / "src"))


def _load_hdglio_helpers():
    """Import hdglio_labels without executing segmentation/__init__.py."""
    import importlib.util
    import types

    if "segmentation" not in sys.modules:
        m = types.ModuleType("segmentation")
        m.__path__ = [str(ROOT / "src" / "segmentation")]
        sys.modules["segmentation"] = m

    name = "segmentation.hdglio_labels"
    path = ROOT / "src" / "segmentation" / "hdglio_labels.py"
    if name in sys.modules and hasattr(sys.modules[name], "compare_v3_hdglio"):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OUT_ROOT = ROOT / "outputs" / "batch2_second_opinion"
BATCH2_ROOT = ROOT / "data" / "processed" / "real_patients_batch2"
LABELING = ROOT / "outputs" / "batch2_labeling"
V3_PREVIEW = ROOT / "outputs" / "batch2_v3_preview"


def _brain_mask_registered(study_dir: Path, target_shape: tuple[int, ...]):
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import zoom

    mp = study_dir / "03_skull" / "T1_brainmask.nii.gz"
    if not mp.is_file():
        mp = study_dir / "03_skull" / "T1c_brainmask.nii.gz"
    if not mp.is_file():
        return np.ones(target_shape, dtype=bool)
    m = np.asanyarray(nib.load(str(mp)).dataobj) > 0
    if m.shape != target_shape:
        factors = [target_shape[i] / m.shape[i] for i in range(3)]
        m = zoom(m.astype(np.float32), factors, order=0) > 0.5
    return m


def prepare_hdglio_inputs(study_id: str, work: Path) -> dict[str, Path]:
    """Brain-mask registered 1 mm volumes (non-brain = 0)."""
    import nibabel as nib
    import numpy as np

    study_dir = BATCH2_ROOT / study_id
    reg = study_dir / "04_registered_1mm"
    mapping = {
        "t1": reg / "T1_1mm.nii.gz",
        "t1c": reg / "T1c_registered_1mm.nii.gz",
        "t2": reg / "T2_registered_1mm.nii.gz",
        "flair": reg / "FLAIR_registered_1mm.nii.gz",
    }
    for k, p in mapping.items():
        if not p.is_file():
            raise FileNotFoundError(f"{study_id}: missing {p}")
    t1_img = nib.load(str(mapping["t1"]))
    mask = _brain_mask_registered(study_dir, t1_img.shape)
    work.mkdir(parents=True, exist_ok=True)
    out_paths = {}
    for key, src in mapping.items():
        img = nib.load(str(src))
        data = np.asanyarray(img.dataobj).astype(np.float32) * mask.astype(np.float32)
        out = work / f"{key}.nii.gz"
        nib.save(nib.Nifti1Image(data, t1_img.affine, t1_img.header), str(out))
        out_paths[key] = out
    return out_paths


def run_hdglio(
    inputs: dict[str, Path],
    out_seg: Path,
    *,
    do_tta: bool = False,
    step_size: float = 1.0,
) -> Path:
    """Call HD-GLIO via nnUNet predict_cases (CPU-safe patches in .venv-hdglio).

    Default ``step_size=1.0`` and ``do_tta=False`` keep Mac CPU RSS lean (~8 tiles).
    Official defaults (0.5 + TTA) thrash on Apple Silicon without CUDA.
    """
    from hd_glio.paths import folder_with_parameter_files
    from hd_glio.setup_hd_glio import maybe_download_weights
    from nnunet.inference.predict import predict_cases

    maybe_download_weights()
    out_seg.parent.mkdir(parents=True, exist_ok=True)
    predict_cases(
        folder_with_parameter_files,
        [[str(inputs["t1"]), str(inputs["t1c"]), str(inputs["t2"]), str(inputs["flair"])]],
        [str(out_seg)],
        (0,),
        False,
        1,
        1,
        None,
        do_tta,
        False,
        True,
        step_size=step_size,
        checkpoint_name="model_final_checkpoint",
    )
    if not out_seg.is_file():
        raise RuntimeError(f"HD-GLIO did not write {out_seg}")
    return out_seg


def t1c_grid_path(study_id: str) -> Path:
    # Prefer labeling package T1c (same grid as v3 draft)
    p = LABELING / study_id / "T1c.nii.gz"
    if p.is_file():
        return p
    p = BATCH2_ROOT / study_id / "04_registered_1mm" / "T1c_registered_1mm.nii.gz"
    if p.is_file():
        return p
    raise FileNotFoundError(f"No T1c grid for {study_id}")


def v3_draft_path(study_id: str) -> Path | None:
    for cand in (
        LABELING / study_id / "v3_draft_label.nii.gz",
        V3_PREVIEW / study_id / "v3_lcc_label.nii.gz",
        V3_PREVIEW / study_id / "label.nii.gz",
    ):
        if cand.is_file():
            return cand
    # search preview folder
    prev = V3_PREVIEW / study_id
    if prev.is_dir():
        hits = list(prev.glob("*label*.nii.gz"))
        if hits:
            return hits[0]
    return None


def studies_to_run(args) -> list[str]:
    summary = json.loads(
        (ROOT / "outputs" / "batch2_native_t1c" / "summary.json").read_text(encoding="utf-8")
    )
    ids = list(summary.get("native_ids", []))
    # Optionally add QC-ready studies with native T1c (native_ids already)
    if args.study:
        ids = [s for s in ids if s in set(args.study)]
    return sorted(ids)


def main(argv: list[str] | None = None) -> int:
    # Refuse to run under main .venv accidentally (resolve() follows brew symlink)
    prefix = Path(sys.prefix).resolve()
    if ".venv-hdglio" not in str(prefix):
        print(
            f"ERROR: must run with .venv-hdglio python (sys.prefix={prefix}).\n"
            "  unset VIRTUAL_ENV PYTHONPATH\n"
            "  .venv-hdglio/bin/python scripts/run_hdglio_second_opinion.py",
            file=sys.stderr,
        )
        return 2

    p = argparse.ArgumentParser()
    p.add_argument("--study", nargs="*", default=None)
    p.add_argument("--tta", action="store_true", help="Enable test-time augmentation (slow)")
    p.add_argument(
        "--step-size",
        type=float,
        default=1.0,
        help="Sliding-window step (1.0=lean CPU; 0.5=official denser)",
    )
    p.add_argument("--skip-predict", action="store_true", help="Only convert existing raw segs")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prediction when hdglio_raw_seg.nii.gz already exists",
    )
    args = p.parse_args(argv)

    helpers = _load_hdglio_helpers()
    DRAFT_DISCLAIMER = helpers.DRAFT_DISCLAIMER
    compare_v3_hdglio = helpers.compare_v3_hdglio
    save_hdglio_draft_on_t1c_grid = helpers.save_hdglio_draft_on_t1c_grid
    import nibabel as nib
    import numpy as np
    import pandas as pd

    ids = studies_to_run(args)
    print(f"Studies: {ids}")
    print(DRAFT_DISCLAIMER)
    agreement_rows = []
    meta = {"disclaimer": DRAFT_DISCLAIMER, "studies": {}}

    for sid in ids:
        print(f"=== {sid} ===", flush=True)
        study_out = OUT_ROOT / sid
        study_out.mkdir(parents=True, exist_ok=True)
        work = study_out / "hdglio_inputs"
        raw_seg = study_out / "hdglio_raw_seg.nii.gz"
        draft = study_out / "hdglio_draft_label.nii.gz"
        readme = study_out / "README.md"
        readme.write_text(
            f"# {sid} second opinion\n\n"
            f"{DRAFT_DISCLAIMER}\n\n"
            "Dark SWI signal inside a contour is a **flag for review**, not a diagnosis.\n"
            "v3 drafts and corrected_label files are left untouched.\n",
            encoding="utf-8",
        )

        if args.skip_predict:
            if not raw_seg.is_file():
                print(f"  SKIP: no raw seg at {raw_seg}", flush=True)
                continue
        elif args.skip_existing and raw_seg.is_file():
            print(f"  reuse existing {raw_seg.name}", flush=True)
        else:
            inputs = prepare_hdglio_inputs(sid, work)
            run_hdglio(
                inputs,
                raw_seg,
                do_tta=args.tta,
                step_size=args.step_size,
            )

        t1c = t1c_grid_path(sid)
        conv = save_hdglio_draft_on_t1c_grid(raw_seg, t1c, draft)
        meta["studies"][sid] = conv

        v3p = v3_draft_path(sid)
        if v3p is not None and draft.is_file():
            v3 = np.rint(np.asanyarray(nib.load(str(v3p)).dataobj)).astype(np.int16)
            hd = np.rint(np.asanyarray(nib.load(str(draft)).dataobj)).astype(np.int16)
            t1c_img = nib.load(str(t1c))
            if v3.shape != hd.shape:
                print(f"  WARN: v3 shape {v3.shape} != hd {hd.shape}; skip compare")
            else:
                spacing = tuple(float(x) for x in t1c_img.header.get_zooms()[:3])
                rows = compare_v3_hdglio(sid, v3, hd, spacing)  # type: ignore[arg-type]
                agreement_rows.extend([r.__dict__ for r in rows])
                wt = next(r for r in rows if r.region == "WT")
                print(
                    f"  WT Dice={wt.dice:.3f} v3={wt.v3_ml:.2f}mL hd={wt.hdglio_ml:.2f}mL "
                    f"class={wt.agreement_class}",
                    flush=True,
                )

    if agreement_rows:
        pd.DataFrame(agreement_rows).to_csv(OUT_ROOT / "agreement.csv", index=False)
    (OUT_ROOT / "hdglio_run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    # Harden isolation
    os.environ.pop("PYTHONPATH", None)
    raise SystemExit(main())
