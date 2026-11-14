"""Patient-grouped cross-validation fine-tuning for the batch2 cohort.

Default: leave-one-patient-out when ≤12 labeled patients, else 5-fold grouped
by patient. Each fold fine-tunes from v3, evaluates baseline vs fine-tuned on
held-out patients, and runs a BraTS forgetting check on a fixed held-out subset.

Reference masks are single-annotator corrections from a v3 draft — **not**
clinical ground truth. That disclaimer is stamped on every report.

Smoke path: ``--dry-run`` (draft labels, 1 fold / 1 epoch / 2 cases).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from config import PROJECT_ROOT, load_config
from device import get_device
from segmentation.batch2_augment import DomainAugmentFlags, log_augment_flags
from segmentation.batch2_dataset import (
    REFERENCE_MASK_DISCLAIMER,
    REGION_SHORT,
    build_batch2_file_list,
    make_patient_folds,
)
from segmentation.batch2_finetune import (
    DEFAULT_V3_CHECKPOINT,
    Batch2FinetuneConfig,
    assert_safe_checkpoint_output_dir,
    finetune_batch2,
    load_brats_replay_pool,
)
from segmentation.dataset import get_val_transforms
from segmentation.evaluate import _evaluate_case, load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS, brats_label_to_regions
from segmentation.postprocess import apply_channel_postprocess
from segmentation.report_metrics import _bootstrap_ci, _bootstrap_mean_diff_ci
from segmentation.splits import read_id_list
from segmentation.train import post_transforms
from validation.ellipsoid import (
    diameters_from_binary,
    ellipsoid_volume_cm3,
    voxel_volume_ml_from_binary,
)

logger = logging.getLogger(__name__)

DRY_RUN_STAMP = "DRY RUN - draft labels, not reportable"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "batch2_finetune"
DRYRUN_RESULTS_DIR = PROJECT_ROOT / "results" / "batch2_finetune_DRYRUN"
FINAL_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "batch2_finetune_final"
HELDOUT_LIST = PROJECT_ROOT / "splits" / "heldout.txt"
FORGETTING_N = 20
FORGETTING_SEED = 42

PER_CASE_FIELDS = [
    "fold",
    "study_id",
    "model",
    "region",
    "dice",
    "hd95",
    "ref_ml",
    "pred_ml",
    "abs_err_ml",
    "pct_err",
    "abc_over_2_ml",
]


def _peak_rss_mb() -> float:
    """Peak resident set size in MiB (macOS: ru_maxrss is bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    if rss > 1e7:  # clearly bytes
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def fixed_heldout_subset(
    *,
    n: int = FORGETTING_N,
    seed: int = FORGETTING_SEED,
    heldout_path: Path | None = None,
    brats_nifti_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Deterministic n-case subset of ``splits/heldout.txt`` (read-only)."""
    from segmentation.dataset import subject_to_datadict

    path = heldout_path or HELDOUT_LIST
    root = Path(brats_nifti_root) if brats_nifti_root else load_config().paths.brats_nifti
    ids = read_id_list(path)
    rng = np.random.default_rng(seed)
    if len(ids) <= n:
        chosen = list(ids)
    else:
        idx = rng.choice(len(ids), size=n, replace=False)
        chosen = [ids[int(i)] for i in sorted(idx)]
    records = []
    for sid in chosen:
        records.append(subject_to_datadict(root / sid))
    return records


@torch.no_grad()
def evaluate_case_detailed(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    postprocess: str = "lcc",
) -> list[dict[str, Any]]:
    """Per-region Dice/HD95 + volume errors + ABC/2 for one case."""
    post_pred, post_label = post_transforms()
    use_amp = device.type == "cuda"
    metrics = _evaluate_case(
        model,
        batch,
        device,
        roi_size,
        use_amp,
        post_pred,
        post_label,
        postprocess=postprocess,  # type: ignore[arg-type]
        pixdim=pixdim,
    )

    # Volumes from the same post-processed prediction as metrics.
    inputs = batch["image"].to(device=device, dtype=torch.float32)
    labels = brats_label_to_regions(batch["label"].to(device))
    logits = sliding_window_inference(
        inputs, roi_size=roi_size, sw_batch_size=1, predictor=model
    )
    from monai.data import decollate_batch

    pred = post_pred(decollate_batch(logits)[0]).detach().float().cpu().numpy()
    label = post_label(decollate_batch(labels)[0]).detach().float().cpu().numpy()
    if postprocess != "raw":
        pred = apply_channel_postprocess(pred.astype(np.float32), mode=postprocess)  # type: ignore[arg-type]

    rows: list[dict[str, Any]] = []
    study_id = batch.get("subject_id", ["unknown"])
    if isinstance(study_id, (list, tuple)):
        study_id = study_id[0]
    for idx, region in enumerate(BRATS_REGIONS):
        short = REGION_SHORT[idx]
        gt = label[idx].astype(bool)
        pr = pred[idx].astype(bool)
        ref_ml = voxel_volume_ml_from_binary(gt, pixdim) if np.any(gt) else 0.0
        pred_ml = voxel_volume_ml_from_binary(pr, pixdim) if np.any(pr) else 0.0
        abs_err = abs(pred_ml - ref_ml)
        pct_err = (100.0 * abs_err / ref_ml) if ref_ml > 0 else float("nan")
        abc_ml = float("nan")
        if np.any(pr):
            try:
                diameters, _ = diameters_from_binary(pr, pixdim, method="radiologist")
                abc_ml = float(ellipsoid_volume_cm3(diameters, formula="abc_over_2"))
            except ValueError:
                abc_ml = float("nan")
        rows.append(
            {
                "study_id": str(study_id),
                "region": short,
                "dice": metrics.get(f"dice_{region}", float("nan")),
                "hd95": metrics.get(f"hd95_{region}", float("nan")),
                "ref_ml": ref_ml,
                "pred_ml": pred_ml,
                "abs_err_ml": abs_err,
                "pct_err": pct_err,
                "abc_over_2_ml": abc_ml,
            }
        )
    return rows


def evaluate_records(
    checkpoint: str | Path,
    records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    model_tag: str,
    fold: int | str,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[dict[str, Any]]:
    model = load_model_from_checkpoint(checkpoint, device)
    model.float().eval()
    loader = DataLoader(
        Dataset(data=list(records), transform=get_val_transforms(pixdim=pixdim)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    rows: list[dict[str, Any]] = []
    for batch in loader:
        for r in evaluate_case_detailed(
            model, batch, device, roi_size=roi_size, pixdim=pixdim, postprocess="lcc"
        ):
            r["fold"] = fold
            r["model"] = model_tag
            rows.append(r)
    return rows


def evaluate_forgetting(
    baseline_ckpt: str | Path,
    fold_ckpt: str | Path,
    heldout_records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    fold: int,
    roi_size: tuple[int, int, int] = (96, 96, 96),
) -> list[dict[str, Any]]:
    """Dice change (fold − baseline) on a fixed BraTS held-out subset."""
    base_rows = evaluate_records(
        baseline_ckpt,
        heldout_records,
        device=device,
        model_tag="v3_baseline",
        fold=fold,
        roi_size=roi_size,
    )
    ft_rows = evaluate_records(
        fold_ckpt,
        heldout_records,
        device=device,
        model_tag="finetuned",
        fold=fold,
        roi_size=roi_size,
    )
    # Index by (study_id, region)
    base_map = {(r["study_id"], r["region"]): r for r in base_rows}
    out: list[dict[str, Any]] = []
    for r in ft_rows:
        key = (r["study_id"], r["region"])
        b = base_map.get(key)
        if b is None:
            continue
        out.append(
            {
                "fold": fold,
                "study_id": r["study_id"],
                "region": r["region"],
                "dice_baseline": b["dice"],
                "dice_finetuned": r["dice"],
                "dice_delta": (
                    float(r["dice"]) - float(b["dice"])
                    if np.isfinite(r["dice"]) and np.isfinite(b["dice"])
                    else float("nan")
                ),
                "hd95_baseline": b["hd95"],
                "hd95_finetuned": r["hd95"],
            }
        )
    return out


def write_summary_md(
    path: Path,
    per_case: pd.DataFrame,
    *,
    dry_run: bool,
    n_patients: int,
    stamp: str,
) -> None:
    lines: list[str] = []
    lines.append(f"# Batch2 fine-tune CV summary")
    lines.append("")
    lines.append(f"**{REFERENCE_MASK_DISCLAIMER}**")
    lines.append("")
    if dry_run:
        lines.append(f"**{stamp}**")
        lines.append("")
    lines.append(
        f"n_patients={n_patients}. Bootstrap 95% CIs are wide when n is small; "
        "treat intervals as descriptive only."
    )
    lines.append("")

    for region in REGION_SHORT:
        sub = per_case[per_case["region"] == region]
        base = sub[sub["model"] == "v3_baseline"]
        ft = sub[sub["model"] == "finetuned"]
        lines.append(f"## {region}")
        lines.append("")
        for metric in ("dice", "hd95"):
            bvals = base[metric].to_numpy(dtype=float)
            fvals = ft[metric].to_numpy(dtype=float)
            # Align by study_id for paired diff.
            merged = base[["study_id", metric]].merge(
                ft[["study_id", metric]],
                on="study_id",
                suffixes=("_base", "_ft"),
            )
            a = merged[f"{metric}_base"].to_numpy(dtype=float)
            b = merged[f"{metric}_ft"].to_numpy(dtype=float)
            mean_diff, lo, hi = _bootstrap_mean_diff_ci(a, b, n_resamples=1000, seed=42)
            b_mean = float(np.nanmean(bvals)) if np.any(np.isfinite(bvals)) else float("nan")
            f_mean = float(np.nanmean(fvals)) if np.any(np.isfinite(fvals)) else float("nan")
            b_med = float(np.nanmedian(bvals)) if np.any(np.isfinite(bvals)) else float("nan")
            f_med = float(np.nanmedian(fvals)) if np.any(np.isfinite(fvals)) else float("nan")
            lines.append(
                f"- **{metric}**: baseline mean={b_mean:.4f} median={b_med:.4f}; "
                f"finetuned mean={f_mean:.4f} median={f_med:.4f}; "
                f"paired Δ(ft−base)={mean_diff:.4f} "
                f"bootstrap 95% CI [{lo:.4f}, {hi:.4f}] "
                f"(n_paired={int(np.sum(np.isfinite(a) & np.isfinite(b)))})"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cv(
    *,
    labeling_root: Path | None = None,
    results_dir: Path | None = None,
    pretrained: Path | None = None,
    device: str | None = None,
    seed: int = 42,
    max_epochs: int = 30,
    dry_run: bool = False,
    train_final: bool = False,
    resume: bool = False,
    domain_aug: DomainAugmentFlags | None = None,
    replay_batch2: int = 1,
    replay_brats: int = 1,
    fold_limit: int | None = None,
    dry_run_n_cases: int = 2,
    dry_run_n_brats: int = 5,
) -> Path:
    """Run patient-grouped CV (or dry-run smoke). Returns results directory."""
    stamp = DRY_RUN_STAMP if dry_run else REFERENCE_MASK_DISCLAIMER
    results_dir = Path(results_dir or (DRYRUN_RESULTS_DIR if dry_run else DEFAULT_RESULTS_DIR))
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = results_dir / "fold_checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    pretrained = Path(pretrained or DEFAULT_V3_CHECKPOINT)
    device_t = get_device(device)
    flags = domain_aug or DomainAugmentFlags(
        # Keep dry-run light: still exercise augs unless disabled by CLI.
        thick_slice=True,
        intensity=True,
        modality_dropout=True,
    )
    log_augment_flags(flags)

    records = build_batch2_file_list(
        labeling_root,
        use_draft_labels=dry_run,
    )
    if dry_run:
        if len(records) < dry_run_n_cases:
            # Fall back: accept whatever drafts exist.
            logger.warning(
                "dry-run requested %d cases but only %d drafts available",
                dry_run_n_cases,
                len(records),
            )
        records = records[:dry_run_n_cases]
        max_epochs = 1
        fold_limit = 1

    if len(records) < 2:
        raise RuntimeError(
            f"Need ≥2 labeled batch2 studies for CV, found {len(records)}. "
            f"(Looking for corrected_label.nii.gz"
            f"{' / v3_draft_label.nii.gz in dry-run' if dry_run else ''}.)"
        )

    folds = make_patient_folds(records, seed=seed)
    if fold_limit is not None:
        folds = folds[:fold_limit]

    brats_pool = load_brats_replay_pool()
    if dry_run:
        brats_pool = brats_pool[:dry_run_n_brats]

    # Forgetting subset: skip the heavy BraTS eval in dry-run (still write a stub).
    heldout_subset: list[dict[str, Any]] = []
    if not dry_run:
        heldout_subset = fixed_heldout_subset(n=FORGETTING_N, seed=FORGETTING_SEED)

    ft_cfg = Batch2FinetuneConfig(
        max_epochs=max_epochs,
        learning_rate=1e-5,
        seed=seed,
        domain_aug=flags,
        replay_batch2=replay_batch2,
        replay_brats=replay_brats,
        # Keep crop/roi at the production 96³ so dry-run epoch timing
        # is a useful lower-bound estimate for full CV (fewer cases/epochs).
        crop_size=(96, 96, 96),
        roi_size=(96, 96, 96),
        num_train_samples=1 if dry_run else 2,
        heartbeat_interval_s=60.0 if dry_run else 300.0,
    )

    cv_config = {
        "disclaimer": stamp,
        "reference_mask_note": REFERENCE_MASK_DISCLAIMER,
        "dry_run": dry_run,
        "seed": seed,
        "max_epochs": max_epochs,
        "pretrained": str(pretrained),
        "device": str(device_t),
        "domain_aug": flags.to_dict(),
        "domain_aug_enabled": flags.enabled_names(),
        "replay_batch2": replay_batch2,
        "replay_brats": replay_brats,
        "n_records": len(records),
        "n_folds": len(folds),
        "fold_mode": folds[0]["mode"] if folds else None,
        "folds": [
            {
                "fold": f["fold"],
                "mode": f["mode"],
                "train_patient_ids": f["train_patient_ids"],
                "held_patient_ids": f["held_patient_ids"],
                "train_study_ids": [r["subject_id"] for r in f["train"]],
                "held_study_ids": [r["subject_id"] for r in f["held"]],
            }
            for f in folds
        ],
        "forgetting_n": 0 if dry_run else FORGETTING_N,
        "forgetting_seed": FORGETTING_SEED,
    }
    (results_dir / "cv_config.json").write_text(
        json.dumps(cv_config, indent=2) + "\n", encoding="utf-8"
    )

    all_case_rows: list[dict[str, Any]] = []
    all_forget_rows: list[dict[str, Any]] = []
    epoch_seconds_all: list[float] = []
    t_cv0 = time.perf_counter()

    for fold in folds:
        fold_idx = int(fold["fold"])
        fold_out = assert_safe_checkpoint_output_dir(ckpt_root / f"fold_{fold_idx}")
        logger.info(
            "=== Fold %d (%s) train_patients=%s held=%s ===",
            fold_idx,
            fold["mode"],
            fold["train_patient_ids"],
            fold["held_patient_ids"],
        )
        # Val = train (tiny n); checkpoint selection still uses val Dice convention.
        best = finetune_batch2(
            train_files=fold["train"],
            val_files=fold["train"],
            output_dir=fold_out,
            pretrained_checkpoint=pretrained,
            brats_replay_pool=brats_pool,
            config=ft_cfg,
            device=str(device_t),
            resume=resume,
            max_brats_per_epoch=dry_run_n_brats if dry_run else None,
        )
        timings_path = fold_out / "epoch_timings.json"
        if timings_path.is_file():
            epoch_seconds_all.extend(
                json.loads(timings_path.read_text(encoding="utf-8")).get(
                    "epoch_seconds", []
                )
            )

        # Baseline v3 + fine-tuned on held-out only.
        all_case_rows.extend(
            evaluate_records(
                pretrained,
                fold["held"],
                device=device_t,
                model_tag="v3_baseline",
                fold=fold_idx,
                roi_size=ft_cfg.roi_size,
            )
        )
        all_case_rows.extend(
            evaluate_records(
                best,
                fold["held"],
                device=device_t,
                model_tag="finetuned",
                fold=fold_idx,
                roi_size=ft_cfg.roi_size,
            )
        )

        if heldout_subset:
            all_forget_rows.extend(
                evaluate_forgetting(
                    pretrained,
                    best,
                    heldout_subset,
                    device=device_t,
                    fold=fold_idx,
                    roi_size=ft_cfg.roi_size,
                )
            )
        elif dry_run:
            all_forget_rows.append(
                {
                    "fold": fold_idx,
                    "study_id": "DRYRUN_SKIPPED",
                    "region": "WT",
                    "dice_baseline": float("nan"),
                    "dice_finetuned": float("nan"),
                    "dice_delta": float("nan"),
                    "hd95_baseline": float("nan"),
                    "hd95_finetuned": float("nan"),
                    "note": DRY_RUN_STAMP,
                }
            )

    # Write per-case metrics.
    per_case_path = results_dir / "per_case_metrics.csv"
    with per_case_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_CASE_FIELDS + (["note"] if dry_run else []))
        writer.writeheader()
        for row in all_case_rows:
            out = {k: row.get(k, "") for k in PER_CASE_FIELDS}
            if dry_run:
                out["note"] = DRY_RUN_STAMP
            writer.writerow(out)

    forget_path = results_dir / "forgetting_check.csv"
    forget_df = pd.DataFrame(all_forget_rows)
    forget_df.to_csv(forget_path, index=False)

    per_case_df = pd.DataFrame(all_case_rows)
    summary_name = "summary_DRYRUN.md" if dry_run else "summary.md"
    write_summary_md(
        results_dir / summary_name,
        per_case_df,
        dry_run=dry_run,
        n_patients=len({r.get("patient_id", r["subject_id"]) for r in records}),
        stamp=stamp,
    )

    # Stamp every text artifact in dry-run.
    if dry_run:
        for path in results_dir.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".json", ".csv", ".log", ".txt"}:
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if DRY_RUN_STAMP not in text:
                    if path.suffix == ".json":
                        try:
                            obj = json.loads(text)
                            if isinstance(obj, dict):
                                obj["dry_run_stamp"] = DRY_RUN_STAMP
                                path.write_text(
                                    json.dumps(obj, indent=2) + "\n", encoding="utf-8"
                                )
                        except json.JSONDecodeError:
                            path.write_text(
                                text + f"\n# {DRY_RUN_STAMP}\n", encoding="utf-8"
                            )
                    else:
                        path.write_text(
                            f"# {DRY_RUN_STAMP}\n" + text, encoding="utf-8"
                        )

    wall = time.perf_counter() - t_cv0
    peak_mb = _peak_rss_mb()
    timing_report = {
        "disclaimer": stamp,
        "wall_clock_s": wall,
        "epoch_seconds": epoch_seconds_all,
        "mean_epoch_seconds": (
            float(np.mean(epoch_seconds_all)) if epoch_seconds_all else float("nan")
        ),
        "peak_rss_mb": peak_mb,
        "device": str(device_t),
        "dry_run": dry_run,
        "n_folds_run": len(folds),
        "max_epochs": max_epochs,
    }
    (results_dir / ("timing_DRYRUN.json" if dry_run else "timing.json")).write_text(
        json.dumps(timing_report, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "CV done | wall=%.1fs | mean_epoch=%.1fs | peak_rss=%.0f MiB | results=%s",
        wall,
        timing_report["mean_epoch_seconds"],
        peak_mb,
        results_dir,
    )

    if train_final and not dry_run:
        final_dir = assert_safe_checkpoint_output_dir(FINAL_CKPT_DIR)
        logger.info("Training final model on ALL labeled cases → %s", final_dir)
        finetune_batch2(
            train_files=records,
            val_files=records,
            output_dir=final_dir,
            pretrained_checkpoint=pretrained,
            brats_replay_pool=load_brats_replay_pool(),
            config=Batch2FinetuneConfig(
                max_epochs=max_epochs,
                learning_rate=1e-5,
                seed=seed,
                domain_aug=flags,
                replay_batch2=replay_batch2,
                replay_brats=replay_brats,
            ),
            device=str(device_t),
            resume=resume,
        )
        # Ensure default inference checkpoint is unchanged.
        default_ckpt = load_config().inference.checkpoint
        logger.info(
            "CLI/API default checkpoint remains %s (not overwritten)", default_ckpt
        )

    return results_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch2 patient-grouped CV fine-tune from v3. "
            f"{REFERENCE_MASK_DISCLAIMER}"
        )
    )
    parser.add_argument(
        "--labeling-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch2_labeling",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_V3_CHECKPOINT,
        help="Initialize from this checkpoint (default: v3 brats_scale_full)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--replay-batch2", type=int, default=1)
    parser.add_argument("--replay-brats", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--train-final",
        action="store_true",
        help="After CV, fine-tune on ALL labeled cases → checkpoints/batch2_finetune_final/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Smoke test: use v3_draft_label.nii.gz, 1 fold / 1 epoch / 2 cases + "
            "5 BraTS replay; write results/batch2_finetune_DRYRUN/ "
            f"({DRY_RUN_STAMP})"
        ),
    )
    parser.add_argument("--no-thick-slice", action="store_true")
    parser.add_argument("--no-intensity-aug", action="store_true")
    parser.add_argument("--no-modality-dropout", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    flags = DomainAugmentFlags(
        thick_slice=not args.no_thick_slice,
        intensity=not args.no_intensity_aug,
        modality_dropout=not args.no_modality_dropout,
    )
    run_cv(
        labeling_root=args.labeling_root,
        results_dir=args.results_dir,
        pretrained=args.pretrained,
        device=args.device,
        seed=args.seed,
        max_epochs=args.epochs,
        dry_run=args.dry_run,
        train_final=args.train_final,
        resume=args.resume,
        domain_aug=flags,
        replay_batch2=args.replay_batch2,
        replay_brats=args.replay_brats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
