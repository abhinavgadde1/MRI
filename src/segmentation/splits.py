"""Reproducible BraTS train / held-out subject splits."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from config import load_config, PROJECT_ROOT
from segmentation.dataset import build_brats_file_list, split_train_val

logger = logging.getLogger(__name__)


def subject_ids(records: Sequence[dict[str, Any]]) -> list[str]:
    return [str(r["subject_id"]) for r in records]


def write_id_list(path: str | Path, ids: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    return path


def read_id_list(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def apply_max_cases_subset(
    train_files: list[dict[str, Any]],
    val_files: list[dict[str, Any]],
    max_cases: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match ``train.train_model`` max_cases truncation."""
    n_total = max(1, max_cases)
    n_val = max(1, min(len(val_files), max(1, n_total // 5)))
    n_train = max(1, n_total - n_val)
    return train_files[:n_train], val_files[:n_val]


def recover_training_cases_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Replay the train/val subset used for a checkpoint from ``train_config``.

    Returns
    -------
    train_ids:
        Subjects that received gradient updates.
    run_val_ids:
        Subjects used as validation during that training run (checkpoint selection).
    meta:
        Relevant config fields.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "train_config" not in ckpt:
        raise KeyError(f"Checkpoint missing train_config: {checkpoint_path}")

    cfg = ckpt["train_config"]
    seed = int(cfg.get("seed", 42))
    val_fraction = float(cfg.get("val_fraction", 0.2))
    max_cases = cfg.get("max_cases")

    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    records = build_brats_file_list(brats_nifti_root)
    train_files, val_files = split_train_val(records, val_fraction=val_fraction, seed=seed)
    if max_cases is not None:
        train_files, val_files = apply_max_cases_subset(train_files, val_files, int(max_cases))

    meta = {
        "seed": seed,
        "val_fraction": val_fraction,
        "max_cases": max_cases,
        "n_train": len(train_files),
        "n_run_val": len(val_files),
    }
    return subject_ids(train_files), subject_ids(val_files), meta


def create_heldout_split(
    *,
    brats_nifti_root: str | Path | None = None,
    train_ids: Sequence[str],
    exclude_ids: Sequence[str] | None = None,
    n_heldout: int = 60,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Sample ``n_heldout`` subjects never in ``exclude_ids`` (default: train_ids)."""
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    all_ids = subject_ids(build_brats_file_list(brats_nifti_root))
    blocked = set(exclude_ids if exclude_ids is not None else train_ids)
    pool = [sid for sid in all_ids if sid not in blocked]
    if len(pool) < n_heldout:
        raise ValueError(
            f"Need {n_heldout} held-out cases from {len(pool)} unused subjects "
            f"(blocked={len(blocked)}, total={len(all_ids)})"
        )
    rng = random.Random(seed)
    heldout = pool.copy()
    rng.shuffle(heldout)
    heldout = heldout[:n_heldout]
    return list(train_ids), heldout


def overlap_report(
    eval_csv: str | Path,
    train_ids: Sequence[str],
    run_val_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare eval CSV subjects to recovered training-run subjects."""
    df = pd.read_csv(eval_csv)
    eval_ids = set(df["subject_id"].astype(str))
    train_set = set(train_ids)
    run_val_set = set(run_val_ids or [])
    used_in_run = train_set | run_val_set
    return {
        "n_eval": len(eval_ids),
        "n_train": len(train_set),
        "n_run_val": len(run_val_set),
        "n_used_in_training_run": len(used_in_run),
        "overlap_eval_vs_train": len(eval_ids & train_set),
        "overlap_eval_vs_run_val": len(eval_ids & run_val_set),
        "overlap_eval_vs_training_run": len(eval_ids & used_in_run),
        "eval_ids_in_train": sorted(eval_ids & train_set),
        "eval_ids_in_run_val": sorted(eval_ids & run_val_set),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover train split and write held-out lists.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "brats_pretrain" / "best_model.pt",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "brats_eval_metrics.csv",
    )
    parser.add_argument("--n-heldout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    train_ids, run_val_ids, meta = recover_training_cases_from_checkpoint(args.checkpoint)
    train_path = write_id_list(
        PROJECT_ROOT / "checkpoints" / "brats_pretrain" / "train_cases.txt",
        train_ids,
    )
    write_id_list(
        PROJECT_ROOT / "checkpoints" / "brats_pretrain" / "train_run_val_cases.txt",
        run_val_ids,
    )

    report = overlap_report(args.eval_csv, train_ids, run_val_ids)
    print("=== Recovered training subset ===")
    print(meta)
    print(f"Wrote {train_path} ({len(train_ids)} train IDs)")
    print(
        "Overlap eval CSV vs train (gradient) cases:",
        report["overlap_eval_vs_train"],
    )
    print(
        "Overlap eval CSV vs training-run val cases:",
        report["overlap_eval_vs_run_val"],
    )
    print(
        "Overlap eval CSV vs any case used in training run:",
        report["overlap_eval_vs_training_run"],
    )

    # Exclude train + training-run val so held-out never saw the model during training.
    train_out, heldout = create_heldout_split(
        train_ids=train_ids,
        exclude_ids=train_ids + run_val_ids,
        n_heldout=args.n_heldout,
        seed=args.seed,
    )
    splits_dir = PROJECT_ROOT / "splits"
    write_id_list(splits_dir / "train.txt", train_out)
    write_id_list(splits_dir / "heldout.txt", heldout)
    print(f"Wrote {splits_dir / 'train.txt'} ({len(train_out)})")
    print(f"Wrote {splits_dir / 'heldout.txt'} ({len(heldout)})")
    assert set(train_out).isdisjoint(heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
