"""Weakly supervised fine-tune of v3 using scan-level tumor / tumor-free labels.

Positives: ``splits/batch2_pos_neg/positives.txt`` (tumor somewhere in the scan).
Negatives: every negative row in ``splits/batch2_pos_neg/manifest.csv`` —
tumor-free status is **verbal and unverified** (radiologist-reported /
claimed_unverified). No pixel-level masks exist for real patients.

BraTS replay uses ``splits/scaleup/cfg_c_all_train.txt`` with full voxel masks
(never ``splits/heldout.txt``). Checkpoints go only under
``checkpoints/batch2_weaksup_fold_k/`` and ``checkpoints/batch2_weaksup_final/``;
``checkpoints/brats_scale_full`` is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, SpatialPadd
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from config import PROJECT_ROOT, load_config
from device import get_device
from preprocessing.h5_to_nifti import MODALITY_NAMES
from preprocessing.registration import REGISTERED_NAMES
from segmentation.batch2_augment import DomainAugmentFlags, DomainAugmentd, log_augment_flags
from segmentation.batch2_cv import evaluate_forgetting, fixed_heldout_subset
from segmentation.batch2_dataset import (
    POS_NEG_SPLIT_DIR,
    make_patient_folds,
    write_empty_label_like,
)
from segmentation.batch2_finetune import (
    DEFAULT_V3_CHECKPOINT,
    HELDOUT_LIST,
    assert_safe_checkpoint_output_dir,
    load_brats_replay_pool,
    sample_replay_mix,
)
from segmentation.batch2_weaksup_loss import (
    ET_CH,
    TC_CH,
    WT_CH,
    negative_scan_loss,
    positive_scan_loss,
    topk_mean,
    weaksup_batch_loss,
)
from segmentation.dataset import IMAGE_KEY, LABEL_KEY, get_train_transforms
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import BRATS_REGIONS, ModelName, brats_label_to_regions, build_brats_model
from segmentation.pos_neg_separation import best_threshold_sens_spec, bootstrap_auc_ci
from segmentation.postprocess import apply_channel_postprocess, enforce_brats_nesting, keep_largest_cc
from segmentation.pseudo_label import get_patient_inference_transforms
from segmentation.train import (
    TrainConfig,
    _HeartbeatWatcher,
    _append_heartbeat,
    _append_metrics_row,
    assert_loss_covers_all_channels,
    build_brats_loss,
    build_lr_scheduler,
    post_transforms,
)

logger = logging.getLogger(__name__)

# Re-export losses so callers can import from this module as specified.
__all__ = [
    "negative_scan_loss",
    "positive_scan_loss",
    "weaksup_batch_loss",
    "topk_mean",
    "run_weaksup_cv",
    "train_weaksup",
    "main",
]

DRY_RUN_STAMP = "DRY RUN"
TUMOR_FREE_WORDING = (
    "tumor-free status is verbal and unverified "
    "(radiologist-reported / claimed_unverified)"
)
LOCALIZATION_DISCLAIMER = (
    "These regions are unvalidated model proposals. No volume or outline "
    "accuracy can be claimed without reference masks."
)
FORGETTING_DICE_DROP_MAX = 0.03
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "batch2_weaksup"
DRYRUN_RESULTS_DIR = PROJECT_ROOT / "results" / "batch2_weaksup_DRYRUN"
FOLD_CKPT_ROOT = PROJECT_ROOT / "checkpoints"
FINAL_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "batch2_weaksup_final"
EMPTY_WEAK_LABEL_DIR = PROJECT_ROOT / "results" / "batch2_weaksup" / "empty_labels"
NEG_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients"
BATCH2_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "real_patients_batch2"


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    if rss > 1e7:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


@dataclass
class WeaksupConfig:
    """Hyperparameters for weakly supervised fine-tuning."""

    max_epochs: int = 15
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    seed: int = 42
    crop_size: tuple[int, int, int] = (96, 96, 96)
    roi_size: tuple[int, int, int] = (96, 96, 96)
    num_train_samples: int = 2
    model_name: ModelName = "segresnet"
    scheduler_type: str = "cosine_warmup"
    warmup_frac: float = 0.05
    cosine_eta_min: float = 1e-7
    grad_clip_max_norm: float = 1.0
    heartbeat_interval_s: float = 300.0
    amp: bool = True
    num_workers: int = 0
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    log_every_n_batches: int = 1
    replay_weak: int = 1
    replay_brats: int = 1
    topk: int = 200
    domain_aug: DomainAugmentFlags = field(
        default_factory=lambda: DomainAugmentFlags(
            thick_slice=True,
            intensity=True,
            modality_dropout=False,
            weaksup_t1c_hide=True,
            weaksup_t1c_hide_pos_prob=0.5,
            weaksup_t1c_hide_neg_always=True,
            inplane_rescale=True,
            blur=True,
            zscore_brain=True,
        )
    )

    def to_train_config(self) -> TrainConfig:
        return TrainConfig(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            seed=self.seed,
            crop_size=self.crop_size,
            roi_size=self.roi_size,
            num_train_samples=self.num_train_samples,
            model_name=self.model_name,
            scheduler_type=self.scheduler_type,
            warmup_frac=self.warmup_frac,
            cosine_eta_min=self.cosine_eta_min,
            grad_clip_max_norm=self.grad_clip_max_norm,
            heartbeat_interval_s=self.heartbeat_interval_s,
            amp=self.amp,
            num_workers=self.num_workers,
            pixdim=self.pixdim,
            log_every_n_batches=self.log_every_n_batches,
            cache_rate=0.0,
        )


def default_weaksup_aug_flags() -> DomainAugmentFlags:
    return DomainAugmentFlags(
        thick_slice=True,
        intensity=True,
        modality_dropout=False,
        weaksup_t1c_hide=True,
        weaksup_t1c_hide_pos_prob=0.5,
        weaksup_t1c_hide_neg_always=True,
        inplane_rescale=True,
        blur=True,
        zscore_brain=True,
    )


def build_weaksup_file_list(
    split_dir: str | Path | None = None,
    *,
    empty_label_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """All 19 positives + all 99 negatives from manifest (empty labels for plumbing).

    Does not modify ``data/processed/real_patients/`` image files or any draft /
    corrected label. Empty labels are written under ``results/batch2_weaksup/``.
    """
    split = Path(split_dir) if split_dir is not None else POS_NEG_SPLIT_DIR
    empty_root = Path(empty_label_dir) if empty_label_dir else EMPTY_WEAK_LABEL_DIR
    empty_root.mkdir(parents=True, exist_ok=True)

    manif = pd.read_csv(split / "manifest.csv")
    pos_ids = [
        ln.strip()
        for ln in (split / "positives.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    neg_ids = (
        manif.loc[manif["cohort"].astype(str) == "negative", "study_id"]
        .astype(str)
        .tolist()
    )
    meta = {str(r.study_id): r for _, r in manif.iterrows()}

    records: list[dict[str, Any]] = []
    for sid in pos_ids:
        reg = BATCH2_PROCESSED_ROOT / sid / "04_registered_1mm"
        if not reg.is_dir():
            logger.warning("Skip positive %s: missing registered dir", sid)
            continue
        img = {m: reg / REGISTERED_NAMES[m] for m in MODALITY_NAMES}
        if not all(p.is_file() for p in img.values()):
            logger.warning("Skip positive %s: missing modality", sid)
            continue
        empty = empty_root / f"{sid}_empty_label.nii.gz"
        write_empty_label_like(img["T1c"], empty)
        row = meta.get(sid)
        pid = str(row.patient_id) if row is not None else sid
        t1c = str(row.t1c_status) if row is not None else "unknown"
        records.append(
            {
                "subject_id": sid,
                "patient_id": pid,
                "study_id": sid,
                "is_negative": False,
                "is_native_t1c": t1c == "native",
                "t1c_status": t1c,
                "cohort": "positive",
                "source": "weak_real",
                "reg_dir": str(reg),
                IMAGE_KEY: [str(img[m]) for m in MODALITY_NAMES],
                LABEL_KEY: str(empty),
            }
        )

    for sid in neg_ids:
        reg = NEG_PROCESSED_ROOT / sid / "04_registered_1mm"
        img = {m: reg / REGISTERED_NAMES[m] for m in MODALITY_NAMES}
        if not all(p.is_file() for p in img.values()):
            logger.warning("Skip negative %s: missing registered modalities", sid)
            continue
        empty = empty_root / f"{sid}_empty_label.nii.gz"
        write_empty_label_like(img["T1c"], empty)
        row = meta.get(sid)
        pid = str(row.patient_id) if row is not None else sid.split("_")[0]
        t1c = str(row.t1c_status) if row is not None else "standin"
        records.append(
            {
                "subject_id": sid,
                "patient_id": pid,
                "study_id": sid,
                "is_negative": True,
                "is_native_t1c": t1c == "native",
                "t1c_status": t1c,
                "cohort": "negative",
                "source": "weak_real",
                "reg_dir": str(reg),
                "tumor_free_note": TUMOR_FREE_WORDING,
                IMAGE_KEY: [str(img[m]) for m in MODALITY_NAMES],
                LABEL_KEY: str(empty),
            }
        )
    n_pos = sum(1 for r in records if not r["is_negative"])
    n_neg = sum(1 for r in records if r["is_negative"])
    logger.info(
        "Weaksup file list: %d positives + %d negatives (%s)",
        n_pos,
        n_neg,
        TUMOR_FREE_WORDING,
    )
    return records


def sample_balanced_pos_neg(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    """Equal positives and negatives each epoch (never the raw ~1:5 ratio)."""
    pos = [dict(r) for r in records if not r.get("is_negative")]
    neg = [dict(r) for r in records if r.get("is_negative")]
    if not pos or not neg:
        raise ValueError("Need both positives and negatives for balanced sampling")
    rng = random.Random(seed + epoch * 9176)
    n = min(len(pos), len(neg))
    pos_s = rng.sample(pos, n) if len(pos) > n else list(pos)
    neg_s = rng.sample(neg, n) if len(neg) > n else list(neg)
    # If one side is smaller, upsample with replacement to match.
    if len(pos_s) < n:
        pos_s = list(pos_s) + rng.choices(pos, k=n - len(pos_s))
    if len(neg_s) < n:
        neg_s = list(neg_s) + rng.choices(neg, k=n - len(neg_s))
    out = pos_s + neg_s
    rng.shuffle(out)
    return out


def report_group_standardized_diffs(
    records: Sequence[dict[str, Any]],
    split_dir: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Standardized differences (pos vs neg) for shortcut-risk covariates."""
    split = split_dir or POS_NEG_SPLIT_DIR
    manif = pd.read_csv(split / "manifest.csv")
    ids = {r["study_id"] for r in records}
    sub = manif[manif["study_id"].astype(str).isin(ids)].copy()
    pos = sub[sub["cohort"] == "positive"]
    neg = sub[sub["cohort"] == "negative"]

    rows: list[dict[str, Any]] = []
    numeric = [
        "inplane_spacing_T1_mm",
        "brain_mask_fraction",
        "field_strength_t",
        "slice_thickness_T1_mm",
    ]
    for col in numeric:
        if col not in sub.columns:
            continue
        a = pd.to_numeric(pos[col], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(neg[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = float(np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0) or 1.0)
        d = float((a.mean() - b.mean()) / pooled)
        rows.append(
            {
                "variable": col,
                "pos_mean": float(a.mean()),
                "neg_mean": float(b.mean()),
                "std_diff": d,
                "note": TUMOR_FREE_WORDING,
            }
        )

    # T1c status
    if "t1c_status" in sub.columns:
        pos_n = float((pos["t1c_status"] == "native").mean()) if len(pos) else float("nan")
        neg_n = float((neg["t1c_status"] == "native").mean()) if len(neg) else float("nan")
        rows.append(
            {
                "variable": "t1c_native_fraction",
                "pos_mean": pos_n,
                "neg_mean": neg_n,
                "std_diff": float(pos_n - neg_n) if pos_n == pos_n else float("nan"),
                "note": TUMOR_FREE_WORDING,
            }
        )
    # Scanner model (top models as fraction of Ingenia among pos)
    if "model" in sub.columns:
        pos_ing = float(pos["model"].astype(str).str.contains("Ingenia", case=False).mean())
        neg_ing = float(neg["model"].astype(str).str.contains("Ingenia", case=False).mean())
        rows.append(
            {
                "variable": "scanner_Ingenia_fraction",
                "pos_mean": pos_ing,
                "neg_mean": neg_ing,
                "std_diff": float(pos_ing - neg_ing),
                "note": TUMOR_FREE_WORDING,
            }
        )

    # Intensity stats from a few registered T1 volumes (cheap proxy)
    def _t1_mean(reg_dir: str) -> float:
        p = Path(reg_dir) / REGISTERED_NAMES["T1"]
        if not p.is_file():
            return float("nan")
        data = np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)
        m = data > 0
        return float(data[m].mean()) if np.any(m) else float("nan")

    pos_int = np.array(
        [_t1_mean(r["reg_dir"]) for r in records if not r["is_negative"]], dtype=float
    )
    neg_int = np.array(
        [_t1_mean(r["reg_dir"]) for r in records if r["is_negative"]], dtype=float
    )
    pos_int = pos_int[np.isfinite(pos_int)]
    neg_int = neg_int[np.isfinite(neg_int)]
    if len(pos_int) >= 2 and len(neg_int) >= 2:
        pooled = float(
            np.sqrt((pos_int.std(ddof=1) ** 2 + neg_int.std(ddof=1) ** 2) / 2.0) or 1.0
        )
        rows.append(
            {
                "variable": "T1_mean_brain_intensity",
                "pos_mean": float(pos_int.mean()),
                "neg_mean": float(neg_int.mean()),
                "std_diff": float((pos_int.mean() - neg_int.mean()) / pooled),
                "note": TUMOR_FREE_WORDING,
            }
        )

    df = pd.DataFrame(rows)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Wrote group standardized diffs → %s", out_path)
        for _, r in df.iterrows():
            logger.info(
                "std_diff %s: %.3f (pos=%.4g neg=%.4g) | %s",
                r["variable"],
                r["std_diff"],
                r["pos_mean"],
                r["neg_mean"],
                TUMOR_FREE_WORDING,
            )
    return df


def _train_transforms_weaksup(config: WeaksupConfig) -> Compose:
    base = get_train_transforms(
        pixdim=config.pixdim,
        crop_size=config.crop_size,
        num_samples=config.num_train_samples,
    )
    transforms = list(base.transforms)
    # Real thick-slice exams can be <96 in one axis after 1 mm spacing; pad before crop.
    pad = SpatialPadd(keys=[IMAGE_KEY, LABEL_KEY], spatial_size=config.crop_size)
    inserted = False
    for i, t in enumerate(transforms):
        if t.__class__.__name__ == "RandCropByPosNegLabeld":
            transforms = transforms[:i] + [pad] + transforms[i:]
            inserted = True
            break
    if not inserted:
        transforms = [pad] + transforms
    aug = DomainAugmentd(
        keys=[IMAGE_KEY],
        label_key=LABEL_KEY,
        flags=config.domain_aug,
    )
    if transforms and transforms[-1].__class__.__name__ == "EnsureTyped":
        transforms = transforms[:-1] + [aug, transforms[-1]]
    else:
        transforms = transforms + [aug]
    return Compose(transforms)


def _batch_flag(batch: dict[str, Any], key: str, default: bool = False) -> torch.Tensor:
    raw = batch.get(key, default)
    if isinstance(raw, torch.Tensor):
        return raw.reshape(-1).bool()
    if isinstance(raw, (list, tuple)):
        return torch.tensor([bool(x) for x in raw], dtype=torch.bool)
    return torch.tensor([bool(raw)], dtype=torch.bool)


def _batch_is_brats(batch: dict[str, Any]) -> bool:
    src = batch.get("source", "weak_real")
    if isinstance(src, (list, tuple)):
        return any(str(s) == "brats_replay" for s in src)
    if isinstance(src, torch.Tensor):
        return False
    return str(src) == "brats_replay"


def _weaksup_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dicece_loss: Any,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    *,
    epoch: int,
    topk: int,
    log_every_n_batches: int,
    scheduler: Any | None,
    scheduler_step_mode: str,
    grad_clip_max_norm: float,
) -> tuple[float, float, float]:
    """Mixed epoch: DiceCE on BraTS, MIL weak loss on real scans."""
    model.train()
    total = 0.0
    weak_total = 0.0
    brats_total = 0.0
    n = 0
    n_weak = 0
    n_brats = 0
    n_total = len(loader)

    for batch in loader:
        inputs = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        is_brats = _batch_is_brats(batch)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            if is_brats:
                labels = brats_label_to_regions(batch["label"].to(device))
                loss = dicece_loss(outputs, labels)
                brats_total += float(loss.item())
                n_brats += 1
            else:
                is_neg = _batch_flag(batch, "is_negative").to(device)
                is_native = _batch_flag(batch, "is_native_t1c").to(device)
                loss = weaksup_batch_loss(
                    outputs, is_negative=is_neg, is_native_t1c=is_native, k=topk
                )
                weak_total += float(loss.item())
                n_weak += 1

        scaler.scale(loss).backward()
        if grad_clip_max_norm and grad_clip_max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(grad_clip_max_norm)
            )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scheduler_step_mode == "batch":
            scheduler.step()

        total += float(loss.item())
        n += 1
        if log_every_n_batches > 0 and (
            n == 1 or n % log_every_n_batches == 0 or n == n_total
        ):
            logger.info(
                "Epoch %d train batch %d/%d | loss=%.4f | %s | lr=%.2e",
                epoch + 1,
                n,
                n_total,
                float(loss.item()),
                "brats_DiceCE" if is_brats else "weak_MIL",
                optimizer.param_groups[0]["lr"],
            )

    return (
        total / max(n, 1),
        weak_total / max(n_weak, 1),
        brats_total / max(n_brats, 1),
    )


def train_weaksup(
    *,
    train_files: Sequence[dict[str, Any]],
    val_files: Sequence[dict[str, Any]] | None = None,
    output_dir: str | Path,
    pretrained_checkpoint: str | Path | None = None,
    brats_replay_pool: Sequence[dict[str, Any]] | None = None,
    config: WeaksupConfig | None = None,
    device: str | None = None,
    resume: bool = False,
    max_brats_per_epoch: int | None = None,
) -> Path:
    """Fine-tune from v3 with balanced weak loss + BraTS DiceCE replay."""
    config = config or WeaksupConfig()
    output_dir = assert_safe_checkpoint_output_dir(output_dir)
    if "brats_scale_full" in Path(output_dir).parts:
        raise ValueError("Refusing to write into brats_scale_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    pretrained = Path(pretrained_checkpoint or DEFAULT_V3_CHECKPOINT)
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")

    train_cfg = config.to_train_config()
    device_t = get_device(device)
    use_amp = config.amp and device_t.type == "cuda"
    scaler = torch.amp.GradScaler(device=device_t.type, enabled=use_amp)

    brats_pool = (
        list(brats_replay_pool) if brats_replay_pool is not None else load_brats_replay_pool()
    )
    # Safety: never include held-out BraTS ids.
    heldout = set()
    if HELDOUT_LIST.is_file():
        heldout = {
            ln.strip()
            for ln in HELDOUT_LIST.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
    brats_pool = [r for r in brats_pool if r.get("subject_id") not in heldout]
    val_files = list(val_files) if val_files else list(train_files)

    log_augment_flags(config.domain_aug)
    logger.info(
        "weaksup FT | %s | lr=%.2e | topk=%d | clip=%.1f | %s",
        TUMOR_FREE_WORDING,
        config.learning_rate,
        config.topk,
        config.grad_clip_max_norm,
        device_t,
    )

    (output_dir / "weaksup_meta.json").write_text(
        json.dumps(
            {
                "tumor_free_note": TUMOR_FREE_WORDING,
                "localization_disclaimer": LOCALIZATION_DISCLAIMER,
                "pretrained_checkpoint": str(pretrained),
                "train_subject_ids": [r["subject_id"] for r in train_files],
                "val_subject_ids": [r["subject_id"] for r in val_files],
                "n_pos_train": sum(1 for r in train_files if not r.get("is_negative")),
                "n_neg_train": sum(1 for r in train_files if r.get("is_negative")),
                "domain_aug": config.domain_aug.to_dict(),
                "domain_aug_enabled": config.domain_aug.enabled_names(),
                "config": asdict(config),
                "loss_real": "MIL top-k BCE + size/component priors",
                "loss_brats": "DiceCELoss(include_background=True, sigmoid=True)",
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    ckpt0 = torch.load(pretrained, map_location=device_t, weights_only=False)
    model_name: ModelName = (
        ckpt0.get("model_name", config.model_name) if isinstance(ckpt0, dict) else config.model_name
    )
    model = build_brats_model(model_name).to(device_t)
    model.load_state_dict(ckpt0["model_state"])

    dicece = build_brats_loss()
    assert_loss_covers_all_channels(dicece)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    balanced0 = sample_balanced_pos_neg(train_files, seed=config.seed, epoch=0)
    mixed0 = sample_replay_mix(
        balanced0,
        brats_pool,
        ratio_batch2=config.replay_weak,
        ratio_brats=config.replay_brats,
        seed=config.seed,
        epoch=0,
        max_brats=max_brats_per_epoch,
    )
    train_tf = _train_transforms_weaksup(config)
    steps_per_epoch = max(
        1, int(len(mixed0) * config.num_train_samples / max(1, config.batch_size))
    )
    scheduler, scheduler_step_mode = build_lr_scheduler(
        optimizer, train_cfg, steps_per_epoch=steps_per_epoch
    )

    writer = SummaryWriter(log_dir=str(log_dir))
    best_proxy = float("inf")  # lower train loss proxy for "best"
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest.pt"
    last_path = output_dir / "last_model.pt"
    metrics_csv = output_dir / "metrics.csv"
    heartbeat_path = output_dir / "heartbeat.log"
    start_epoch = 0
    heartbeat = _HeartbeatWatcher(
        heartbeat_path, interval_s=float(config.heartbeat_interval_s)
    )

    if resume and latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=device_t, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_proxy = float(ckpt.get("best_train_loss", best_proxy))
        raw_epoch = int(ckpt.get("epoch", -1))
        start_epoch = raw_epoch if ckpt.get("epoch_1based") else raw_epoch + 1
        _append_heartbeat(
            heartbeat_path,
            f"resume from latest.pt → next epoch {start_epoch + 1}",
        )

    heartbeat.start()
    epoch_timings: list[float] = []
    try:
        for epoch in range(start_epoch, config.max_epochs):
            heartbeat.set_epoch(epoch + 1)
            _append_heartbeat(
                heartbeat_path,
                f"epoch {epoch + 1}/{config.max_epochs} START",
                epoch=epoch + 1,
            )
            balanced = sample_balanced_pos_neg(
                train_files, seed=config.seed, epoch=epoch
            )
            mixed = sample_replay_mix(
                balanced,
                brats_pool,
                ratio_batch2=config.replay_weak,
                ratio_brats=config.replay_brats,
                seed=config.seed,
                epoch=epoch,
                max_brats=max_brats_per_epoch,
            )
            train_loader = DataLoader(
                Dataset(data=mixed, transform=train_tf),
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )
            t0 = time.perf_counter()
            mean_loss, weak_loss, brats_loss = _weaksup_train_epoch(
                model,
                train_loader,
                optimizer,
                dicece,
                device_t,
                scaler,
                use_amp,
                epoch=epoch,
                topk=config.topk,
                log_every_n_batches=config.log_every_n_batches,
                scheduler=scheduler,
                scheduler_step_mode=scheduler_step_mode,
                grad_clip_max_norm=config.grad_clip_max_norm,
            )
            epoch_seconds = time.perf_counter() - t0
            epoch_timings.append(epoch_seconds)
            if scheduler_step_mode == "epoch_val":
                # Cosine warmup steps per batch; plateau would need a val metric.
                pass
            lr = float(optimizer.param_groups[0]["lr"])
            writer.add_scalar("train/loss", mean_loss, epoch)
            writer.add_scalar("train/weak_loss", weak_loss, epoch)
            writer.add_scalar("train/brats_loss", brats_loss, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)

            _append_metrics_row(
                metrics_csv,
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{mean_loss:.6f}",
                    "weak_loss": f"{weak_loss:.6f}",
                    "brats_loss": f"{brats_loss:.6f}",
                    "val_dice_mean": f"{-mean_loss:.6f}",  # proxy for metadata assert
                    "learning_rate": f"{lr:.8e}",
                    "epoch_seconds": f"{epoch_seconds:.3f}",
                },
            )
            logger.info(
                "Epoch %d/%d | loss=%.4f (weak=%.4f brats=%.4f) | %.1fs | lr=%.2e | mixed=%d",
                epoch + 1,
                config.max_epochs,
                mean_loss,
                weak_loss,
                brats_loss,
                epoch_seconds,
                lr,
                len(mixed),
            )
            _append_heartbeat(
                heartbeat_path,
                f"epoch {epoch + 1}/{config.max_epochs} END "
                f"loss={mean_loss:.4f} sec={epoch_seconds:.1f}",
                epoch=epoch + 1,
            )

            is_best = mean_loss < best_proxy
            if is_best:
                best_proxy = mean_loss
            checkpoint = {
                "epoch": epoch + 1,
                "epoch_1based": True,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_loss": mean_loss,
                "best_train_loss": best_proxy,
                "val_dice": -mean_loss,
                "best_val_dice": -best_proxy,
                "model_name": model_name,
                "pretrained_checkpoint": str(pretrained),
                "train_config": asdict(config),
                "loss": "weaksup MIL + BraTS DiceCE",
                "tumor_free_note": TUMOR_FREE_WORDING,
                "regions": list(BRATS_REGIONS),
            }
            torch.save(checkpoint, latest_path)
            torch.save(checkpoint, last_path)
            if is_best:
                torch.save(checkpoint, best_path)
                logger.info("New best weaksup loss %.4f → %s", best_proxy, best_path)
    finally:
        heartbeat.stop("training loop exited")
        writer.close()

    if not best_path.is_file():
        # Fall back to last
        if last_path.is_file():
            import shutil

            shutil.copy(last_path, best_path)
        else:
            raise RuntimeError("Weaksup training finished without a checkpoint")
    (output_dir / "epoch_timings.json").write_text(
        json.dumps({"epoch_seconds": epoch_timings}, indent=2) + "\n",
        encoding="utf-8",
    )
    return best_path


@torch.no_grad()
def score_scan_level(
    model: torch.nn.Module,
    record: dict[str, Any],
    device: torch.device,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
) -> dict[str, Any]:
    """Scan-level score: max WT prob on LCC + predicted WT mL."""
    model.eval()
    tf = get_patient_inference_transforms()
    data = tf(
        {
            "study_id": record["study_id"],
            "image": record[IMAGE_KEY],
        }
    )
    image = data["image"].unsqueeze(0).to(device, dtype=torch.float32)
    logits = sliding_window_inference(
        image, roi_size=roi_size, sw_batch_size=1, predictor=model
    )
    probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    post_pred, _ = post_transforms()
    pred_raw = (
        post_pred(decollate_batch(logits)[0]).detach().cpu().numpy().astype(np.float32)
    )
    pred_lcc = enforce_brats_nesting(
        apply_channel_postprocess(pred_raw.copy(), mode="lcc")
    )

    reg = Path(record.get("reg_dir", ""))
    t1c_path = reg / REGISTERED_NAMES["T1c"] if reg.is_dir() else None
    if t1c_path is not None and t1c_path.is_file():
        spacing = tuple(float(x) for x in nib.load(str(t1c_path)).header.get_zooms()[:3])
    else:
        spacing = (1.0, 1.0, 1.0)
    voxel_ml = float(np.prod(spacing) / 1000.0)

    wt_lcc = pred_lcc[WT_CH].astype(bool)
    wt_prob = probs[WT_CH]
    on_lcc = wt_prob[wt_lcc]
    max_prob = float(on_lcc.max()) if on_lcc.size else 0.0
    wt_ml = float(np.count_nonzero(wt_lcc) * voxel_ml)

    return {
        "study_id": record["study_id"],
        "patient_id": record.get("patient_id"),
        "is_negative": bool(record.get("is_negative")),
        "t1c_status": record.get("t1c_status"),
        "WT_max_prob_on_lcc": max_prob,
        "WT_lcc_ml": wt_ml,
        "score": max_prob,
        "pred_lcc": pred_lcc,
        "probs": probs,
        "spacing": spacing,
        "voxel_ml": voxel_ml,
        "reg_dir": str(reg),
    }


def classic_label_from_regions(regions: np.ndarray) -> np.ndarray:
    """ET=4, edema(WT\\TC)=2, necrosis(TC\\ET)=1 from nested region channels."""
    et = regions[ET_CH] > 0.5
    tc = regions[TC_CH] > 0.5
    wt = regions[WT_CH] > 0.5
    out = np.zeros(wt.shape, dtype=np.int16)
    out[wt & ~tc] = 2
    out[tc & ~et] = 1
    out[et] = 4
    return out


def top1_component_mask(wt_binary: np.ndarray) -> np.ndarray:
    return keep_largest_cc(wt_binary)


def _norm_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(float)
    p2, p98 = np.percentile(sl, (2, 98))
    if p98 <= p2:
        return np.zeros_like(sl)
    return np.clip((sl - p2) / (p98 - p2), 0, 1)


def _contour_edges(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    eroded = ndi.binary_erosion(m, iterations=1)
    return m & ~eroded


def _pick_slices(mask: np.ndarray, n: int = 6) -> list[int]:
    m = np.asarray(mask, dtype=bool)
    z = m.shape[2]
    if not np.any(m):
        return [int(round(i * (z - 1) / max(n - 1, 1))) for i in range(n)]
    zs = np.where(m.any(axis=(0, 1)))[0]
    if len(zs) <= n:
        mid = list(map(int, zs))
        while len(mid) < n:
            mid.append(int(zs[len(zs) // 2]))
        return sorted(mid)[:n]
    idx = np.linspace(0, len(zs) - 1, n).astype(int)
    return [int(zs[i]) for i in idx]


def save_localization(
    scored: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Save top-1 WT component as weaksup_pred_label.nii.gz + 6-slice montage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = Path(scored["reg_dir"])
    t1c_p = reg / REGISTERED_NAMES["T1c"]
    flair_p = reg / REGISTERED_NAMES["FLAIR"]
    t2_p = reg / REGISTERED_NAMES["T2"]
    t1c_img = nib.load(str(t1c_p))
    t1c = np.asanyarray(t1c_img.dataobj).astype(np.float32)
    flair = np.asanyarray(nib.load(str(flair_p)).dataobj).astype(np.float32)
    t2 = np.asanyarray(nib.load(str(t2_p)).dataobj).astype(np.float32)

    pred_lcc = scored["pred_lcc"]
    # Resample / crop prediction to T1c grid if needed (inference may be 1mm same grid).
    wt = pred_lcc[WT_CH]
    if wt.shape != t1c.shape:
        # Nearest resize to T1c grid
        wt_t = torch.from_numpy(wt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        wt_t = torch.nn.functional.interpolate(
            wt_t, size=t1c.shape, mode="nearest"
        )
        wt = wt_t.squeeze().numpy()
        regions = np.zeros((3,) + t1c.shape, dtype=np.float32)
        for i in range(3):
            ch = torch.from_numpy(pred_lcc[i].astype(np.float32)).unsqueeze(0).unsqueeze(0)
            ch = torch.nn.functional.interpolate(ch, size=t1c.shape, mode="nearest")
            regions[i] = ch.squeeze().numpy()
    else:
        regions = pred_lcc

    top1 = top1_component_mask(regions[WT_CH] > 0.5)
    # Zero other channels outside top-1 WT, then classic label
    regs = regions.copy()
    regs[WT_CH] = top1.astype(np.float32)
    regs[TC_CH] = (regs[TC_CH] > 0.5) & top1
    regs[ET_CH] = (regs[ET_CH] > 0.5) & top1
    classic = classic_label_from_regions(regs.astype(np.float32))
    label_path = out_dir / "weaksup_pred_label.nii.gz"
    nib.save(nib.Nifti1Image(classic, t1c_img.affine, t1c_img.header), str(label_path))

    # SWI panel if present
    swi = None
    for cand in (
        reg / "SWI_registered_1mm.nii.gz",
        reg.parent / "SWI_registered_1mm.nii.gz",
        PROJECT_ROOT / "results" / "batch2_swi" / scored["study_id"] / "SWI_on_T1c.nii.gz",
    ):
        if cand.is_file():
            swi_data = np.asanyarray(nib.load(str(cand)).dataobj).astype(np.float32)
            if swi_data.shape == t1c.shape:
                swi = swi_data
            break

    slices = _pick_slices(top1, 6)
    n_rows = 4 if swi is not None else 3
    fig, axs = plt.subplots(n_rows, 6, figsize=(14, 2.4 * n_rows))
    row_titles = ["T1c + contour", "FLAIR + contour", "T2 + contour"]
    if swi is not None:
        row_titles.append("SWI")
    for col, z in enumerate(slices):
        edge = _contour_edges(top1[:, :, z])
        for r, vol in enumerate((t1c, flair, t2)):
            base = _norm_slice(vol[:, :, z])
            rgb = np.stack([base, base, base], axis=-1)
            rgb[edge] = (0.95, 0.2, 0.15)
            axs[r, col].imshow(np.rot90(rgb), origin="lower")
            axs[r, col].axis("off")
            if col == 0:
                axs[r, col].set_ylabel(row_titles[r], fontsize=8)
        if swi is not None:
            base = _norm_slice(swi[:, :, z])
            rgb = np.stack([base, base, base], axis=-1)
            rgb[edge] = (0.95, 0.2, 0.15)
            axs[3, col].imshow(np.rot90(rgb), origin="lower")
            axs[3, col].axis("off")
            if col == 0:
                axs[3, col].set_ylabel(row_titles[3], fontsize=8)
    fig.suptitle(
        f"{scored['study_id']} weaksup proposal\n{LOCALIZATION_DISCLAIMER}",
        fontsize=9,
    )
    fig.tight_layout()
    montage = out_dir / "weaksup_montage.png"
    fig.savefig(montage, dpi=120, bbox_inches="tight")
    plt.close(fig)
    (out_dir / "README.md").write_text(
        f"# {scored['study_id']} weaksup localization\n\n"
        f"{LOCALIZATION_DISCLAIMER}\n\n"
        f"Negatives: {TUMOR_FREE_WORDING}.\n",
        encoding="utf-8",
    )
    return {"label": str(label_path), "montage": str(montage)}


def shuffle_labels(
    records: Sequence[dict[str, Any]], *, seed: int = 42
) -> list[dict[str, Any]]:
    """Shuffle is_negative among scans (same pos/neg counts) for control."""
    out = [dict(r) for r in records]
    n_neg = sum(1 for r in out if r.get("is_negative"))
    rng = random.Random(seed + 4242)
    flags = [True] * n_neg + [False] * (len(out) - n_neg)
    rng.shuffle(flags)
    for r, f in zip(out, flags):
        r["is_negative"] = f
        r["cohort"] = "negative" if f else "positive"
        r["shuffled_control"] = True
    return out


def _metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, thr: float
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "fpr_negatives": float(fpr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_heldout_scan_level(
    checkpoint: str | Path,
    held_records: Sequence[dict[str, Any]],
    train_records: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    model_tag: str,
    fold: int,
    thr: float | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Score held-out scans; threshold from train scores if thr is None."""
    model = load_model_from_checkpoint(checkpoint, device)
    model.float().eval()

    # Threshold on training folds only (balanced subsample — full train SWI is too slow).
    if thr is None:
        pos_tr = [r for r in train_records if not r.get("is_negative")]
        neg_tr = [r for r in train_records if r.get("is_negative")]
        rng = random.Random(42 + int(fold) * 17)
        n_cap = 16
        if len(pos_tr) > n_cap:
            pos_tr = rng.sample(pos_tr, n_cap)
        if len(neg_tr) > n_cap:
            neg_tr = rng.sample(neg_tr, n_cap)
        train_scores = []
        train_y = []
        for rec in list(pos_tr) + list(neg_tr):
            s = score_scan_level(model, rec, device, roi_size=roi_size)
            train_scores.append(s["score"])
            train_y.append(0 if rec.get("is_negative") else 1)
        thr, _, _ = best_threshold_sens_spec(
            np.asarray(train_y), np.asarray(train_scores, dtype=float)
        )
        if thr != thr or not np.isfinite(thr):
            thr = 0.5

    rows = []
    for rec in held_records:
        s = score_scan_level(model, rec, device, roi_size=roi_size)
        y = 0 if rec.get("is_negative") else 1
        rows.append(
            {
                "fold": fold,
                "model": model_tag,
                "study_id": s["study_id"],
                "patient_id": s["patient_id"],
                "is_negative": s["is_negative"],
                "t1c_status": s["t1c_status"],
                "y_true": y,
                "WT_max_prob_on_lcc": s["WT_max_prob_on_lcc"],
                "WT_lcc_ml": s["WT_lcc_ml"],
                "score": s["score"],
                "threshold": float(thr),
                "pred_positive": int(s["score"] >= thr),
                "_scored": s,
            }
        )
    y_true = np.array([r["y_true"] for r in rows], dtype=int)
    y_score = np.array([r["score"] for r in rows], dtype=float)
    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = float("nan")
    mets = _metrics_at_threshold(y_true, y_score, float(thr))
    mets["auc"] = auc
    mets["threshold"] = float(thr)
    return rows, float(thr), mets


def run_weaksup_cv(
    *,
    results_dir: Path | None = None,
    pretrained: Path | None = None,
    device: str | None = None,
    seed: int = 42,
    max_epochs: int = 15,
    n_folds: int = 5,
    dry_run: bool = False,
    train_final: bool = False,
    resume: bool = False,
    fold_limit: int | None = None,
    run_shuffle_control: bool = True,
    domain_aug: DomainAugmentFlags | None = None,
    dry_run_n_brats: int = 4,
    estimate_only: bool = False,
) -> Path:
    """5-fold patient-grouped weaksup CV (+ optional shuffle control)."""
    stamp_parts = [TUMOR_FREE_WORDING]
    if dry_run:
        stamp_parts.insert(0, DRY_RUN_STAMP)
    stamp = " | ".join(stamp_parts)

    results_dir = Path(
        results_dir or (DRYRUN_RESULTS_DIR if dry_run else DEFAULT_RESULTS_DIR)
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    pretrained = Path(pretrained or DEFAULT_V3_CHECKPOINT)
    device_t = get_device(device)
    flags = domain_aug or default_weaksup_aug_flags()
    log_augment_flags(flags)

    records = build_weaksup_file_list()
    if dry_run:
        # Keep class balance but shrink: 2 pos + 2 neg, 1 fold, 2 epochs
        pos = [r for r in records if not r["is_negative"]][:2]
        neg = [r for r in records if r["is_negative"]][:2]
        records = pos + neg
        max_epochs = 2
        fold_limit = 1
        run_shuffle_control = False
        n_folds = 2

    report_group_standardized_diffs(
        records, out_path=results_dir / "group_standardized_diffs.csv"
    )

    folds = make_patient_folds(
        records, seed=seed, n_folds=n_folds, balance_negatives=True
    )
    if fold_limit is not None:
        folds = folds[:fold_limit]

    brats_pool = load_brats_replay_pool()
    if dry_run:
        brats_pool = brats_pool[:dry_run_n_brats]

    heldout_subset: list[dict[str, Any]] = []
    if not dry_run:
        heldout_subset = fixed_heldout_subset(n=20, seed=42)

    cfg = WeaksupConfig(
        max_epochs=max_epochs,
        learning_rate=1e-5,
        seed=seed,
        domain_aug=flags,
        # num_train_samples=1 keeps 5×15×2 (real+shuffle) CV under ~8h on MPS.
        num_train_samples=1,
        heartbeat_interval_s=60.0 if dry_run else 300.0,
    )

    cv_config = {
        "stamp": stamp,
        "tumor_free_note": TUMOR_FREE_WORDING,
        "localization_disclaimer": LOCALIZATION_DISCLAIMER,
        "dry_run": dry_run,
        "seed": seed,
        "max_epochs": max_epochs,
        "n_folds": len(folds),
        "pretrained": str(pretrained),
        "device": str(device_t),
        "domain_aug": flags.to_dict(),
        "domain_aug_enabled": flags.enabled_names(),
        "n_pos": sum(1 for r in records if not r["is_negative"]),
        "n_neg": sum(1 for r in records if r["is_negative"]),
        "folds": [
            {
                "fold": f["fold"],
                "mode": f["mode"],
                "train_patient_ids": f["train_patient_ids"],
                "held_patient_ids": f["held_patient_ids"],
                "n_pos_held": sum(1 for r in f["held"] if not r.get("is_negative")),
                "n_neg_held": sum(1 for r in f["held"] if r.get("is_negative")),
            }
            for f in folds
        ],
    }
    (results_dir / "cv_config.json").write_text(
        json.dumps(cv_config, indent=2) + "\n", encoding="utf-8"
    )

    all_rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    forget_rows: list[dict[str, Any]] = []
    epoch_times: list[float] = []
    t_wall0 = time.perf_counter()

    def _run_folds(
        fold_list: list[dict[str, Any]],
        *,
        tag: str,
        recs: list[dict[str, Any]],
        save_loc: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows_out: list[dict[str, Any]] = []
        mets_out: list[dict[str, Any]] = []
        for fold in fold_list:
            fi = int(fold["fold"])
            if tag == "real":
                ckpt_dir = assert_safe_checkpoint_output_dir(
                    FOLD_CKPT_ROOT / f"batch2_weaksup_fold_{fi}"
                )
            else:
                ckpt_dir = assert_safe_checkpoint_output_dir(
                    results_dir / "shuffle_fold_checkpoints" / f"fold_{fi}"
                )
            logger.info(
                "=== %s fold %d | train=%d held=%d | %s ===",
                tag,
                fi,
                len(fold["train"]),
                len(fold["held"]),
                stamp,
            )
            best = train_weaksup(
                train_files=fold["train"],
                val_files=fold["held"],
                output_dir=ckpt_dir,
                pretrained_checkpoint=pretrained,
                brats_replay_pool=brats_pool,
                config=cfg,
                device=str(device_t),
                resume=resume,
                max_brats_per_epoch=dry_run_n_brats if dry_run else None,
            )
            timings_path = ckpt_dir / "epoch_timings.json"
            if timings_path.is_file():
                epoch_times.extend(
                    json.loads(timings_path.read_text(encoding="utf-8")).get(
                        "epoch_seconds", []
                    )
                )

            # Baseline v3 + fine-tuned on held-out
            for model_tag, ckpt in (
                ("v3_baseline", pretrained),
                ("weaksup", best),
            ):
                rows, thr, mets = evaluate_heldout_scan_level(
                    ckpt,
                    fold["held"],
                    fold["train"],
                    device=device_t,
                    model_tag=model_tag,
                    fold=fi,
                )
                for r in rows:
                    r["experiment"] = tag
                    scored = r.pop("_scored", None)
                    rows_out.append({k: v for k, v in r.items() if not k.startswith("_")})
                    if (
                        save_loc
                        and tag == "real"
                        and model_tag == "weaksup"
                        and scored is not None
                    ):
                        is_pos = not r["is_negative"]
                        above = bool(r["pred_positive"])
                        if is_pos or above:
                            loc_dir = results_dir / str(r["study_id"])
                            save_localization(scored, loc_dir)
                mets_out.append(
                    {
                        "experiment": tag,
                        "fold": fi,
                        "model": model_tag,
                        **mets,
                        "tumor_free_note": TUMOR_FREE_WORDING,
                    }
                )
                logger.info(
                    "%s fold %d %s AUC=%.3f sens=%.3f spec=%.3f FPR_neg=%.3f thr=%.4f",
                    tag,
                    fi,
                    model_tag,
                    mets["auc"],
                    mets["sensitivity"],
                    mets["specificity"],
                    mets["fpr_negatives"],
                    mets["threshold"],
                )

            if heldout_subset and tag == "real":
                forget_rows.extend(
                    evaluate_forgetting(
                        pretrained,
                        best,
                        heldout_subset,
                        device=device_t,
                        fold=fi,
                    )
                )
        return rows_out, mets_out

    # Optional early estimate after configuring
    if estimate_only:
        # Train 0 epochs — just write estimate placeholder
        pass

    rows, mets = _run_folds(folds, tag="real", recs=records, save_loc=not dry_run)
    all_rows.extend(rows)
    fold_metrics.extend(mets)

    # Time estimate: scale dry-run epoch by expected steps (balanced pos/neg + BraTS).
    mean_epoch_raw = float(np.mean(epoch_times)) if epoch_times else float("nan")
    n_epochs_full = int(max_epochs) if not dry_run else 15
    n_folds_full = 5
    if dry_run and mean_epoch_raw == mean_epoch_raw:
        # Dry: 2+2 balanced → 4 weak + ~4 BraTS, num_samples=1 → ~6–8 steps.
        # Full: 19+19 weak + 38 BraTS, num_samples=2 → ~152 steps.
        # Dry: 2+2 balanced weak + dry_run_n_brats BraTS, num_samples=1.
        # Full: 19+19 weak + 38 BraTS; num_samples matches cfg (1 keeps 15-ep CV <8h).
        dry_steps = max(1, (2 + 2) + dry_run_n_brats)
        n_weak_full = 19 + 19
        n_brats_full = n_weak_full  # 1:1 replay
        full_samples = 1  # intentional for wall-clock budget
        full_steps = (n_weak_full + n_brats_full) * full_samples
        scale = full_steps / float(dry_steps)
        mean_epoch = mean_epoch_raw * scale
        # Eval overhead per fold (2 models × ~32 train thr + ~24 held) ≈ 90 SWI @ ~7s
        eval_per_fold_s = 90.0 * 7.0
        n_cv = n_folds_full * 2  # real + shuffle
        est_hours = (
            mean_epoch * n_epochs_full * n_cv + eval_per_fold_s * n_cv
        ) / 3600.0
    else:
        mean_epoch = mean_epoch_raw
        est_hours = (
            (mean_epoch * n_epochs_full * n_folds_full * (2 if run_shuffle_control else 1))
            / 3600.0
            if mean_epoch == mean_epoch
            else float("nan")
        )

    if dry_run:
        (results_dir / "timing_DRYRUN.json").write_text(
            json.dumps(
                {
                    "stamp": DRY_RUN_STAMP,
                    "epoch_seconds_raw": epoch_times,
                    "mean_epoch_seconds_raw": mean_epoch_raw,
                    "mean_epoch_seconds_scaled": mean_epoch,
                    "estimated_full_cv_hours": est_hours,
                    "assumed_epochs": n_epochs_full,
                    "assumed_folds": n_folds_full,
                    "includes_shuffle_control": True,
                    "tumor_free_note": TUMOR_FREE_WORDING,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "%s mean epoch raw=%.1fs scaled=%.1fs | estimated full CV "
            "(5×%d×2 + eval)=%.2f h",
            DRY_RUN_STAMP,
            mean_epoch_raw,
            mean_epoch,
            n_epochs_full,
            est_hours,
        )

    shuffle_auc = float("nan")
    if run_shuffle_control and not dry_run:
        shuffled = shuffle_labels(records, seed=seed)
        sh_folds = make_patient_folds(
            shuffled, seed=seed, n_folds=n_folds, balance_negatives=True
        )
        if fold_limit is not None:
            sh_folds = sh_folds[:fold_limit]
        srows, smets = _run_folds(
            sh_folds, tag="shuffle", recs=shuffled, save_loc=False
        )
        all_rows.extend(srows)
        fold_metrics.extend(smets)
        # Pooled shuffle AUC for weaksup
        sdf = pd.DataFrame(
            [r for r in srows if r["model"] == "weaksup" and r["experiment"] == "shuffle"]
        )
        if len(sdf) and sdf["y_true"].nunique() >= 2:
            shuffle_auc = float(roc_auc_score(sdf["y_true"], sdf["score"]))

    # Pool real weaksup predictions for overall AUC + bootstrap CI
    rdf = pd.DataFrame(
        [r for r in all_rows if r["model"] == "weaksup" and r.get("experiment") == "real"]
    )
    overall = {}
    if len(rdf) and rdf["y_true"].nunique() >= 2:
        auc, lo, hi = bootstrap_auc_ci(
            rdf["y_true"].to_numpy(), rdf["score"].to_numpy()
        )
        overall["auc"] = auc
        overall["auc_ci_lo"] = lo
        overall["auc_ci_hi"] = hi
        # Threshold: median of fold thresholds
        thr = float(np.nanmedian(rdf["threshold"].to_numpy(dtype=float)))
        overall.update(_metrics_at_threshold(rdf["y_true"], rdf["score"], thr))
        overall["threshold"] = thr

    # Stand-in positives subset (9)
    standin_ids = set(
        pd.read_csv(POS_NEG_SPLIT_DIR / "manifest.csv")
        .query("cohort == 'positive' and t1c_status == 'standin'")["study_id"]
        .astype(str)
    )
    standin_metrics = {}
    if len(rdf):
        # Restrict positives to stand-in; keep all held-out negatives in those folds
        pos_stand = rdf[
            (rdf["y_true"] == 1) & rdf["study_id"].astype(str).isin(standin_ids)
        ]
        neg_all = rdf[rdf["y_true"] == 0]
        if len(pos_stand) and len(neg_all):
            y = np.concatenate(
                [np.ones(len(pos_stand)), np.zeros(len(neg_all))]
            )
            s = np.concatenate(
                [
                    pos_stand["score"].to_numpy(dtype=float),
                    neg_all["score"].to_numpy(dtype=float),
                ]
            )
            if len(np.unique(y)) >= 2:
                a, lo, hi = bootstrap_auc_ci(y, s)
                thr = float(np.nanmedian(rdf["threshold"]))
                standin_metrics = {
                    "auc": a,
                    "auc_ci_lo": lo,
                    "auc_ci_hi": hi,
                    "n_standin_pos": int(len(pos_stand)),
                    **_metrics_at_threshold(y, s, thr),
                    "threshold": thr,
                }

    # Forgetting flag
    forget_flag = False
    forget_summary = {}
    if forget_rows:
        fdf = pd.DataFrame(forget_rows)
        # Mean Dice delta per region
        for region in ("ET", "TC", "WT"):
            sub = fdf[fdf["region"] == region]
            if sub.empty:
                continue
            mean_delta = float(sub["dice_delta"].mean())
            forget_summary[region] = {
                "mean_dice_baseline": float(sub["dice_baseline"].mean()),
                "mean_dice_finetuned": float(sub["dice_finetuned"].mean()),
                "mean_dice_delta": mean_delta,
            }
            if mean_delta < -FORGETTING_DICE_DROP_MAX:
                forget_flag = True
        forget_summary["flag"] = forget_flag
        forget_summary["threshold"] = -FORGETTING_DICE_DROP_MAX

    # Persist tables
    per_case = pd.DataFrame(all_rows)
    per_case.to_csv(results_dir / "per_case_scores.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(results_dir / "fold_metrics.csv", index=False)
    if forget_rows:
        pd.DataFrame(forget_rows).to_csv(
            results_dir / "forgetting_check.csv", index=False
        )

    wall = time.perf_counter() - t_wall0
    timing = {
        "wall_seconds": wall,
        "epoch_seconds": epoch_times,
        "mean_epoch_seconds": mean_epoch,
        "estimated_full_cv_hours": est_hours,
        "peak_rss_mb": _peak_rss_mb(),
        "tumor_free_note": TUMOR_FREE_WORDING,
    }
    timing_name = "timing_DRYRUN.json" if dry_run else "timing.json"
    (results_dir / timing_name).write_text(
        json.dumps(timing, indent=2) + "\n", encoding="utf-8"
    )

    # SUMMARY.md — only stamp DRY RUN for dry-run; full run is clean
    summary_path = results_dir / ("SUMMARY_DRYRUN.md" if dry_run else "SUMMARY.md")
    _write_summary(
        summary_path,
        overall=overall,
        standin=standin_metrics,
        shuffle_auc=shuffle_auc,
        fold_metrics=fold_metrics,
        forget=forget_summary,
        dry_run=dry_run,
        timing=timing,
    )

    if train_final and not dry_run:
        logger.info(
            "train_final requested → %s (will not replace v3 default)", FINAL_CKPT_DIR
        )
        train_weaksup(
            train_files=records,
            val_files=records,
            output_dir=assert_safe_checkpoint_output_dir(FINAL_CKPT_DIR),
            pretrained_checkpoint=pretrained,
            brats_replay_pool=load_brats_replay_pool(),
            config=WeaksupConfig(
                max_epochs=max_epochs,
                learning_rate=1e-5,
                seed=seed,
                domain_aug=flags,
            ),
            device=str(device_t),
            resume=resume,
        )
        logger.info(
            "CLI/API default checkpoint remains %s",
            load_config().inference.checkpoint,
        )

    return results_dir


def _write_summary(
    path: Path,
    *,
    overall: dict[str, Any],
    standin: dict[str, Any],
    shuffle_auc: float,
    fold_metrics: list[dict[str, Any]],
    forget: dict[str, Any],
    dry_run: bool,
    timing: dict[str, Any],
) -> None:
    lines = [
        "# Batch2 weakly supervised CV summary",
        "",
    ]
    if dry_run:
        lines += [f"**{DRY_RUN_STAMP}**", ""]
    lines += [
        f"Negatives: {TUMOR_FREE_WORDING}.",
        "",
        f"Localization: {LOCALIZATION_DISCLAIMER}",
        "",
        "## Overall (pooled held-out weaksup)",
        "",
    ]
    if overall:
        lines.append(
            f"- ROC AUC: **{overall.get('auc', float('nan')):.3f}** "
            f"95% bootstrap CI "
            f"[{overall.get('auc_ci_lo', float('nan')):.3f}, "
            f"{overall.get('auc_ci_hi', float('nan')):.3f}]"
        )
        lines.append(
            f"- Threshold (median of train-fold Youden): {overall.get('threshold', float('nan')):.4f}"
        )
        lines.append(
            f"- Sensitivity: {overall.get('sensitivity', float('nan')):.3f}; "
            f"Specificity: {overall.get('specificity', float('nan')):.3f}; "
            f"FPR on held-out negatives: {overall.get('fpr_negatives', float('nan')):.3f}"
        )
    else:
        lines.append("- Insufficient held-out predictions for pooled AUC.")
    lines += ["", "## Shuffle control", ""]
    lines.append(
        f"- Shuffled-label weaksup AUC: **{shuffle_auc:.3f}** "
        "(must be near 0.5; high values imply bug or leakage)."
    )
    if overall.get("auc") == overall.get("auc"):
        lines.append(
            f"- Real vs shuffled: {overall.get('auc', float('nan')):.3f} vs {shuffle_auc:.3f}"
        )
    lines += ["", "## Stand-in positives only (n≈9)", ""]
    if standin:
        lines.append(
            f"- AUC: {standin.get('auc', float('nan')):.3f} "
            f"[{standin.get('auc_ci_lo', float('nan')):.3f}, "
            f"{standin.get('auc_ci_hi', float('nan')):.3f}] "
            f"(n_standin_pos={standin.get('n_standin_pos')})"
        )
        lines.append(
            f"- Sens={standin.get('sensitivity', float('nan')):.3f} "
            f"Spec={standin.get('specificity', float('nan')):.3f} "
            f"FPR_neg={standin.get('fpr_negatives', float('nan')):.3f}"
        )
    else:
        lines.append("- Not available.")
    lines += ["", "## Per-fold metrics", ""]
    lines.append("| exp | fold | model | AUC | sens | spec | FPR_neg | thr |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for m in fold_metrics:
        lines.append(
            f"| {m.get('experiment')} | {m.get('fold')} | {m.get('model')} | "
            f"{m.get('auc', float('nan')):.3f} | {m.get('sensitivity', float('nan')):.3f} | "
            f"{m.get('specificity', float('nan')):.3f} | "
            f"{m.get('fpr_negatives', float('nan')):.3f} | "
            f"{m.get('threshold', float('nan')):.4f} |"
        )
    lines += ["", "## BraTS held-out forgetting (20-case subset)", ""]
    if forget:
        flag = forget.get("flag")
        lines.append(
            f"- Flag if mean Dice drops by > {FORGETTING_DICE_DROP_MAX}: "
            f"{'**FLAGGED**' if flag else 'ok'}"
        )
        for region in ("ET", "TC", "WT"):
            if region in forget:
                d = forget[region]
                lines.append(
                    f"- {region}: baseline={d['mean_dice_baseline']:.3f} "
                    f"weaksup={d['mean_dice_finetuned']:.3f} "
                    f"Δ={d['mean_dice_delta']:.3f}"
                )
    else:
        lines.append("- Skipped (dry-run) or unavailable.")
    lines += [
        "",
        "## Timing",
        "",
        f"- Mean epoch: {timing.get('mean_epoch_seconds', float('nan')):.1f}s",
        f"- Wall: {timing.get('wall_seconds', float('nan')):.1f}s",
        f"- Estimated full CV hours (from smoke): "
        f"{timing.get('estimated_full_cv_hours', float('nan'))}",
        "",
        f"Negatives reminder: {TUMOR_FREE_WORDING}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Weakly supervised batch2 CV from v3 (scan-level labels). "
            f"Negatives: {TUMOR_FREE_WORDING}."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_V3_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--train-final",
        action="store_true",
        help=(
            "After CV, train on all 19 positives + all negatives → "
            "checkpoints/batch2_weaksup_final/. Do not use until confirmed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Smoke: 1 fold, 2 epochs, tiny cohort ({DRY_RUN_STAMP})",
    )
    parser.add_argument("--no-shuffle-control", action="store_true")
    parser.add_argument("--no-thick-slice", action="store_true")
    parser.add_argument("--no-intensity-aug", action="store_true")
    parser.add_argument("--no-t1c-hide", action="store_true")
    parser.add_argument("--no-inplane-rescale", action="store_true")
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--no-zscore-brain", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.train_final and not args.dry_run:
        logger.warning(
            "--train-final is implemented but should only be run after explicit confirmation."
        )

    flags = DomainAugmentFlags(
        thick_slice=not args.no_thick_slice,
        intensity=not args.no_intensity_aug,
        modality_dropout=False,
        weaksup_t1c_hide=not args.no_t1c_hide,
        weaksup_t1c_hide_pos_prob=0.5,
        weaksup_t1c_hide_neg_always=True,
        inplane_rescale=not args.no_inplane_rescale,
        blur=not args.no_blur,
        zscore_brain=not args.no_zscore_brain,
    )
    run_weaksup_cv(
        results_dir=args.results_dir,
        pretrained=args.pretrained,
        device=args.device,
        seed=args.seed,
        max_epochs=args.epochs,
        n_folds=args.folds,
        dry_run=args.dry_run,
        train_final=args.train_final,
        resume=args.resume,
        run_shuffle_control=not args.no_shuffle_control,
        domain_aug=flags,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
