"""Fine-tune v3 on batch2 reference masks with BraTS replay + domain augs.

Reuses :mod:`segmentation.train` helpers (loss, epoch loops, scheduler,
heartbeat, best-checkpoint metadata). Never writes into
``checkpoints/brats_scale_full``.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric
from monai.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

from config import PROJECT_ROOT, load_config
from device import get_device
from segmentation.batch2_augment import DomainAugmentFlags, DomainAugmentd, log_augment_flags
from segmentation.batch2_dataset import REFERENCE_MASK_DISCLAIMER
from segmentation.dataset import (
    IMAGE_KEY,
    LABEL_KEY,
    get_train_transforms,
    get_val_transforms,
    subject_to_datadict,
)
from segmentation.model import BRATS_REGIONS, ModelName, build_brats_model
from segmentation.splits import read_id_list
from segmentation.train import (
    TrainConfig,
    _HeartbeatWatcher,
    _append_heartbeat,
    _append_metrics_row,
    _train_epoch,
    _validate_epoch,
    assert_best_checkpoint_matches_metrics,
    assert_loss_covers_all_channels,
    build_brats_loss,
    build_channel_dice_loss,
    build_lr_scheduler,
    post_transforms,
    update_best_dice,
)

logger = logging.getLogger(__name__)

PROTECTED_CHECKPOINT_DIRNAME = "brats_scale_full"
DEFAULT_V3_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"
)
DEFAULT_BRATS_REPLAY_LIST = (
    PROJECT_ROOT / "splits" / "scaleup" / "cfg_c_all_train.txt"
)
HELDOUT_LIST = PROJECT_ROOT / "splits" / "heldout.txt"


def assert_safe_checkpoint_output_dir(output_dir: str | Path) -> Path:
    """Refuse to write into ``checkpoints/brats_scale_full`` (v3 default)."""
    out = Path(output_dir).resolve()
    protected = (PROJECT_ROOT / "checkpoints" / PROTECTED_CHECKPOINT_DIRNAME).resolve()
    if out == protected or protected in out.parents or out == protected.parent / PROTECTED_CHECKPOINT_DIRNAME:
        raise ValueError(
            f"Refusing to write checkpoints into protected directory {protected}. "
            "Use checkpoints/batch2_finetune_* (or another non-v3 path)."
        )
    # Also refuse if the leaf name is the protected dirname under checkpoints/.
    if out.name == PROTECTED_CHECKPOINT_DIRNAME and out.parent == (
        PROJECT_ROOT / "checkpoints"
    ).resolve():
        raise ValueError(
            f"Refusing to write into protected checkpoint dir {protected}"
        )
    return out


@dataclass
class Batch2FinetuneConfig:
    """Hyperparameters for batch2 fine-tuning with BraTS replay."""

    max_epochs: int = 30
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
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    grad_clip_max_norm: float = 1.0
    heartbeat_interval_s: float = 300.0
    amp: bool = True
    num_workers: int = 0
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    log_every_n_batches: int = 1
    # Replay mixing: batch2 : BraTS ratio (default 1:1).
    replay_batch2: int = 1
    replay_brats: int = 1
    domain_aug: DomainAugmentFlags = field(default_factory=DomainAugmentFlags)

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
            scheduler_patience=self.scheduler_patience,
            scheduler_factor=self.scheduler_factor,
            grad_clip_max_norm=self.grad_clip_max_norm,
            heartbeat_interval_s=self.heartbeat_interval_s,
            amp=self.amp,
            num_workers=self.num_workers,
            pixdim=self.pixdim,
            log_every_n_batches=self.log_every_n_batches,
            cache_rate=0.0,
        )


def load_brats_replay_pool(
    *,
    brats_nifti_root: str | Path | None = None,
    replay_list: str | Path | None = None,
    heldout_list: str | Path | None = None,
) -> list[dict[str, Any]]:
    """BraTS cases from the scale-up train list; never from held-out."""
    root = Path(brats_nifti_root) if brats_nifti_root else load_config().paths.brats_nifti
    replay_path = Path(replay_list) if replay_list else DEFAULT_BRATS_REPLAY_LIST
    heldout_path = Path(heldout_list) if heldout_list else HELDOUT_LIST
    replay_ids = read_id_list(replay_path)
    heldout_ids = set(read_id_list(heldout_path)) if heldout_path.is_file() else set()
    overlap = [s for s in replay_ids if s in heldout_ids]
    if overlap:
        raise ValueError(
            f"Replay list intersects held-out ({len(overlap)} ids); refusing to sample"
        )
    records: list[dict[str, Any]] = []
    for sid in replay_ids:
        subject_dir = root / sid
        if not subject_dir.is_dir():
            logger.warning("Replay subject missing on disk, skipping: %s", sid)
            continue
        rec = subject_to_datadict(subject_dir)
        rec["patient_id"] = sid  # BraTS subject == patient for grouping
        rec["source"] = "brats_replay"
        records.append(rec)
    if not records:
        raise RuntimeError(f"No BraTS replay cases found under {root} from {replay_path}")
    logger.info("BraTS replay pool: %d cases (from %s)", len(records), replay_path)
    return records


def sample_replay_mix(
    batch2_files: Sequence[dict[str, Any]],
    brats_pool: Sequence[dict[str, Any]],
    *,
    ratio_batch2: int = 1,
    ratio_brats: int = 1,
    seed: int = 42,
    epoch: int = 0,
    max_brats: int | None = None,
) -> list[dict[str, Any]]:
    """Build one epoch's file list: all batch2 + BraTS samples at the given ratio."""
    if ratio_batch2 < 1 or ratio_brats < 0:
        raise ValueError("ratio_batch2 must be >=1 and ratio_brats >=0")
    b2 = [dict(r, source="batch2") for r in batch2_files]
    if ratio_brats == 0 or not brats_pool:
        return b2
    n_b2 = len(b2)
    # Target BraTS count ≈ n_b2 * (ratio_brats / ratio_batch2)
    n_brats = max(1, int(round(n_b2 * (ratio_brats / float(ratio_batch2)))))
    if max_brats is not None:
        # When set (e.g. dry-run), use exactly this many BraTS cases (cap at pool size).
        n_brats = min(int(max_brats), len(brats_pool))
    rng = random.Random(seed + epoch * 1009)
    if n_brats <= len(brats_pool):
        brats = [dict(r) for r in rng.sample(list(brats_pool), n_brats)]
    else:
        brats = [dict(r) for r in rng.choices(list(brats_pool), k=n_brats)]
    for r in brats:
        r["source"] = "brats_replay"
    mixed = b2 + brats
    rng.shuffle(mixed)
    return mixed


def _train_transforms_with_domain_aug(
    config: Batch2FinetuneConfig,
) -> Compose:
    base = get_train_transforms(
        pixdim=config.pixdim,
        crop_size=config.crop_size,
        num_samples=config.num_train_samples,
    )
    # Insert domain augs after intensity normalize / crop (last EnsureTyped stays last).
    transforms = list(base.transforms)
    aug = DomainAugmentd(
        keys=[IMAGE_KEY],
        label_key=LABEL_KEY,
        flags=config.domain_aug,
    )
    # Place before final EnsureTyped if present, else append.
    if transforms and transforms[-1].__class__.__name__ == "EnsureTyped":
        transforms = transforms[:-1] + [aug, transforms[-1]]
    else:
        transforms = transforms + [aug]
    return Compose(transforms)


def finetune_batch2(
    *,
    train_files: Sequence[dict[str, Any]],
    val_files: Sequence[dict[str, Any]] | None = None,
    output_dir: str | Path,
    pretrained_checkpoint: str | Path | None = None,
    brats_replay_pool: Sequence[dict[str, Any]] | None = None,
    config: Batch2FinetuneConfig | None = None,
    device: str | None = None,
    resume: bool = False,
    max_brats_per_epoch: int | None = None,
) -> Path:
    """Fine-tune from v3 on ``train_files`` with BraTS replay mixing.

    Returns path to ``best_model.pt``. Never overwrites
    ``checkpoints/brats_scale_full``.
    """
    config = config or Batch2FinetuneConfig()
    output_dir = assert_safe_checkpoint_output_dir(output_dir)
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

    brats_pool = list(brats_replay_pool) if brats_replay_pool is not None else load_brats_replay_pool()
    val_files = list(val_files) if val_files else list(train_files)
    if not train_files:
        raise ValueError("train_files is empty")

    log_augment_flags(config.domain_aug)
    logger.info(
        "batch2 fine-tune | %s | lr=%.2e | replay ratio batch2:brats = %d:%d | %s",
        REFERENCE_MASK_DISCLAIMER,
        config.learning_rate,
        config.replay_batch2,
        config.replay_brats,
        device_t,
    )

    # Persist fold / mix metadata.
    (output_dir / "finetune_meta.json").write_text(
        json.dumps(
            {
                "disclaimer": REFERENCE_MASK_DISCLAIMER,
                "pretrained_checkpoint": str(pretrained),
                "train_subject_ids": [r["subject_id"] for r in train_files],
                "val_subject_ids": [r["subject_id"] for r in val_files],
                "replay_ratio_batch2": config.replay_batch2,
                "replay_ratio_brats": config.replay_brats,
                "max_brats_per_epoch": max_brats_per_epoch,
                "domain_aug": config.domain_aug.to_dict(),
                "config": asdict(config),
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

    loss_fn = build_brats_loss()
    assert_loss_covers_all_channels(loss_fn)
    channel_dice_loss = build_channel_dice_loss()
    assert_loss_covers_all_channels(channel_dice_loss)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Build an initial mixed list so we know steps_per_epoch for the scheduler.
    mixed0 = sample_replay_mix(
        train_files,
        brats_pool,
        ratio_batch2=config.replay_batch2,
        ratio_brats=config.replay_brats,
        seed=config.seed,
        epoch=0,
        max_brats=max_brats_per_epoch,
    )
    train_tf = _train_transforms_with_domain_aug(config)
    val_tf = get_val_transforms(pixdim=config.pixdim)
    # Approximate steps: len(mixed) * num_train_samples / batch_size (RandCrop expands).
    steps_per_epoch = max(
        1, int(len(mixed0) * config.num_train_samples / max(1, config.batch_size))
    )
    scheduler, scheduler_step_mode = build_lr_scheduler(
        optimizer, train_cfg, steps_per_epoch=steps_per_epoch
    )

    dice_metric = DiceMetric(include_background=True, reduction="none")
    post_pred, post_label = post_transforms()

    writer = SummaryWriter(log_dir=str(log_dir))
    best_dice = -1.0
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
        best_dice = float(ckpt.get("best_val_dice", -1.0))
        raw_epoch = int(ckpt.get("epoch", -1))
        if ckpt.get("epoch_1based"):
            start_epoch = raw_epoch
        else:
            start_epoch = raw_epoch + 1
        logger.info(
            "Resumed from %s at epoch %d (best_val_dice=%.4f)",
            latest_path,
            start_epoch + 1,
            best_dice,
        )
        _append_heartbeat(
            heartbeat_path,
            f"resume from latest.pt → next epoch {start_epoch + 1} "
            f"(best={best_dice:.4f})",
        )

    val_loader = DataLoader(
        Dataset(data=list(val_files), transform=val_tf),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
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
            mixed = sample_replay_mix(
                train_files,
                brats_pool,
                ratio_batch2=config.replay_batch2,
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
            train_loss, (dl_et, dl_tc, dl_wt) = _train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                channel_dice_loss,
                device_t,
                scaler,
                use_amp,
                epoch=epoch,
                log_every_n_batches=config.log_every_n_batches,
                scheduler=scheduler,
                scheduler_step_mode=scheduler_step_mode,
                grad_clip_max_norm=config.grad_clip_max_norm,
            )
            val_dice, (vd_et, vd_tc, vd_wt) = _validate_epoch(
                model,
                val_loader,
                dice_metric,
                post_pred,
                post_label,
                device_t,
                config.roi_size,
                use_amp,
                epoch=epoch,
                log_every_n_batches=config.log_every_n_batches,
            )
            epoch_seconds = time.perf_counter() - t0
            epoch_timings.append(epoch_seconds)
            if scheduler_step_mode == "epoch_val":
                scheduler.step(val_dice)
            lr = float(optimizer.param_groups[0]["lr"])

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/dice_mean", val_dice, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            writer.add_scalar("time/epoch_seconds", epoch_seconds, epoch)

            _append_metrics_row(
                metrics_csv,
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{train_loss:.6f}",
                    "dice_loss_et": f"{dl_et:.6f}",
                    "dice_loss_tc": f"{dl_tc:.6f}",
                    "dice_loss_wt": f"{dl_wt:.6f}",
                    "val_dice_et": f"{vd_et:.6f}",
                    "val_dice_tc": f"{vd_tc:.6f}",
                    "val_dice_wt": f"{vd_wt:.6f}",
                    "val_dice_mean": f"{val_dice:.6f}",
                    "learning_rate": f"{lr:.8e}",
                    "epoch_seconds": f"{epoch_seconds:.3f}",
                },
            )
            logger.info(
                "Epoch %d/%d | loss=%.4f | val_dice=%.4f "
                "(ET=%.3f TC=%.3f WT=%.3f) | %.1fs | lr=%.2e | mixed=%d",
                epoch + 1,
                config.max_epochs,
                train_loss,
                val_dice,
                vd_et,
                vd_tc,
                vd_wt,
                epoch_seconds,
                lr,
                len(mixed),
            )
            _append_heartbeat(
                heartbeat_path,
                f"epoch {epoch + 1}/{config.max_epochs} END "
                f"val_dice={val_dice:.4f} lr={lr:.2e} sec={epoch_seconds:.1f}",
                epoch=epoch + 1,
            )

            checkpoint = {
                "epoch": epoch + 1,
                "epoch_1based": True,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_dice": val_dice,
                "val_dice_per_channel": {
                    "enhancing_tumor": vd_et,
                    "tumor_core": vd_tc,
                    "whole_tumor": vd_wt,
                },
                "best_val_dice": best_dice,
                "model_name": model_name,
                "pretrained_checkpoint": str(pretrained),
                "train_config": asdict(config),
                "loss": "DiceCELoss(include_background=True, sigmoid=True)",
                "disclaimer": REFERENCE_MASK_DISCLAIMER,
                "regions": list(BRATS_REGIONS),
            }
            best_dice, is_best = update_best_dice(val_dice, best_dice)
            checkpoint["best_val_dice"] = best_dice
            torch.save(checkpoint, latest_path)
            torch.save(checkpoint, last_path)
            if is_best:
                torch.save(checkpoint, best_path)
                logger.info("New best mean val Dice %.4f -> %s", best_dice, best_path)
    finally:
        heartbeat.stop("training loop exited")
        writer.close()

    if not best_path.is_file():
        raise RuntimeError("Fine-tuning finished without saving a best checkpoint")
    assert_best_checkpoint_matches_metrics(best_path, metrics_csv)
    (output_dir / "epoch_timings.json").write_text(
        json.dumps({"epoch_seconds": epoch_timings}, indent=2) + "\n",
        encoding="utf-8",
    )
    return best_path
