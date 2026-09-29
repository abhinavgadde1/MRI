"""MONAI training loop for BraTS multi-modal segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.tensorboard import SummaryWriter

from config import load_config
from device import get_device
from segmentation.dataset import (
    build_brats_file_list,
    get_train_transforms,
    get_val_transforms,
    split_train_val,
    subject_to_datadict,
)
from segmentation.model import (
    BRATS_REGIONS,
    OUT_CHANNELS,
    BraTSModelConfig,
    ModelName,
    brats_label_to_regions,
    build_brats_model,
)
from segmentation.splits import read_id_list

logger = logging.getLogger(__name__)

METRICS_CSV_FIELDS = [
    "epoch",
    "train_loss",
    "dice_loss_et",
    "dice_loss_tc",
    "dice_loss_wt",
    "val_dice_et",
    "val_dice_tc",
    "val_dice_wt",
    "val_dice_mean",
    "learning_rate",
    "epoch_seconds",
]


@dataclass
class TrainConfig:
    """Hyperparameters for BraTS segmentation training."""

    max_epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    val_fraction: float = 0.2
    seed: int = 42
    crop_size: tuple[int, int, int] = (96, 96, 96)
    roi_size: tuple[int, int, int] = (96, 96, 96)
    num_train_samples: int = 2
    model_name: ModelName = "segresnet"
    # Legacy plateau knobs (used when scheduler_type == "plateau").
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    # Default: linear warmup (~5% of steps) then cosine annealing.
    # Rationale (LIMITATIONS §7): ReduceLROnPlateau with patience=10 left the
    # DiceCELoss run on a flat 1e-4 LR through noisy val Dice swings; cosine
    # with warmup damps early spikes and anneals without waiting for plateaus.
    scheduler_type: str = "cosine_warmup"  # "cosine_warmup" | "plateau"
    warmup_frac: float = 0.05
    cosine_eta_min: float = 1e-6
    grad_clip_max_norm: float = 1.0
    # MONAI CacheDataset fraction. Default 0 = plain Dataset (lazy). On a
    # 16 GB laptop, cache_rate>0 for 150+ full volumes risks OOM; prefer
    # PersistentDataset (cache_dir) if caching is needed.
    cache_rate: float = 0.0
    cache_dir: str | None = None
    # Heartbeat file interval (seconds) while an epoch is in progress.
    heartbeat_interval_s: float = 300.0
    amp: bool = True
    num_workers: int = 0
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_cases: int | None = None
    log_every_n_batches: int = 1


def build_brats_loss() -> DiceCELoss:
    """Multi-label Dice+CE over all [ET, TC, WT] channels (no background skip)."""
    return DiceCELoss(
        sigmoid=True,
        include_background=True,
        to_onehot_y=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )


def build_channel_dice_loss() -> DiceLoss:
    """Per-channel Dice loss monitor (reduction=none → shape (B, 3, ...))."""
    return DiceLoss(
        sigmoid=True,
        include_background=True,
        squared_pred=True,
        reduction="none",
    )


def assert_loss_covers_all_channels(
    loss_fn: DiceCELoss | DiceLoss,
    *,
    n_channels: int = OUT_CHANNELS,
) -> None:
    """Fail fast if the loss would skip channel 0 (ET) or not cover all regions."""
    dice_part = getattr(loss_fn, "dice", loss_fn)
    include_bg = bool(getattr(dice_part, "include_background", True))
    if not include_bg:
        raise AssertionError(
            "include_background=False skips channel 0 (ET) for multi-label BraTS; "
            "use include_background=True"
        )
    # Probe reduction=none Dice on a tiny batch.
    probe = DiceLoss(
        sigmoid=True,
        include_background=include_bg,
        reduction="none",
    )
    logits = torch.zeros(1, n_channels, 4, 4, 4)
    target = torch.zeros(1, n_channels, 4, 4, 4)
    target[:, :, 1:3, 1:3, 1:3] = 1.0
    per_ch = probe(logits, target)
    flat = per_ch.reshape(per_ch.shape[0], per_ch.shape[1], -1).mean(dim=-1)
    if flat.shape[1] != n_channels:
        raise AssertionError(
            f"Expected per-channel Dice loss with {n_channels} channels, got shape {tuple(per_ch.shape)}"
        )


def split_brats_cases(
    brats_nifti_root: str | Path | None = None,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split BraTS NIfTI cases into train and validation sets (default 80/20)."""
    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    records = build_brats_file_list(brats_nifti_root)
    train_files, val_files = split_train_val(
        records, val_fraction=val_fraction, seed=seed
    )
    logger.info(
        "BraTS split %.0f/%.0f: train=%d val=%d",
        (1 - val_fraction) * 100,
        val_fraction * 100,
        len(train_files),
        len(val_files),
    )
    return train_files, val_files


def _records_from_id_list(
    brats_nifti_root: str | Path,
    subject_ids: list[str],
) -> list[dict[str, Any]]:
    root = Path(brats_nifti_root)
    return [subject_to_datadict(root / sid) for sid in subject_ids]


def assert_disjoint_from_heldout(
    train_ids: list[str],
    val_ids: list[str],
    heldout_ids: list[str],
) -> None:
    """Raise if any held-out subject leaks into train or val."""
    held = set(heldout_ids)
    leak_train = sorted(set(train_ids) & held)
    leak_val = sorted(set(val_ids) & held)
    if leak_train or leak_val:
        raise AssertionError(
            f"Held-out leakage: train∩heldout={leak_train} val∩heldout={leak_val}"
        )


def _build_dataloaders(
    train_files: list[dict[str, Any]],
    val_files: list[dict[str, Any]],
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    train_tf = get_train_transforms(
        pixdim=config.pixdim,
        crop_size=config.crop_size,
        num_samples=config.num_train_samples,
    )
    val_tf = get_val_transforms(pixdim=config.pixdim)

    cache_rate = float(config.cache_rate)
    if config.cache_dir:
        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using PersistentDataset cache_dir=%s", cache_dir)
        train_ds = PersistentDataset(
            data=train_files, transform=train_tf, cache_dir=str(cache_dir / "train")
        )
        val_ds = PersistentDataset(
            data=val_files, transform=val_tf, cache_dir=str(cache_dir / "val")
        )
    elif cache_rate > 0.0:
        logger.info(
            "Using CacheDataset cache_rate=%.2f (set 0.0 on low-RAM hosts)",
            cache_rate,
        )
        train_ds = CacheDataset(
            data=train_files, transform=train_tf, cache_rate=cache_rate, num_workers=0
        )
        val_ds = CacheDataset(
            data=val_files, transform=val_tf, cache_rate=min(1.0, cache_rate), num_workers=0
        )
    else:
        logger.info("Using lazy Dataset (cache_rate=0) — safest for laptop RAM")
        train_ds = Dataset(data=train_files, transform=train_tf)
        val_ds = Dataset(data=val_files, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    *,
    steps_per_epoch: int,
) -> tuple[Any, str]:
    """Build LR scheduler. Returns ``(scheduler, step_mode)`` where step_mode is
    ``"batch"`` (cosine warmup) or ``"epoch_val"`` (plateau on val Dice).
    """
    if config.scheduler_type == "plateau":
        # Retuned vs historical patience=10: react within ~1/4 of a 20-epoch run.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
        logger.info(
            "LR schedule: ReduceLROnPlateau(factor=%.2f, patience=%d)",
            config.scheduler_factor,
            config.scheduler_patience,
        )
        return sched, "epoch_val"

    if config.scheduler_type != "cosine_warmup":
        raise ValueError(f"Unknown scheduler_type={config.scheduler_type!r}")

    total_steps = max(1, int(config.max_epochs) * max(1, steps_per_epoch))
    warmup_steps = max(1, int(round(config.warmup_frac * total_steps)))
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=float(config.cosine_eta_min),
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    logger.info(
        "LR schedule: linear warmup %d steps (%.0f%%) then cosine %d steps "
        "(eta_min=%.1e); total_steps=%d",
        warmup_steps,
        100 * config.warmup_frac,
        cosine_steps,
        config.cosine_eta_min,
        total_steps,
    )
    return sched, "batch"


def post_transforms() -> tuple[Compose, Compose]:
    """Sigmoid + threshold for multi-label BraTS region predictions."""
    pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    label = Compose([])
    return pred, label


def _post_transforms() -> tuple[Compose, Compose]:
    """Backward-compatible alias for :func:`post_transforms`."""
    return post_transforms()


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DiceCELoss,
    channel_dice_loss: DiceLoss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    *,
    epoch: int = 0,
    log_every_n_batches: int = 1,
    scheduler: Any | None = None,
    scheduler_step_mode: str = "epoch_val",
    grad_clip_max_norm: float = 1.0,
) -> tuple[float, tuple[float, float, float]]:
    model.train()
    total_loss = 0.0
    channel_sums = torch.zeros(OUT_CHANNELS, dtype=torch.float64)
    n_batches = 0
    n_total = len(loader)

    for batch in loader:
        inputs = batch["image"].to(device)
        labels = brats_label_to_regions(batch["label"].to(device))

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            with torch.no_grad():
                per_ch = channel_dice_loss(outputs.detach(), labels)
                # (B, C, 1, 1, 1) → (C,)
                per_ch_mean = per_ch.reshape(per_ch.shape[0], per_ch.shape[1], -1).mean(
                    dim=(0, 2)
                )
                if per_ch_mean.numel() != OUT_CHANNELS:
                    raise AssertionError(
                        f"Per-channel Dice loss must have {OUT_CHANNELS} entries, "
                        f"got shape {tuple(per_ch.shape)}"
                    )

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

        total_loss += float(loss.item())
        channel_sums += per_ch_mean.detach().float().cpu().double()
        n_batches += 1
        if log_every_n_batches > 0 and (
            n_batches == 1 or n_batches % log_every_n_batches == 0 or n_batches == n_total
        ):
            logger.info(
                "Epoch %d train batch %d/%d | loss=%.4f | dice_ch=%s | lr=%.2e",
                epoch + 1,
                n_batches,
                n_total,
                float(loss.item()),
                [round(float(x), 4) for x in per_ch_mean.tolist()],
                optimizer.param_groups[0]["lr"],
            )

    mean_loss = total_loss / max(n_batches, 1)
    ch = tuple(float(x) for x in (channel_sums / max(n_batches, 1)).tolist())
    assert len(ch) == OUT_CHANNELS
    logger.info(
        "Epoch %d train channel Dice losses ET/TC/WT = %.4f / %.4f / %.4f",
        epoch + 1,
        ch[0],
        ch[1],
        ch[2],
    )
    return mean_loss, ch  # type: ignore[return-value]


@torch.no_grad()
def _validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    dice_metric: DiceMetric,
    post_pred: Compose,
    post_label: Compose,
    device: torch.device,
    roi_size: tuple[int, int, int],
    use_amp: bool,
    *,
    epoch: int = 0,
    log_every_n_batches: int = 1,
) -> tuple[float, tuple[float, float, float]]:
    """Return mean Dice and per-channel Dice (ET, TC, WT) on raw sigmoid>0.5 preds."""
    model.eval()
    dice_metric.reset()
    n_batches = 0
    n_total = len(loader)

    for batch in loader:
        inputs = batch["image"].to(device)
        labels = brats_label_to_regions(batch["label"].to(device))

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = sliding_window_inference(
                inputs,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
            )

        outputs_list = decollate_batch(outputs)
        labels_list = decollate_batch(labels)
        outputs_list = [post_pred(o) for o in outputs_list]
        labels_list = [post_label(l) for l in labels_list]
        dice_metric(y_pred=outputs_list, y=labels_list)
        n_batches += 1
        if log_every_n_batches > 0 and (
            n_batches == 1 or n_batches % log_every_n_batches == 0 or n_batches == n_total
        ):
            logger.info(
                "Epoch %d val batch %d/%d",
                epoch + 1,
                n_batches,
                n_total,
            )

    agg = dice_metric.aggregate()  # (N, C) with reduction=none
    if agg.ndim == 0:
        mean = float(agg.item())
        return mean, (mean, mean, mean)
    # Average over cases; keep channels.
    if agg.ndim == 1:
        per_ch = agg
    else:
        per_ch = torch.nanmean(agg, dim=0)
    if per_ch.numel() != OUT_CHANNELS:
        raise AssertionError(
            f"Expected {OUT_CHANNELS} val Dice channels, got shape {tuple(agg.shape)}"
        )
    ch = tuple(float(x) for x in per_ch.tolist())
    mean = float(sum(ch) / len(ch))
    return mean, ch  # type: ignore[return-value]


def _append_heartbeat(
    path: Path,
    message: str,
    *,
    epoch: int | None = None,
) -> None:
    """Append a timestamped heartbeat line (survives crashes for post-mortem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    ep = f" epoch={epoch}" if epoch is not None else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{ts}{ep} | {message}\n")


class _HeartbeatWatcher:
    """Daemon thread writing heartbeat.log every ``interval_s`` seconds."""

    def __init__(self, path: Path, interval_s: float = 300.0) -> None:
        self.path = path
        self.interval_s = max(30.0, float(interval_s))
        self._stop = threading.Event()
        self._epoch = 0
        self._thread = threading.Thread(
            target=self._run, name="train-heartbeat", daemon=True
        )

    def start(self) -> None:
        self._thread.start()
        _append_heartbeat(self.path, "heartbeat watcher started")

    def set_epoch(self, epoch_1based: int) -> None:
        self._epoch = int(epoch_1based)

    def stop(self, message: str = "heartbeat watcher stopped") -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5)
        _append_heartbeat(self.path, message, epoch=self._epoch or None)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            _append_heartbeat(
                self.path,
                f"alive (interval={self.interval_s:.0f}s)",
                epoch=self._epoch or None,
            )


def _append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRICS_CSV_FIELDS})


def best_epoch_from_metrics(metrics_csv: str | Path | Any) -> int:
    """Return the 1-based epoch with the highest ``val_dice_mean`` (ties → first)."""
    import pandas as pd

    df = (
        metrics_csv
        if hasattr(metrics_csv, "loc")
        else pd.read_csv(metrics_csv)
    )
    if df.empty or "val_dice_mean" not in df.columns:
        raise ValueError("metrics table is empty or missing val_dice_mean")
    idx = df["val_dice_mean"].astype(float).idxmax()
    return int(df.loc[idx, "epoch"])


def checkpoint_epoch_1based(ckpt: dict[str, Any]) -> int:
    """Normalize checkpoint epoch to 1-based (matches metrics.csv).

    New checkpoints store 1-based ``epoch`` with ``epoch_1based=True``.
    Legacy checkpoints stored 0-based ``epoch``.
    """
    raw = int(ckpt["epoch"])
    if ckpt.get("epoch_1based"):
        return raw
    return raw + 1


def assert_best_checkpoint_matches_metrics(
    best_path: str | Path,
    metrics_csv: str | Path,
    *,
    atol: float = 1e-5,
) -> None:
    """Fail if ``best_model.pt`` is not the metrics.csv argmax of val_dice_mean.

    Catches stale ``best_val_dice`` / wrong-epoch saves: if any later epoch has a
    strictly higher ``val_dice_mean`` than the epoch stored in the checkpoint,
    this raises.
    """
    import pandas as pd

    best_path = Path(best_path)
    metrics_csv = Path(metrics_csv)
    df = pd.read_csv(metrics_csv)
    expected_epoch = best_epoch_from_metrics(df)
    expected_dice = float(df.loc[df["epoch"] == expected_epoch, "val_dice_mean"].iloc[0])

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    ckpt_epoch = checkpoint_epoch_1based(ckpt)
    ckpt_dice = float(ckpt.get("val_dice", float("nan")))

    if ckpt_epoch != expected_epoch:
        raise AssertionError(
            f"best_model.pt epoch={ckpt_epoch} but metrics.csv argmax "
            f"val_dice_mean is epoch={expected_epoch} "
            f"(mean={expected_dice:.6f}); checkpoint val_dice={ckpt_dice:.6f}"
        )
    if abs(ckpt_dice - expected_dice) > atol:
        raise AssertionError(
            f"best_model.pt val_dice={ckpt_dice:.6f} != metrics.csv "
            f"epoch {expected_epoch} val_dice_mean={expected_dice:.6f}"
        )


def update_best_dice(val_dice: float, best_dice: float) -> tuple[float, bool]:
    """Strictly improve best mean val Dice. Returns ``(new_best, is_new_best)``."""
    if val_dice > best_dice:
        return float(val_dice), True
    return float(best_dice), False


def train_model(
    output_dir: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    config: TrainConfig | None = None,
    model_config: BraTSModelConfig | None = None,
    device: str | None = None,
    log_dir: str | Path | None = None,
    train_subjects_file: str | Path | None = None,
    val_subjects_file: str | Path | None = None,
    heldout_subjects_file: str | Path | None = None,
    resume: bool = False,
) -> Path:
    """Train a BraTS segmentation model and save the best validation checkpoint.

    Uses ``DiceCELoss`` (all three [ET, TC, WT] channels), Adam,
    cosine warmup (default) or ``ReduceLROnPlateau``, gradient clipping,
    mixed precision (when CUDA is available), TensorBoard logging, and either
    explicit subject lists or an 80/20 split.

    Returns
    -------
    Path
        Path to ``best_model.pt`` under ``output_dir``.
    """
    config = config or TrainConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir is not None else output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    if brats_nifti_root is None:
        brats_nifti_root = load_config().paths.brats_nifti
    brats_nifti_root = Path(brats_nifti_root)

    device_t = get_device(device)
    use_amp = config.amp and device_t.type == "cuda"
    scaler = torch.amp.GradScaler(device=device_t.type, enabled=use_amp)

    if train_subjects_file is not None and val_subjects_file is not None:
        train_ids = read_id_list(train_subjects_file)
        val_ids = read_id_list(val_subjects_file)
        if heldout_subjects_file is not None:
            heldout_ids = read_id_list(heldout_subjects_file)
            assert_disjoint_from_heldout(train_ids, val_ids, heldout_ids)
            logger.info(
                "Confirmed train/val disjoint from held-out (%d subjects)",
                len(heldout_ids),
            )
        train_files = _records_from_id_list(brats_nifti_root, train_ids)
        val_files = _records_from_id_list(brats_nifti_root, val_ids)
    else:
        train_files, val_files = split_brats_cases(
            brats_nifti_root,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )
        if config.max_cases is not None:
            n_total = max(1, config.max_cases)
            n_val = max(1, min(len(val_files), max(1, n_total // 5)))
            n_train = max(1, n_total - n_val)
            train_files = train_files[:n_train]
            val_files = val_files[:n_val]
            logger.info(
                "Limited to max_cases=%d -> train=%d val=%d",
                config.max_cases,
                len(train_files),
                len(val_files),
            )
        if heldout_subjects_file is not None:
            heldout_ids = read_id_list(heldout_subjects_file)
            assert_disjoint_from_heldout(
                [r["subject_id"] for r in train_files],
                [r["subject_id"] for r in val_files],
                heldout_ids,
            )

    # Persist subject lists + full config next to checkpoints.
    train_list_path = output_dir / "train_cases.txt"
    val_list_path = output_dir / "train_run_val_cases.txt"
    train_list_path.write_text(
        "\n".join(r["subject_id"] for r in train_files) + "\n",
        encoding="utf-8",
    )
    val_list_path.write_text(
        "\n".join(r["subject_id"] for r in val_files) + "\n",
        encoding="utf-8",
    )
    config_path = output_dir / "train_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote train/val lists + config -> %s", output_dir)

    train_loader, val_loader = _build_dataloaders(train_files, val_files, config)

    model = build_brats_model(config.model_name, model_config).to(device_t)
    loss_fn = build_brats_loss()
    assert_loss_covers_all_channels(loss_fn)
    channel_dice_loss = build_channel_dice_loss()
    assert_loss_covers_all_channels(channel_dice_loss)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler, scheduler_step_mode = build_lr_scheduler(
        optimizer, config, steps_per_epoch=len(train_loader)
    )
    dice_metric = DiceMetric(include_background=True, reduction="none")
    post_pred, post_label = _post_transforms()

    writer = SummaryWriter(log_dir=str(log_dir))
    best_dice = -1.0
    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest.pt"
    # Keep last_model.pt as alias for backward compatibility with older scripts.
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
        # New ckpts: 1-based completed epoch → next 0-based index is that value.
        # Legacy: 0-based epoch → next is epoch+1.
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
            f"(completed={checkpoint_epoch_1based(ckpt)}, best={best_dice:.4f})",
        )

    logger.info(
        "Training %s on %s | train=%d val=%d | amp=%s | loss=DiceCELoss(bg=True) | "
        "sched=%s | grad_clip=%.1f | cache_rate=%.2f",
        config.model_name,
        device_t,
        len(train_files),
        len(val_files),
        use_amp,
        config.scheduler_type,
        config.grad_clip_max_norm,
        config.cache_rate,
    )
    logger.info("Regions / channels: %s", list(BRATS_REGIONS))

    heartbeat.start()
    try:
        for epoch in range(start_epoch, config.max_epochs):
            heartbeat.set_epoch(epoch + 1)
            _append_heartbeat(
                heartbeat_path,
                f"epoch {epoch + 1}/{config.max_epochs} START",
                epoch=epoch + 1,
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
            if scheduler_step_mode == "epoch_val":
                scheduler.step(val_dice)
            lr = float(optimizer.param_groups[0]["lr"])

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/dice_loss_et", dl_et, epoch)
            writer.add_scalar("train/dice_loss_tc", dl_tc, epoch)
            writer.add_scalar("train/dice_loss_wt", dl_wt, epoch)
            writer.add_scalar("val/dice_mean", val_dice, epoch)
            writer.add_scalar("val/dice_et", vd_et, epoch)
            writer.add_scalar("val/dice_tc", vd_tc, epoch)
            writer.add_scalar("val/dice_wt", vd_wt, epoch)
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
                "Epoch %d/%d | train_loss=%.4f | dice_ch=[%.3f,%.3f,%.3f] | "
                "val_dice=%.4f (ET=%.3f TC=%.3f WT=%.3f) | %.1fs | lr=%.2e",
                epoch + 1,
                config.max_epochs,
                train_loss,
                dl_et,
                dl_tc,
                dl_wt,
                val_dice,
                vd_et,
                vd_tc,
                vd_wt,
                epoch_seconds,
                lr,
            )
            _append_heartbeat(
                heartbeat_path,
                f"epoch {epoch + 1}/{config.max_epochs} END "
                f"val_dice={val_dice:.4f} lr={lr:.2e} sec={epoch_seconds:.1f}",
                epoch=epoch + 1,
            )

            checkpoint = {
                # 1-based epoch to match metrics.csv (legacy ckpts used 0-based).
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
                # Update best BEFORE writing so latest.pt never carries a stale
                # best_val_dice (resume would otherwise overwrite best_model with
                # a worse later epoch that still beats the stale value).
                "best_val_dice": best_dice,
                "model_name": config.model_name,
                "train_config": asdict(config),
                "loss": "DiceCELoss(include_background=True, sigmoid=True)",
            }
            # Select best by mean val Dice over the 3 channels (val set ONLY).
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
        raise RuntimeError("Training finished without saving a best checkpoint")
    assert_best_checkpoint_matches_metrics(best_path, metrics_csv)
    return best_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train BraTS segmentation model.")
    parser.add_argument("--config", type=Path, default=None, help="Project config.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: checkpoints/brats_pretrain)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model", choices=("segresnet", "unet"), default="segresnet")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit total BraTS cases (smoke / CPU debug). Splits ~80/20 of this subset.",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=None,
        help="Explicit train subject IDs (one per line)",
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        default=None,
        help="Explicit val subject IDs (one per line)",
    )
    parser.add_argument(
        "--heldout-list",
        type=Path,
        default=None,
        help="Held-out IDs to assert disjoint from train/val",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output_dir/latest.pt if present",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log every N train/val batches (default: 1)",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    train_cfg = app_cfg.training.pretrain
    max_cases = args.max_cases if args.max_cases is not None else train_cfg.max_cases
    # When explicit lists are provided, ignore max_cases truncation.
    if args.train_list is not None and args.val_list is not None:
        max_cases = None

    config = TrainConfig(
        max_epochs=args.epochs if args.epochs is not None else train_cfg.epochs,
        batch_size=args.batch_size if args.batch_size is not None else train_cfg.batch_size,
        learning_rate=args.lr if args.lr is not None else train_cfg.learning_rate,
        amp=not args.no_amp,
        model_name=args.model,
        max_cases=max_cases,
        log_every_n_batches=max(1, args.log_every),
        seed=42,
    )
    if config.max_cases is not None:
        logging.getLogger(__name__).info(
            "Using BraTS subset max_cases=%d (from CLI or config.yaml)",
            config.max_cases,
        )
    output_dir = args.output_dir or (app_cfg.paths.checkpoints / "brats_pretrain")
    train_model(
        output_dir,
        config=config,
        device=args.device,
        train_subjects_file=args.train_list,
        val_subjects_file=args.val_list,
        heldout_subjects_file=args.heldout_list,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
