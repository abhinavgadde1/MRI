"""MONAI training loop for BraTS multi-modal segmentation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.tensorboard import SummaryWriter

from config import load_config
from segmentation.dataset import (
    build_brats_file_list,
    get_train_transforms,
    get_val_transforms,
    split_train_val,
)
from segmentation.model import (
    OUT_CHANNELS,
    BraTSModelConfig,
    ModelName,
    brats_label_to_regions,
    build_brats_model,
)

logger = logging.getLogger(__name__)


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
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    amp: bool = True
    num_workers: int = 0
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)


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


def _build_dataloaders(
    train_files: list[dict[str, Any]],
    val_files: list[dict[str, Any]],
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    train_ds = Dataset(
        data=train_files,
        transform=get_train_transforms(
            pixdim=config.pixdim,
            crop_size=config.crop_size,
            num_samples=config.num_train_samples,
        ),
    )
    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms(pixdim=config.pixdim),
    )
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
    loss_fn: DiceLoss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inputs = batch["image"].to(device)
        labels = brats_label_to_regions(batch["label"].to(device))

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


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
) -> float:
    model.eval()
    dice_metric.reset()

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

    return float(dice_metric.aggregate().item())


def train_model(
    output_dir: str | Path,
    *,
    brats_nifti_root: str | Path | None = None,
    config: TrainConfig | None = None,
    model_config: BraTSModelConfig | None = None,
    device: str | None = None,
    log_dir: str | Path | None = None,
) -> Path:
    """Train a BraTS segmentation model and save the best validation checkpoint.

    Uses ``DiceLoss``, Adam, ``ReduceLROnPlateau``, mixed precision (when CUDA
    is available), TensorBoard logging, and an 80/20 train/val split by default.

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

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = config.amp and device_t.type == "cuda"
    scaler = torch.amp.GradScaler(device=device_t.type, enabled=use_amp)

    train_files, val_files = split_brats_cases(
        brats_nifti_root,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_loader, val_loader = _build_dataloaders(train_files, val_files, config)

    model = build_brats_model(config.model_name, model_config).to(device_t)
    loss_fn = DiceLoss(include_background=False, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred, post_label = _post_transforms()

    writer = SummaryWriter(log_dir=str(log_dir))
    best_dice = -1.0
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"

    logger.info(
        "Training %s on %s | train=%d val=%d | amp=%s",
        config.model_name,
        device_t,
        len(train_files),
        len(val_files),
        use_amp,
    )

    try:
        for epoch in range(config.max_epochs):
            train_loss = _train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device_t,
                scaler,
                use_amp,
            )
            val_dice = _validate_epoch(
                model,
                val_loader,
                dice_metric,
                post_pred,
                post_label,
                device_t,
                config.roi_size,
                use_amp,
            )
            scheduler.step(val_dice)
            lr = optimizer.param_groups[0]["lr"]

            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/dice_mean", val_dice, epoch)
            writer.add_scalar("train/lr", lr, epoch)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_dice=%.4f | lr=%.2e",
                epoch + 1,
                config.max_epochs,
                train_loss,
                val_dice,
                lr,
            )

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_dice": val_dice,
                "best_val_dice": best_dice,
                "model_name": config.model_name,
                "train_config": asdict(config),
            }
            torch.save(checkpoint, last_path)

            if val_dice > best_dice:
                best_dice = val_dice
                checkpoint["best_val_dice"] = best_dice
                torch.save(checkpoint, best_path)
                logger.info("New best val Dice %.4f -> %s", best_dice, best_path)
    finally:
        writer.close()

    if not best_path.is_file():
        raise RuntimeError("Training finished without saving a best checkpoint")
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
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    train_cfg = app_cfg.training.pretrain
    config = TrainConfig(
        max_epochs=args.epochs or train_cfg.epochs,
        batch_size=args.batch_size or train_cfg.batch_size,
        learning_rate=args.lr or train_cfg.learning_rate,
        amp=not args.no_amp,
        model_name=args.model,
    )
    output_dir = args.output_dir or (app_cfg.paths.checkpoints / "brats_pretrain")
    train_model(output_dir, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
