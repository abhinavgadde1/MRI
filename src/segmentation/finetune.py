"""Fine-tune a BraTS-pretrained model on manually corrected clinical cases."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter

from config import load_config
from segmentation.dataset import get_train_transforms, get_val_transforms
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.model import ModelName, build_brats_model
from segmentation.pseudo_label import REGISTERED_NAMES
from segmentation.train import (
    _train_epoch,
    _validate_epoch,
    post_transforms,
)

logger = logging.getLogger(__name__)

CORRECTED_LABEL_NAMES: tuple[str, ...] = (
    "corrected_tumor_label.nii.gz",
    "manual_tumor_label.nii.gz",
)


@dataclass
class FinetuneConfig:
    """Hyperparameters for clinical fine-tuning."""

    max_epochs: int = 50
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    seed: int = 42
    crop_size: tuple[int, int, int] = (96, 96, 96)
    roi_size: tuple[int, int, int] = (96, 96, 96)
    num_train_samples: int = 2
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    amp: bool = True
    num_workers: int = 0
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_test_cases: int = 4
    min_test_cases: int = 3
    max_test_cases: int = 5
    val_fraction: float = 0.15
    freeze_encoder: bool = False
    n_frozen_down_blocks: int = 2


def _find_corrected_label(reg_dir: Path) -> Path | None:
    for name in CORRECTED_LABEL_NAMES:
        path = reg_dir / name
        if path.is_file():
            return path
    return None


def discover_corrected_cases(corrected_root: str | Path) -> list[Path]:
    """Find study folders with registered scans and a manually corrected label."""
    root = Path(corrected_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Corrected cases root not found: {root}")

    cases: list[Path] = []
    for study_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        reg_dir = study_dir / "04_registered_1mm"
        if not reg_dir.is_dir():
            continue
        label_path = _find_corrected_label(reg_dir)
        if label_path is None:
            logger.debug("Skipping %s: no corrected label", study_dir.name)
            continue
        missing = [
            name for name in REGISTERED_NAMES.values() if not (reg_dir / name).is_file()
        ]
        if missing:
            logger.warning("Skipping %s: missing registered scans %s", study_dir.name, missing)
            continue
        cases.append(study_dir)
    logger.info("Discovered %d corrected patient cases under %s", len(cases), root)
    return cases


def corrected_case_to_datadict(study_dir: str | Path) -> dict[str, Any]:
    """MONAI record for one manually corrected clinical study."""
    from preprocessing.h5_to_nifti import MODALITY_NAMES

    study_dir = Path(study_dir)
    reg_dir = study_dir / "04_registered_1mm"
    label_path = _find_corrected_label(reg_dir)
    if label_path is None:
        raise FileNotFoundError(f"No corrected label in {reg_dir}")

    return {
        "subject_id": study_dir.name,
        "image": [str(reg_dir / REGISTERED_NAMES[m]) for m in MODALITY_NAMES],
        "label": str(label_path),
    }


def build_corrected_file_list(corrected_root: str | Path) -> list[dict[str, Any]]:
    return [corrected_case_to_datadict(p) for p in discover_corrected_cases(corrected_root)]


def split_finetune_sets(
    records: Sequence[dict[str, Any]],
    *,
    n_test_cases: int = 4,
    min_test_cases: int = 3,
    max_test_cases: int = 5,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out 3–5 test cases; split the remainder into train and val."""
    items = list(records)
    if len(items) < min_test_cases + 2:
        raise ValueError(
            f"Need at least {min_test_cases + 2} corrected cases for fine-tuning; got {len(items)}"
        )

    n_test = max(min_test_cases, min(n_test_cases, max_test_cases, len(items) - 2))
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    test_files = shuffled[:n_test]
    remaining = shuffled[n_test:]
    n_val = max(1, int(round(len(remaining) * val_fraction)))
    val_files = remaining[:n_val]
    train_files = remaining[n_val:]
    if not train_files:
        raise ValueError("Train split empty after holding out test/val cases")

    logger.info(
        "Fine-tune split: train=%d val=%d test=%d (test holdout=%d)",
        len(train_files),
        len(val_files),
        len(test_files),
        n_test,
    )
    return train_files, val_files, test_files


def freeze_early_encoder(
    model: torch.nn.Module,
    model_name: ModelName,
    *,
    n_down_blocks: int = 2,
) -> int:
    """Freeze early encoder blocks; return number of frozen parameter tensors."""
    frozen = 0
    if model_name == "segresnet":
        blocks: list[torch.nn.Module] = [model.convInit]
        blocks.extend(list(model.down_layers[:n_down_blocks]))
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False
                frozen += 1
    elif model_name == "unet":
        prefixes = _unet_encoder_prefixes(n_down_blocks)
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                param.requires_grad = False
                frozen += 1
    else:
        raise ValueError(f"Unsupported model for freezing: {model_name}")
    logger.info(
        "Froze %d parameter tensors in early encoder (%s, n_down=%d)",
        frozen,
        model_name,
        n_down_blocks,
    )
    return frozen


def _unet_encoder_prefixes(n_down_blocks: int) -> list[str]:
    """Encoder path prefixes for MONAI UNet (``model.0.``, ``model.1.submodule.0.``, ...)."""
    prefixes = ["model.0."]
    path = "model.1"
    for _ in range(max(0, n_down_blocks - 1)):
        prefixes.append(f"{path}.submodule.0.")
        path = f"{path}.submodule.1"
    return prefixes


def _build_finetune_dataloaders(
    train_files: list[dict[str, Any]],
    val_files: list[dict[str, Any]],
    config: FinetuneConfig,
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


def finetune_model(
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    corrected_root: str | Path,
    *,
    config: FinetuneConfig | None = None,
    device: str | None = None,
    log_dir: str | Path | None = None,
) -> Path:
    """Fine-tune a BraTS checkpoint on manually corrected real-patient cases.

    Holds out ``3–5`` test cases that never receive gradient updates. Saves
    ``best_model.pt`` based on validation Dice from the non-test training pool.
    """
    config = config or FinetuneConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir is not None else output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    records = build_corrected_file_list(corrected_root)
    train_files, val_files, test_files = split_finetune_sets(
        records,
        n_test_cases=config.n_test_cases,
        min_test_cases=config.min_test_cases,
        max_test_cases=config.max_test_cases,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    split_manifest = {
        "train": [r["subject_id"] for r in train_files],
        "val": [r["subject_id"] for r in val_files],
        "test": [r["subject_id"] for r in test_files],
    }
    (output_dir / "finetune_split.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = config.amp and device_t.type == "cuda"
    scaler = torch.amp.GradScaler(device=device_t.type, enabled=use_amp)

    ckpt = torch.load(pretrained_checkpoint, map_location=device_t)
    model_name: ModelName = ckpt.get("model_name", "segresnet") if isinstance(ckpt, dict) else "segresnet"
    model = build_brats_model(model_name).to(device_t)
    model.load_state_dict(ckpt["model_state"])

    if config.freeze_encoder:
        freeze_early_encoder(
            model,
            model_name,
            n_down_blocks=config.n_frozen_down_blocks,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Fine-tuning %s | trainable params=%d / %d (%.1f%%)",
        model_name,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )

    train_loader, val_loader = _build_finetune_dataloaders(train_files, val_files, config)
    test_loader = DataLoader(
        Dataset(data=test_files, transform=get_val_transforms(pixdim=config.pixdim)),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
    )

    loss_fn = DiceLoss(include_background=False, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
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
    post_pred, post_label = post_transforms()

    writer = SummaryWriter(log_dir=str(log_dir))
    best_dice = -1.0
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"

    logger.info(
        "Fine-tuning on %s | lr=%.2e | train=%d val=%d test=%d | freeze_encoder=%s",
        device_t,
        config.learning_rate,
        len(train_files),
        len(val_files),
        len(test_files),
        config.freeze_encoder,
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
            test_dice = _validate_epoch(
                model,
                test_loader,
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
            writer.add_scalar("test/dice_mean", test_dice, epoch)
            writer.add_scalar("train/lr", lr, epoch)

            logger.info(
                "Epoch %d/%d | loss=%.4f | val_dice=%.4f | test_dice=%.4f | lr=%.2e",
                epoch + 1,
                config.max_epochs,
                train_loss,
                val_dice,
                test_dice,
                lr,
            )

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_dice": val_dice,
                "test_dice": test_dice,
                "best_val_dice": best_dice,
                "model_name": model_name,
                "pretrained_checkpoint": str(pretrained_checkpoint),
                "finetune_config": asdict(config),
                "finetune_split": split_manifest,
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
        raise RuntimeError("Fine-tuning finished without saving a best checkpoint")
    return best_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune segmentation on corrected cases.")
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="BraTS-pretrained checkpoint (default: checkpoints/brats_pretrain/best_model.pt)",
    )
    parser.add_argument(
        "--corrected-root",
        type=Path,
        default=None,
        help="Directory of manually corrected patient studies",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--test-cases", type=int, default=4, help="Test holdout count (3-5)")
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze early encoder blocks (convInit + down layers)",
    )
    parser.add_argument("--frozen-down-blocks", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app_cfg = load_config(args.config)
    ft_cfg = app_cfg.training.finetune
    config = FinetuneConfig(
        max_epochs=args.epochs or ft_cfg.epochs,
        batch_size=args.batch_size or ft_cfg.batch_size,
        learning_rate=args.lr or ft_cfg.learning_rate,
        amp=not args.no_amp,
        n_test_cases=max(3, min(5, args.test_cases)),
        freeze_encoder=args.freeze_encoder,
        n_frozen_down_blocks=args.frozen_down_blocks,
    )

    pretrained = args.pretrained or (app_cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt")
    corrected_root = args.corrected_root or (app_cfg.paths.processed / "real_patients" / "corrected")
    output_dir = args.output_dir or (app_cfg.paths.checkpoints / "clinical_finetune")

    finetune_model(pretrained, output_dir, corrected_root, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
