"""Experimental fine-tune from v3 on 4 Claude drafts + BraTS replay.

**Experimental / unverified.** Trained on machine-assisted drafts that are
``draft_unverified``. Does **not** overwrite ``checkpoints/brats_*`` and does
**not** change the default CLI/API checkpoint.

Channel policy: batch2_01 / batch2_12 → WT loss only; batch2_07 / batch2_10 →
ET+TC+WT. Replay ≥3 BraTS train cases per draft case.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, SpatialPadd
from scipy import ndimage as ndi

from config import PROJECT_ROOT, load_config
from device import get_device
from preprocessing.h5_to_nifti import MODALITY_NAMES
from segmentation.batch2_augment import DomainAugmentFlags, DomainAugmentd, log_augment_flags
from segmentation.batch2_cv import fixed_heldout_subset
from segmentation.batch2_finetune import (
    DEFAULT_V3_CHECKPOINT,
    assert_safe_checkpoint_output_dir,
    load_brats_replay_pool,
    sample_replay_mix,
)
from segmentation.dataset import IMAGE_KEY, LABEL_KEY, get_train_transforms, get_val_transforms
from segmentation.evaluate import load_model_from_checkpoint
from segmentation.hdglio_labels import brats_regions_from_classic
from segmentation.model import BRATS_REGIONS, ModelName, brats_label_to_regions, build_brats_model
from segmentation.postprocess import apply_channel_postprocess, enforce_brats_nesting
from segmentation.pseudo_label import get_patient_inference_transforms
from segmentation.train import (
    TrainConfig,
    _HeartbeatWatcher,
    _append_heartbeat,
    _append_metrics_row,
    build_lr_scheduler,
    post_transforms,
)

logger = logging.getLogger(__name__)

DRAFT_STUDIES = ("batch2_01", "batch2_07", "batch2_10", "batch2_12")
WT_ONLY_STUDIES = frozenset({"batch2_01", "batch2_12"})
NON_DRAFT_STUDIES = ("batch2_02", "batch2_03", "batch2_05", "batch2_06", "batch2_08", "batch2_13")
LABELING_ROOT = PROJECT_ROOT / "outputs" / "batch2_labeling"
HDGLIO_ROOT = PROJECT_ROOT / "outputs" / "batch2_second_opinion"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "experimental_draft_ft"
RESULTS_DIR = PROJECT_ROOT / "results" / "experimental_draft_ft"
CLAUDE_DRAFT = "claude_draft_label.nii.gz"
DISCLAIMER = (
    "Experimental checkpoint trained on unverified machine-assisted drafts "
    "(draft_unverified). Agreement with drafts is not accuracy. Not for clinical use. "
    "Not the default CLI/API checkpoint."
)
ET_CH, TC_CH, WT_CH = 0, 1, 2
REGRESSION_WT_DROP = 0.02
HELDOUT_N = 20
HELDOUT_EARLYSTOP_N = 5  # cheap monitor; SUMMARY uses HELDOUT_N
EARLY_STOP_PATIENCE = 6


@dataclass
class ExpFTConfig:
    max_epochs: int = 30
    batch_size: int = 1
    learning_rate: float = 2e-5
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
    heartbeat_interval_s: float = 120.0
    amp: bool = True
    num_workers: int = 0
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    log_every_n_batches: int = 5
    # ≥3 BraTS per draft case
    replay_draft: int = 1
    replay_brats: int = 3
    domain_aug: DomainAugmentFlags = field(
        default_factory=lambda: DomainAugmentFlags(
            thick_slice=True,
            intensity=True,
            modality_dropout=True,
            t1c_dropout_prob=0.1,
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


def build_draft_records(
    study_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    ids = list(study_ids or DRAFT_STUDIES)
    records: list[dict[str, Any]] = []
    for sid in ids:
        pkg = LABELING_ROOT / sid
        label = pkg / CLAUDE_DRAFT
        if not label.is_file():
            raise FileNotFoundError(f"Missing Claude draft: {label}")
        imgs = {m: pkg / f"{m}.nii.gz" for m in MODALITY_NAMES}
        missing = [m for m, p in imgs.items() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{sid}: missing modalities {missing}")
        wt_only = sid in WT_ONLY_STUDIES
        # np.ndarray (3,) so default_collate → (B, 3), not a list of scalars.
        weights = (
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if wt_only
            else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )
        records.append(
            {
                "subject_id": sid,
                "patient_id": sid,
                "study_id": sid,
                "is_draft_label": True,
                "draft_status": "draft_unverified",
                "wt_only": wt_only,
                "source": "claude_draft",
                "channel_weights": weights,
                IMAGE_KEY: [str(imgs[m]) for m in MODALITY_NAMES],
                LABEL_KEY: str(label),
            }
        )
    return records


def _train_transforms(cfg: ExpFTConfig) -> Compose:
    base = get_train_transforms(
        pixdim=cfg.pixdim,
        crop_size=cfg.crop_size,
        num_samples=cfg.num_train_samples,
    )
    transforms = list(base.transforms)
    pad = SpatialPadd(keys=[IMAGE_KEY, LABEL_KEY], spatial_size=cfg.crop_size)
    inserted = False
    for i, t in enumerate(transforms):
        if t.__class__.__name__ == "RandCropByPosNegLabeld":
            transforms = transforms[:i] + [pad] + transforms[i:]
            inserted = True
            break
    if not inserted:
        transforms = [pad] + transforms
    aug = DomainAugmentd(keys=[IMAGE_KEY], label_key=LABEL_KEY, flags=cfg.domain_aug)
    if transforms and transforms[-1].__class__.__name__ == "EnsureTyped":
        transforms = transforms[:-1] + [aug, transforms[-1]]
    else:
        transforms = transforms + [aug]
    return Compose(transforms)


def channel_weighted_dicece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-channel soft Dice + BCE, mean over weighted channels.

    ``weights`` is (C,) or (B, C) with C=3 [ET, TC, WT].
    """
    if weights.ndim == 1:
        w = weights.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    else:
        w = weights.to(device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    # (B, C, ...)
    dims = tuple(range(2, probs.ndim))
    inter = (probs * labels).sum(dim=dims)
    denom = probs.sum(dim=dims) + labels.sum(dim=dims)
    dice = 1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    bce = bce.mean(dim=dims)
    per = dice + bce  # (B, C)
    # Normalize by sum of weights per sample
    w_sum = w.sum(dim=1).clamp_min(1e-8)
    return ((per * w).sum(dim=1) / w_sum).mean()


def _batch_channel_weights(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """(B, 3) weights; BraTS / missing → all ones."""
    raw = batch.get("channel_weights")
    bsz = batch["image"].shape[0]
    if raw is None:
        return torch.ones(bsz, 3, device=device)
    if isinstance(raw, torch.Tensor):
        w = raw.detach().float()
        if w.ndim == 1 and w.numel() == 3:
            w = w.view(1, 3).expand(bsz, -1)
        elif w.ndim == 1:
            w = w.view(bsz, -1)
        return w.to(device)[:, :3]
    # Collate sometimes yields list-of-3 tensors (one per channel).
    if isinstance(raw, (list, tuple)) and len(raw) == 3 and all(
        isinstance(x, torch.Tensor) for x in raw
    ):
        w = torch.stack([x.reshape(-1).float() for x in raw], dim=1)
        return w.to(device)
    try:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.broadcast_to(arr.reshape(1, -1), (bsz, arr.size)).copy()
        return torch.as_tensor(arr, device=device)[:, :3]
    except Exception:  # noqa: BLE001
        return torch.ones(bsz, 3, device=device)


def _is_brats_batch(batch: dict[str, Any]) -> bool:
    src = batch.get("source", "")
    if isinstance(src, (list, tuple)):
        return any(str(s) == "brats_replay" for s in src)
    return str(src) == "brats_replay"


@torch.no_grad()
def eval_brats_heldout_dice(
    model_or_ckpt: torch.nn.Module | str | Path,
    records: Sequence[dict[str, Any]],
    device: torch.device,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
) -> dict[str, float]:
    """Mean LCC Dice ET/TC/WT on BraTS records with masks."""
    if isinstance(model_or_ckpt, (str, Path)):
        model = load_model_from_checkpoint(model_or_ckpt, device)
    else:
        model = model_or_ckpt
    model.float().eval()
    post_pred, post_label = post_transforms()
    loader = DataLoader(
        Dataset(data=list(records), transform=get_val_transforms()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    sums = {r: 0.0 for r in BRATS_REGIONS}
    counts = {r: 0 for r in BRATS_REGIONS}
    for batch in loader:
        inputs = batch["image"].to(device=device, dtype=torch.float32)
        labels = brats_label_to_regions(batch["label"].to(device))
        logits = sliding_window_inference(
            inputs, roi_size=roi_size, sw_batch_size=1, predictor=model
        )
        pred = post_pred(decollate_batch(logits)[0]).detach().float().cpu().numpy()
        lab = post_label(decollate_batch(labels)[0]).detach().float().cpu().numpy()
        pred = apply_channel_postprocess(pred.astype(np.float32), mode="lcc")
        for i, region in enumerate(BRATS_REGIONS):
            p = pred[i] > 0.5
            g = lab[i] > 0.5
            inter = float(np.count_nonzero(p & g))
            denom = float(np.count_nonzero(p) + np.count_nonzero(g))
            if denom == 0:
                continue
            sums[region] += 2.0 * inter / denom
            counts[region] += 1
    out: dict[str, float] = {}
    for r in BRATS_REGIONS:
        out[r] = sums[r] / counts[r] if counts[r] else float("nan")
    out["mean"] = float(np.nanmean([out[r] for r in BRATS_REGIONS]))
    return out


@torch.no_grad()
def predict_study_regions(
    checkpoint: str | Path,
    study_id: str,
    device: torch.device,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
) -> dict[str, Any]:
    """Infer on a labeling-package study; return WT mask, volume mL, centroid mm."""
    pkg = LABELING_ROOT / study_id
    record = {
        "study_id": study_id,
        "image": [str(pkg / f"{m}.nii.gz") for m in MODALITY_NAMES],
    }
    model = load_model_from_checkpoint(checkpoint, device)
    model.float().eval()
    data = get_patient_inference_transforms()(record)
    image = data["image"].unsqueeze(0).to(device, dtype=torch.float32)
    logits = sliding_window_inference(
        image, roi_size=roi_size, sw_batch_size=1, predictor=model
    )
    post_pred, _ = post_transforms()
    pred = post_pred(decollate_batch(logits)[0]).detach().float().cpu().numpy()
    pred = enforce_brats_nesting(
        apply_channel_postprocess(pred.astype(np.float32), mode="lcc")
    )
    t1c = nib.load(str(pkg / "T1c.nii.gz"))
    spacing = tuple(float(z) for z in t1c.header.get_zooms()[:3])
    vox_ml = float(np.prod(spacing) / 1000.0)
    # Resize to T1c if needed
    if pred.shape[1:] != t1c.shape:
        out = np.zeros((3,) + t1c.shape, dtype=np.float32)
        for i in range(3):
            t = torch.from_numpy(pred[i]).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=t1c.shape, mode="nearest")
            out[i] = t.squeeze().numpy()
        pred = out
    wt = pred[WT_CH] > 0.5
    et = pred[ET_CH] > 0.5
    vol = float(wt.sum() * vox_ml)
    mask_for_cent = et if np.any(et) else wt
    if np.any(mask_for_cent):
        c = ndi.center_of_mass(mask_for_cent)
        centroid = tuple(float(c[i] * spacing[i]) for i in range(3))
    else:
        centroid = None
    return {
        "study_id": study_id,
        "WT_ml": vol,
        "ET_ml": float(et.sum() * vox_ml),
        "centroid_mm": centroid,
        "pred": pred,
        "spacing": spacing,
    }


def dice_vs_label(
    pred_regions: np.ndarray,
    classic_label: np.ndarray,
    region: str = "WT",
) -> float:
    gt = brats_regions_from_classic(classic_label)
    key = {"ET": "ET", "TC": "TC", "WT": "WT"}[region]
    ch = {"ET": ET_CH, "TC": TC_CH, "WT": WT_CH}[region]
    p = pred_regions[ch] > 0.5
    g = gt[key]
    inter = float(np.count_nonzero(p & g))
    denom = float(np.count_nonzero(p) + np.count_nonzero(g))
    if denom == 0:
        return float("nan")
    return 2.0 * inter / denom


def centroid_mm_from_classic(
    classic: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    regs = brats_regions_from_classic(classic)
    mask = regs["ET"] if np.any(regs["ET"]) else regs["WT"]
    if not np.any(mask):
        return None
    c = ndi.center_of_mass(mask)
    return tuple(float(c[i] * spacing[i]) for i in range(3))  # type: ignore[return-value]


def dist_mm(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def train_experimental(
    *,
    train_drafts: Sequence[dict[str, Any]],
    output_dir: str | Path,
    brats_pool: Sequence[dict[str, Any]],
    heldout_records: Sequence[dict[str, Any]],
    pretrained: str | Path | None = None,
    config: ExpFTConfig | None = None,
    device: str | None = None,
    resume: bool = False,
) -> Path:
    """Fine-tune with channel-masked draft loss + BraTS DiceCE; early-stop on heldout WT."""
    cfg = config or ExpFTConfig()
    output_dir = assert_safe_checkpoint_output_dir(output_dir)
    # Extra guard: never write under any checkpoints/brats_* tree.
    resolved = Path(output_dir).resolve()
    ckpt_root = (PROJECT_ROOT / "checkpoints").resolve()
    try:
        rel = resolved.relative_to(ckpt_root)
        if rel.parts and str(rel.parts[0]).startswith("brats_"):
            raise ValueError(f"Refusing to write under checkpoints/brats_*: {output_dir}")
    except ValueError as exc:
        if "Refusing" in str(exc):
            raise
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained = Path(pretrained or DEFAULT_V3_CHECKPOINT)
    device_t = get_device(device)
    use_amp = cfg.amp and device_t.type == "cuda"
    scaler = torch.amp.GradScaler(device=device_t.type, enabled=use_amp)

    log_augment_flags(cfg.domain_aug)
    logger.info(
        "%s | lr=%.2e | replay draft:brats=%d:%d | %s",
        DISCLAIMER,
        cfg.learning_rate,
        cfg.replay_draft,
        cfg.replay_brats,
        device_t,
    )

    (output_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "disclaimer": DISCLAIMER,
                "pretrained": str(pretrained),
                "train_studies": [r["study_id"] for r in train_drafts],
                "wt_only": [r["study_id"] for r in train_drafts if r.get("wt_only")],
                "config": asdict(cfg),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    ckpt0 = torch.load(pretrained, map_location=device_t, weights_only=False)
    model_name: ModelName = ckpt0.get("model_name", cfg.model_name)
    model = build_brats_model(model_name).to(device_t)
    model.load_state_dict(ckpt0["model_state"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_cfg = cfg.to_train_config()
    mixed0 = sample_replay_mix(
        train_drafts,
        brats_pool,
        ratio_batch2=cfg.replay_draft,
        ratio_brats=cfg.replay_brats,
        seed=cfg.seed,
        epoch=0,
    )
    # Ensure BraTS records carry full channel weights
    for r in mixed0:
        if r.get("source") == "brats_replay" or "channel_weights" not in r:
            r["channel_weights"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            r["wt_only"] = False
    steps = max(1, int(len(mixed0) * cfg.num_train_samples / max(1, cfg.batch_size)))
    scheduler, sched_mode = build_lr_scheduler(
        optimizer, train_cfg, steps_per_epoch=steps
    )
    train_tf = _train_transforms(cfg)

    best_path = output_dir / "best_model.pt"
    latest_path = output_dir / "latest.pt"
    metrics_csv = output_dir / "metrics.csv"
    heartbeat_path = output_dir / "heartbeat.log"
    heartbeat = _HeartbeatWatcher(heartbeat_path, interval_s=cfg.heartbeat_interval_s)
    start_epoch = 0
    best_wt = -1.0
    patience = 0

    if resume and latest_path.is_file():
        ck = torch.load(latest_path, map_location=device_t, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            optimizer.load_state_dict(ck["optimizer_state"])
        best_wt = float(ck.get("best_heldout_wt_dice", -1.0))
        start_epoch = int(ck.get("epoch", 0)) if ck.get("epoch_1based") else int(ck.get("epoch", -1)) + 1

    heartbeat.start()
    try:
        for epoch in range(start_epoch, cfg.max_epochs):
            heartbeat.set_epoch(epoch + 1)
            _append_heartbeat(
                heartbeat_path, f"epoch {epoch + 1}/{cfg.max_epochs} START", epoch=epoch + 1
            )
            mixed = sample_replay_mix(
                train_drafts,
                brats_pool,
                ratio_batch2=cfg.replay_draft,
                ratio_brats=cfg.replay_brats,
                seed=cfg.seed,
                epoch=epoch,
            )
            for r in mixed:
                if r.get("source") == "brats_replay" or "channel_weights" not in r:
                    r["channel_weights"] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                    r["wt_only"] = False
            loader = DataLoader(
                Dataset(data=mixed, transform=train_tf),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
            )
            model.train()
            total_loss = 0.0
            n_batches = 0
            t0 = time.perf_counter()
            for batch in loader:
                inputs = batch["image"].to(device_t)
                labels = brats_label_to_regions(batch["label"].to(device_t))
                weights = _batch_channel_weights(batch, device_t)
                # BraTS always full supervision
                if _is_brats_batch(batch):
                    weights = torch.ones_like(weights)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_t.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = channel_weighted_dicece(outputs, labels, weights)
                scaler.scale(loss).backward()
                if cfg.grad_clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=cfg.grad_clip_max_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                if sched_mode == "batch":
                    scheduler.step()
                total_loss += float(loss.item())
                n_batches += 1
                if cfg.log_every_n_batches > 0 and (
                    n_batches == 1 or n_batches % cfg.log_every_n_batches == 0
                ):
                    logger.info(
                        "Epoch %d batch %d/%d loss=%.4f lr=%.2e",
                        epoch + 1,
                        n_batches,
                        len(loader),
                        float(loss.item()),
                        optimizer.param_groups[0]["lr"],
                    )

            mean_loss = total_loss / max(n_batches, 1)
            # Early-stop metric: BraTS held-out WT Dice (LCC) on monitor subset
            held = eval_brats_heldout_dice(
                model, heldout_records, device_t, roi_size=cfg.roi_size
            )
            wt = float(held["whole_tumor"])
            epoch_s = time.perf_counter() - t0
            lr = float(optimizer.param_groups[0]["lr"])
            torch.save(
                {
                    "epoch": epoch + 1,
                    "epoch_1based": True,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "model_name": model_name,
                    "best_heldout_wt_dice": best_wt,
                    "disclaimer": DISCLAIMER,
                },
                latest_path,
            )
            _append_metrics_row(
                metrics_csv,
                {
                    "epoch": epoch + 1,
                    "train_loss": f"{mean_loss:.6f}",
                    "heldout_dice_et": f"{held['enhancing_tumor']:.6f}",
                    "heldout_dice_tc": f"{held['tumor_core']:.6f}",
                    "heldout_dice_wt": f"{wt:.6f}",
                    "heldout_dice_mean": f"{held['mean']:.6f}",
                    "learning_rate": f"{lr:.8e}",
                    "epoch_seconds": f"{epoch_s:.3f}",
                },
            )
            logger.info(
                "Epoch %d/%d loss=%.4f heldout WT=%.4f ET=%.3f TC=%.3f mean=%.3f | %.1fs",
                epoch + 1,
                cfg.max_epochs,
                mean_loss,
                wt,
                held["enhancing_tumor"],
                held["tumor_core"],
                held["mean"],
                epoch_s,
            )
            _append_heartbeat(
                heartbeat_path,
                f"epoch {epoch + 1} END wt={wt:.4f} loss={mean_loss:.4f}",
                epoch=epoch + 1,
            )

            improved = wt > best_wt + 1e-4
            if improved:
                best_wt = wt
                patience = 0
                ckpt = {
                    "epoch": epoch + 1,
                    "epoch_1based": True,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "model_name": model_name,
                    "best_heldout_wt_dice": best_wt,
                    "heldout_dice": held,
                    "pretrained_checkpoint": str(pretrained),
                    "disclaimer": DISCLAIMER,
                    "draft_status": "draft_unverified",
                    "regions": list(BRATS_REGIONS),
                    "val_dice": best_wt,
                    "best_val_dice": best_wt,
                }
                torch.save(ckpt, best_path)
                logger.info("New best heldout WT Dice %.4f → %s", best_wt, best_path)
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    logger.info(
                        "Early stop at epoch %d (patience=%d, best WT=%.4f)",
                        epoch + 1,
                        EARLY_STOP_PATIENCE,
                        best_wt,
                    )
                    break
    finally:
        heartbeat.stop("training exited")

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint saved")
    return best_path


def evaluate_heldout_study(
    checkpoint: str | Path,
    study_id: str,
    device: torch.device,
) -> dict[str, Any]:
    pred = predict_study_regions(checkpoint, study_id, device)
    pkg = LABELING_ROOT / study_id
    draft = np.rint(
        np.asanyarray(nib.load(str(pkg / CLAUDE_DRAFT)).dataobj)
    ).astype(np.int16)
    hd_path = HDGLIO_ROOT / study_id / "hdglio_draft_label.nii.gz"
    hd = (
        np.rint(np.asanyarray(nib.load(str(hd_path)).dataobj)).astype(np.int16)
        if hd_path.is_file()
        else None
    )
    spacing = pred["spacing"]
    dice_draft = dice_vs_label(pred["pred"], draft, "WT")
    dice_hd = (
        dice_vs_label(pred["pred"], hd, "WT") if hd is not None else float("nan")
    )
    c_draft = centroid_mm_from_classic(draft, spacing)
    return {
        "study_id": study_id,
        "WT_dice_vs_draft": dice_draft,
        "WT_dice_vs_hdglio": dice_hd,
        "centroid_dist_to_draft_mm": dist_mm(pred["centroid_mm"], c_draft),
        "WT_ml": pred["WT_ml"],
        "centroid_mm": pred["centroid_mm"],
    }


def write_model_card(path: Path) -> None:
    path.write_text(
        "# Experimental draft fine-tune — MODEL CARD\n\n"
        f"**{DISCLAIMER}**\n\n"
        "## Training data\n\n"
        "- Base: `checkpoints/brats_scale_full/best_model.pt` (v3).\n"
        "- Drafts: `batch2_01`, `07`, `10`, `12` "
        "`claude_draft_label.nii.gz` (`draft_unverified`).\n"
        "- Channel policy: `01`/`12` → WT loss only; `07`/`10` → ET+TC+WT.\n"
        "- BraTS replay ≥3:1 from `splits/scaleup/cfg_c_all_train.txt` "
        "(never held-out).\n"
        "- Domain augs: thick-slice + T1c→T1 dropout (`batch2_augment`).\n\n"
        "## Not the default\n\n"
        "CLI, API, and UI continue to use `brats_scale_full`. This checkpoint "
        "lives only under `checkpoints/experimental_draft_ft/`.\n\n"
        "## Do not use clinically\n\n"
        "Drafts are unverified machine-assisted labels. Metrics vs drafts are "
        "agreement, not accuracy.\n",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    *,
    v3_heldout: dict[str, float],
    ft_heldout: dict[str, float],
    loso_rows: list[dict[str, Any]],
    nondraft_rows: list[dict[str, Any]],
    regressed: bool,
) -> None:
    lines = [
        "# Experimental draft fine-tune SUMMARY",
        "",
        f"**{DISCLAIMER}**",
        "",
    ]
    if regressed:
        lines += [
            f"> **REGRESSED**: BraTS held-out WT Dice dropped by more than "
            f"{REGRESSION_WT_DROP} vs v3.",
            "",
        ]
    lines += [
        "## 1. BraTS held-out Dice (LCC, n="
        f"{HELDOUT_N} fixed subset)",
        "",
        "| model | ET | TC | WT | mean |",
        "|---|---:|---:|---:|---:|",
        f"| v3 | {v3_heldout['enhancing_tumor']:.3f} | "
        f"{v3_heldout['tumor_core']:.3f} | {v3_heldout['whole_tumor']:.3f} | "
        f"{v3_heldout['mean']:.3f} |",
        f"| experimental_draft_ft | {ft_heldout['enhancing_tumor']:.3f} | "
        f"{ft_heldout['tumor_core']:.3f} | {ft_heldout['whole_tumor']:.3f} | "
        f"{ft_heldout['mean']:.3f} |",
        f"| Δ(ft−v3) WT |  |  | "
        f"{ft_heldout['whole_tumor'] - v3_heldout['whole_tumor']:+.3f} |  |",
        "",
        "## 2. Leave-one-study-out (held-out draft study)",
        "",
        "Agreement with drafts is **not accuracy** (drafts are unverified).",
        "",
        "| study | model | WT Dice vs draft | WT Dice vs HD-GLIO | "
        "centroid dist to draft (mm) | WT mL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in loso_rows:
        lines.append(
            f"| {r['study_id']} | {r['model']} | {r['WT_dice_vs_draft']:.3f} | "
            f"{r['WT_dice_vs_hdglio']:.3f} | {r['centroid_dist_to_draft_mm']:.1f} | "
            f"{r['WT_ml']:.2f} |"
        )
    lines += [
        "",
        "## 3. Non-draft batch2 studies (inference only — no accuracy claim)",
        "",
        "| study | model | WT mL | centroid mm |",
        "|---|---|---:|---|",
    ]
    for r in nondraft_rows:
        c = r.get("centroid_mm")
        cstr = (
            f"({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})" if c is not None else "—"
        )
        lines.append(
            f"| {r['study_id']} | {r['model']} | {r['WT_ml']:.2f} | {cstr} |"
        )
    lines += [
        "",
        "## 4. Plain statement",
        "",
        "Agreement with the Claude drafts is **not accuracy**, because the drafts "
        "are unverified (`draft_unverified`). This checkpoint is experimental and "
        "is not the default model for CLI/API/UI.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(
    *,
    device: str | None = None,
    max_epochs: int = 30,
    learning_rate: float = 2e-5,
    seed: int = 42,
    skip_loso: bool = False,
) -> Path:
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = assert_safe_checkpoint_output_dir(CKPT_DIR)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    device_t = get_device(device)
    pretrained = DEFAULT_V3_CHECKPOINT
    cfg = ExpFTConfig(
        max_epochs=max_epochs, learning_rate=learning_rate, seed=seed
    )
    drafts = build_draft_records()
    brats_pool = load_brats_replay_pool()
    heldout_report = fixed_heldout_subset(n=HELDOUT_N, seed=42)
    heldout_monitor = fixed_heldout_subset(n=HELDOUT_EARLYSTOP_N, seed=42)

    # Baseline v3 heldout (full report subset; done after training for timing,
    # but compute early so SUMMARY always has a v3 reference).
    logger.info("Evaluating v3 on BraTS held-out subset (n=%d)…", HELDOUT_N)
    v3_held = eval_brats_heldout_dice(pretrained, heldout_report, device_t)

    loso_rows: list[dict[str, Any]] = []
    if not skip_loso:
        for held in drafts:
            sid = held["study_id"]
            train = [r for r in drafts if r["study_id"] != sid]
            fold_dir = ckpt_root / f"loso_hold_{sid}"
            logger.info("=== LOSO hold out %s (train %s) ===", sid, [r["study_id"] for r in train])
            best = train_experimental(
                train_drafts=train,
                output_dir=fold_dir,
                brats_pool=brats_pool,
                heldout_records=heldout_monitor,
                pretrained=pretrained,
                config=cfg,
                device=str(device_t),
            )
            for tag, ckpt in (("v3", pretrained), ("experimental_loso", best)):
                row = evaluate_heldout_study(ckpt, sid, device_t)
                row["model"] = tag
                row["fold"] = f"hold_{sid}"
                loso_rows.append(row)
                logger.info(
                    "LOSO %s %s: dice_draft=%.3f dice_hd=%.3f dist=%.1f WT_ml=%.2f",
                    sid,
                    tag,
                    row["WT_dice_vs_draft"],
                    row["WT_dice_vs_hdglio"],
                    row["centroid_dist_to_draft_mm"],
                    row["WT_ml"],
                )

    # Final train on all 4
    logger.info("=== Final train on all 4 drafts → %s ===", ckpt_root)
    final_best = train_experimental(
        train_drafts=drafts,
        output_dir=ckpt_root,
        brats_pool=brats_pool,
        heldout_records=heldout_monitor,
        pretrained=pretrained,
        config=cfg,
        device=str(device_t),
    )
    write_model_card(ckpt_root / "MODEL_CARD.md")

    logger.info("Evaluating FT checkpoint on BraTS held-out subset (n=%d)…", HELDOUT_N)
    ft_held = eval_brats_heldout_dice(final_best, heldout_report, device_t)
    wt_drop = float(v3_held["whole_tumor"] - ft_held["whole_tumor"])
    regressed = wt_drop > REGRESSION_WT_DROP

    # Non-draft inference
    nondraft_rows: list[dict[str, Any]] = []
    for sid in NON_DRAFT_STUDIES:
        for tag, ckpt in (("v3", pretrained), ("experimental_draft_ft", final_best)):
            pred = predict_study_regions(ckpt, sid, device_t)
            nondraft_rows.append(
                {
                    "study_id": sid,
                    "model": tag,
                    "WT_ml": pred["WT_ml"],
                    "centroid_mm": pred["centroid_mm"],
                }
            )

    # Also append final-model LOSO-style metrics on each draft for completeness
    # (optional — already have LOSO). Keep LOSO table as-is.

    write_summary(
        results_dir / "SUMMARY.md",
        v3_heldout=v3_held,
        ft_heldout=ft_held,
        loso_rows=loso_rows,
        nondraft_rows=nondraft_rows,
        regressed=regressed,
    )
    pd.DataFrame(loso_rows).to_csv(results_dir / "loso_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "study_id": r["study_id"],
                "model": r["model"],
                "WT_ml": r["WT_ml"],
                "centroid_mm": str(r["centroid_mm"]),
            }
            for r in nondraft_rows
        ]
    ).to_csv(results_dir / "nondraft_inference.csv", index=False)

    (results_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "disclaimer": DISCLAIMER,
                "checkpoint": str(final_best),
                "default_unchanged": str(load_config().inference.checkpoint),
                "v3_heldout": v3_held,
                "ft_heldout": ft_held,
                "regressed": regressed,
                "wt_drop_v3_minus_ft": wt_drop,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Done. Checkpoint %s | default remains %s | regressed=%s",
        final_best,
        load_config().inference.checkpoint,
        regressed,
    )
    return final_best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_all(
        device=args.device,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        skip_loso=args.skip_loso,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
