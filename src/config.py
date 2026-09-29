"""Project configuration loader (YAML + Pydantic v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class StageHyperparams(BaseModel):
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    epochs: int = Field(ge=1)
    max_cases: int | None = Field(default=None, ge=1)


class TrainingConfig(BaseModel):
    pretrain: StageHyperparams
    finetune: StageHyperparams


class PathsConfig(BaseModel):
    raw_dicom: Path
    brats: Path
    brats_metadata: Path
    processed: Path
    brats_nifti: Path
    checkpoints: Path

    @field_validator(
        "raw_dicom",
        "brats",
        "brats_metadata",
        "processed",
        "brats_nifti",
        "checkpoints",
        mode="before",
    )
    @classmethod
    def _resolve_path(cls, value: Any) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


class InferenceConfig(BaseModel):
    checkpoint: Path = Field(default=Path("checkpoints/brats_scale_full/best_model.pt"))
    postprocess: str = "lcc"

    @field_validator("checkpoint", mode="before")
    @classmethod
    def _resolve_ckpt(cls, value: Any) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


class AppConfig(BaseModel):
    paths: PathsConfig
    training: TrainingConfig
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate ``config.yaml`` (default: project root)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
