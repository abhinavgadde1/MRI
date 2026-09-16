"""Load and validate pipeline configuration from YAML via Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class StageHyperparameters(BaseModel):
    """Hyperparameters for a single training stage."""

    batch_size: int = Field(..., ge=1, description="Mini-batch size")
    learning_rate: float = Field(..., gt=0.0, description="Optimizer learning rate")
    epochs: int = Field(..., ge=1, description="Number of training epochs")


class TrainingConfig(BaseModel):
    """Pretraining (e.g. BraTS) and fine-tuning (clinical) stages."""

    pretrain: StageHyperparameters
    finetune: StageHyperparameters


class PathsConfig(BaseModel):
    """Filesystem locations for data and artifacts."""

    raw_dicom: Path
    brats: Path
    brats_metadata: Path | None = None
    processed: Path
    brats_nifti: Path | None = None
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
    def _coerce_path(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        return Path(value)


class AppConfig(BaseModel):
    """Root configuration for the MRI pipeline."""

    paths: PathsConfig
    training: TrainingConfig
    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT)

    @model_validator(mode="after")
    def _resolve_paths(self) -> AppConfig:
        root = self.project_root.resolve()
        self.project_root = root
        brats_nifti = self.paths.brats_nifti
        if brats_nifti is None:
            brats_nifti = self.paths.processed / "brats_nifti"
        self.paths = PathsConfig(
            raw_dicom=_resolve(root, self.paths.raw_dicom),
            brats=_resolve(root, self.paths.brats),
            brats_metadata=(
                _resolve(root, self.paths.brats_metadata)
                if self.paths.brats_metadata is not None
                else None
            ),
            processed=_resolve(root, self.paths.processed),
            brats_nifti=_resolve(root, brats_nifti),
            checkpoints=_resolve(root, self.paths.checkpoints),
        )
        return self


def _resolve(root: Path, path: Path) -> Path:
    """Resolve a path relative to the project root; leave absolutes unchanged."""
    return path if path.is_absolute() else (root / path).resolve()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load ``config.yaml`` and return a validated ``AppConfig``.

    Parameters
    ----------
    config_path:
        Optional path to a YAML file. Defaults to ``<project_root>/config.yaml``.

    Returns
    -------
    AppConfig
        Validated configuration with absolute paths.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    return AppConfig.model_validate(raw)


def ensure_output_dirs(config: AppConfig) -> None:
    """Create processed and checkpoint directories if they do not exist."""
    config.paths.processed.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoints.mkdir(parents=True, exist_ok=True)
