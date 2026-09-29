"""Shim: v1/v2 ablation comparison lives under experiments/ablation_v1_v2/.

Import helpers from ``segmentation.report_metrics`` for new code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from segmentation.report_metrics import _bootstrap_mean_diff_ci  # noqa: F401

_EXP = Path(__file__).resolve().parents[2] / "experiments" / "ablation_v1_v2" / "compare_v1_v2.py"


def _load_ablation():
    spec = importlib.util.spec_from_file_location("ablation_compare_v1_v2", _EXP)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ablation module at {_EXP}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def __getattr__(name: str):
    mod = _load_ablation()
    if hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(name)
