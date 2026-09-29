"""Device selection for inference / training."""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return a torch device.

    Order: explicit ``prefer`` → CUDA → MPS → CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
