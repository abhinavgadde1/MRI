"""Marching-cubes mesh extraction from label masks.

Canonical implementation lives in :mod:`reconstruction.mesh`.
This module re-exports :func:`mask_to_mesh` for backward compatibility.
"""

from __future__ import annotations

from reconstruction.mesh import mask_to_mesh

__all__ = ["mask_to_mesh"]
