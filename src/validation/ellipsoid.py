"""Clinical ellipsoid volume estimate (A × B × C / 2) for radiologist-style comparison."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from reconstruction.measurements import mask_coordinates_mm, principal_axes_pca
from reconstruction.volume import compute_mask_volume_ml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClinicalEllipsoid:
    """Standard clinical ellipsoid volume from three orthogonal diameters.

    Volume uses the radiology formula ``V = A × B × C / 2`` with diameters in
    millimeters, reported in cm³ (numerically equal to mL).
    """

    diameters_mm: tuple[float, float, float]
    volume_cm3: float
    center_mm: tuple[float, float, float] | None = None
    source: str = "manual"  # "manual" | "principal_axes"

    @property
    def volume_ml(self) -> float:
        """Alias: 1 cm³ = 1 mL."""
        return self.volume_cm3

    @property
    def radii_mm(self) -> tuple[float, float, float]:
        """Half-diameters (for compatibility with geometric ellipsoid APIs)."""
        a, b, c = self.diameters_mm
        return (a / 2.0, b / 2.0, c / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EllipsoidComparison:
    """Voxel / mesh volume vs clinical ellipsoid (A×B×C/2) estimate."""

    mask_volume_ml: float
    ellipsoid_volume_ml: float
    volume_difference_ml: float
    volume_ratio: float
    diameters_mm: tuple[float, float, float]
    source: str


# Backward-compatible alias used by older imports / README wording.
EllipsoidFit = ClinicalEllipsoid


def clinical_ellipsoid_volume_cm3(diameters_mm: Sequence[float]) -> float:
    """Clinical ellipsoid volume ``V = A × B × C / 2`` in cm³.

    Parameters
    ----------
    diameters_mm:
        Three diameters in millimeters (order does not matter for the product).
    """
    if len(diameters_mm) != 3:
        raise ValueError(f"Expected 3 diameters, got {len(diameters_mm)}")
    a, b, c = (float(x) for x in diameters_mm)
    if min(a, b, c) <= 0.0:
        raise ValueError(f"Diameters must be positive, got {(a, b, c)}")
    # mm³ → cm³: / 1000
    return float((a * b * c) / 2.0 / 1000.0)


def clinical_ellipsoid_from_diameters(
    diameters_mm: Sequence[float],
    *,
    center_mm: tuple[float, float, float] | None = None,
) -> ClinicalEllipsoid:
    """Build a clinical ellipsoid from manually entered diameters (mm)."""
    diameters = tuple(sorted((float(x) for x in diameters_mm), reverse=True))
    if len(diameters) != 3:
        raise ValueError("Provide exactly three diameters")
    return ClinicalEllipsoid(
        diameters_mm=diameters,  # type: ignore[arg-type]
        volume_cm3=clinical_ellipsoid_volume_cm3(diameters),
        center_mm=center_mm,
        source="manual",
    )


def principal_axis_bounding_diameters(
    points_mm: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray]:
    """Bounding-box extents of points along the three PCA axes (mm).

    Projects mask coordinates onto principal components and takes
    ``max − min`` along each axis — analogous to a radiologist measuring the
    longest extent on three orthogonal planes aligned with the lesion.
    """
    lengths_pca, center, components = principal_axes_pca(points_mm)
    # ``components`` are rows = principal directions (major → minor).
    centered = points_mm - np.asarray(center, dtype=float)
    projected = centered @ components.T  # (N, 3) coords in PC frame
    extents = projected.max(axis=0) - projected.min(axis=0)
    # Sort major → minor to match clinical A ≥ B ≥ C convention.
    order = np.argsort(extents)[::-1]
    diameters = tuple(float(extents[i]) for i in order)
    return diameters, center, components[order]  # type: ignore[return-value]


def diameters_from_mask_principal_axes(
    mask_path: str | Path | np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] | None = None,
    label: int | None = 1,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Estimate A, B, C as PCA-axis bounding extents of the segmentation."""
    if isinstance(mask_path, (str, Path)):
        img = nib.load(str(mask_path))
        data = np.asanyarray(img.dataobj)
        zooms = img.header.get_zooms()[:3]
        spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        binary = data != 0 if label is None else data == label
    else:
        if spacing_mm is None:
            raise ValueError("spacing_mm is required when mask is an array")
        arr = np.asarray(mask_path)
        binary = arr != 0 if label is None else arr == label
        spacing = spacing_mm

    if not np.any(binary):
        raise ValueError("Mask has no foreground voxels for diameter estimation")

    points = mask_coordinates_mm(binary.astype(bool), spacing)
    diameters, center, _axes = principal_axis_bounding_diameters(points)
    return diameters, center


def fit_ellipsoid(
    mask_path: str | Path | None = None,
    *,
    diameters_mm: Sequence[float] | None = None,
    label: int = 1,
    spacing_mm: tuple[float, float, float] | None = None,
) -> ClinicalEllipsoid:
    """Clinical ellipsoid volume from manual diameters or PCA bounding extents.

    Provide either ``diameters_mm=(A, B, C)`` or a segmentation ``mask_path``.
    Automatic mode measures bounding-box extents of mask voxels along the three
    principal axes (radiologist-style orthogonal diameters).
    """
    if diameters_mm is not None:
        return clinical_ellipsoid_from_diameters(diameters_mm)

    if mask_path is None:
        raise ValueError("Provide diameters_mm or mask_path")

    diameters, center = diameters_from_mask_principal_axes(
        mask_path,
        spacing_mm=spacing_mm,
        label=label,
    )
    return ClinicalEllipsoid(
        diameters_mm=diameters,
        volume_cm3=clinical_ellipsoid_volume_cm3(diameters),
        center_mm=center,
        source="principal_axes",
    )


def compare_to_ellipsoid(
    mask_path: str | Path,
    *,
    label: int = 1,
    diameters_mm: Sequence[float] | None = None,
) -> EllipsoidComparison:
    """Compare voxel mask volume to the clinical A×B×C/2 estimate."""
    fit = fit_ellipsoid(mask_path, diameters_mm=diameters_mm, label=label)
    mask_vol = compute_mask_volume_ml(mask_path, label=label)
    ellip_vol = fit.volume_ml
    ratio = mask_vol / ellip_vol if ellip_vol else float("nan")
    return EllipsoidComparison(
        mask_volume_ml=mask_vol,
        ellipsoid_volume_ml=ellip_vol,
        volume_difference_ml=mask_vol - ellip_vol,
        volume_ratio=ratio,
        diameters_mm=fit.diameters_mm,
        source=fit.source,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clinical ellipsoid volume V = A×B×C/2 from diameters or mask PCA extents.",
    )
    parser.add_argument("--mask", type=Path, default=None, help="Segmentation NIfTI")
    parser.add_argument(
        "--diameters",
        type=float,
        nargs=3,
        metavar=("A", "B", "C"),
        default=None,
        help="Manual diameters in millimeters",
    )
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("-o", "--output-json", type=Path, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.diameters is None and args.mask is None:
        parser.error("Provide --diameters A B C and/or --mask")

    if args.diameters is not None:
        result = fit_ellipsoid(diameters_mm=args.diameters)
        payload: dict[str, Any] = result.to_dict()
        if args.mask is not None:
            comparison = compare_to_ellipsoid(
                args.mask,
                label=args.label,
                diameters_mm=args.diameters,
            )
            payload["comparison"] = asdict(comparison)
    else:
        result = fit_ellipsoid(args.mask, label=args.label)
        payload = result.to_dict()
        comparison = compare_to_ellipsoid(args.mask, label=args.label)
        payload["comparison"] = asdict(comparison)

    text = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
