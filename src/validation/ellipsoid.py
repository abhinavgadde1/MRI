"""Clinical ellipsoid volume estimates and diameter definitions for BraTS masks."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import nibabel as nib
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes

from reconstruction.measurements import mask_coordinates_mm, principal_axes_pca
from reconstruction.volume import compute_mask_volume_ml

logger = logging.getLogger(__name__)

DiameterMethod = Literal["aabb", "pca", "radiologist"]
VolumeFormula = Literal["abc_over_2", "pi_over_6"]

DIAMETER_METHODS: tuple[DiameterMethod, ...] = ("aabb", "pca", "radiologist")
VOLUME_FORMULAS: tuple[VolumeFormula, ...] = ("abc_over_2", "pi_over_6")


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


def geometric_ellipsoid_volume_cm3(diameters_mm: Sequence[float]) -> float:
    """True ellipsoid volume ``V = π/6 × A × B × C`` in cm³."""
    if len(diameters_mm) != 3:
        raise ValueError(f"Expected 3 diameters, got {len(diameters_mm)}")
    a, b, c = (float(x) for x in diameters_mm)
    if min(a, b, c) <= 0.0:
        raise ValueError(f"Diameters must be positive, got {(a, b, c)}")
    return float((np.pi / 6.0) * a * b * c / 1000.0)


def ellipsoid_volume_cm3(
    diameters_mm: Sequence[float],
    formula: VolumeFormula = "abc_over_2",
) -> float:
    """Volume from diameters using ``abc_over_2`` or ``pi_over_6``."""
    if formula == "abc_over_2":
        return clinical_ellipsoid_volume_cm3(diameters_mm)
    if formula == "pi_over_6":
        return geometric_ellipsoid_volume_cm3(diameters_mm)
    raise ValueError(f"Unknown formula {formula!r}")


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
    _lengths_pca, center, components = principal_axes_pca(points_mm)
    # ``components`` are rows = principal directions (major → minor).
    centered = points_mm - np.asarray(center, dtype=float)
    projected = centered @ components.T  # (N, 3) coords in PC frame
    extents = projected.max(axis=0) - projected.min(axis=0)
    # Sort major → minor to match clinical A ≥ B ≥ C convention.
    order = np.argsort(extents)[::-1]
    diameters = tuple(float(extents[i]) for i in order)
    return diameters, center, components[order]  # type: ignore[return-value]


def axis_aligned_bounding_diameters(
    points_mm: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Axis-aligned bounding-box extents (sorted major → minor) in mm."""
    pts = np.asarray(points_mm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {pts.shape}")
    extents = pts.max(axis=0) - pts.min(axis=0)
    order = np.argsort(extents)[::-1]
    diameters = tuple(float(extents[i]) for i in order)
    center = tuple(float(x) for x in pts.mean(axis=0))
    return diameters, center  # type: ignore[return-value]


def _max_diameter_2d(points_xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Longest pairwise extent on a 2D point set via convex hull."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need ≥2 points for a 2D diameter")
    if pts.shape[0] == 2:
        direction = pts[1] - pts[0]
        return float(np.linalg.norm(direction)), direction

    try:
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
    except Exception:  # noqa: BLE001 — degenerate slices fall back to all points
        hp = pts

    best_d = -1.0
    best_dir = np.array([1.0, 0.0])
    for i in range(len(hp)):
        for j in range(i + 1, len(hp)):
            delta = hp[j] - hp[i]
            dist = float(np.linalg.norm(delta))
            if dist > best_d:
                best_d = dist
                best_dir = delta
    if best_d <= 0.0:
        raise ValueError("Degenerate 2D diameter")
    return best_d, best_dir


def radiologist_axial_diameters(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Radiologist-style A/B/C diameters in mm.

    * **A** — longest diameter on any axial (``z``) slice
    * **B** — longest extent perpendicular to A on that same slice
    * **C** — craniocaudal extent (``z`` bounding extent of voxel centers)
    """
    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {binary.shape}")
    sx, sy, sz = (float(x) for x in spacing_mm)

    z_indices = np.flatnonzero(binary.any(axis=(0, 1)))
    if z_indices.size == 0:
        raise ValueError("Mask has no foreground voxels for diameter estimation")

    c_mm = float((z_indices.max() - z_indices.min()) * sz)
    if c_mm <= 0.0:
        # Single-slice tumor: use slice thickness as a minimal CC extent.
        c_mm = sz

    best_a = -1.0
    best_b = -1.0
    best_center_xy = np.zeros(2)

    for z in z_indices:
        sl = binary[:, :, int(z)]
        xs, ys = np.nonzero(sl)
        pts = np.column_stack([xs.astype(float) * sx, ys.astype(float) * sy])
        if pts.shape[0] < 2:
            continue
        a_mm, direction = _max_diameter_2d(pts)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            continue
        unit = direction / norm
        perp = np.array([-unit[1], unit[0]])
        proj = pts @ perp
        b_mm = float(proj.max() - proj.min())
        if b_mm <= 0.0:
            b_mm = min(sx, sy)
        if a_mm > best_a:
            best_a = a_mm
            best_b = b_mm
            best_center_xy = pts.mean(axis=0)

    if best_a <= 0.0:
        raise ValueError("Could not measure axial diameters")

    diameters = tuple(
        sorted((float(best_a), float(best_b), float(c_mm)), reverse=True)
    )
    ijk = np.column_stack(np.nonzero(binary)).astype(float)
    center = (
        float(ijk[:, 0].mean() * sx),
        float(ijk[:, 1].mean() * sy),
        float(ijk[:, 2].mean() * sz),
    )
    # Prefer axial-slice centroid for A/B plane when available.
    if best_center_xy is not None:
        center = (float(best_center_xy[0]), float(best_center_xy[1]), center[2])
    return diameters, center  # type: ignore[return-value]


def diameters_from_binary(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
    method: DiameterMethod = "pca",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Estimate A, B, C diameters from a binary mask using ``method``."""
    binary = np.asarray(binary, dtype=bool)
    if not np.any(binary):
        raise ValueError("Mask has no foreground voxels for diameter estimation")

    if method == "radiologist":
        return radiologist_axial_diameters(binary, spacing_mm)

    points = mask_coordinates_mm(binary, spacing_mm)
    if method == "aabb":
        return axis_aligned_bounding_diameters(points)
    if method == "pca":
        diameters, center, _axes = principal_axis_bounding_diameters(points)
        return diameters, center
    raise ValueError(f"Unknown diameter method {method!r}")


def mesh_volume_ml_from_binary(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    level: float = 0.5,
) -> float:
    """Marching-cubes mesh volume (mL) from a binary mask."""
    binary = np.asarray(binary, dtype=bool)
    if not np.any(binary):
        raise ValueError("Cannot build mesh from an empty mask")
    verts, faces, _normals, _values = marching_cubes(
        binary.astype(np.float32),
        level=level,
        spacing=spacing_mm,
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.is_empty:
        raise ValueError("Marching cubes produced an empty mesh")
    # Ensure a consistent orientation for volume sign.
    if not mesh.is_watertight:
        mesh.fill_holes()
    return float(abs(mesh.volume) / 1000.0)


def voxel_volume_ml_from_binary(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> float:
    """Voxel-count volume in mL."""
    voxel_ml = float(np.prod(spacing_mm) / 1000.0)
    return float(np.count_nonzero(binary) * voxel_ml)


def mesh_voxel_relative_error(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> float:
    """Absolute relative error ``|mesh − voxel| / voxel``."""
    voxel = voxel_volume_ml_from_binary(binary, spacing_mm)
    if voxel <= 0.0:
        raise ValueError("Empty mask")
    mesh = mesh_volume_ml_from_binary(binary, spacing_mm)
    return float(abs(mesh - voxel) / voxel)


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

    return diameters_from_binary(binary.astype(bool), spacing, method="pca")


def fit_ellipsoid(
    mask_path: str | Path | None = None,
    *,
    diameters_mm: Sequence[float] | None = None,
    label: int | None = 1,
    spacing_mm: tuple[float, float, float] | None = None,
) -> ClinicalEllipsoid:
    """Clinical ellipsoid volume from manual diameters or PCA bounding extents.

    Provide either ``diameters_mm=(A, B, C)`` or a segmentation ``mask_path``.
    Automatic mode measures bounding-box extents of mask voxels along the three
    principal axes (radiologist-style orthogonal diameters).
    Use ``label=None`` for all non-zero voxels (whole tumor).
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
    label: int | None = 1,
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
    parser.add_argument(
        "--whole-tumor",
        action="store_true",
        help="Use all non-zero mask labels (1+2+3) instead of --label",
    )
    parser.add_argument("-o", "--output-json", type=Path, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.diameters is None and args.mask is None:
        parser.error("Provide --diameters A B C and/or --mask")

    label: int | None = None if args.whole_tumor else args.label

    if args.diameters is not None:
        result = fit_ellipsoid(diameters_mm=args.diameters)
        payload: dict[str, Any] = result.to_dict()
        if args.mask is not None:
            comparison = compare_to_ellipsoid(
                args.mask,
                label=label,
                diameters_mm=args.diameters,
            )
            payload["comparison"] = asdict(comparison)
    else:
        result = fit_ellipsoid(args.mask, label=label)
        payload = result.to_dict()
        comparison = compare_to_ellipsoid(args.mask, label=label)
        payload["comparison"] = asdict(comparison)

    text = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
