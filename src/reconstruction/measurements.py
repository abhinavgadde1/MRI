"""Tumor morphometric measurements from voxel masks and surface meshes."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import trimesh
from skimage.measure import marching_cubes, mesh_surface_area

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TumorMeasurements:
    """Morphometrics for one tumor / region (physical units).

    ``principal_axis_lengths_mm`` are full diameters (major → minor) from PCA
    on mask coordinates in millimeter space, suitable for ellipsoid comparison.
    """

    volume_cm3: float
    surface_area_mm2: float
    principal_axis_lengths_mm: tuple[float, float, float]
    sphericity: float
    center_mm: tuple[float, float, float]
    source: str  # "mask" | "mesh" | "mask+mesh"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def volume_cm3_from_mask(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> float:
    """Voxel count × voxel volume → cubic centimeters (1 cm³ = 1 mL)."""
    voxel_mm3 = float(np.prod(spacing_mm))
    return float(np.count_nonzero(binary) * voxel_mm3 / 1000.0)


def volume_cm3_from_mesh(mesh: trimesh.Trimesh) -> float:
    """Watertight mesh volume in cm³ (vertices assumed millimeters)."""
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    return float(abs(mesh.volume) / 1000.0)


def surface_area_mm2_from_mesh(mesh: trimesh.Trimesh) -> float:
    """Triangle mesh surface area in mm²."""
    return float(mesh.area)


def surface_area_mm2_from_mask(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    level: float = 0.5,
) -> float:
    """Approximate surface area via marching cubes on the binary mask."""
    vol = binary.astype(np.float32)
    if not np.any(vol):
        raise ValueError("Cannot measure surface area of an empty mask")
    verts, faces, _normals, _values = marching_cubes(
        vol,
        level=level,
        spacing=spacing_mm,
    )
    return float(mesh_surface_area(verts, faces))


def sphericity_index(volume_cm3: float, surface_area_mm2: float) -> float:
    """Wadell sphericity: ``π^(1/3) (6V)^(2/3) / A`` (1 = perfect sphere).

    ``volume_cm3`` is converted to mm³ so it matches ``surface_area_mm2``.
    """
    if volume_cm3 <= 0.0 or surface_area_mm2 <= 0.0:
        return float("nan")
    volume_mm3 = volume_cm3 * 1000.0
    return float((np.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2)


def principal_axes_pca(
    points_mm: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray]:
    """PCA on 3D points in millimeters.

    Returns
    -------
    axis_lengths_mm:
        Full diameters along principal axes, sorted major → minor:
        ``2 * sqrt(λ)`` for eigenvalues of the covariance matrix.
    center_mm:
        Centroid of the point cloud.
    components:
        ``(3, 3)`` eigenvectors as rows (same order as axis lengths).
    """
    pts = np.asarray(points_mm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points for PCA")

    center = pts.mean(axis=0)
    centered = pts - center
    # Covariance of coordinates (mm²); ddof=0 matches population PCA on voxels.
    cov = np.cov(centered, rowvar=False, ddof=0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Full axis lengths (diameters) from 1-σ extents along each PC.
    lengths = tuple(float(2.0 * np.sqrt(max(ev, 0.0))) for ev in eigenvalues)
    center_t = tuple(float(x) for x in center)
    return lengths, center_t, eigenvectors.T


def mask_coordinates_mm(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    """Foreground voxel indices → physical coordinates in millimeters."""
    ijk = np.column_stack(np.nonzero(binary)).astype(float)
    if ijk.size == 0:
        raise ValueError("Mask has no foreground voxels")
    spacing = np.asarray(spacing_mm, dtype=float)
    return ijk * spacing


def _load_mask_array(
    mask_path: str | Path,
    *,
    label: int | None = 1,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    if label is None:
        binary = data != 0
    else:
        binary = data == label
    if not np.any(binary):
        raise ValueError(
            f"No foreground voxels in {mask_path}"
            + (f" for label={label}" if label is not None else "")
        )
    return binary.astype(bool), spacing


def measure_from_mask(
    mask: str | Path | np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] | None = None,
    label: int | None = 1,
) -> TumorMeasurements:
    """Compute volume, surface area, PCA axes, and sphericity from a voxel mask.

    Parameters
    ----------
    mask:
        NIfTI path or boolean / integer ndarray.
    spacing_mm:
        Required when ``mask`` is an array; ignored for NIfTI (read from header).
    label:
        Integer label for NIfTI / integer arrays. ``None`` = any non-zero.
    """
    if isinstance(mask, (str, Path)):
        binary, spacing = _load_mask_array(mask, label=label)
    else:
        if spacing_mm is None:
            raise ValueError("spacing_mm is required when mask is an array")
        arr = np.asarray(mask)
        binary = arr != 0 if label is None else arr == label
        if not np.any(binary):
            raise ValueError("Mask has no foreground voxels")
        spacing = spacing_mm

    volume = volume_cm3_from_mask(binary, spacing)
    area = surface_area_mm2_from_mask(binary, spacing)
    points = mask_coordinates_mm(binary, spacing)
    axes, center, _components = principal_axes_pca(points)
    sph = sphericity_index(volume, area)

    return TumorMeasurements(
        volume_cm3=volume,
        surface_area_mm2=area,
        principal_axis_lengths_mm=axes,
        sphericity=sph,
        center_mm=center,
        source="mask",
    )


def measure_from_mesh(
    mesh: str | Path | trimesh.Trimesh,
    *,
    mask: str | Path | np.ndarray | None = None,
    spacing_mm: tuple[float, float, float] | None = None,
    label: int | None = 1,
) -> TumorMeasurements:
    """Compute volume and surface area from a mesh; PCA from mask when available.

    If ``mask`` is provided, principal axes use mask voxel coordinates (preferred
    for ellipsoid comparison). Otherwise PCA falls back to mesh vertices.
    """
    if isinstance(mesh, (str, Path)):
        loaded = trimesh.load(str(mesh), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected a triangle mesh at {mesh}")
        tri = loaded
    else:
        tri = mesh

    volume = volume_cm3_from_mesh(tri)
    area = surface_area_mm2_from_mesh(tri)

    if mask is not None:
        if isinstance(mask, (str, Path)):
            binary, spacing = _load_mask_array(mask, label=label)
        else:
            if spacing_mm is None:
                raise ValueError("spacing_mm is required when mask is an array")
            arr = np.asarray(mask)
            binary = arr != 0 if label is None else arr == label
            spacing = spacing_mm
        points = mask_coordinates_mm(binary, spacing)
        source = "mask+mesh"
    else:
        points = np.asarray(tri.vertices, dtype=float)
        source = "mesh"

    axes, center, _components = principal_axes_pca(points)
    sph = sphericity_index(volume, area)

    return TumorMeasurements(
        volume_cm3=volume,
        surface_area_mm2=area,
        principal_axis_lengths_mm=axes,
        sphericity=sph,
        center_mm=center,
        source=source,
    )


def measure_tumor(
    *,
    mask: str | Path | np.ndarray | None = None,
    mesh: str | Path | trimesh.Trimesh | None = None,
    spacing_mm: tuple[float, float, float] | None = None,
    label: int | None = 1,
) -> TumorMeasurements:
    """Measure from mesh and/or mask (prefer mesh volume/area when both given)."""
    if mesh is not None:
        return measure_from_mesh(
            mesh,
            mask=mask,
            spacing_mm=spacing_mm,
            label=label,
        )
    if mask is not None:
        return measure_from_mask(mask, spacing_mm=spacing_mm, label=label)
    raise ValueError("Provide at least one of mask or mesh")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute tumor volume, surface area, PCA axes, and sphericity.",
    )
    parser.add_argument("--mask", type=Path, default=None, help="Segmentation NIfTI")
    parser.add_argument("--mesh", type=Path, default=None, help="Surface mesh (.obj/.stl)")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument(
        "--any-nonzero",
        action="store_true",
        help="Use all non-zero mask voxels instead of --label",
    )
    parser.add_argument("-o", "--output-json", type=Path, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.mask is None and args.mesh is None:
        parser.error("Provide --mask and/or --mesh")

    label: int | None = None if args.any_nonzero else args.label
    result = measure_tumor(mask=args.mask, mesh=args.mesh, label=label)
    payload = result.to_dict()
    text = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output_json)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
