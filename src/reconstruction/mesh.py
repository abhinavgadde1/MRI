"""Marching-cubes surface meshes from segmentation masks (physical millimeters)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from skimage.measure import marching_cubes

logger = logging.getLogger(__name__)

SUPPORTED_MESH_SUFFIXES = {".obj", ".stl"}


def _load_binary_mask(
    mask_path: Path,
    *,
    label: int | None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI mask and return a float binary volume plus voxel spacing (mm)."""
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    if label is None:
        binary = (data != 0).astype(np.float32)
    else:
        binary = (data == label).astype(np.float32)

    if not np.any(binary):
        raise ValueError(
            f"No foreground voxels in {mask_path}"
            + (f" for label={label}" if label is not None else "")
        )
    return binary, spacing


def mask_to_mesh(
    mask_path: str | Path,
    output_path: str | Path,
    *,
    label: int | None = 1,
    level: float = 0.5,
    step_size: int = 1,
) -> Path:
    """Extract a surface mesh with ``skimage.measure.marching_cubes``.

    Voxel spacing from the NIfTI header is passed as ``spacing`` so mesh
    vertices are in physical millimeters (array origin at the volume corner).

    Parameters
    ----------
    mask_path:
        Segmentation NIfTI (``.nii`` / ``.nii.gz``).
    output_path:
        Destination mesh (``.obj`` or ``.stl``).
    label:
        Integer label to extract. Use ``None`` for any non-zero voxel.
    level:
        Isosurface level on the binary volume (typically ``0.5``).
    step_size:
        Marching-cubes step size (``1`` = full resolution).

    Returns
    -------
    Path
        Path to the written mesh.
    """
    mask_path = Path(mask_path)
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(
            f"Unsupported mesh format {suffix!r}; use one of {sorted(SUPPORTED_MESH_SUFFIXES)}"
        )

    binary, spacing = _load_binary_mask(mask_path, label=label)
    logger.info(
        "Marching cubes on %s | spacing_mm=%s | label=%s",
        mask_path.name,
        spacing,
        label,
    )

    verts, faces, normals, _values = marching_cubes(
        binary,
        level=level,
        spacing=spacing,
        step_size=step_size,
    )
    # ``spacing`` already places vertices in physical mm; do not re-scale by affine zooms.
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    logger.info(
        "Wrote mesh %s (%d vertices, %d faces)",
        output_path,
        len(mesh.vertices),
        len(mesh.faces),
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract an OBJ/STL mesh from a NIfTI segmentation mask.",
    )
    parser.add_argument("mask", type=Path, help="Input segmentation NIfTI")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output mesh path (.obj or .stl)",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Integer label to extract (default: 1). Use -1 for any non-zero.",
    )
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    label: int | None = None if args.label < 0 else args.label
    mask_to_mesh(
        args.mask,
        args.output,
        label=label,
        level=args.level,
        step_size=args.step_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
