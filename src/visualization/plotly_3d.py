"""Plotly helpers for 3D mesh rendering.

Standard color / legend convention used across paper figures and demos:

- Brain surface — light gray, semi-transparent
- Whole Tumor (WT) — green, opaque
- Tumor Core (TC) — blue, opaque
- Enhancing Tumor (ET) — red, opaque
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import trimesh
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

log = logging.getLogger(__name__)

# Canonical display names / colors / opacities (project-wide)
MESH_REGION_ORDER = (
    "brain",
    "whole_tumor",
    "tumor_core",
    "enhancing_tumor",
)

MESH_COLORS: dict[str, str] = {
    # Soft cortical tissue tone (reads as brain, not metal/cloud)
    "brain": "#C9B8A6",
    "whole_tumor": "#55A868",
    "tumor_core": "#4C72B0",
    "enhancing_tumor": "#C44E52",
}

MESH_LABELS: dict[str, str] = {
    "brain": "Brain surface",
    "whole_tumor": "Whole Tumor (WT)",
    "tumor_core": "Tumor Core (TC)",
    "enhancing_tumor": "Enhancing Tumor (ET)",
}

MESH_OPACITY: dict[str, float] = {
    "brain": 0.28,
    "whole_tumor": 0.92,
    "tumor_core": 0.95,
    "enhancing_tumor": 1.0,
}

# Short keys used in morphometrics CSVs / OBJ filenames
SHORT_TO_REGION = {
    "WT": "whole_tumor",
    "TC": "tumor_core",
    "ET": "enhancing_tumor",
    "brain": "brain",
    "whole_tumor": "whole_tumor",
    "tumor_core": "tumor_core",
    "enhancing_tumor": "enhancing_tumor",
}


def mesh3d_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    name: str = "mesh",
    color: str = "#C44E52",
    opacity: float = 0.85,
    hovertemplate: str | None = None,
    showlegend: bool = True,
    flatshading: bool = True,
) -> go.Mesh3d:
    """Build a Plotly ``Mesh3d`` trace from vertex / face arrays."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Expected vertices (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Expected faces (M, 3), got {f.shape}")
    kwargs: dict[str, Any] = dict(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=flatshading,
        showlegend=showlegend,
        lighting=dict(
            ambient=0.35,
            diffuse=0.85,
            specular=0.45 if not flatshading else 0.15,
            roughness=0.35 if not flatshading else 0.6,
            fresnel=0.15,
        ),
        lightposition=dict(x=120, y=180, z=280),
    )
    if hovertemplate is not None:
        kwargs["hovertemplate"] = hovertemplate
        kwargs["hoverinfo"] = "text"
    else:
        kwargs["hoverinfo"] = "name"
    return go.Mesh3d(**kwargs)


def mesh3d_from_trimesh(
    mesh: trimesh.Trimesh | str | Path,
    *,
    name: str = "mesh",
    color: str = "#C44E52",
    opacity: float = 0.85,
    hovertemplate: str | None = None,
    showlegend: bool = True,
    flatshading: bool = True,
) -> go.Mesh3d:
    """Build a Plotly ``Mesh3d`` from a ``trimesh.Trimesh`` or mesh file path."""
    if isinstance(mesh, (str, Path)):
        loaded = trimesh.load(str(mesh), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected a triangle mesh at {mesh}")
        tri = loaded
    else:
        tri = mesh
    return mesh3d_from_arrays(
        tri.vertices,
        tri.faces,
        name=name,
        color=color,
        opacity=opacity,
        hovertemplate=hovertemplate,
        showlegend=showlegend,
        flatshading=flatshading,
    )


def _spacing_tuple(
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def mesh_index_affine(spacing: tuple[float, float, float]) -> np.ndarray:
    """Voxel-index → mm affine used by marching-cubes meshes in this project.

    Vertices are ``ijk * spacing`` with origin at the array corner (no NIfTI
    rotation/origin). All of brain / WT / TC / ET must share this same affine.
    """
    sx, sy, sz = _spacing_tuple(spacing)
    return np.diag([sx, sy, sz, 1.0]).astype(np.float64)


def _marching_cubes_padded(
    field: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    level: float = 0.5,
    step_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Marching cubes with 1-voxel zero pad so FOV-edge masks get a closed surface.

    Without padding, masks that touch the array border omit an exterior face and
    the isosurface sits inward of hard tumor meshes extracted from the same grid.
    """
    sp = _spacing_tuple(spacing)
    padded = np.pad(np.asarray(field, dtype=np.float32), 1, mode="constant", constant_values=0.0)
    try:
        verts, faces, normals, _ = marching_cubes(
            padded,
            level=float(level),
            spacing=sp,
            step_size=max(1, int(step_size)),
        )
    except (ValueError, RuntimeError):
        return None
    # Undo the pad offset (1 voxel along each axis, in mm).
    verts = np.asarray(verts, dtype=np.float64) - np.asarray(sp, dtype=np.float64)
    return verts, faces, normals


def binary_mask_to_trimesh(
    binary: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    step_size: int = 1,
    smooth: bool = True,
    smooth_sigma: float = 0.9,
    taubin_iterations: int = 8,
    target_faces: int = 40_000,
) -> trimesh.Trimesh | None:
    """Marching-cubes surface from a boolean / binary volume (vertices in mm).

    With ``smooth=True`` (default for viz), applies a light Gaussian on the mask
    then Taubin smoothing so tumor surfaces are organic rather than voxel stairs.
    """
    if binary is None or not np.any(binary):
        return None
    mask = np.asarray(binary, dtype=bool)
    sp = _spacing_tuple(spacing)

    if smooth and float(smooth_sigma) > 0:
        # Soft band around the mask → LS isosurface (less blocky than raw MC).
        field = ndi.gaussian_filter(mask.astype(np.float32), sigma=float(smooth_sigma))
        result = _marching_cubes_padded(
            field, spacing, level=0.5, step_size=step_size
        )
    else:
        result = _marching_cubes_padded(
            mask.astype(np.float32), spacing, level=0.5, step_size=step_size
        )
    if result is None:
        return None
    verts, faces, normals = result
    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_normals=normals, process=True
    )
    if len(mesh.faces) == 0:
        return None
    mesh = _keep_largest_mesh_body(mesh)
    if smooth and taubin_iterations > 0:
        try:
            from trimesh.smoothing import filter_taubin

            filter_taubin(mesh, iterations=int(taubin_iterations))
        except Exception:  # noqa: BLE001
            pass
        mesh = _keep_largest_mesh_body(mesh)
    if smooth and target_faces > 0 and len(mesh.faces) > int(target_faces):
        mesh = _decimate_to_face_count(mesh, int(target_faces))
        mesh = _keep_largest_mesh_body(mesh)
    try:
        mesh.fix_normals()
    except Exception:  # noqa: BLE001
        pass
    return mesh


def clean_brain_mask(binary: np.ndarray) -> np.ndarray:
    """Fill holes and keep the largest connected component of a brain mask.

    Removes small disconnected islands that otherwise become floating speckles.

    Intentionally avoids morphological opening/closing: those erode FOV-border
    slices (structuring element overhang) and let tumor meshes poke outside the
    brain surface at the volume edge.
    """
    mask = np.asarray(binary, dtype=bool)
    if not np.any(mask):
        return mask

    labeled, n_comp = ndi.label(mask)
    if n_comp == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = labeled == int(sizes.argmax())
    keep = ndi.binary_fill_holes(keep)
    return keep


def brain_mask_from_volume(
    volume: np.ndarray,
    *,
    percentile: float = 15.0,
    include_masks: Mapping[str, np.ndarray] | None = None,
    dilate_voxels: int = 1,
) -> np.ndarray:
    """Brain support mask from a skull-stripped intensity volume.

    Steps:
    1. Threshold at ``percentile`` of nonzero intensities (default 15th — tighter
       than a 5th-pct “everything nonzero” blob).
    2. Union optional tumor masks so predicted lesion voxels stay inside the shell.
    3. Keep largest connected component + hole fill (no open/close).
    4. Small dilation (default 1) only — heavy dilation turns the head into a potato.
    """
    vol = np.asarray(volume, dtype=np.float32)
    nz = vol[vol > 0]
    if nz.size == 0:
        raw = vol > 0
    else:
        thr = float(np.percentile(nz, percentile))
        raw = vol > thr

    if include_masks:
        for mask in include_masks.values():
            if mask is None:
                continue
            m = np.asarray(mask, dtype=bool)
            if m.shape != raw.shape:
                raise ValueError(
                    f"include_masks shape {m.shape} != volume shape {raw.shape}"
                )
            raw = raw | m

    keep = clean_brain_mask(raw)
    if dilate_voxels > 0 and np.any(keep):
        keep = ndi.binary_dilation(keep, iterations=int(dilate_voxels))
        keep = ndi.binary_fill_holes(keep)
        keep = clean_brain_mask(keep)
    return keep


def _keep_largest_mesh_body(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop disconnected mesh islands left by marching-cubes / smoothing."""
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:  # noqa: BLE001
        return mesh
    if not parts:
        return mesh
    parts = sorted(parts, key=lambda m: len(m.faces), reverse=True)
    return parts[0]


def _decimate_to_face_count(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric-collapse *mesh* to about ``target_faces``.

    trimesh 5.x implements this as a wrapper around ``fast_simplification``.
    Swallowing that import error used to leave the raw marching-cubes mesh
    (e.g. 10k faces on the unit-test ellipsoid, ~200k on a real T1), so
    ``target_faces`` — including the 55k gyri-preserving production default —
    was silently ignored.
    """
    target = int(target_faces)
    if target <= 0 or len(mesh.faces) <= target:
        return mesh
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target)
    except ModuleNotFoundError as exc:
        log.warning(
            "Mesh decimation skipped (%d faces, target %d): %s. "
            "Install the fast-simplification package (required by trimesh 5).",
            len(mesh.faces),
            target,
            exc,
        )
        return mesh
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Mesh decimation failed (%d faces, target %d): %s: %s",
            len(mesh.faces),
            target,
            type(exc).__name__,
            exc,
        )
        return mesh
    if simplified is None or len(getattr(simplified, "faces", [])) == 0:
        return mesh
    return simplified


def brain_surface_mesh(
    volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    percentile: float = 12.0,
    include_masks: Mapping[str, np.ndarray] | None = None,
    dilate_voxels: int = 1,
    peel_voxels: int = 1,
    smooth_sigma: float = 1.4,
    surface_offset_mm: float = 0.0,
    taubin_iterations: int = 8,
    target_faces: int = 55_000,
) -> trimesh.Trimesh | None:
    """Build a smooth, recognizable brain shell from a skull-stripped volume.

    Uses a **smoothed signed-distance** isosurface (not raw voxel stairs):

    1. Support = LCC(intensity > ``percentile`` ∪ tumor) + light dilate
    2. Peel so the shell sits on parenchyma, then re-union tumor
    3. Signed distance field → Gaussian smooth → marching cubes near 0
    4. Keep largest body → Taubin → decimate

    This is the standard “organic organ surface” approach and reads far more like
    a brain than binary-mask marching cubes.
    """
    vol = np.asarray(volume, dtype=np.float32)
    mask = brain_mask_from_volume(
        vol,
        percentile=percentile,
        include_masks=include_masks,
        dilate_voxels=dilate_voxels,
    )
    if not np.any(mask):
        return None

    if peel_voxels > 0 and int(mask.sum()) > 100_000:
        peeled = ndi.binary_erosion(mask, iterations=int(peel_voxels))
        if include_masks:
            for m in include_masks.values():
                if m is not None and np.any(m):
                    peeled = peeled | np.asarray(m, dtype=bool)
        peeled = clean_brain_mask(peeled)
        if np.any(peeled):
            mask = peeled

    sp = _spacing_tuple(spacing)
    # Pad + close so tunnels to the FOV edge seal into a single outer shell.
    # (Unpadded closing would erase border slices; padding avoids that.)
    pad = 4
    mask_p = np.pad(mask, pad, mode="constant", constant_values=False)
    mask_p = ndi.binary_closing(
        mask_p, structure=np.ones((3, 3, 3), dtype=bool), iterations=2
    )
    mask_p = ndi.binary_fill_holes(mask_p)
    mask_p = clean_brain_mask(mask_p)

    dist = ndi.distance_transform_edt(mask_p, sampling=sp).astype(np.float32)
    dist_out = ndi.distance_transform_edt(~mask_p, sampling=sp).astype(np.float32)
    sdf = dist - dist_out
    if float(smooth_sigma) > 0:
        sdf = ndi.gaussian_filter(sdf, sigma=float(smooth_sigma))

    level = float(surface_offset_mm)
    try:
        verts, faces, normals, _ = marching_cubes(
            sdf, level=level, spacing=sp, step_size=1
        )
    except (ValueError, RuntimeError):
        return None
    offset = np.asarray(sp, dtype=np.float64) * float(pad)
    verts = np.asarray(verts, dtype=np.float64) - offset

    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_normals=normals, process=True
    )
    if len(mesh.faces) == 0:
        return None

    mesh = _keep_largest_mesh_body(mesh)

    try:
        from trimesh.smoothing import filter_taubin

        filter_taubin(mesh, iterations=int(taubin_iterations))
    except Exception:  # noqa: BLE001
        pass

    mesh = _keep_largest_mesh_body(mesh)

    if target_faces > 0 and len(mesh.faces) > int(target_faces):
        mesh = _decimate_to_face_count(mesh, int(target_faces))

    mesh = _keep_largest_mesh_body(mesh)
    try:
        mesh.fix_normals()
    except Exception:  # noqa: BLE001
        pass
    return mesh


def tumor_outside_brain_fraction(
    brain_mask: np.ndarray,
    tumor_mask: np.ndarray,
) -> float:
    """Fraction of tumor voxels that fall outside the (boolean) brain mask."""
    tumor = np.asarray(tumor_mask, dtype=bool)
    brain = np.asarray(brain_mask, dtype=bool)
    if tumor.shape != brain.shape:
        raise ValueError(f"shape mismatch tumor {tumor.shape} vs brain {brain.shape}")
    n = int(tumor.sum())
    if n == 0:
        return 0.0
    return float((tumor & ~brain).sum()) / float(n)


def validate_tumor_inside_brain(
    brain_mask: np.ndarray,
    region_masks: Mapping[str, np.ndarray | None],
    *,
    max_outside_fraction: float = 0.01,
    logger: Any | None = None,
) -> dict[str, float]:
    """Warn when tumor voxels poke outside the brain mask used for meshing.

    Returns ``{region: outside_fraction}`` for every non-empty tumor mask.
    """
    import logging as _logging

    log = logger or _logging.getLogger(__name__)
    fractions: dict[str, float] = {}
    for key, mask in region_masks.items():
        if mask is None or not np.any(mask):
            continue
        frac = tumor_outside_brain_fraction(brain_mask, mask)
        fractions[str(key)] = frac
        if frac > float(max_outside_fraction):
            log.warning(
                "Tumor region %s has %.2f%% voxels outside the brain meshing mask "
                "(threshold %.2f%%) — coordinate/shrinkage bug likely",
                key,
                100.0 * frac,
                100.0 * max_outside_fraction,
            )
    return fractions


def _normalize_region_key(key: str) -> str:
    return SHORT_TO_REGION.get(key, key)


def _volume_ml_for(
    region_key: str,
    volumes_ml: Mapping[str, float] | None,
    mesh: trimesh.Trimesh | None,
) -> float | None:
    if volumes_ml:
        for cand in (region_key, MESH_LABELS.get(region_key, ""), region_key.upper()):
            if cand in volumes_ml:
                return float(volumes_ml[cand])
        short = {v: k for k, v in SHORT_TO_REGION.items() if k in ("WT", "TC", "ET")}.get(
            region_key
        )
        if short and short in volumes_ml:
            return float(volumes_ml[short])
    if mesh is not None and getattr(mesh, "is_watertight", False):
        try:
            return float(abs(mesh.volume) / 1000.0)
        except Exception:  # noqa: BLE001
            return None
    return None


def _hovertemplate(label: str, volume_ml: float | None) -> str:
    if volume_ml is None or not np.isfinite(volume_ml):
        vol_txt = "volume n/a"
    else:
        vol_txt = f"volume = {volume_ml:.2f} mL"
    return f"<b>{label}</b><br>{vol_txt}<extra></extra>"


def _default_camera(meshes: list[trimesh.Trimesh]) -> dict[str, Any]:
    """Lateral 3/4 view so the brain silhouette reads as a head in profile."""
    if not meshes:
        return dict(eye=dict(x=2.1, y=0.55, z=0.65), up=dict(x=0, y=0, z=1))
    pts = np.concatenate([np.asarray(m.vertices) for m in meshes], axis=0)
    center = pts.mean(axis=0)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) + 1e-6)
    # Stronger +X (lateral) so gyral contour is readable; slight superior tilt.
    eye = dict(
        x=float(center[0] / extent + 2.1),
        y=float(center[1] / extent + 0.55),
        z=float(center[2] / extent + 0.65),
    )
    return dict(eye=eye, up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))


def build_labeled_mesh_figure(
    meshes: Mapping[str, trimesh.Trimesh | None],
    *,
    subject_id: str,
    checkpoint: str = "checkpoints/brats_scale_full/best_model.pt",
    postprocess: str = "lcc",
    volumes_ml: Mapping[str, float] | None = None,
    title: str | None = None,
    height: int = 560,
) -> go.Figure:
    """Multi-region Plotly figure with legend, hover volumes, orientation, camera.

    ``meshes`` keys may be region names (``whole_tumor`` …) or short codes (``WT``).
    """
    fig = go.Figure()
    used: list[trimesh.Trimesh] = []

    # Stable legend order
    normalized: dict[str, trimesh.Trimesh] = {}
    for key, mesh in meshes.items():
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            continue
        normalized[_normalize_region_key(str(key))] = mesh

    for region in MESH_REGION_ORDER:
        mesh = normalized.get(region)
        if mesh is None:
            continue
        label = MESH_LABELS[region]
        vol = _volume_ml_for(region, volumes_ml, mesh)
        fig.add_trace(
            mesh3d_from_trimesh(
                mesh,
                name=label,
                color=MESH_COLORS[region],
                opacity=MESH_OPACITY[region],
                hovertemplate=_hovertemplate(label, vol),
                showlegend=True,
                # Smooth shading for all regions (reads less “voxelly”).
                flatshading=False,
            )
        )
        used.append(mesh)

    if title is None:
        ckpt_short = Path(str(checkpoint)).name if checkpoint else "model"
        title = f"{subject_id} · predicted meshes ({postprocess}) · {ckpt_short}"

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=14)),
        height=height,
        margin=dict(l=0, r=0, t=60, b=40),
        legend=dict(
            title=dict(text="Regions"),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
            itemsizing="constant",
            font=dict(size=12),
        ),
        scene=dict(
            aspectmode="data",
            xaxis=dict(
                title="X (mm) · L ← → R",
                backgroundcolor="#f7f9fc",
                gridcolor="#dde3ea",
                showbackground=True,
            ),
            yaxis=dict(
                title="Y (mm) · P ← → A",
                backgroundcolor="#f7f9fc",
                gridcolor="#dde3ea",
                showbackground=True,
            ),
            zaxis=dict(
                title="Z (mm) · I ← → S",
                backgroundcolor="#f7f9fc",
                gridcolor="#dde3ea",
                showbackground=True,
            ),
            camera=_default_camera(used),
            annotations=[
                dict(
                    showarrow=False,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    text="",
                    xanchor="left",
                )
            ],
        ),
        annotations=[
            dict(
                text=(
                    "Anatomical orientation (approx. RAS after 1 mm resample): "
                    "<b>L/R</b> along X · <b>P/A</b> along Y · <b>I/S</b> along Z"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.02,
                showarrow=False,
                font=dict(size=11, color="#444"),
                align="center",
            )
        ],
        showlegend=True,
    )
    return fig


def write_labeled_mesh_html(
    out_path: str | Path,
    meshes: Mapping[str, trimesh.Trimesh | None],
    *,
    subject_id: str,
    checkpoint: str = "checkpoints/brats_scale_full/best_model.pt",
    postprocess: str = "lcc",
    volumes_ml: Mapping[str, float] | None = None,
    title: str | None = None,
) -> Path:
    """Build labeled figure and write a self-contained Plotly HTML file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_labeled_mesh_figure(
        meshes,
        subject_id=subject_id,
        checkpoint=checkpoint,
        postprocess=postprocess,
        volumes_ml=volumes_ml,
        title=title,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path


def mesh_figure(
    mesh: str | Path | trimesh.Trimesh | None = None,
    *,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    name: str = "mesh",
    color: str = "#C44E52",
    opacity: float = 0.85,
) -> go.Figure:
    """Create a Plotly figure from a mesh file/trimesh or raw vertices+faces."""
    if mesh is not None:
        trace = mesh3d_from_trimesh(mesh, name=name, color=color, opacity=opacity)
    elif vertices is not None and faces is not None:
        trace = mesh3d_from_arrays(vertices, faces, name=name, color=color, opacity=opacity)
    else:
        raise ValueError("Provide mesh or both vertices and faces")

    fig = go.Figure(data=[trace])
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm) · L ← → R",
            yaxis_title="Y (mm) · P ← → A",
            zaxis_title="Z (mm) · I ← → S",
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.95)),
        ),
        margin=dict(l=0, r=0, t=40, b=30),
        title=name,
        showlegend=True,
        legend=dict(orientation="v", y=0.98, x=0.01),
    )
    return fig


def overlay_tumor_and_tracts(
    tumor_mesh: str | Path,
    tract_meshes: list[str | Path],
    *,
    tumor_color: str = "#C44E52",
    tract_color: str = "#4C72B0",
) -> go.Figure:
    """Overlay a tumor mesh with one or more tract meshes in a single scene."""
    fig = mesh_figure(tumor_mesh, name="tumor", color=tumor_color, opacity=1.0)
    for idx, path in enumerate(tract_meshes):
        fig.add_trace(
            mesh3d_from_trimesh(
                path,
                name=f"tract_{idx}",
                color=tract_color,
                opacity=0.35,
            )
        )
    fig.update_layout(title="Tumor + tracts")
    return fig
