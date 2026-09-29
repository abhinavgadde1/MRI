"""Plotly helpers for 3D mesh rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh


def mesh3d_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    name: str = "mesh",
    color: str = "#C44E52",
    opacity: float = 0.85,
) -> go.Mesh3d:
    """Build a Plotly ``Mesh3d`` trace from vertex / face arrays."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Expected vertices (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Expected faces (M, 3), got {f.shape}")
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
    )


def mesh3d_from_trimesh(
    mesh: trimesh.Trimesh | str | Path,
    *,
    name: str = "mesh",
    color: str = "#C44E52",
    opacity: float = 0.85,
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
    )


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
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=name,
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
    fig = mesh_figure(tumor_mesh, name="tumor", color=tumor_color, opacity=0.8)
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
