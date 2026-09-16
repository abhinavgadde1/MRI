"""Plotly helpers for 3D mesh and tract rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh


def mesh_figure(
    mesh_path: str | Path,
    *,
    name: str = "mesh",
    color: str = "#C44E52",
    opacity: float = 0.85,
) -> go.Figure:
    """Create a Plotly Mesh3d figure from a triangle mesh file."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {mesh_path}")

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    fig = go.Figure(
        data=[
            go.Mesh3d(
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
        ]
    )
    fig.update_layout(
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data"),
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
        mesh = trimesh.load(str(path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=tract_color,
                opacity=0.35,
                name=f"tract_{idx}",
                flatshading=True,
                showlegend=True,
            )
        )
    fig.update_layout(title="Tumor + tracts")
    return fig
