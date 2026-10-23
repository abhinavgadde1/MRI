"""Interactive Plotly 3D visualization."""

from visualization.plotly_3d import (
    MESH_COLORS,
    MESH_LABELS,
    MESH_OPACITY,
    binary_mask_to_trimesh,
    brain_mask_from_volume,
    brain_surface_mesh,
    build_labeled_mesh_figure,
    clean_brain_mask,
    mesh_index_affine,
    mesh3d_from_arrays,
    mesh3d_from_trimesh,
    mesh_figure,
    overlay_tumor_and_tracts,
    tumor_outside_brain_fraction,
    validate_tumor_inside_brain,
    write_labeled_mesh_html,
)

__all__ = [
    "MESH_COLORS",
    "MESH_LABELS",
    "MESH_OPACITY",
    "binary_mask_to_trimesh",
    "brain_mask_from_volume",
    "brain_surface_mesh",
    "build_labeled_mesh_figure",
    "clean_brain_mask",
    "mesh_index_affine",
    "mesh3d_from_arrays",
    "mesh3d_from_trimesh",
    "mesh_figure",
    "overlay_tumor_and_tracts",
    "tumor_outside_brain_fraction",
    "validate_tumor_inside_brain",
    "write_labeled_mesh_html",
]
