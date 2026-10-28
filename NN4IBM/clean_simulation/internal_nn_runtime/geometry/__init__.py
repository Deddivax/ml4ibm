"""Geometry preparation and FEM-facing neural-runtime assembly."""

from .cutcell_runtime import (
    compute_cell_node_sdf_csc,
    compute_cell_volume_fractions_ray_parity,
    compute_nn_cutcell_stiffness,
    extract_volume_triangles,
    points_inside_mesh_ray_parity,
    prepare_nn_inputs,
    reorder_refined_sdf_for_nn,
    write_geometry_cutcell_training_h5,
)

__all__ = [
    "compute_cell_node_sdf_csc",
    "compute_cell_volume_fractions_ray_parity",
    "compute_nn_cutcell_stiffness",
    "extract_volume_triangles",
    "points_inside_mesh_ray_parity",
    "prepare_nn_inputs",
    "reorder_refined_sdf_for_nn",
    "write_geometry_cutcell_training_h5",
]