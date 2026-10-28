"""
Export functions for hanging sphere results to VTU/STL formats.
Handles displaced surface export and integration point visualization.
"""

import numpy as np
import pyvista as pv
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree


def export_displaced_sphere(
    sphere_volume,
    u_nodes: np.ndarray,
    grid,
    poly_order: int,
    work_dir: Path,
) -> Path:
    """
    Export displaced sphere surface as STL and VTU.
    
    Parameters
    ----------
    sphere_volume : Volume
        Original sphere geometry with mesh.
    u_nodes : ndarray, shape (n_nodes, 3)
        Nodal displacements.
    grid : HexGrid
        FEM grid for KDTree search.
    poly_order : int
        Polynomial order.
    work_dir : Path
        Output directory.
    
    Returns
    -------
    distorted_sphere_vtu : Path
        Path to displaced sphere VTU file.
    """
    # Map grid nodes to sphere surface via nearest neighbor
    tree = cKDTree(grid.node_coordinates(poly_order=poly_order))
    _, nn = tree.query(sphere_volume.mesh.vertices, k=1)
    surface_disp = u_nodes[nn]
    deformed_vertices = sphere_volume.mesh.vertices + surface_disp
    
    # Export as STL
    distorted_sphere_mesh = trimesh.Trimesh(
        vertices=deformed_vertices,
        faces=sphere_volume.mesh.faces,
        process=False,
    )
    distorted_sphere_stl = work_dir / "sphere_deformed.stl"
    distorted_sphere_mesh.export(str(distorted_sphere_stl))
    
    # Export as VTU with displacement data
    n_faces = sphere_volume.mesh.faces.shape[0]
    tri_cells = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int64),
        sphere_volume.mesh.faces.astype(np.int64),
    ]).ravel()
    cell_types = np.full(n_faces, pv.CellType.TRIANGLE, dtype=np.uint8)
    vtu_grid = pv.UnstructuredGrid(tri_cells, cell_types, deformed_vertices)
    vtu_grid.point_data["Displacement"] = surface_disp
    vtu_grid.point_data["DisplacementMagnitude"] = np.linalg.norm(surface_disp, axis=1)
    distorted_sphere_vtu = work_dir / "sphere_deformed.vtu"
    vtu_grid.save(str(distorted_sphere_vtu))
    
    return distorted_sphere_vtu


def export_integration_points(
    ips: np.ndarray,
    work_dir: Path,
) -> Path:
    """
    Export integration points as VTU with weight data.
    
    Parameters
    ----------
    ips : ndarray, shape (n_points, 4)
        Integration points [X, Y, Z, Weight].
    work_dir : Path
        Output directory.
    
    Returns
    -------
    integration_points_vtu : Path
        Path to integration points VTU file.
    """
    ip_xyz = ips[:, :3]
    ip_w = ips[:, 3]
    
    ip_cells = np.hstack([
        np.ones((ips.shape[0], 1), dtype=np.int64),
        np.arange(ips.shape[0], dtype=np.int64)[:, None],
    ]).ravel()
    ip_ctypes = np.full(ips.shape[0], pv.CellType.VERTEX, dtype=np.uint8)
    ip_grid = pv.UnstructuredGrid(ip_cells, ip_ctypes, ip_xyz)
    ip_grid.point_data["Weight"] = ip_w
    integration_points_vtu = work_dir / "queso_integration_points.vtu"
    ip_grid.save(str(integration_points_vtu))
    
    return integration_points_vtu
