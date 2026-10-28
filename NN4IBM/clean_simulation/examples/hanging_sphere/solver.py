"""
Hanging sphere assembly, solving, and export functions.
Encapsulates stiffness assembly, body force computation, Nitsche BC, and system solve.
"""

import numpy as np
import torch
import trimesh
from typing import Tuple
from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from core.grid_indexing import active_element_nodes
from fem import FEM
from shape import Shape


def assemble_and_solve_system(
    grid,
    material,
    config: dict,
    active_indices: np.ndarray,
    ips: np.ndarray,
    top_mesh: trimesh.Trimesh,
    poly_order: int,
    g_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble stiffness and load, apply Nitsche BC, and solve system.
    
    Parameters
    ----------
    grid : HexGrid
        FEM grid.
    material : Material
        Material properties.
    config : dict
        Configuration with loads, bc settings.
    active_indices : ndarray
        Active element indices.
    ips : ndarray, shape (n_ips, 4)
        Integration points [X, Y, Z, Weight].
    top_mesh : trimesh.Trimesh
        Top cap mesh for Nitsche BC.
    poly_order : int
        Polynomial order of elements.
    g_order : int
        Gauss quadrature order.
    
    Returns
    -------
    u_nodes : ndarray, shape (n_nodes, 3)
        Nodal displacements.
    u_vec : ndarray, shape (n_nodes*3,)
        Flattened displacement vector.
    n_nodes_total : int
        Total number of nodes.
    """
    # Extract IPs
    ip_xyz = ips[:, :3]
    ip_w = ips[:, 3]
    
    # 1) Stiffness from QuESo quadrature
    ke_list, _ = FEM.StiffnessAssembly.compute_element_matrices_from_queso_points(
        grid=grid,
        material=material,
        ips=ips,
        poly_order=poly_order,
        device="cpu",
    )
    k_base = FEM.StiffnessAssembly.assemble_global_stiffness_sparse(
        ke_list,
        active_indices,
        grid,
        poly_order=poly_order,
    ).tocsr()
    
    # 2) Volumetric gravity load (body force)
    rho = float(config["material"]["density"])
    g_vec = np.array(config["loads"]["gravity"], dtype=np.float64)
    body_force = rho * g_vec
    
    # Map IPs to local coords
    nx, ny, nz = grid.resolution
    bounds_min = np.array(grid.bounds[0], dtype=np.float64)
    dx = np.array([grid.dx, grid.dy, grid.dz], dtype=np.float64)
    ijk = np.floor((ip_xyz - bounds_min) / dx).astype(np.int64)
    ijk = np.clip(ijk, [0, 0, 0], [nx - 1, ny - 1, nz - 1])
    cell_min = bounds_min + ijk * dx
    xi_all = np.clip(2.0 * (ip_xyz - cell_min) / dx - 1.0, -1.0, 1.0)
    
    # Evaluate shape functions
    xi_t = torch.tensor(xi_all[:, 0], dtype=torch.float64)
    et_t = torch.tensor(xi_all[:, 1], dtype=torch.float64)
    ze_t = torch.tensor(xi_all[:, 2], dtype=torch.float64)
    N_all = Shape.N3(xi_t, et_t, ze_t, poly_order=poly_order).cpu().numpy()
    
    # Assemble element body forces
    flat = ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]
    flat_to_pos = np.full(nx * ny * nz, -1, dtype=np.int64)
    flat_to_pos[active_indices] = np.arange(len(active_indices), dtype=np.int64)
    pos = flat_to_pos[flat]
    valid = pos >= 0
    
    n_active = len(active_indices)
    n_nodes_per_elem = (poly_order + 1) ** 3
    f_elem = np.zeros((n_active, n_nodes_per_elem, 3), dtype=np.float64)
    np.add.at(
        f_elem,
        pos[valid],
        ip_w[valid, None, None] * N_all[valid, :, None] * body_force[None, None, :],
    )
    
    # Assemble global force vector
    nodes_conn = active_element_nodes(active_indices, grid.resolution, poly_order=poly_order)
    n_nodes_total = grid.total_nodes(poly_order=poly_order)
    f_nodes = np.zeros((n_nodes_total, 3), dtype=np.float64)
    np.add.at(f_nodes, nodes_conn.ravel(), f_elem.reshape(-1, 3))
    f_base = f_nodes.ravel()
    
    # 3) Nitsche BC on top cap (homogeneous displacement)
    nitsche_bcs = [{
        "triangles": top_mesh.triangles,
        "prescribed_displacement": None,
        "penalty": float(config["bc"]["nitsche_penalty"]),
        "order": g_order,
    }]
    
    nitsche_sets = FEM.NitscheAssembly.compute_terms(
        grid=grid,
        material=material,
        active_indices=active_indices,
        nitsche_bcs=nitsche_bcs,
        device="cpu",
        poly_order=poly_order,
    )
    
    # 4) Assemble global system
    k_global, f_global = FEM.GlobalAssembly.assemble_system(
        k_global=k_base,
        f_global=f_base,
        k_add=nitsche_sets,
    )
    k_global = k_global.tocsr()
    
    # 5) Solve full system
    active_dofs = np.where(np.diff(k_global.indptr) > 0)[0]
    k_aa = k_global[active_dofs][:, active_dofs]
    k_aa = 0.5 * (k_aa + k_aa.T)
    diag_m = np.clip(k_aa.diagonal().mean(), 1e-30, None)
    k_reg = k_aa + 1e-12 * diag_m * speye(k_aa.shape[0], format="csr")
    u_a = spsolve(k_reg, f_global[active_dofs])
    
    u_vec = np.zeros(n_nodes_total * 3, dtype=np.float64)
    u_vec[active_dofs] = u_a
    u_nodes = u_vec.reshape(-1, 3)
    
    return u_nodes, u_vec, n_nodes_total
