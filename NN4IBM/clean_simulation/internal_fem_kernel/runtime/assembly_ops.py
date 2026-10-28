from __future__ import annotations

import torch

from ..core.grid_indexing import active_element_nodes
from ..fem import FEM
from ..shape import Shape
from .common import as_tensor

def build_body_force_from_ips(
    grid,
    active_indices,
    ips,
    config: dict,
    poly_order: int,
    device: str = "cuda",
) -> torch.Tensor:

    active_indices_t = as_tensor(active_indices, dtype=torch.int64, device=device).reshape(-1)

    ips_t = as_tensor(ips, dtype=torch.float64, device=device)
    if ips_t.numel() == 0:
        n_nodes_total = grid.total_nodes(poly_order=poly_order)
        return torch.zeros((n_nodes_total * 3,), dtype=torch.float64, device=device)

    ip_coords = ips_t[:, :3]
    ip_w = ips_t[:, 3]

    nx, ny, nz = grid.resolution
    bounds_min = as_tensor(grid.bounds[0], dtype=torch.float64, device=device)
    bounds_max = as_tensor(grid.bounds[1], dtype=torch.float64, device=device)
    dx = torch.tensor([grid.dx, grid.dy, grid.dz], dtype=torch.float64, device=device)

    # QuESo points may be returned in UVW ([0,1]^3) even when grid bounds are physical XYZ.
    in_unit_box = bool(torch.all((ip_coords >= -1.0e-12) & (ip_coords <= 1.0 + 1.0e-12)).item())
    bounds_are_unit = bool(torch.all((bounds_min >= -1.0e-12) & (bounds_max <= 1.0 + 1.0e-12)).item())
    if in_unit_box and not bounds_are_unit:
        ip_xyz = bounds_min + ip_coords * (bounds_max - bounds_min)
        ip_w = ip_w * torch.prod(bounds_max - bounds_min)
    else:
        ip_xyz = ip_coords

    rho = float(config["material"]["density"])
    g_vec = as_tensor(config["loads"]["gravity"], dtype=torch.float64, device=device)
    body_force = rho * g_vec

    ijk = torch.floor((ip_xyz - bounds_min) / dx).to(torch.int64)
    ijk_min = torch.zeros(3, dtype=torch.int64, device=device)
    ijk_max = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=torch.int64, device=device)
    ijk = torch.minimum(torch.maximum(ijk, ijk_min), ijk_max)

    flat = ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]

    cell_min = bounds_min + ijk.to(torch.float64) * dx
    xi_all = torch.clamp(2.0 * (ip_xyz - cell_min) / dx - 1.0, -1.0, 1.0)
    n_all = Shape.N3(xi_all[:, 0], xi_all[:, 1], xi_all[:, 2], poly_order=poly_order)

    flat_to_pos = torch.full((nx * ny * nz,), -1, dtype=torch.int64, device=device)
    flat_to_pos[active_indices_t] = torch.arange(active_indices_t.numel(), dtype=torch.int64, device=device)
    pos = flat_to_pos[flat]
    valid = pos >= 0

    n_active = int(active_indices_t.numel())
    n_nodes_per_elem = (poly_order + 1) ** 3
    f_elem = torch.zeros((n_active, n_nodes_per_elem, 3), dtype=torch.float64, device=device)

    contrib = ip_w[valid, None, None] * n_all[valid, :, None] * body_force[None, None, :]
    elem_index = pos[valid][:, None, None].expand(-1, n_nodes_per_elem, 3)
    f_elem.scatter_add_(0, elem_index, contrib)

    nodes_conn = active_element_nodes(
        active_indices_t,
        grid.resolution,
        poly_order=poly_order,
        device=device,
    )

    n_nodes_total = grid.total_nodes(poly_order=poly_order)
    f_nodes = torch.zeros((n_nodes_total, 3), dtype=torch.float64, device=device)
    f_nodes.index_add_(0, nodes_conn.reshape(-1), f_elem.reshape(-1, 3))
    return f_nodes.reshape(-1)


def assemble_stiffness_base(
    *,
    ke_list,
    active_indices,
    grid,
    poly_order: int,
    device: str,
    sparse_backend: str = "torch",
):
    return FEM.StiffnessAssembly.assemble_global_stiffness_sparse(
        ke_list,
        active_indices,
        grid,
        poly_order=poly_order,
        device=device,
        sparse_backend=sparse_backend,
    )


def assemble_force_base(*, grid, active_indices, ips, config: dict, poly_order: int, device: str = "cuda") -> torch.Tensor:
    return build_body_force_from_ips(
        grid=grid,
        active_indices=active_indices,
        ips=ips,
        config=config,
        poly_order=poly_order,
        device=device,
    )
