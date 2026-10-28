import numpy as np
import torch

from core.grid_indexing import active_element_nodes, structured_total_nodes
from integration.gauss_integration import Gauss
from integration.octreesampler import OctreeSampler
from shape import Shape


def _is_axis_aligned_box_volume(volume, bounds_min, dx, resolution, tol=1e-10):
    if volume is None or not hasattr(volume, "mesh"):
        return False
    mesh = volume.mesh
    mesh_bounds = np.asarray(getattr(mesh, "bounds", None), dtype=np.float64)
    if mesh_bounds.shape != (2, 3):
        return False
    bounds_min = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    dx = np.asarray(dx, dtype=np.float64).reshape(3)
    resolution = np.asarray(resolution, dtype=np.int64).reshape(3)
    bounds_max = bounds_min + dx * resolution
    if not np.allclose(mesh_bounds[0], bounds_min, atol=tol, rtol=0.0):
        return False
    if not np.allclose(mesh_bounds[1], bounds_max, atol=tol, rtol=0.0):
        return False
    face_normals = np.asarray(getattr(mesh, "face_normals", None), dtype=np.float64)
    if face_normals.ndim != 2 or face_normals.shape[1] != 3 or face_normals.shape[0] == 0:
        return False
    abs_normals = np.abs(face_normals)
    dominant_axis = abs_normals.argmax(axis=1)
    dominant_mag = abs_normals[np.arange(abs_normals.shape[0]), dominant_axis]
    off_axis = abs_normals.copy()
    off_axis[np.arange(abs_normals.shape[0]), dominant_axis] = 0.0
    return bool(np.allclose(dominant_mag, 1.0, atol=tol, rtol=0.0) and np.all(off_axis <= tol))


def _volume_rule_key(order, poly_order, device, dtype):
    return int(order), int(poly_order), str(device), str(dtype)


def _get_volume_rule(order, poly_order, device, dtype):
    pts, gw = Gauss.gauss_3d(order, device=device, dtype=dtype)
    return pts, gw, Shape.N3(pts[:, 0], pts[:, 1], pts[:, 2], poly_order=poly_order), Shape.dN3_dxi(pts[:, 0], pts[:, 1], pts[:, 2], poly_order=poly_order)


def _cut_points_from_order(order):
    return (2 * max(1, int(order))) ** 3


def _refine_triangles_to_cell_size(triangles, dx, max_refine=8, tol=1e-12):
    refined = triangles
    cell_span = dx.view(1, 3)
    for _ in range(max_refine):
        spans = refined.amax(dim=1) - refined.amin(dim=1)
        needs_split = (spans > (cell_span + tol)).any(dim=1)
        if not torch.any(needs_split):
            break
        keep = refined[~needs_split]
        tri = refined[needs_split]
        a = tri[:, 0, :]
        b = tri[:, 1, :]
        c = tri[:, 2, :]
        ab = 0.5 * (a + b)
        bc = 0.5 * (b + c)
        ca = 0.5 * (c + a)
        children = torch.stack([torch.stack([a, ab, ca], dim=1), torch.stack([ab, b, bc], dim=1), torch.stack([ca, bc, c], dim=1), torch.stack([ab, bc, ca], dim=1)], dim=1).reshape(-1, 3, 3)
        refined = children if keep.numel() == 0 else torch.cat([keep, children], dim=0)
    return refined


def integrate_structured_box_surface_force(bounds_min, dx, resolution, traction_func, active_cell_mask=None, order=2, poly_order=1, device="cpu"):
    dtype = torch.float64
    bounds_min_t = torch.as_tensor(bounds_min, dtype=dtype, device=device)
    dx_t = torch.as_tensor(dx, dtype=dtype, device=device)
    nx, ny, nz = [int(v) for v in resolution]
    E = nx * ny * nz
    n_nodes = (int(poly_order) + 1) ** 3
    active_mask = None
    if active_cell_mask is not None:
        active_mask = np.asarray(active_cell_mask, dtype=bool).reshape(-1)
        if active_mask.shape[0] != E:
            raise ValueError("active_cell_mask must have one entry per grid cell.")
    pts_2d, w_2d = Gauss.gauss_2d(order=order, device=device, dtype=dtype)
    elem_ids = np.arange(E, dtype=np.int64)
    i = elem_ids // (ny * nz)
    j = (elem_ids % (ny * nz)) // nz
    k = elem_ids % nz
    F = torch.zeros((E, n_nodes, 3), dtype=dtype, device=device)
    ref = torch.zeros((pts_2d.shape[0], 3), dtype=dtype, device=device)
    face_specs = [(0, 0, -1.0, 0.25 * float(dx_t[1] * dx_t[2]), (-1.0, 0.0, 0.0)), (0, nx - 1, 1.0, 0.25 * float(dx_t[1] * dx_t[2]), (1.0, 0.0, 0.0)), (1, 0, -1.0, 0.25 * float(dx_t[0] * dx_t[2]), (0.0, -1.0, 0.0)), (1, ny - 1, 1.0, 0.25 * float(dx_t[0] * dx_t[2]), (0.0, 1.0, 0.0)), (2, 0, -1.0, 0.25 * float(dx_t[0] * dx_t[1]), (0.0, 0.0, -1.0)), (2, nz - 1, 1.0, 0.25 * float(dx_t[0] * dx_t[1]), (0.0, 0.0, 1.0))]
    for axis_idx, target, fixed, jac_face, normal_vals in face_specs:
        ref.zero_()
        if axis_idx == 0:
            boundary_elems = elem_ids[i == target]
            ref[:, 0] = fixed
            ref[:, 1] = pts_2d[:, 0]
            ref[:, 2] = pts_2d[:, 1]
        elif axis_idx == 1:
            boundary_elems = elem_ids[j == target]
            ref[:, 0] = pts_2d[:, 0]
            ref[:, 1] = fixed
            ref[:, 2] = pts_2d[:, 1]
        else:
            boundary_elems = elem_ids[k == target]
            ref[:, 0] = pts_2d[:, 0]
            ref[:, 1] = pts_2d[:, 1]
            ref[:, 2] = fixed
        if boundary_elems.size == 0:
            continue
        if active_mask is not None:
            boundary_elems = boundary_elems[active_mask[boundary_elems]]
            if boundary_elems.size == 0:
                continue
        ijk = np.stack([i[boundary_elems], j[boundary_elems], k[boundary_elems]], axis=1)
        ijk_t = torch.as_tensor(ijk, dtype=dtype, device=device)
        cell_min = bounds_min_t.unsqueeze(0) + ijk_t * dx_t.unsqueeze(0)
        x_phys = cell_min[:, None, :] + 0.5 * (ref.unsqueeze(0) + 1.0) * dx_t.view(1, 1, 3)
        normal = torch.tensor(normal_vals, dtype=dtype, device=device)
        normals = normal.view(1, 1, 3).expand(boundary_elems.size, ref.shape[0], -1)
        traction = traction_func(x_phys, normals)
        N3 = Shape.N3(ref[:, 0], ref[:, 1], ref[:, 2], poly_order=poly_order)
        contrib = (N3.unsqueeze(0).unsqueeze(-1) * traction.unsqueeze(2) * (w_2d * jac_face).view(1, -1, 1, 1)).sum(dim=1)
        F[torch.as_tensor(boundary_elems, dtype=torch.long, device=device)] += contrib
    return F


def integrate_surface_force(triangles, bounds_min, dx, resolution, traction_func, active_cell_mask=None, order=2, poly_order=1, device="cpu"):
    dtype = torch.float64
    triangles = torch.as_tensor(triangles, dtype=dtype, device=device)
    bounds_min = torch.as_tensor(bounds_min, dtype=dtype, device=device)
    dx = torch.as_tensor(dx, dtype=dtype, device=device)
    nx, ny, nz = resolution
    E = nx * ny * nz
    n_nodes = (int(poly_order) + 1) ** 3
    triangles = _refine_triangles_to_cell_size(triangles, dx)
    x1 = triangles[:, 0]
    x2 = triangles[:, 1]
    x3 = triangles[:, 2]
    e1 = x2 - x1
    e2 = x3 - x1
    cross = torch.cross(e1, e2, dim=-1)
    norm_cross = torch.norm(cross, dim=-1)
    valid_tri = norm_cross > 1e-14
    J_tri = norm_cross
    n = torch.zeros_like(cross)
    n[valid_tri] = cross[valid_tri] / norm_cross[valid_tri, None]
    gp, gw = Gauss.gauss_triangle(order, device=device, dtype=dtype)
    T = triangles.shape[0]
    F = torch.zeros((E, n_nodes, 3), dtype=dtype, device=device)
    elem_ids = torch.arange(E, device=device)
    if active_cell_mask is not None:
        active_mask = torch.as_tensor(active_cell_mask, dtype=torch.bool, device=device).reshape(-1)
    else:
        active_mask = None
    i = elem_ids // (ny * nz)
    j = (elem_ids % (ny * nz)) // nz
    k = elem_ids % nz
    cell_index_cache = None
    for axis_idx, target, fixed in ((0, 0, -1.0), (0, nx - 1, 1.0), (1, 0, -1.0), (1, ny - 1, 1.0), (2, 0, -1.0), (2, nz - 1, 1.0)):
        if axis_idx == 0:
            boundary_elems = elem_ids[i == target]
        elif axis_idx == 1:
            boundary_elems = elem_ids[j == target]
        else:
            boundary_elems = elem_ids[k == target]
        if boundary_elems.numel() == 0:
            continue
        if active_mask is not None:
            boundary_elems = boundary_elems[active_mask[boundary_elems]]
            if boundary_elems.numel() == 0:
                continue
        ijk = torch.stack([i[boundary_elems], j[boundary_elems], k[boundary_elems]], dim=1).to(dtype=dtype)
        cell_min = bounds_min.unsqueeze(0) + ijk * dx.unsqueeze(0)
        ref = torch.zeros((gp.shape[0], 3), dtype=dtype, device=device)
        if axis_idx == 0:
            ref[:, 0] = fixed
            ref[:, 1] = gp[:, 0]
            ref[:, 2] = gp[:, 1]
        elif axis_idx == 1:
            ref[:, 0] = gp[:, 0]
            ref[:, 1] = fixed
            ref[:, 2] = gp[:, 1]
        else:
            ref[:, 0] = gp[:, 0]
            ref[:, 1] = gp[:, 1]
            ref[:, 2] = fixed
        x_phys = cell_min[:, None, :] + 0.5 * (ref.unsqueeze(0) + 1.0) * dx.view(1, 1, 3)
        normal = torch.zeros(3, dtype=dtype, device=device)
        normal[axis_idx] = fixed
        normals = normal.view(1, 1, 3).expand(boundary_elems.numel(), ref.shape[0], -1)
        traction = traction_func(x_phys, normals)
        N3 = Shape.N3(ref[:, 0], ref[:, 1], ref[:, 2], poly_order=poly_order)
        w_face = (gw * J_tri[boundary_elems]).view(-1, 1)
        contrib = (N3.unsqueeze(0).unsqueeze(-1) * traction.unsqueeze(2) * w_face.view(-1, 1, 1, 1)).sum(dim=1)
        F[boundary_elems.long()] += contrib
    return F


def integrate_volume_force(hex_coords, body_force_func, order=2, poly_order=1, device="cpu", volume=None, inside_mask=None, partial_mask=None, cut_max_depth=3):
    if hex_coords is None:
        raise ValueError("hex_coords is required")
    _, gw, N3, dN_ref = _get_volume_rule(order, poly_order, device, torch.float64)
    _ = dN_ref
    elem = torch.as_tensor(hex_coords, dtype=torch.float64, device=device)
    coords = elem.reshape(-1, 8, 3)
    x = torch.einsum("gq,eqd->egd", N3, coords)
    f = body_force_func(x.reshape(-1, 3)).reshape(coords.shape[0], -1, 3)
    weighted = f * gw.view(1, -1, 1)
    out = weighted.sum(dim=1)
    return out


def compute_element_force(triangles, hex_coords, bounds_min, dx, resolution, traction_func, body_force_func=None, inside=None, partial=None, volume=None, cut_max_depth=3, order=2, poly_order=1, device="cpu"):
    active_cell_mask = None
    if inside is not None and partial is not None:
        active_cell_mask = np.asarray(inside, dtype=bool).reshape(-1) | np.asarray(partial, dtype=bool).reshape(-1)
    if _is_axis_aligned_box_volume(volume, bounds_min, dx, resolution):
        f_surface = integrate_structured_box_surface_force(bounds_min, dx, resolution, traction_func, active_cell_mask=active_cell_mask, order=order, poly_order=poly_order, device=device)
    else:
        f_surface = integrate_surface_force(triangles, bounds_min, dx, resolution, traction_func, active_cell_mask=active_cell_mask, order=order, poly_order=poly_order, device=device)
    if body_force_func is not None:
        f_volume = integrate_volume_force(hex_coords, body_force_func, order=order, poly_order=poly_order, device=device, volume=volume, inside_mask=inside, partial_mask=partial, cut_max_depth=cut_max_depth)
        return f_surface + f_volume
    return f_surface


def assemble_plane_traction_force(grid, active_indices, axis, side, traction, order=2, poly_order=1, device="cpu"):
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_key = str(axis).strip().lower()
    side_key = str(side).strip().lower()
    if axis_key not in axis_map:
        raise ValueError("axis must be one of 'x', 'y', 'z'.")
    if side_key not in {"min", "max"}:
        raise ValueError("side must be one of 'min', 'max'.")
    axis_idx = axis_map[axis_key]
    nx, ny, nz = grid.resolution
    e = np.asarray(active_indices, dtype=np.int64).reshape(-1)
    if e.size == 0:
        n_nodes = structured_total_nodes(grid.resolution, poly_order=poly_order)
        return torch.zeros(3 * n_nodes, dtype=torch.float64, device=device)
    i = e // (ny * nz)
    j = (e % (ny * nz)) // nz
    k = e % nz
    if axis_idx == 0:
        target = int(i.min()) if side_key == "min" else int(i.max())
        boundary_elems = e[i == target]
        jac_face = 0.25 * float(grid.dy) * float(grid.dz)
        fixed = -1.0 if side_key == "min" else 1.0
    elif axis_idx == 1:
        target = int(j.min()) if side_key == "min" else int(j.max())
        boundary_elems = e[j == target]
        jac_face = 0.25 * float(grid.dx) * float(grid.dz)
        fixed = -1.0 if side_key == "min" else 1.0
    else:
        target = int(k.min()) if side_key == "min" else int(k.max())
        boundary_elems = e[k == target]
        jac_face = 0.25 * float(grid.dx) * float(grid.dy)
        fixed = -1.0 if side_key == "min" else 1.0
    pts_2d, w_2d = Gauss.gauss_2d(order=order, device=device, dtype=torch.float64)
    ref = torch.zeros((pts_2d.shape[0], 3), dtype=torch.float64, device=device)
    if axis_idx == 0:
        ref[:, 0] = fixed
        ref[:, 1] = pts_2d[:, 0]
        ref[:, 2] = pts_2d[:, 1]
    elif axis_idx == 1:
        ref[:, 0] = pts_2d[:, 0]
        ref[:, 1] = fixed
        ref[:, 2] = pts_2d[:, 1]
    else:
        ref[:, 0] = pts_2d[:, 0]
        ref[:, 1] = pts_2d[:, 1]
        ref[:, 2] = fixed
    n_vals = Shape.N3(ref[:, 0], ref[:, 1], ref[:, 2], poly_order=poly_order)
    w_face = w_2d * jac_face
    n_nodes = structured_total_nodes(grid.resolution, poly_order=poly_order)
    nodes = active_element_nodes(boundary_elems, grid.resolution, poly_order=poly_order)
    nodes_t = torch.as_tensor(nodes, dtype=torch.long, device=device)
    local = torch.zeros((nodes_t.shape[1], 3), dtype=torch.float64, device=device)
    traction_t = torch.as_tensor(traction, dtype=torch.float64, device=device).reshape(1, 3)
    for gp in range(ref.shape[0]):
        local += n_vals[gp].reshape(-1, 1) * traction_t * w_face[gp]
    f_global = torch.zeros((n_nodes, 3), dtype=torch.float64, device=device)
    for e_idx in range(nodes_t.shape[0]):
        f_global.index_add_(0, nodes_t[e_idx], local)
    return f_global.reshape(-1)


def assemble_global_force(f_elem, active_indices, grid, device="cpu"):
    n_nodes_per_elem = f_elem.shape[1]
    poly_order = round(n_nodes_per_elem ** (1.0 / 3.0)) - 1
    n_nodes = structured_total_nodes(grid.resolution, poly_order=poly_order)
    nodes = active_element_nodes(active_indices, grid.resolution, poly_order=poly_order)
    nodes = torch.tensor(nodes, dtype=torch.long, device=device)
    f_active = f_elem[active_indices]
    nodes_flat = nodes.reshape(-1)
    forces_flat = f_active.reshape(-1, 3)
    f_global = torch.zeros((n_nodes, 3), dtype=torch.float64, device=device)
    f_global.index_add_(0, nodes_flat, forces_flat)
    return f_global.reshape(-1)


class Force:
    _volume_rule_cache = None
    _is_axis_aligned_box_volume = staticmethod(_is_axis_aligned_box_volume)
    _volume_rule_key = staticmethod(_volume_rule_key)
    _get_volume_rule = staticmethod(_get_volume_rule)
    _cut_points_from_order = staticmethod(_cut_points_from_order)
    _refine_triangles_to_cell_size = staticmethod(_refine_triangles_to_cell_size)
    integrate_structured_box_surface_force = staticmethod(integrate_structured_box_surface_force)
    integrate_surface_force = staticmethod(integrate_surface_force)
    integrate_volume_force = staticmethod(integrate_volume_force)
    compute_element_force = staticmethod(compute_element_force)
    assemble_plane_traction_force = staticmethod(assemble_plane_traction_force)


class ForceAssembly:
    assemble_plane_traction_force = staticmethod(assemble_plane_traction_force)
    compute_element_force = staticmethod(compute_element_force)
    assemble_global_force = staticmethod(assemble_global_force)
