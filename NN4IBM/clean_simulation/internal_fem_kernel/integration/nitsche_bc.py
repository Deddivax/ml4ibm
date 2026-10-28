import numpy as np
import torch
from scipy.sparse import coo_matrix, issparse

from core.grid_indexing import active_element_nodes, cell_index, global_to_local, structured_total_nodes
from geometry import HexGrid
from integration.element_force import Force
from integration.gauss_integration import Gauss
from kinematics import Kinematics
from shape import Shape


def _to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


class NitscheBC:
    """Weak Dirichlet imposition on structured tensor-product hex grids via symmetric Nitsche terms."""

    @staticmethod
    def _axis_to_index(axis):
        if isinstance(axis, int):
            if axis in (0, 1, 2):
                return axis
            raise ValueError("Axis index must be 0, 1, or 2.")
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_key = str(axis).strip().lower()
        if axis_key not in axis_map:
            raise ValueError("Axis must be one of: 'x', 'y', 'z', 0, 1, 2.")
        return axis_map[axis_key]

    @staticmethod
    def _resolve_side(bounds, axis_idx, value=None, side=None, tol=1e-10):
        if side is not None:
            side_key = str(side).strip().lower()
            if side_key not in {"min", "max"}:
                raise ValueError("side must be 'min' or 'max'.")
            return side_key
        if value is None:
            raise ValueError("Provide either side=('min'|'max') or a boundary value.")
        vmin = float(bounds[0][axis_idx])
        vmax = float(bounds[1][axis_idx])
        if abs(float(value) - vmin) <= tol:
            return "min"
        if abs(float(value) - vmax) <= tol:
            return "max"
        raise ValueError(
            "For structured-grid boundary Nitsche terms, value must match the domain min/max plane."
        )

    @staticmethod
    def _boundary_elements(active_indices, resolution, axis_idx, side):
        nx, ny, nz = resolution
        e = np.asarray(active_indices, dtype=np.int64).reshape(-1)
        if e.size == 0:
            return e
        i = e // (ny * nz)
        j = (e % (ny * nz)) // nz
        k = e % nz
        if axis_idx == 0:
            target = int(i.min()) if side == "min" else int(i.max())
            mask = i == target
        elif axis_idx == 1:
            target = int(j.min()) if side == "min" else int(j.max())
            mask = j == target
        else:
            target = int(k.min()) if side == "min" else int(k.max())
            mask = k == target
        return e[mask]

    @staticmethod
    def _face_quadrature_and_geometry(grid, axis_idx, side, order, device):
        pts_2d, w_2d = Gauss.gauss_2d(order=order, device=device, dtype=torch.float64)
        u = pts_2d[:, 0]
        v = pts_2d[:, 1]
        fixed = -1.0 if side == "min" else 1.0

        ref = torch.zeros((len(pts_2d), 3), dtype=torch.float64, device=device)
        if axis_idx == 0:
            ref[:, 0] = fixed; ref[:, 1] = u; ref[:, 2] = v
            jac_face = 0.25 * grid.dy * grid.dz
            h_n = float(grid.dx)
        elif axis_idx == 1:
            ref[:, 0] = u; ref[:, 1] = fixed; ref[:, 2] = v
            jac_face = 0.25 * grid.dx * grid.dz
            h_n = float(grid.dy)
        else:
            ref[:, 0] = u; ref[:, 1] = v; ref[:, 2] = fixed
            jac_face = 0.25 * grid.dx * grid.dy
            h_n = float(grid.dz)

        n = torch.zeros(3, dtype=torch.float64, device=device)
        n[axis_idx] = -1.0 if side == "min" else 1.0
        weights = w_2d * jac_face
        return ref, weights, n, h_n

    @staticmethod
    def _N_matrix_batch(n_vals, device):
        """
        Batched N-matrix construction.
        n_vals : (G, n_nodes)
        Returns : (G, 3, 3*n_nodes)
        """
        G, n_nodes = n_vals.shape
        eye3 = torch.eye(3, dtype=torch.float64, device=device)          # (3, 3)
        # n_vals[:, a] * eye3  for each node a → (G, n_nodes, 3, 3)
        nmat = (n_vals[:, :, None, None] * eye3[None, None, :, :])       # (G, n_nodes, 3, 3)
        # rearrange to (G, 3, n_nodes*3)
        nmat = nmat.permute(0, 2, 1, 3).reshape(G, 3, 3 * n_nodes)
        return nmat

    @staticmethod
    def _traction_map_batch(normal, c_t, b_gp):
        """
        normal : (3,)
        c_t    : (6, 6)
        b_gp   : (G, 6, n_dofs)
        Returns: (G, 3, n_dofs)
        """
        nx, ny, nz = normal[0], normal[1], normal[2]
        t_n = torch.tensor(
            [[nx,  0.,  0., ny,  0., nz],
             [ 0., ny,  0., nx, nz,  0.],
             [ 0.,  0., nz,  0., ny, nx]],
            dtype=torch.float64, device=b_gp.device,
        )  # (3, 6)
        # (3,6)@(6,6) = (3,6);  then (3,6) @ (G,6,nd) via einsum
        tc = t_n @ c_t                                                    # (3, 6)
        return torch.einsum("ab,gbc->gac", tc, b_gp)                     # (G, 3, n_dofs)

    @staticmethod
    def _traction_map(normal, c_t, b_gp):
        nx, ny, nz = normal[0], normal[1], normal[2]
        t_n = torch.tensor(
            [[nx,  0.,  0., ny,  0., nz],
             [ 0., ny,  0., nx, nz,  0.],
             [ 0.,  0., nz,  0., ny, nx]],
            dtype=torch.float64, device=b_gp.device,
        )
        return t_n @ c_t @ b_gp

    @staticmethod
    def _normal_cell_size(normal, grid):
        normal_abs = np.abs(np.asarray(normal, dtype=np.float64))
        inv_h = (
            normal_abs[0] / float(grid.dx)
            + normal_abs[1] / float(grid.dy)
            + normal_abs[2] / float(grid.dz)
        )
        return 1.0 / max(inv_h, 1e-30)

    @staticmethod
    def _cells_to_ijk(cells, resolution):
        ny, nz = resolution[1], resolution[2]
        c = np.asarray(cells, dtype=np.int64).reshape(-1)
        i = c // (ny * nz)
        j = (c % (ny * nz)) // nz
        k = c % nz
        return np.stack([i, j, k], axis=1)

    @staticmethod
    def _map_points_to_cells(points, bounds_min, dx, resolution):
        p = np.asarray(points, dtype=np.float64)
        b = np.asarray(bounds_min, dtype=np.float64)
        h = np.asarray(dx, dtype=np.float64)
        ijk = np.floor((p - b[None, :]) / h[None, :]).astype(np.int64)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, resolution[0] - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, resolution[1] - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, resolution[2] - 1)
        cells = (
            ijk[:, 0] * (resolution[1] * resolution[2])
            + ijk[:, 1] * resolution[2]
            + ijk[:, 2]
        )
        return ijk, cells

    @staticmethod
    def _snap_boundary_cells_to_active(x_flat, n_flat, base_cells, grid, active_indices):
        active_set = set(_to_numpy(active_indices, dtype=np.int64).reshape(-1).tolist())
        cells = np.asarray(base_cells, dtype=np.int64).copy()
        if cells.size == 0 or len(active_set) == 0:
            return cells

        active_mask = np.asarray([c in active_set for c in cells], dtype=bool)
        if np.all(active_mask):
            return cells

        bounds_min = np.asarray(_to_numpy(grid.bounds[0]), dtype=np.float64)
        dx = np.array([float(grid.dx), float(grid.dy), float(grid.dz)], dtype=np.float64)
        eta = 0.25 * float(np.min(dx))
        p = _to_numpy(x_flat, dtype=np.float64)
        n = _to_numpy(n_flat, dtype=np.float64)

        idx_bad = np.where(~active_mask)[0]
        if idx_bad.size == 0:
            return cells

        _, cells_plus  = NitscheBC._map_points_to_cells(p[idx_bad] + eta * n[idx_bad], bounds_min, dx, grid.resolution)
        _, cells_minus = NitscheBC._map_points_to_cells(p[idx_bad] - eta * n[idx_bad], bounds_min, dx, grid.resolution)

        for loc, gp_idx in enumerate(idx_bad):
            cp, cm = int(cells_plus[loc]), int(cells_minus[loc])
            cp_ok, cm_ok = cp in active_set, cm in active_set
            if cp_ok and not cm_ok:
                cells[gp_idx] = cp
            elif cm_ok and not cp_ok:
                cells[gp_idx] = cm
            elif cp_ok and cm_ok:
                c0 = int(cells[gp_idx])
                cells[gp_idx] = cp if abs(cp - c0) <= abs(cm - c0) else cm
        return cells

    # ── coords template (zero-center reference cell) ─────────────────────────
    @staticmethod
    def _ref_coords_template(grid, poly_order, device):
        """(1, K, 3) float64 tensor — reference cell nodes at origin."""
        center = torch.zeros(3, dtype=torch.float64, device=device)
        nodes  = HexGrid.cell_nodes_from_centers(
            center, float(grid.dx), float(grid.dy), float(grid.dz), poly_order=poly_order
        )  # (K, 3)
        return nodes.unsqueeze(0)  # (1, K, 3)

    # ── elem dof indices (vectorised) ─────────────────────────────────────────
    @staticmethod
    def _elem_dofs_np(nodes_per_elem_np):
        """(E, n_dofs) int64 numpy — DOF indices for each element."""
        # nodes_per_elem_np : (E, n_nodes)
        n = nodes_per_elem_np                              # (E, n_nodes)
        dof_offsets = np.array([0, 1, 2], dtype=np.int64) # (3,)
        # (E, n_nodes, 3) → (E, n_dofs)
        return (n[:, :, None] * 3 + dof_offsets[None, None, :]).reshape(n.shape[0], -1)

    # ─────────────────────────────────────────────────────────────────────────
    # compute_surface_terms  (arbitrary embedded triangles)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_surface_terms(
        grid,
        material,
        triangles,
        active_indices=None,
        prescribed_displacement=None,
        penalty=10.0,
        order=2,
        poly_order=1,
        device="cpu",
    ):
        dtype = torch.float64

        triangles_t = torch.as_tensor(
            triangles,
            dtype=dtype,
            device=device,
        )

        n_nodes = structured_total_nodes(
            grid.resolution,
            poly_order=poly_order,
        )

        ndof = 3 * n_nodes

        if triangles_t.numel() == 0:

            idx = torch.zeros((2, 0), dtype=torch.long, device=device)
            vals = torch.zeros((0,), dtype=dtype, device=device)

            k_empty = torch.sparse_coo_tensor(
                idx,
                vals,
                (ndof, ndof),
                device=device,
            ).coalesce()

            f_empty = torch.zeros(
                ndof,
                dtype=dtype,
                device=device,
            )

            return (
                k_empty,
                f_empty,
                np.array([], dtype=np.int64),
            )

        bounds_min = torch.as_tensor(
            _to_numpy(grid.bounds[0]),
            dtype=dtype,
            device=device,
        )

        dx = torch.as_tensor(
            [
                float(grid.dx),
                float(grid.dy),
                float(grid.dz),
            ],
            dtype=dtype,
            device=device,
        )

        triangles_t = Force._refine_triangles_to_cell_size(
            triangles_t,
            dx,
        )

        x1 = triangles_t[:, 0]
        x2 = triangles_t[:, 1]
        x3 = triangles_t[:, 2]

        e1 = x2 - x1
        e2 = x3 - x1

        cross = torch.cross(e1, e2, dim=-1)
        norm_cross = torch.norm(cross, dim=-1)

        valid_tri = norm_cross > 1e-14

        normals = torch.zeros_like(cross)

        normals[valid_tri] = (
            cross[valid_tri]
            / norm_cross[valid_tri, None]
        )

        j_tri = norm_cross

        gp, gw = Gauss.gauss_triangle(
            order=order,
            device=device,
            dtype=dtype,
        )

        u = gp[:, 0]
        v = gp[:, 1]
        w_tri = 1.0 - u - v

        g = gp.shape[0]

        x_gp = (
            w_tri[None, :, None] * x1[:, None, :]
            + u[None, :, None] * x2[:, None, :]
            + v[None, :, None] * x3[:, None, :]
        )

        n_gp = normals[:, None, :].expand(-1, g, -1)

        ijk = ((x_gp - bounds_min) / dx).floor().long()

        nx, ny, nz = grid.resolution

        ijk[..., 0] = torch.clamp(ijk[..., 0], 0, nx - 1)
        ijk[..., 1] = torch.clamp(ijk[..., 1], 0, ny - 1)
        ijk[..., 2] = torch.clamp(ijk[..., 2], 0, nz - 1)

        cell = cell_index(
            ijk,
            grid.resolution,
        )

        x_flat = x_gp.reshape(-1, 3)
        n_flat = n_gp.reshape(-1, 3)
        cell_flat = cell.reshape(-1)

        valid_gp = (
            valid_tri[:, None]
            .expand(-1, g)
            .reshape(-1)
        )

        if active_indices is not None:

            active_indices_np = _to_numpy(
                active_indices,
                dtype=np.int64,
            )

            snapped_cells_np = (
                NitscheBC._snap_boundary_cells_to_active(
                    _to_numpy(x_flat),
                    _to_numpy(n_flat),
                    _to_numpy(cell_flat, dtype=np.int64),
                    grid,
                    active_indices_np,
                )
            )

            cell_flat = torch.as_tensor(
                snapped_cells_np,
                dtype=torch.long,
                device=device,
            )

            snapped_ijk_np = NitscheBC._cells_to_ijk(
                snapped_cells_np,
                grid.resolution,
            )

            ijk = torch.as_tensor(
                snapped_ijk_np.reshape(ijk.shape),
                dtype=torch.long,
                device=device,
            )

            active_set = set(
                active_indices_np.reshape(-1).tolist()
            )

            gp_cells_np = _to_numpy(
                cell_flat,
                dtype=np.int64,
            )

            valid_gp_np = _to_numpy(
                valid_gp,
                dtype=bool,
            )

            valid_gp_np &= np.array(
                [c in active_set for c in gp_cells_np],
                dtype=bool,
            )

            valid_gp = torch.from_numpy(
                valid_gp_np
            ).to(device=device)

        touched_cells = np.unique(
            _to_numpy(
                cell_flat[valid_gp],
                dtype=np.int64,
            )
        )

        if touched_cells.size == 0:

            idx = torch.zeros((2, 0), dtype=torch.long, device=device)
            vals = torch.zeros((0,), dtype=dtype, device=device)

            k_empty = torch.sparse_coo_tensor(
                idx,
                vals,
                (ndof, ndof),
                device=device,
            ).coalesce()

            f_empty = torch.zeros(
                ndof,
                dtype=dtype,
                device=device,
            )

            return (
                k_empty,
                f_empty,
                touched_cells,
            )

        cell_min_all = bounds_min + ijk * dx

        xi_flat = torch.clamp(
            global_to_local(
                x_gp,
                cell_min_all,
                dx,
            ),
            -1.0,
            1.0,
        ).reshape(-1, 3)

        coords_template = NitscheBC._ref_coords_template(
            grid,
            poly_order,
            device,
        )

        n_all = Shape.N3(
            xi_flat[:, 0],
            xi_flat[:, 1],
            xi_flat[:, 2],
            poly_order=poly_order,
        )

        d_n_ref = Shape.dN3_dxi(
            xi_flat[:, 0],
            xi_flat[:, 1],
            xi_flat[:, 2],
            poly_order=poly_order,
        )

        d_n_phys, _ = Shape.dN3_dx(
            d_n_ref,
            coords_template,
        )

        b_all = Kinematics.B_matrix(d_n_phys)

        c_t = torch.as_tensor(
            material.C,
            dtype=dtype,
            device=device,
        )

        e_eff = float(
            getattr(
                material,
                "E",
                np.max(np.diag(np.asarray(material.C))),
            )
        )

        weights_flat = (
            gw[None, :] * j_tri[:, None]
        ).reshape(-1)

        cell_to_pos = {
            int(c): pos
            for pos, c in enumerate(touched_cells)
        }

        nodes_per_elem = active_element_nodes(
            touched_cells,
            grid.resolution,
            poly_order=poly_order,
            device=device,
        ).long()

        dof_offsets = torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
            device=device,
        )

        elem_dofs = (
            nodes_per_elem[:, :, None] * 3
            + dof_offsets[None, None, :]
        ).reshape(nodes_per_elem.shape[0], -1)

        n_elem_dofs = elem_dofs.shape[1]
        E = elem_dofs.shape[0]

        ke_accum = torch.zeros(
            (E, n_elem_dofs, n_elem_dofs),
            dtype=dtype,
            device=device,
        )

        fe_accum = torch.zeros(
            (E, n_elem_dofs),
            dtype=dtype,
            device=device,
        )

        valid_idx = torch.where(valid_gp)[0]

        if valid_idx.numel() > 0:

            n_v = n_all[valid_idx]
            b_v = b_all[valid_idx]
            nf_v = n_flat[valid_idx]
            w_v = weights_flat[valid_idx]
            cell_v = cell_flat[valid_idx]

            nf_abs = nf_v.abs()

            inv_h = (
                nf_abs[:, 0] / float(grid.dx)
                + nf_abs[:, 1] / float(grid.dy)
                + nf_abs[:, 2] / float(grid.dz)
            ).clamp(min=1e-30)

            h_n_v = 1.0 / inv_h

            gamma_v = (
                float(penalty)
                * e_eff
                / h_n_v
            )

            n_mat_v = NitscheBC._N_matrix_batch(
                n_v,
                device,
            )

            zeros = torch.zeros_like(nf_v[:, 0])

            t_n_v = torch.stack(
                [
                    torch.stack(
                        [
                            nf_v[:, 0],
                            zeros,
                            zeros,
                            nf_v[:, 1],
                            zeros,
                            nf_v[:, 2],
                        ],
                        dim=1,
                    ),
                    torch.stack(
                        [
                            zeros,
                            nf_v[:, 1],
                            zeros,
                            nf_v[:, 0],
                            nf_v[:, 2],
                            zeros,
                        ],
                        dim=1,
                    ),
                    torch.stack(
                        [
                            zeros,
                            zeros,
                            nf_v[:, 2],
                            zeros,
                            nf_v[:, 1],
                            nf_v[:, 0],
                        ],
                        dim=1,
                    ),
                ],
                dim=1,
            )

            h_mat_v = torch.einsum(
                "gab,bc,gcd->gad",
                t_n_v,
                c_t,
                b_v,
            )

            nt_h = torch.bmm(
                n_mat_v.transpose(1, 2),
                h_mat_v,
            )

            ht_n = torch.bmm(
                h_mat_v.transpose(1, 2),
                n_mat_v,
            )

            nt_n = torch.bmm(
                n_mat_v.transpose(1, 2),
                n_mat_v,
            )

            ke_gp = (
                -nt_h * w_v[:, None, None]
                - ht_n * w_v[:, None, None]
                + nt_n * (
                    w_v * gamma_v
                )[:, None, None]
            )

            pos_v = torch.tensor(
                [
                    cell_to_pos[int(c)]
                    for c in _to_numpy(
                        cell_v,
                        dtype=np.int64,
                    )
                ],
                dtype=torch.long,
                device=device,
            )

            ke_accum.index_add_(
                0,
                pos_v,
                ke_gp,
            )

            if prescribed_displacement is not None:

                x_v = x_flat[valid_idx]

                g_val = prescribed_displacement(
                    x_v,
                    nf_v,
                )

                g_t = torch.as_tensor(
                    g_val,
                    dtype=dtype,
                    device=device,
                )

                if g_t.ndim == 1:
                    g_t = g_t.unsqueeze(0).expand(
                        valid_idx.numel(),
                        -1,
                    )

                ht_g = torch.bmm(
                    h_mat_v.transpose(1, 2),
                    g_t.unsqueeze(2),
                ).squeeze(2)

                nt_g = torch.bmm(
                    n_mat_v.transpose(1, 2),
                    g_t.unsqueeze(2),
                ).squeeze(2)

                fe_gp = (
                    -ht_g * w_v[:, None]
                    + nt_g * (
                        w_v * gamma_v
                    )[:, None]
                )

                fe_accum.index_add_(
                    0,
                    pos_v,
                    fe_gp,
                )

        rows = torch.repeat_interleave(
            elem_dofs,
            n_elem_dofs,
            dim=1,
        ).reshape(-1)

        cols = torch.tile(
            elem_dofs,
            (1, n_elem_dofs),
        ).reshape(-1)

        vals = ke_accum.reshape(-1)

        k_add = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            (ndof, ndof),
            dtype=dtype,
            device=device,
        ).coalesce()

        f_add = torch.zeros(
            ndof,
            dtype=dtype,
            device=device,
        )

        f_add.index_add_(
            0,
            elem_dofs.reshape(-1),
            fe_accum.reshape(-1),
        )

        return (
            k_add,
            f_add,
            touched_cells,
            gamma_v
        )

    # ─────────────────────────────────────────────────────────────────────────
    # compute_plane_terms  (structured boundary plane)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_plane_terms(
        grid,
        material,
        active_indices,
        axis,
        value=None,
        side=None,
        prescribed_displacement=None,
        penalty=10.0,
        order=2,
        poly_order=1,
        device="cuda",
    ):

        axis_idx = NitscheBC._axis_to_index(axis)

        side_key = NitscheBC._resolve_side(
            grid.bounds,
            axis_idx,
            value=value,
            side=side,
        )

        if not torch.is_tensor(active_indices):

            active_indices = torch.as_tensor(
                active_indices,
                dtype=torch.int64,
                device=device,
            )

        boundary_elems = NitscheBC._boundary_elements(
            active_indices,
            grid.resolution,
            axis_idx,
            side_key,
        )

        n_nodes = structured_total_nodes(
            grid.resolution,
            poly_order=poly_order,
        )

        ndof = 3 * n_nodes

        # ------------------------------------------------------------
        # empty boundary
        # ------------------------------------------------------------
        if boundary_elems.numel() == 0:

            indices = torch.empty(
                (2, 0),
                dtype=torch.int64,
                device=device,
            )

            values = torch.empty(
                (0,),
                dtype=torch.float64,
                device=device,
            )

            k_add = torch.sparse_coo_tensor(
                indices,
                values,
                (ndof, ndof),
                device=device,
            ).coalesce()

            f_add = torch.zeros(
                ndof,
                dtype=torch.float64,
                device=device,
            )

            return k_add, f_add, boundary_elems, None

        # ------------------------------------------------------------
        # quadrature
        # ------------------------------------------------------------
        ref_pts, w_face, normal, h_n = (
            NitscheBC._face_quadrature_and_geometry(
                grid,
                axis_idx,
                side_key,
                order,
                device,
            )
        )

        gq = ref_pts.shape[0]

        coords_template = NitscheBC._ref_coords_template(
            grid,
            poly_order,
            device,
        )

        d_n_ref = Shape.dN3_dxi(
            ref_pts[:, 0],
            ref_pts[:, 1],
            ref_pts[:, 2],
            poly_order=poly_order,
        )

        d_n_phys, _ = Shape.dN3_dx(
            d_n_ref,
            coords_template,
        )

        b_all = Kinematics.B_matrix(d_n_phys)

        n_all = Shape.N3(
            ref_pts[:, 0],
            ref_pts[:, 1],
            ref_pts[:, 2],
            poly_order=poly_order,
        )

        n_nodes_per_elem = int(n_all.shape[1])

        n_elem_dofs = 3 * n_nodes_per_elem

        c_t = torch.as_tensor(
            material.C,
            dtype=torch.float64,
            device=device,
        )

        e_eff = float(
            getattr(
                material,
                "E",
                torch.max(torch.diag(c_t)).item(),
            )
        )

        gamma = float(penalty) * e_eff / float(h_n)

        # ------------------------------------------------------------
        # local stiffness
        # ------------------------------------------------------------
        n_mat_all = NitscheBC._N_matrix_batch(
            n_all,
            device,
        )

        h_mat_all = NitscheBC._traction_map_batch(
            normal,
            c_t,
            b_all,
        )

        nt_h = torch.bmm(
            n_mat_all.transpose(1, 2),
            h_mat_all,
        )

        ht_n = torch.bmm(
            h_mat_all.transpose(1, 2),
            n_mat_all,
        )

        nt_n = torch.bmm(
            n_mat_all.transpose(1, 2),
            n_mat_all,
        )

        ke_local = (
            (-nt_h - ht_n + gamma * nt_n)
            * w_face[:, None, None]
        ).sum(dim=0)

        # ------------------------------------------------------------
        # connectivity
        # ------------------------------------------------------------
        nodes_per_elem = active_element_nodes(
            boundary_elems,
            grid.resolution,
            poly_order=poly_order,
            device=device,
        )

        dof_offsets = torch.tensor(
            [0, 1, 2],
            dtype=torch.int64,
            device=device,
        )

        elem_dofs = (
            nodes_per_elem[:, :, None] * 3
            + dof_offsets[None, None, :]
        ).reshape(-1, n_elem_dofs)

        n_boundary = boundary_elems.shape[0]

        # ------------------------------------------------------------
        # sparse assembly
        # ------------------------------------------------------------
        rows = torch.repeat_interleave(
            elem_dofs,
            n_elem_dofs,
            dim=1,
        ).reshape(-1)

        cols = torch.tile(
            elem_dofs,
            (1, n_elem_dofs),
        ).reshape(-1)

        vals = ke_local.reshape(1, -1).repeat(
            n_boundary,
            1,
        ).reshape(-1)

        indices = torch.stack(
            [rows, cols],
            dim=0,
        )

        k_add = torch.sparse_coo_tensor(
            indices,
            vals,
            (ndof, ndof),
            dtype=torch.float64,
            device=device,
        ).coalesce()

        # ------------------------------------------------------------
        # rhs
        # ------------------------------------------------------------
        f_add = torch.zeros(
            ndof,
            dtype=torch.float64,
            device=device,
        )

        if prescribed_displacement is not None:

            nx, ny, nz = grid.resolution

            bounds_min = torch.as_tensor(
                grid.bounds[0],
                dtype=torch.float64,
                device=device,
            )

            dxyz = torch.tensor(
                [grid.dx, grid.dy, grid.dz],
                dtype=torch.float64,
                device=device,
            )

            normal_gp = normal.unsqueeze(0).expand(gq, -1)

            for e_pos, e_idx in enumerate(boundary_elems):

                dofs = elem_dofs[e_pos]

                i = e_idx // (ny * nz)

                j = (e_idx % (ny * nz)) // nz

                k = e_idx % nz

                cell_min = bounds_min + torch.stack(
                    [i, j, k]
                ).to(torch.float64) * dxyz

                x_gp = (
                    cell_min.unsqueeze(0)
                    + 0.5 * (ref_pts + 1.0) * dxyz.unsqueeze(0)
                )

                g_t = prescribed_displacement(
                    x_gp,
                    normal_gp,
                )

                if g_t.ndim == 1:

                    g_t = g_t.unsqueeze(0).expand(gq, -1)

                ht_g = torch.bmm(
                    h_mat_all.transpose(1, 2),
                    g_t.unsqueeze(2),
                ).squeeze(2)

                nt_g = torch.bmm(
                    n_mat_all.transpose(1, 2),
                    g_t.unsqueeze(2),
                ).squeeze(2)

                fe_local = (
                    (-ht_g + gamma * nt_g)
                    * w_face[:, None]
                ).sum(dim=0)

                f_add.index_add_(
                    0,
                    dofs,
                    fe_local,
                )

        return k_add, f_add, boundary_elems

        # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def apply_to_system(
        k,
        f,
        grid,
        material,
        active_indices,
        axis,
        value=None,
        side=None,
        prescribed_displacement=None,
        penalty=10.0,
        order=2,
        poly_order=1,
        device="cuda",
    ):

        k_add, f_add, boundary_elems = NitscheBC.compute_plane_terms(
            grid=grid,
            material=material,
            active_indices=active_indices,
            axis=axis,
            value=value,
            side=side,
            prescribed_displacement=prescribed_displacement,
            penalty=penalty,
            order=order,
            poly_order=poly_order,
            device=device,
        )

        # ------------------------------------------------------------
        # force torch tensors
        # ------------------------------------------------------------
        if not torch.is_tensor(k):
            k = torch.as_tensor(
                k,
                dtype=torch.float64,
                device=device,
            )

        if not torch.is_tensor(f):
            f = torch.as_tensor(
                f,
                dtype=torch.float64,
                device=device,
            )

        if not torch.is_tensor(k_add):
            k_add = torch.as_tensor(
                k_add,
                dtype=k.dtype,
                device=device,
            )

        if not torch.is_tensor(f_add):
            f_add = torch.as_tensor(
                f_add,
                dtype=f.dtype,
                device=device,
            )

        # ------------------------------------------------------------
        # sparse handling
        # ------------------------------------------------------------
        if k.is_sparse:

            if not k_add.is_sparse:
                k_add = k_add.to_sparse()

            k_out = (k + k_add).coalesce()

        else:

            if k_add.is_sparse:
                k_add = k_add.to_dense()

            k_out = k + k_add

        # ------------------------------------------------------------
        # rhs
        # ------------------------------------------------------------
        f_out = f + f_add

        return k_out, f_out, boundary_elems