import numpy as np
import torch
import warnings
from shape import Shape
from integration.gauss_integration import Gauss
from integration.octreesampler import OctreeSampler
from core.grid_indexing import global_to_local, cell_index


class Force:

    _volume_rule_cache = {}

    @staticmethod
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

        return bool(
            np.allclose(dominant_mag, 1.0, atol=tol, rtol=0.0)
            and np.all(off_axis <= tol)
        )

    @staticmethod
    def _volume_rule_key(order, poly_order, device, dtype):
        return (int(order), int(poly_order), str(device), str(dtype))

    @staticmethod
    def _get_volume_rule(order, poly_order, device, dtype):
        pts, gw = Gauss.gauss_3d(order, device=device, dtype=dtype)
        N3 = Shape.N3(pts[:, 0], pts[:, 1], pts[:, 2], poly_order=poly_order)
        dN_ref = Shape.dN3_dxi(pts[:, 0], pts[:, 1], pts[:, 2], poly_order=poly_order)
        return pts, gw, N3, dN_ref

    @staticmethod
    def _cut_points_from_order(order):
        p = max(1, int(order))
        return (2 * p) ** 3

    @staticmethod
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

            children = torch.stack(
                [
                    torch.stack([a, ab, ca], dim=1),
                    torch.stack([ab, b, bc], dim=1),
                    torch.stack([ca, bc, c], dim=1),
                    torch.stack([ab, bc, ca], dim=1),
                ],
                dim=1,
            ).reshape(-1, 3, 3)

            if keep.numel() == 0:
                refined = children
            else:
                refined = torch.cat([keep, children], dim=0)

        return refined

    @staticmethod
    def integrate_structured_box_surface_force(
        bounds_min,
        dx,
        resolution,
        traction_func,
        active_cell_mask=None,
        order=2,
        poly_order=1,
        device="cpu"
    ):
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

        face_specs = [
            (0, 0, -1.0, 0.25 * float(dx_t[1] * dx_t[2]), (-1.0, 0.0, 0.0)),
            (0, nx - 1, 1.0, 0.25 * float(dx_t[1] * dx_t[2]), (1.0, 0.0, 0.0)),
            (1, 0, -1.0, 0.25 * float(dx_t[0] * dx_t[2]), (0.0, -1.0, 0.0)),
            (1, ny - 1, 1.0, 0.25 * float(dx_t[0] * dx_t[2]), (0.0, 1.0, 0.0)),
            (2, 0, -1.0, 0.25 * float(dx_t[0] * dx_t[1]), (0.0, 0.0, -1.0)),
            (2, nz - 1, 1.0, 0.25 * float(dx_t[0] * dx_t[1]), (0.0, 0.0, 1.0)),
        ]

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

            contrib = (
                N3.unsqueeze(0).unsqueeze(-1)
                * traction.unsqueeze(2)
                * (w_2d * jac_face).view(1, -1, 1, 1)
            ).sum(dim=1)

            F[torch.as_tensor(boundary_elems, dtype=torch.long, device=device)] += contrib

        return F

    # ==========================================================
    # SURFACE FORCE (ROBUST + CONSISTENT)
    # ==========================================================
    @staticmethod
    def integrate_surface_force(
        triangles,        # (T,3,3)
        bounds_min,       # (3,)
        dx,               # (3,)
        resolution,       # (nx,ny,nz)
        traction_func,
        active_cell_mask=None,
        order=2,
        poly_order=1,
        device="cpu"
    ):
        dtype = torch.float64

        triangles = torch.as_tensor(triangles, dtype=dtype, device=device)
        bounds_min = torch.as_tensor(bounds_min, dtype=dtype, device=device)
        dx = torch.as_tensor(dx, dtype=dtype, device=device)

        nx, ny, nz = resolution
        E = nx * ny * nz
        n_nodes = (int(poly_order) + 1) ** 3

        triangles = Force._refine_triangles_to_cell_size(triangles, dx)

        T = triangles.shape[0]

        # =====================================================
        # TRIANGLE GEOMETRY
        # =====================================================
        x1 = triangles[:, 0]
        x2 = triangles[:, 1]
        x3 = triangles[:, 2]

        e1 = x2 - x1
        e2 = x3 - x1

        cross = torch.cross(e1, e2, dim=-1)
        norm_cross = torch.norm(cross, dim=-1)

        eps = 1e-14
        valid_tri = norm_cross > eps

        # Surface mapping Jacobian from reference triangle coordinates.
        # Quadrature weights already sum to 0.5 on the reference triangle,
        # so the correct factor here is |e1 x e2| (not area).
        J_tri = norm_cross  # (T,)

        # normals (safe)
        n = torch.zeros_like(cross)
        n[valid_tri] = cross[valid_tri] / norm_cross[valid_tri, None]

        # =====================================================
        # GAUSS POINTS (REFERENCE TRIANGLE)
        # =====================================================
        gp, gw = Gauss.gauss_triangle(order, device=device, dtype=dtype)
        # Keep raw quadrature weights. They already integrate over the
        # reference triangle area and must not be normalized here.

        G = gp.shape[0]

        u = gp[:, 0]
        v = gp[:, 1]
        w_tri = 1.0 - u - v

        # =====================================================
        # MAP TO PHYSICAL TRIANGLE (BARYCENTRIC)
        # =====================================================
        x = (
            w_tri[None,:,None] * x1[:,None,:] +
            u[None,:,None]     * x2[:,None,:] +
            v[None,:,None]     * x3[:,None,:]
        )  # (T,G,3)

        n = n[:, None, :].expand(-1, G, -1)  # (T,G,3)

        # =====================================================
        # TRACTION
        # =====================================================
        t = traction_func(x, n)  # (T,G,3)

        # =====================================================
        # FIND HOST CELL (ROBUST)
        # =====================================================
        ijk = ((x - bounds_min) / dx).floor().long()

        ijk[..., 0] = torch.clamp(ijk[..., 0], 0, nx - 1)
        ijk[..., 1] = torch.clamp(ijk[..., 1], 0, ny - 1)
        ijk[..., 2] = torch.clamp(ijk[..., 2], 0, nz - 1)

        cell = cell_index(ijk, resolution)  # (T,G)

        valid_host = torch.ones_like(cell, dtype=torch.bool)
        if active_cell_mask is not None:
            active_mask_np = np.asarray(active_cell_mask, dtype=bool).reshape(-1)
            cell_np = cell.detach().cpu().numpy().astype(np.int64)
            ijk_np = ijk.detach().cpu().numpy().astype(np.int64)
            valid_np = np.ones(cell_np.shape, dtype=bool)

            offsets = [
                (-1, 0, 0), (1, 0, 0),
                (0, -1, 0), (0, 1, 0),
                (0, 0, -1), (0, 0, 1),
                (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
                (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
                (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
            ]

            for ti in range(cell_np.shape[0]):
                for gi in range(cell_np.shape[1]):
                    c = int(cell_np[ti, gi])
                    if active_mask_np[c]:
                        continue

                    i0, j0, k0 = ijk_np[ti, gi]
                    found = False
                    for di, dj, dk in offsets:
                        ii = int(np.clip(i0 + di, 0, nx - 1))
                        jj = int(np.clip(j0 + dj, 0, ny - 1))
                        kk = int(np.clip(k0 + dk, 0, nz - 1))
                        cc = ii * (ny * nz) + jj * nz + kk
                        if active_mask_np[cc]:
                            cell_np[ti, gi] = cc
                            found = True
                            break

                    if not found:
                        valid_np[ti, gi] = False

            cell = torch.as_tensor(cell_np, dtype=cell.dtype, device=device)
            valid_host = torch.as_tensor(valid_np, dtype=torch.bool, device=device)

            # Keep local coordinates consistent with the possibly remapped host
            # cell ids used for assembly.
            i_new = cell_np // (ny * nz)
            rem_new = cell_np % (ny * nz)
            j_new = rem_new // nz
            k_new = rem_new % nz
            ijk_np = np.stack([i_new, j_new, k_new], axis=-1)
            ijk = torch.as_tensor(ijk_np, dtype=ijk.dtype, device=device)

        # =====================================================
        # LOCAL HEX COORDS
        # =====================================================
        cell_min = bounds_min + ijk * dx

        xi = global_to_local(x, cell_min, dx)

        # clamp for stability
        xi = torch.clamp(xi, -1.0, 1.0)

        xi_flat = xi.reshape(-1, 3)

        N3 = Shape.N3(
            xi_flat[:, 0],
            xi_flat[:, 1],
            xi_flat[:, 2],
            poly_order=poly_order,
        ).reshape(T, G, n_nodes)

        # =====================================================
        # WEIGHTS
        # =====================================================
        w = gw.view(1, G) * J_tri[:, None]  # (T,G)

        # =====================================================
        # CONTRIBUTION
        # =====================================================
        contrib = (
            N3.unsqueeze(-1) *
            t.unsqueeze(2) *
            w.unsqueeze(-1).unsqueeze(-1)
        )

        valid_tri = valid_tri.to(contrib.dtype)
        contrib = contrib * valid_tri[:, None, None, None]
        contrib = contrib * valid_host[:, :, None, None].to(contrib.dtype)

        # =====================================================
        # ASSEMBLY
        # =====================================================
        F = torch.zeros((E, n_nodes, 3), dtype=dtype, device=device)

        F.index_add_(0, cell.reshape(-1), contrib.reshape(-1, n_nodes, 3))

        return F


    # ==========================================================
    # VOLUME FORCE (CLEAN + CONSISTENT)
    # ==========================================================
    @staticmethod
    def integrate_volume_force(
        hex_coords,
        body_force_func,
        order=2,
        poly_order=1,
        device="cpu",
        volume=None,
        inside_mask=None,
        partial_mask=None,
        cut_max_depth=3,
    ):
        dtype = torch.float64

        coords = torch.as_tensor(hex_coords, dtype=dtype, device=device)
        E = coords.shape[0]
        n_nodes = coords.shape[1]

        inside_idx = None
        partial_idx = None
        if inside_mask is not None and partial_mask is not None:
            inside_idx = torch.as_tensor(inside_mask, device=device).reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            partial_idx = torch.as_tensor(partial_mask, device=device).reshape(-1).nonzero(as_tuple=False).squeeze(-1)

        # =====================================================
        # GAUSS POINTS + SHAPE TABLES (HEX)
        # =====================================================
        _, gw, N3, dN_ref = Force._get_volume_rule(order, poly_order, device, dtype)

        G = N3.shape[0]

        x = torch.einsum('ga,eaq->egq', N3, coords)
        f = body_force_func(x)

        # =====================================================
        # JACOBIAN
        # =====================================================
        dN = dN_ref.unsqueeze(0).expand(E, -1, -1, -1)

        _, J = Shape.dN3_dx(dN, coords)

        detJ = torch.linalg.det(J)

        if torch.any(detJ <= 0):
            warnings.warn("negative or zero detJ detected", RuntimeWarning)

        # =====================================================
        # WEIGHTS
        # =====================================================
        w = gw.view(1, G) * detJ

        # =====================================================
        # INTEGRATION
        # =====================================================
        contrib = (
            N3.unsqueeze(0).unsqueeze(-1) *
            f.unsqueeze(2) *
            w.unsqueeze(-1).unsqueeze(-1)
        )

        F_vol = contrib.sum(dim=1)

        # If masks/volume are provided, replace partial-cell contributions with
        # cut-volume integration to avoid overloading tiny boundary cells.
        if volume is not None and inside_idx is not None and partial_idx is not None and partial_idx.numel() > 0:
            F_vol[~torch.isfinite(F_vol)] = 0.0
            cut_n_pts = Force._cut_points_from_order(order)

            pts_ref_list, w_list, _, _, _ = OctreeSampler.sample_batch(
                coords[partial_idx],
                volume,
                cut_n_pts,
                device=device,
                max_depth=cut_max_depth,
                poly_order=poly_order,
                compute_physical=False,
            )

            F_partial = torch.zeros((partial_idx.numel(), n_nodes, 3), dtype=dtype, device=device)

            for local_e in range(partial_idx.numel()):
                xi_ref = pts_ref_list[local_e]
                if xi_ref is None:
                    continue

                w_ref = w_list[local_e]
                e_coords = coords[partial_idx[local_e]].unsqueeze(0)

                N_cut = Shape.N3(
                    xi_ref[:, 0],
                    xi_ref[:, 1],
                    xi_ref[:, 2],
                    poly_order=poly_order,
                )
                x_cut = torch.einsum('ga,eaq->egq', N_cut, e_coords).squeeze(0)  # (G, 3)
                f_cut = body_force_func(x_cut)  # (G, 3)

                dN = Shape.dN3_dxi(
                    xi_ref[:, 0],
                    xi_ref[:, 1],
                    xi_ref[:, 2],
                    poly_order=poly_order,
                ).unsqueeze(0)
                _, J = Shape.dN3_dx(dN, e_coords)
                detJ = torch.linalg.det(J).squeeze(0)  # (G,)

                inside_cut = (volume.signed_distance_torch(x_cut) > 0).to(detJ.dtype)

                # w_ref is normalized to ref-volume fraction in [0,1], so
                # multiply by 8 for integration over [-1,1]^3.
                w_eff = (8.0 * w_ref) * detJ * inside_cut
                F_partial[local_e] = torch.einsum('ga,gq,g->aq', N_cut, f_cut, w_eff)

            F_vol[partial_idx] = F_partial

        return F_vol