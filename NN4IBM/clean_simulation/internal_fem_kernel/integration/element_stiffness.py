import torch
from tensor import _tensor
from kinematics import Kinematics
from shape import Shape
from integration.gauss_integration import Gauss
from integration.octreesampler import OctreeSampler


class Stiffness:

    @staticmethod
    def _effective_subcell_point_count(n_subcell_points, reduction_factor=1.0):
        """Return the execution-time point count used per accepted subcell.

        The quadrature is tensor-product Gauss-Legendre in 3D, so the final
        number of points must be a perfect cube (p^3).
        """
        n_req = int(n_subcell_points)
        if n_req < 1:
            raise ValueError("n_subcell_points must be >= 1")

        rf = float(reduction_factor)
        if rf <= 0.0:
            raise ValueError("subcell_point_reduction_factor must be > 0")

        n_target = max(1, int(round(n_req * rf)))
        order_eff = int(round(n_target ** (1.0 / 3.0)))
        order_eff = max(1, order_eff)

        # Ensure we never increase execution count when reducing.
        while (order_eff ** 3) > n_target and order_eff > 1:
            order_eff -= 1

        return order_eff ** 3, order_eff

    @staticmethod
    def _core(coords, C_t, gauss_pts, gauss_w, poly_order=1):

        xi, eta, zeta = gauss_pts[:, 0], gauss_pts[:, 1], gauss_pts[:, 2]

        dN3_ref  = Shape.dN3_dxi(xi, eta, zeta, poly_order=poly_order)
        dN3_phys, J = Shape.dN3_dx(dN3_ref, coords)
        detJ = torch.linalg.det(J)

        B = Kinematics.B_matrix(dN3_phys)

        w = gauss_w * detJ

        CB   = torch.einsum('ij,gjk->gik', C_t, B)
        BtCB = torch.einsum('gij,gik->gjk', B, CB)
        return torch.einsum('g,gjk->jk', w, BtCB)

    @staticmethod
    def elements(coords, C, order=2, device='cuda', poly_order=1):

        coords_t = _tensor(coords, device)
        C_t      = _tensor(C, device)

        pts, w = Gauss.gauss_3d(order, device)

        return Stiffness._core(coords_t, C_t, pts, w, poly_order=poly_order)
    
    @staticmethod
    def elements_cut(
        coords_batch,
        C,
        volume,
        device='cpu',
        n_pts=2,
        max_depth=4,
        order=1,
        n_subcell_points=None,
        subcell_point_reduction_factor=1.0,
        poly_order=1,
        octree_debug=False,
        octree_debug_prefix='',
    ):
        coords_batch = torch.as_tensor(coords_batch, dtype=torch.float64, device=device)
        C_t = _tensor(C, device).to(torch.float64)
        E = coords_batch.shape[0]
        n_nodes = coords_batch.shape[1]
        elem_dofs = 3 * n_nodes

        (pts_ref_list, w_list, h_ref_list, _, _) = OctreeSampler.sample_batch(
            coords_batch,
            volume,
            n_pts,
            device,
            max_depth=max_depth,
            poly_order=poly_order,
            compute_physical=False,
            debug=octree_debug,
            debug_prefix=octree_debug_prefix,
        )

        # --- resolve effective integration order (unchanged logic) -------------
        if n_subcell_points is not None:
            n_subcell_points = int(n_subcell_points)
            order_req = int(round(n_subcell_points ** (1.0 / 3.0)))
            if order_req ** 3 != n_subcell_points:
                raise ValueError(
                    "n_subcell_points must be a perfect cube for tensor-product "
                    "Gauss-Legendre (e.g. 1, 8, 27, 64, ...)."
                )
            _, order_eff = Stiffness._effective_subcell_point_count(
                n_subcell_points,
                reduction_factor=subcell_point_reduction_factor,
            )
        else:
            order_req = int(order)
            if order_req < 1:
                raise ValueError("order must be >= 1")
            n_req = order_req ** 3
            _, order_eff = Stiffness._effective_subcell_point_count(
                n_req,
                reduction_factor=subcell_point_reduction_factor,
            )

        # Gauss rule for order_eff > 1 — computed once, shared across all elements
        if order_eff > 1:
            pts_1d, w_1d = Gauss.gauss_1d(order_eff, device)
            xi_g = torch.stack(
                torch.meshgrid(pts_1d, pts_1d, pts_1d, indexing='ij'), dim=-1
            ).reshape(-1, 3)
            w_g = (
                w_1d[:, None, None] * w_1d[None, :, None] * w_1d[None, None, :]
            ).reshape(-1)
            w_g = w_g * (8.0 / w_g.sum())
        else:
            xi_g = None
            w_g  = None

        # -----------------------------------------------------------------------
        # Build per-element (xi_pts, w_pts) lists — O(E) Python but lightweight
        # -----------------------------------------------------------------------
        xi_list = []
        w_list_out = []

        for e in range(E):
            if pts_ref_list[e] is None:
                # No inside points → zero contribution; placeholder with 0 rows
                xi_list.append(torch.empty((0, 3), dtype=torch.float64, device=device))
                w_list_out.append(torch.empty((0,), dtype=torch.float64, device=device))
                continue

            subcell_ctr = pts_ref_list[e]   # (S, 3)
            w_ref       = w_list[e]         # (S,)
            h_ref_e     = h_ref_list[e]     # (S,)
            S = subcell_ctr.shape[0]

            if order_eff == 1:
                xi_list.append(subcell_ctr)
                w_list_out.append(8.0 * w_ref)
            else:
                P       = xi_g.shape[0]
                ctr_exp = subcell_ctr[:, None, :]        # (S, 1, 3)
                h_exp   = h_ref_e[:, None, None]         # (S, 1, 1)
                xi      = (ctr_exp + h_exp * xi_g[None]).reshape(S * P, 3)
                w       = (w_ref[:, None] * w_g[None] / 8.0).reshape(S * P) * 8.0
                xi_list.append(xi)
                w_list_out.append(w)

        # -----------------------------------------------------------------------
        # Pad all elements to N_pts_max — zero weights mask out padding in einsum
        # -----------------------------------------------------------------------
        N_pts_per_elem = [x.shape[0] for x in xi_list]
        N_pts_max = max(N_pts_per_elem) if N_pts_per_elem else 0

        if N_pts_max == 0:
            return (
                torch.zeros((E, elem_dofs, elem_dofs), dtype=torch.float64, device=device),
                None, None,
            )

        xi_padded = torch.zeros((E, N_pts_max, 3),    dtype=torch.float64, device=device)
        w_padded  = torch.zeros((E, N_pts_max),        dtype=torch.float64, device=device)

        for e, (xi, w) in enumerate(zip(xi_list, w_list_out)):
            n = xi.shape[0]
            if n > 0:
                xi_padded[e, :n] = xi
                w_padded[e, :n]  = w

        # -----------------------------------------------------------------------
        # Vectorised FEM kernel — single pass over all (E × N_pts_max) points
        # -----------------------------------------------------------------------
        # Flatten spatial dims: treat every (element, point) as one big batch
        # Shape: (E*N_pts_max, 3)
        xi_flat = xi_padded.reshape(E * N_pts_max, 3)

        # Reference-space shape function derivatives: (E*N_pts_max, n_nodes, 3)
        dN_ref = Shape.dN3_dxi(
            xi_flat[:, 0], xi_flat[:, 1], xi_flat[:, 2], poly_order=poly_order
        )
        if dN_ref.ndim == 3 and dN_ref.shape[0] == 1:
            dN_ref = dN_ref.squeeze(0)

        # Physical-space coords, one copy per integration point of each element
        # (E, 1, n_nodes, 3) → (E, N_pts_max, n_nodes, 3) → (E*N_pts_max, n_nodes, 3)
        coords_flat = (
            coords_batch[:, None, :, :]
            .expand(E, N_pts_max, n_nodes, 3)
            .reshape(E * N_pts_max, n_nodes, 3)
        )

        # Jacobian:  J[p,i,j] = sum_n dN_ref[p,n,i] * coords[p,n,j]
        # (E*P, n_nodes, 3) × (E*P, n_nodes, 3) → (E*P, 3, 3)
        J = torch.einsum('pni,pnj->pij', dN_ref, coords_flat)   # (E*P, 3, 3)

        detJ  = torch.linalg.det(J)                              # (E*P,)
        J_inv = torch.linalg.inv(J)                              # (E*P, 3, 3)

        # Physical shape-function derivatives:  dN_phys = dN_ref @ J^{-1}
        # (E*P, n_nodes, 3) @ (E*P, 3, 3) → (E*P, n_nodes, 3)
        dN_phys = torch.bmm(dN_ref, J_inv)

        # Strain-displacement matrix B: (E*P, 6, elem_dofs)
        B = Kinematics.B_matrix(dN_phys)
        if B.ndim == 4 and B.shape[0] == 1:
            B = B.squeeze(0)

        # Reshape back to element-batched form
        B     = B.reshape(E, N_pts_max, 6, elem_dofs)            # (E, P, 6, n_dofs)
        detJ  = detJ.reshape(E, N_pts_max)                       # (E, P)
        w_eff = w_padded * detJ                                   # (E, P)  — zeros at padding

        # Stiffness:  Ke = sum_p w_eff[p] * B[p]^T C B[p]
        # C_t: (6,6);  CB = C @ B: (E, P, 6, n_dofs)
        CB    = torch.matmul(C_t[None, None], B)                  # broadcast (1,1,6,6)
        BtCB  = torch.matmul(B.transpose(-2, -1), CB)             # (E, P, n_dofs, n_dofs)
        Ke_all = (w_eff[..., None, None] * BtCB).sum(dim=1)       # (E, n_dofs, n_dofs)

        return Ke_all, None, None

    @staticmethod
    def elements_from_queso_points_batched(
        coords_batch,
        C,
        xi_pts_list,
        w_phys_list,
        poly_order=1,
        device='cuda',
        element_batch_size=512,
    ):
        """Evaluate QuESo-point stiffness matrices in bounded vectorized batches."""
        if element_batch_size < 1:
            raise ValueError("element_batch_size must be positive.")

        device_t = torch.device(device)
        coords_all = torch.as_tensor(coords_batch, dtype=torch.float64, device=device_t)
        c_t = _tensor(C, device_t).to(torch.float64)
        n_elements, n_nodes, _ = coords_all.shape
        element_dofs = 3 * n_nodes
        result_chunks = []

        for start in range(0, n_elements, element_batch_size):
            stop = min(start + element_batch_size, n_elements)
            coords = coords_all[start:stop]
            xi_chunk = xi_pts_list[start:stop]
            weight_chunk = w_phys_list[start:stop]
            max_points = max((int(points.shape[0]) for points in xi_chunk), default=0)
            if max_points == 0:
                result_chunks.append(torch.zeros(
                    (stop - start, element_dofs, element_dofs),
                    dtype=torch.float64,
                    device=device_t,
                ))
                continue

            xi_padded = torch.zeros((stop - start, max_points, 3), dtype=torch.float64, device=device_t)
            weights_padded = torch.zeros((stop - start, max_points), dtype=torch.float64, device=device_t)
            for local_index, (xi, weights) in enumerate(zip(xi_chunk, weight_chunk)):
                point_count = int(xi.shape[0])
                if point_count:
                    xi_padded[local_index, :point_count] = xi.to(device=device_t, dtype=torch.float64)
                    weights_padded[local_index, :point_count] = weights.to(device=device_t, dtype=torch.float64)

            batch_size = stop - start
            xi_flat = xi_padded.reshape(batch_size * max_points, 3)
            d_n_ref = Shape.dN3_dxi(
                xi_flat[:, 0], xi_flat[:, 1], xi_flat[:, 2], poly_order=poly_order,
            )
            coords_flat = coords[:, None].expand(batch_size, max_points, n_nodes, 3).reshape(
                batch_size * max_points, n_nodes, 3,
            )
            jacobian = torch.einsum('pni,pnj->pij', d_n_ref, coords_flat)
            d_n_phys = torch.bmm(d_n_ref, torch.linalg.inv(jacobian))
            b_matrix = Kinematics.B_matrix(d_n_phys).reshape(
                batch_size, max_points, 6, element_dofs,
            )
            cb_matrix = torch.matmul(c_t[None, None], b_matrix)
            bt_cb = torch.matmul(b_matrix.transpose(-2, -1), cb_matrix)
            result_chunks.append((weights_padded[..., None, None] * bt_cb).sum(dim=1))

        return torch.cat(result_chunks, dim=0)

    @staticmethod
    def elements_from_packed_queso_points(
        coords_batch,
        C,
        xi_sorted,
        weights_sorted,
        point_counts,
        poly_order=1,
        device='cuda',
        element_batch_size=512,
    ):
        """Evaluate sorted QuESo points without host-side per-element packing."""
        if element_batch_size < 1:
            raise ValueError("element_batch_size must be positive.")

        device_t = torch.device(device)
        coords_all = torch.as_tensor(coords_batch, dtype=torch.float64, device=device_t)
        c_t = _tensor(C, device_t).to(torch.float64)
        n_elements, n_nodes, _ = coords_all.shape
        element_dofs = 3 * n_nodes
        result_chunks = []
        point_offsets = torch.cat((
            torch.zeros(1, dtype=torch.int64, device=device_t),
            torch.cumsum(point_counts, dim=0),
        ))

        for start in range(0, n_elements, element_batch_size):
            stop = min(start + element_batch_size, n_elements)
            counts = point_counts[start:stop]
            max_points = int(counts.max().item())
            batch_size = stop - start
            xi_padded = torch.zeros((batch_size, max_points, 3), dtype=torch.float64, device=device_t)
            weights_padded = torch.zeros((batch_size, max_points), dtype=torch.float64, device=device_t)
            source_start = point_offsets[start]
            source_stop = point_offsets[stop]
            element_ids = torch.repeat_interleave(torch.arange(batch_size, device=device_t), counts)
            local_offsets = torch.repeat_interleave(
                torch.cumsum(counts, dim=0) - counts,
                counts,
            )
            local_points = torch.arange(source_stop - source_start, device=device_t) - local_offsets
            xi_padded[element_ids, local_points] = xi_sorted[source_start:source_stop]
            weights_padded[element_ids, local_points] = weights_sorted[source_start:source_stop]

            xi_flat = xi_padded.reshape(batch_size * max_points, 3)
            d_n_ref = Shape.dN3_dxi(
                xi_flat[:, 0], xi_flat[:, 1], xi_flat[:, 2], poly_order=poly_order,
            )
            coords_flat = coords_all[start:stop, None].expand(batch_size, max_points, n_nodes, 3).reshape(
                batch_size * max_points, n_nodes, 3,
            )
            jacobian = torch.einsum('pni,pnj->pij', d_n_ref, coords_flat)
            d_n_phys = torch.bmm(d_n_ref, torch.linalg.inv(jacobian))
            b_matrix = Kinematics.B_matrix(d_n_phys).reshape(
                batch_size, max_points, 6, element_dofs,
            )
            cb_matrix = torch.matmul(c_t[None, None], b_matrix)
            bt_cb = torch.matmul(b_matrix.transpose(-2, -1), cb_matrix)
            result_chunks.append((weights_padded[..., None, None] * bt_cb).sum(dim=1))

        return torch.cat(result_chunks, dim=0)

    @staticmethod
    def elements_from_queso_points(coords_list, C, xi_pts_list, w_phys_list, poly_order=1, device='cuda'):
        """Compute element stiffness matrices directly from QuESo physical-space integration points.

        QuESo weights are physical (m^3) and already encode det(J), so:
            K_e = sum_g  w_phys_g * B(xi_g)^T C B(xi_g)
        with no additional det(J) factor.

        Args:
            coords_list: iterable of (n_nodes, 3) node coords per active element
            C:           (6,6) constitutive matrix
            xi_pts_list: iterable of (n_pts, 3) reference coords in [-1,1]^3 per element
            w_phys_list: iterable of (n_pts,) physical weights per element
        Returns:
            list of (n_dofs, n_dofs) torch tensors, one per element
        """
        C_t = _tensor(C, device).to(torch.float64)
        ke_list = []
        for coords, xi_pts, w_phys in zip(coords_list, xi_pts_list, w_phys_list):
            coords_t = torch.as_tensor(coords, dtype=torch.float64, device=device)
            xi_t  = torch.as_tensor(xi_pts,  dtype=torch.float64, device=device)  # (g, 3)
            w_t   = torch.as_tensor(w_phys,  dtype=torch.float64, device=device)  # (g,)

            dN3_ref  = Shape.dN3_dxi(xi_t[:, 0], xi_t[:, 1], xi_t[:, 2], poly_order=poly_order)
            dN3_phys, _ = Shape.dN3_dx(dN3_ref, coords_t)
            B = Kinematics.B_matrix(dN3_phys)
            CB   = torch.einsum('ij,gjk->gik', C_t, B)
            BtCB = torch.einsum('gij,gik->gjk', B, CB)
            ke_list.append(torch.einsum('g,gjk->jk', w_t, BtCB))

        if len(ke_list) == 0:
            ndof = 3 * (poly_order + 1) ** 3
            return torch.empty((0, ndof, ndof), dtype=torch.float64, device=device)

        return torch.stack(ke_list, dim=0)