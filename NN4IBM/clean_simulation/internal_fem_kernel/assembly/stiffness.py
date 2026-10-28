import time
import torch

from core.grid_indexing import active_element_nodes, structured_total_nodes
from geometry import HexGrid
from integration.element_stiffness import Stiffness
from assembly.global_system import GlobalAssembly
from assembly.nitsche import NitscheAssembly


class StiffnessAssembly:
    _inside_ref_ke_cache = {}
    _sparse_pattern_cache = {}
    USE_INSIDE_REF_CACHE = False
    USE_SPARSE_PATTERN_CACHE = False

    @staticmethod
    def _inside_ref_cache_key(grid, material, order, poly_order, device):
        c_t = torch.as_tensor(material.C, dtype=torch.float64, device=device)
        return (
            float(grid.dx),
            float(grid.dy),
            float(grid.dz),
            int(order),
            int(poly_order),
            str(device),
            tuple(c_t.shape),
            hash(c_t.detach().cpu().numpy().tobytes()),
        )

    @staticmethod
    def _cell_nodes_batched(centers_t, dx, dy, dz, poly_order=1):
        device = centers_t.device
        offsets = HexGrid._local_node_offsets(
            dx,
            dy,
            dz,
            poly_order=poly_order,
            device=device,
        )
        return centers_t[:, None, :] + offsets[None, :, :]

    @staticmethod
    def compute_element_matrices_from_queso_points(
        grid,
        material,
        ips,
        poly_order=1,
        device="cuda",
        element_batch_size=512,
    ):
        dtype = torch.float64

        bounds_min = torch.as_tensor(
            grid.bounds[0],
            dtype=dtype,
            device=device,
        )
        bounds_max = torch.as_tensor(
            grid.bounds[1],
            dtype=dtype,
            device=device,
        )

        dx_arr = torch.tensor(
            [grid.dx, grid.dy, grid.dz],
            dtype=dtype,
            device=device,
        )

        nx, ny, nz = grid.resolution

        ips = torch.as_tensor(
            ips,
            dtype=dtype,
            device=device,
        )

        ip_coords = ips[:, :3]
        ip_w = ips[:, 3]

        in_unit_box = bool(torch.all((ip_coords >= -1.0e-12) & (ip_coords <= 1.0 + 1.0e-12)).item())
        bounds_are_unit = bool(torch.all((bounds_min >= -1.0e-12) & (bounds_max <= 1.0 + 1.0e-12)).item())
        if in_unit_box and not bounds_are_unit:
            ip_xyz = bounds_min + ip_coords * (bounds_max - bounds_min)
            # QuESo can provide integration points in UVW [0,1]^3 with UVW-domain
            # weights. After mapping to physical XYZ we must scale by det(J).
            jac_det = torch.prod(bounds_max - bounds_min)
            ip_w = ip_w * jac_det
        else:
            ip_xyz = ip_coords

        ijk = torch.floor(
            (ip_xyz - bounds_min) / dx_arr
        ).to(torch.int64)

        ijk[:, 0] = torch.clamp(ijk[:, 0], 0, nx - 1)
        ijk[:, 1] = torch.clamp(ijk[:, 1], 0, ny - 1)
        ijk[:, 2] = torch.clamp(ijk[:, 2], 0, nz - 1)

        flat = (
            ijk[:, 0] * (ny * nz)
            + ijk[:, 1] * nz
            + ijk[:, 2]
        )

        cell_min_all = bounds_min + ijk * dx_arr

        xi_all = torch.clamp(
            2.0 * (ip_xyz - cell_min_all) / dx_arr - 1.0,
            -1.0,
            1.0,
        )

        if torch.device(device).type == "cuda":
            sort_idx = torch.argsort(flat)
            flat_s = flat[sort_idx]
            xi_s = xi_all[sort_idx]
            w_s = ip_w[sort_idx]
            active_elements, point_counts = torch.unique_consecutive(flat_s, return_counts=True)

            elem_i = active_elements // (ny * nz)
            elem_rem = active_elements % (ny * nz)
            elem_j = elem_rem // nz
            elem_k = elem_rem % nz
            centers = bounds_min + (
                torch.stack([elem_i, elem_j, elem_k], dim=1).to(dtype=dtype) + 0.5
            ) * dx_arr
            coords_all = StiffnessAssembly._cell_nodes_batched(
                centers,
                grid.dx,
                grid.dy,
                grid.dz,
                poly_order=poly_order,
            )
            ke_list = Stiffness.elements_from_packed_queso_points(
                coords_all,
                material.C,
                xi_s,
                w_s,
                point_counts,
                poly_order=poly_order,
                device=device,
                element_batch_size=element_batch_size,
            )
            return ke_list, active_elements.cpu()

        # Group point ranges on CPU. Slicing GPU tensors once per element forces
        # a synchronization per active element, overwhelming the batched kernel.
        flat_cpu = flat.cpu()
        xi_cpu = xi_all.cpu()
        weight_cpu = ip_w.cpu()
        active_elements = torch.unique(flat_cpu)

        sort_idx = torch.argsort(flat_cpu)

        flat_s = flat_cpu[sort_idx]
        xi_s = xi_cpu[sort_idx]
        w_s = weight_cpu[sort_idx]

        starts = torch.searchsorted(
            flat_s,
            active_elements,
            side="left",
        )

        ends = torch.searchsorted(
            flat_s,
            active_elements,
            side="right",
        )

        elem_i = active_elements // (ny * nz)

        elem_rem = active_elements % (ny * nz)

        elem_j = elem_rem // nz

        elem_k = elem_rem % nz

        centers = bounds_min + (
            torch.stack(
                [elem_i, elem_j, elem_k],
                dim=1,
            ).to(device=device, dtype=dtype) + 0.5
        ) * dx_arr

        coords_all = StiffnessAssembly._cell_nodes_batched(
            centers,
            grid.dx,
            grid.dy,
            grid.dz,
            poly_order=poly_order,
        )

        xi_pts_list = []
        w_phys_list = []

        for i in range(len(active_elements)):
            xi_pts_list.append(
                xi_s[starts[i]:ends[i]]
            )

            w_phys_list.append(
                w_s[starts[i]:ends[i]]
            )

        ke_list = Stiffness.elements_from_queso_points_batched(
            coords_all,
            material.C,
            xi_pts_list,
            w_phys_list,
            poly_order=poly_order,
            device=device,
            element_batch_size=element_batch_size,
        )

        return ke_list, active_elements

    @staticmethod
    def compute_element_matrices_from_queso_points_fast(
        grid,
        material,
        ips,
        trimmed_ip_mask,
        poly_order=1,
        order=2,
        device="cuda",
        element_batch_size=512,
    ):
        """Assemble QuESo local stiffness without reintegrating full cells.

        A uniform full grid cell has the same Gauss-integrated stiffness at
        every location. QuESo points remain the authoritative quadrature for
        trimmed cells, while the identical analytic full-cell matrix is
        broadcast to all interior cells.
        """
        ips_cpu = torch.as_tensor(ips, dtype=torch.float64, device="cpu")
        trimmed_mask = torch.as_tensor(trimmed_ip_mask, dtype=torch.bool, device="cpu").reshape(-1)
        if trimmed_mask.numel() != ips_cpu.shape[0]:
            raise ValueError("trimmed_ip_mask must have one value per integration point.")

        nx, ny, nz = grid.resolution
        bounds_min = torch.as_tensor(grid.bounds[0], dtype=torch.float64)
        bounds_max = torch.as_tensor(grid.bounds[1], dtype=torch.float64)
        spacing = torch.tensor([grid.dx, grid.dy, grid.dz], dtype=torch.float64)
        coords = ips_cpu[:, :3]
        if bool(torch.all((coords >= -1.0e-12) & (coords <= 1.0 + 1.0e-12))) and not bool(torch.all((bounds_min >= -1.0e-12) & (bounds_max <= 1.0 + 1.0e-12))):
            coords = bounds_min + coords * (bounds_max - bounds_min)
        ijk = torch.floor((coords - bounds_min) / spacing).to(torch.int64)
        ijk[:, 0].clamp_(0, nx - 1)
        ijk[:, 1].clamp_(0, ny - 1)
        ijk[:, 2].clamp_(0, nz - 1)
        flat = ijk[:, 0] * (ny * nz) + ijk[:, 1] * nz + ijk[:, 2]

        inside_mask = torch.zeros(nx * ny * nz, dtype=torch.bool)
        inside_mask[torch.unique(flat[~trimmed_mask])] = True
        ke_inside, idx_inside = StiffnessAssembly.compute_inside_element_matrices(
            grid=grid,
            material=material,
            inside=inside_mask,
            order=order,
            device="cpu",
            poly_order=poly_order,
        )

        if bool(trimmed_mask.any()):
            ke_trimmed, idx_trimmed = StiffnessAssembly.compute_element_matrices_from_queso_points(
                grid=grid,
                material=material,
                ips=ips_cpu[trimmed_mask],
                poly_order=poly_order,
                device=device,
                element_batch_size=element_batch_size,
            )
            ke_trimmed = ke_trimmed.cpu()
            idx_trimmed = idx_trimmed.cpu()
            ke_all = torch.cat((ke_inside, ke_trimmed), dim=0)
            idx_all = torch.cat((idx_inside, idx_trimmed), dim=0)
        else:
            ke_all, idx_all = ke_inside, idx_inside

        order_idx = torch.argsort(idx_all)
        return ke_all[order_idx], idx_all[order_idx]

    @staticmethod
    def _cut_points_from_order(order):
        p = max(1, int(order))
        return p ** 3

    @staticmethod
    def _normalize_cell_mask(mask, n_cells, device):
        if mask is None:
            return torch.zeros(n_cells, dtype=torch.bool, device=device)

        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).reshape(-1)
        if mask_t.numel() == 0:
            return torch.zeros(n_cells, dtype=torch.bool, device=device)

        if mask_t.numel() != n_cells:
            raise ValueError(
                f"Cell mask size mismatch: expected {n_cells}, got {mask_t.numel()}."
            )

        return mask_t

    @staticmethod
    def _filter_by_ke_norm(
        ke_array,
        active_indices_all,
        *,
        ke_tol,
        ke_rel_tol,
        device,
        dtype,
    ):
        if active_indices_all.numel() == 0:
            empty_ke = torch.empty(
                (0,) + tuple(ke_array.shape[1:]),
                dtype=dtype,
                device=device,
            )
            empty_idx = torch.empty(
                (0,),
                dtype=torch.int64,
                device=device,
            )
            return empty_ke, empty_idx

        ke_norm = torch.linalg.norm(
            ke_array.reshape(active_indices_all.numel(), -1),
            dim=1,
        )

        finite_pos = (
            torch.isfinite(ke_norm)
            & (ke_norm > 0.0)
        )

        ref_norm = (
            torch.median(ke_norm[finite_pos])
            if torch.any(finite_pos)
            else torch.tensor(
                0.0,
                dtype=dtype,
                device=device,
            )
        )

        ke_floor = max(
            float(max(0.0, ke_tol)),
            float(max(0.0, ke_rel_tol)) * float(ref_norm),
        )

        effective_mask = (
            torch.isfinite(ke_norm)
            & (ke_norm > ke_floor)
        )

        return ke_array[effective_mask], active_indices_all[effective_mask]

    @staticmethod
    def compute_inside_element_matrices(
        grid,
        material,
        inside,
        order=2,
        device="cuda",
        poly_order=1,
        ke_tol=0.0,
        ke_rel_tol=0.0,
        return_diagnostics=False,
    ):
        t_total_ns = time.perf_counter_ns()
        diagnostics = {
            "inside_count": 0,
            "inside_ref_cache_miss": 0,
            "inside_ref_compute": 0.0,
            "inside_broadcast": 0.0,
            "norm_filter": 0.0,
            "total": 0.0,
        }

        dtype = torch.float64
        n_cells = int(grid.shape[0] * grid.shape[1] * grid.shape[2])
        inside_f = StiffnessAssembly._normalize_cell_mask(inside, n_cells, device)
        inside_idx = torch.where(inside_f)[0]
        diagnostics["inside_count"] = int(inside_idx.numel())

        n_elem_dofs = 3 * ((poly_order + 1) ** 3)
        if inside_idx.numel() == 0:
            empty_ke = torch.empty((0, n_elem_dofs, n_elem_dofs), dtype=dtype, device=device)
            empty_idx = torch.empty((0,), dtype=torch.int64, device=device)
            diagnostics["total"] = (time.perf_counter_ns() - t_total_ns) * 1e-9
            if return_diagnostics:
                return empty_ke, empty_idx, diagnostics
            return empty_ke, empty_idx

        centers = grid.centers.reshape(-1, 3).to(device=device, dtype=dtype)

        cache_key = StiffnessAssembly._inside_ref_cache_key(
            grid,
            material,
            order,
            poly_order,
            device,
        )

        if (
            (not StiffnessAssembly.USE_INSIDE_REF_CACHE)
            or (cache_key not in StiffnessAssembly._inside_ref_ke_cache)
        ):
            t_ref_ns = time.perf_counter_ns()
            ref_center = centers[inside_idx[0]]
            ref_coords = HexGrid.cell_nodes_from_centers(
                ref_center,
                grid.dx,
                grid.dy,
                grid.dz,
                poly_order=poly_order,
            )
            k_inside_ref = Stiffness.elements(
                ref_coords,
                material.C,
                order=order,
                device=device,
                poly_order=poly_order,
            )

            if StiffnessAssembly.USE_INSIDE_REF_CACHE:
                StiffnessAssembly._inside_ref_ke_cache[cache_key] = k_inside_ref

            diagnostics["inside_ref_cache_miss"] = 1
            diagnostics["inside_ref_compute"] = (time.perf_counter_ns() - t_ref_ns) * 1e-9

        if StiffnessAssembly.USE_INSIDE_REF_CACHE:
            k_inside_ref = StiffnessAssembly._inside_ref_ke_cache[cache_key]

        t_bc_ns = time.perf_counter_ns()
        ke_inside = k_inside_ref.expand(inside_idx.numel(), -1, -1).clone()
        diagnostics["inside_broadcast"] = (time.perf_counter_ns() - t_bc_ns) * 1e-9

        t_filter_ns = time.perf_counter_ns()
        filtered_ke, filtered_idx = StiffnessAssembly._filter_by_ke_norm(
            ke_inside,
            inside_idx,
            ke_tol=ke_tol,
            ke_rel_tol=ke_rel_tol,
            device=device,
            dtype=dtype,
        )
        diagnostics["norm_filter"] = (time.perf_counter_ns() - t_filter_ns) * 1e-9
        diagnostics["total"] = (time.perf_counter_ns() - t_total_ns) * 1e-9

        if return_diagnostics:
            return filtered_ke, filtered_idx, diagnostics

        return filtered_ke, filtered_idx

    @staticmethod
    def compute_trimmed_element_matrices(
        grid,
        volume,
        material,
        partial,
        order=2,
        device="cuda",
        batch_size=4096,
        cut_max_depth=4,
        subcell_point_reduction_factor=1.0,
        poly_order=1,
        ke_tol=0.0,
        ke_rel_tol=0.0,
        return_diagnostics=False,
    ):
        t_total_ns = time.perf_counter_ns()
        diagnostics = {
            "partial_count": 0,
            "partial_coords_build": 0.0,
            "partial_compute": 0.0,
            "norm_filter": 0.0,
            "total": 0.0,
        }

        dtype = torch.float64
        n_cells = int(grid.shape[0] * grid.shape[1] * grid.shape[2])

        # --- fast-path: skip normalize+where if caller already gave us a 1-D index tensor ---
        if (
            isinstance(partial, torch.Tensor)
            and partial.ndim == 1
            and partial.dtype in (torch.int32, torch.int64)
            and partial.numel() < n_cells        # heuristic: looks like indices not a mask
        ):
            partial_idx = partial.to(device=device, dtype=torch.int64)
        else:
            partial_f = StiffnessAssembly._normalize_cell_mask(partial, n_cells, device)
            partial_idx = torch.where(partial_f)[0]

        diagnostics["partial_count"] = int(partial_idx.numel())

        n_elem_dofs = 3 * ((poly_order + 1) ** 3)
        if partial_idx.numel() == 0:
            empty_ke  = torch.empty((0, n_elem_dofs, n_elem_dofs), dtype=dtype, device=device)
            empty_idx = torch.empty((0,), dtype=torch.int64, device=device)
            diagnostics["total"] = (time.perf_counter_ns() - t_total_ns) * 1e-9
            if return_diagnostics:
                return empty_ke, empty_idx, diagnostics
            return empty_ke, empty_idx

        # centers kept on GPU; no full coord tensor built upfront
        centers = grid.centers.reshape(-1, 3).to(device=device, dtype=dtype)

        n_partial   = partial_idx.numel()
        cut_n_pts   = StiffnessAssembly._cut_points_from_order(order)
        material_C  = material.C   # avoid attribute lookup in loop

        # pre-allocate output for full set (one malloc beats n_batches mallocs)
        ke_partial = torch.empty(
            (n_partial, n_elem_dofs, n_elem_dofs),
            dtype=dtype,
            device=device,
        )

        # --- two CUDA streams: compute stream + prefetch stream ----------------
        use_streams = (device != "cpu") and torch.cuda.is_available()
        if use_streams:
            compute_stream = torch.cuda.Stream(device=device)
            prefetch_stream = torch.cuda.Stream(device=device)
        else:
            compute_stream = prefetch_stream = None

        def _build_coords_for_batch(idx_slice):
            """Build element-node coords for a slice of partial_idx."""
            batch_centers = centers[partial_idx[idx_slice]]
            return StiffnessAssembly._cell_nodes_batched(
                batch_centers,
                grid.dx,
                grid.dy,
                grid.dz,
                poly_order=poly_order,
            )

        t_part_ns = time.perf_counter_ns()

        if use_streams:
            # Prefetch coords for first batch on the prefetch stream
            first_slice = slice(0, min(batch_size, n_partial))
            with torch.cuda.stream(prefetch_stream):
                next_coords = _build_coords_for_batch(first_slice)

            for start in range(0, n_partial, batch_size):
                stop = min(start + batch_size, n_partial)
                cur_slice  = slice(start, stop)

                # Coords for this batch are ready from the previous prefetch
                prefetch_stream.synchronize()
                cur_coords = next_coords

                # Kick off prefetch of NEXT batch coords while this batch computes
                next_start = stop
                if next_start < n_partial:
                    next_stop  = min(next_start + batch_size, n_partial)
                    next_slice = slice(next_start, next_stop)
                    with torch.cuda.stream(prefetch_stream):
                        next_coords = _build_coords_for_batch(next_slice)

                # Compute on the compute stream
                with torch.cuda.stream(compute_stream):
                    k_batch, _, _ = Stiffness.elements_cut(
                        cur_coords,
                        material_C,
                        volume,
                        device=device,
                        n_pts=cut_n_pts,
                        max_depth=cut_max_depth,
                        order=order,
                        subcell_point_reduction_factor=subcell_point_reduction_factor,
                        poly_order=poly_order,
                    )
                    ke_partial[cur_slice] = k_batch

            compute_stream.synchronize()

        else:
            # CPU / no-stream fallback — still builds coords lazily per batch
            t_pc_ns = time.perf_counter_ns()
            diagnostics["partial_coords_build"] = 0.0   # measured inline below

            for start in range(0, n_partial, batch_size):
                stop = min(start + batch_size, n_partial)

                t_pc_batch = time.perf_counter_ns()
                cur_coords = _build_coords_for_batch(slice(start, stop))
                diagnostics["partial_coords_build"] += (
                    (time.perf_counter_ns() - t_pc_batch) * 1e-9
                )

                k_batch, _, _ = Stiffness.elements_cut(
                    cur_coords,
                    material_C,
                    volume,
                    device=device,
                    n_pts=cut_n_pts,
                    max_depth=cut_max_depth,
                    order=order,
                    subcell_point_reduction_factor=subcell_point_reduction_factor,
                    poly_order=poly_order,
                )
                ke_partial[start:stop] = k_batch

        diagnostics["partial_compute"] = (time.perf_counter_ns() - t_part_ns) * 1e-9

        t_filter_ns = time.perf_counter_ns()
        filtered_ke, filtered_idx = StiffnessAssembly._filter_by_ke_norm(
            ke_partial,
            partial_idx,
            ke_tol=ke_tol,
            ke_rel_tol=ke_rel_tol,
            device=device,
            dtype=dtype,
        )
        diagnostics["norm_filter"] = (time.perf_counter_ns() - t_filter_ns) * 1e-9
        diagnostics["total"] = (time.perf_counter_ns() - t_total_ns) * 1e-9

        if return_diagnostics:
            return filtered_ke, filtered_idx, diagnostics
        return filtered_ke, filtered_idx

    @staticmethod
    def compute_element_matrices(
        grid,
        volume,
        material,
        inside,
        partial,
        order=2,
        device="cuda",
        batch_size=4096,
        cut_max_depth=4,
        subcell_point_reduction_factor=1.0,
        poly_order=1,
        ke_tol=0.0,
        ke_rel_tol=0.0,
        return_diagnostics=False,
    ):
        t_total_ns = time.perf_counter_ns()

        diagnostics = {
            "inside_count": 0,
            "partial_count": 0,
            "inside_ref_cache_miss": 0,
            "inside_ref_compute": 0.0,
            "inside_broadcast": 0.0,
            "partial_coords_build": 0.0,
            "partial_compute": 0.0,
            "norm_filter": 0.0,
            "total": 0.0,
        }

        n_cells = int(grid.shape[0] * grid.shape[1] * grid.shape[2])
        inside_f = StiffnessAssembly._normalize_cell_mask(inside, n_cells, device)
        partial_f = StiffnessAssembly._normalize_cell_mask(partial, n_cells, device)

        ke_inside, active_inside, inside_diag = StiffnessAssembly.compute_inside_element_matrices(
            grid=grid,
            material=material,
            inside=inside_f,
            order=order,
            device=device,
            poly_order=poly_order,
            ke_tol=0.0,
            ke_rel_tol=0.0,
            return_diagnostics=True,
        )

        ke_partial, active_partial, partial_diag = StiffnessAssembly.compute_trimmed_element_matrices(
            grid=grid,
            volume=volume,
            material=material,
            partial=partial_f,
            order=order,
            device=device,
            batch_size=batch_size,
            cut_max_depth=cut_max_depth,
            subcell_point_reduction_factor=subcell_point_reduction_factor,
            poly_order=poly_order,
            ke_tol=0.0,
            ke_rel_tol=0.0,
            return_diagnostics=True,
        )

        diagnostics["inside_count"] = int(inside_diag.get("inside_count", 0))
        diagnostics["partial_count"] = int(partial_diag.get("partial_count", 0))
        diagnostics["inside_ref_cache_miss"] = int(inside_diag.get("inside_ref_cache_miss", 0))
        diagnostics["inside_ref_compute"] = float(inside_diag.get("inside_ref_compute", 0.0))
        diagnostics["inside_broadcast"] = float(inside_diag.get("inside_broadcast", 0.0))
        diagnostics["partial_coords_build"] = float(partial_diag.get("partial_coords_build", 0.0))
        diagnostics["partial_compute"] = float(partial_diag.get("partial_compute", 0.0))

        if active_inside.numel() == 0 and active_partial.numel() == 0:
            diagnostics["norm_filter"] = 0.0
            diagnostics["total"] = (time.perf_counter_ns() - t_total_ns) * 1e-9
            n_elem_dofs = 3 * ((poly_order + 1) ** 3)
            empty_ke = torch.empty((0, n_elem_dofs, n_elem_dofs), dtype=torch.float64, device=device)
            empty_idx = torch.empty((0,), dtype=torch.int64, device=device)
            if return_diagnostics:
                return empty_ke, empty_idx, diagnostics
            return empty_ke, empty_idx

        if active_inside.numel() == 0:
            ke_array = ke_partial
            active_indices_all = active_partial
        elif active_partial.numel() == 0:
            ke_array = ke_inside
            active_indices_all = active_inside
        else:
            ke_array = torch.cat([ke_inside, ke_partial], dim=0)
            active_indices_all = torch.cat([active_inside, active_partial], dim=0)

        t_filter_ns = time.perf_counter_ns()
        filtered_ke, active_indices = StiffnessAssembly._filter_by_ke_norm(
            ke_array,
            active_indices_all,
            ke_tol=ke_tol,
            ke_rel_tol=ke_rel_tol,
            device=device,
            dtype=torch.float64,
        )

        diagnostics["norm_filter"] = (time.perf_counter_ns() - t_filter_ns) * 1e-9
        diagnostics["total"] = (time.perf_counter_ns() - t_total_ns) * 1e-9

        if return_diagnostics:
            return (
                filtered_ke,
                active_indices,
                diagnostics,
            )

        return filtered_ke, active_indices

    @staticmethod
    def assemble_global_stiffness(
        ke_list,
        active_indices,
        grid,
        device="cuda",
        poly_order=1,
    ):
        dtype = torch.float64

        n_nodes = structured_total_nodes(grid.resolution, poly_order=poly_order)
        ndof = 3 * n_nodes

        if len(ke_list) == 0:
            return torch.zeros((ndof, ndof), dtype=dtype, device=device)

        nodes_per_elem = active_element_nodes(active_indices, grid.resolution, poly_order=poly_order, device=device)
        n_nodes_per_elem = nodes_per_elem.shape[1]
        n_elem_dofs = 3 * n_nodes_per_elem
        dof_offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
        elem_dofs = ( nodes_per_elem[:, :, None] * 3 + dof_offsets[None, None, :]).reshape(-1, n_elem_dofs)

        ke_array = torch.as_tensor(ke_list, dtype=dtype, device=device).reshape(-1, n_elem_dofs, n_elem_dofs)

        rows = torch.repeat_interleave(
            elem_dofs,
            n_elem_dofs,
            dim=1,
        ).reshape(-1)

        cols = torch.tile(
            elem_dofs,
            (1, n_elem_dofs),
        ).reshape(-1)

        vals = ke_array.reshape(-1)

        k = torch.zeros(
            (ndof, ndof),
            dtype=dtype,
            device=device,
        )

        k.index_put_(
            (rows, cols),
            vals,
            accumulate=True,
        )

        return k

    @staticmethod
    def assemble_global_stiffness_sparse(
        ke_list,
        active_indices,
        grid,
        device="cuda",
        poly_order=1,
        sparse_backend="torch",
    ):
        dtype = torch.float64
        device_t = torch.device(device)
        if sparse_backend not in {"torch", "scipy"}:
            raise ValueError(f"Unsupported sparse backend: {sparse_backend!r}.")
        if sparse_backend == "scipy" and device_t.type != "cpu":
            raise ValueError("The SciPy sparse backend requires device='cpu'.")

        n_nodes = structured_total_nodes(
            grid.resolution,
            poly_order=poly_order,
        )

        ndof = 3 * n_nodes

        if len(ke_list) == 0:
            indices = torch.empty(
                (2, 0),
                dtype=torch.int64,
                device=device_t,
            )

            values = torch.empty(
                (0,),
                dtype=dtype,
                device=device_t,
            )

            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    indices,
                    values,
                    (ndof, ndof),
                    device=device_t,
                    dtype=dtype,
                )

        active_idx_cpu = torch.as_tensor(
            active_indices,
            dtype=torch.int64,
            device="cpu",
        ).contiguous()

        cache_key = (
            tuple(grid.resolution),
            int(poly_order),
            str(device_t),
            int(active_idx_cpu.numel()),
            hash(active_idx_cpu.numpy().tobytes()),
        )

        cached = None
        if StiffnessAssembly.USE_SPARSE_PATTERN_CACHE:
            cached = StiffnessAssembly._sparse_pattern_cache.get(cache_key)

        if cached is None:
            nodes_per_elem = active_element_nodes(
                active_idx_cpu,
                grid.resolution,
                poly_order=poly_order,
                device=device_t,
            )

            n_nodes_per_elem = (
                nodes_per_elem.shape[1]
            )

            n_elem_dofs = 3 * n_nodes_per_elem

            dof_offsets = torch.tensor(
                [0, 1, 2],
                dtype=torch.int64,
                device=device_t,
            )

            elem_dofs = (
                nodes_per_elem[:, :, None] * 3
                + dof_offsets[None, None, :]
            ).reshape(-1, n_elem_dofs)

            local_dof_idx = torch.arange(
                n_elem_dofs,
                dtype=torch.int64,
                device=device_t,
            )

            local_rows = torch.repeat_interleave(
                local_dof_idx,
                n_elem_dofs,
            )

            local_cols = torch.tile(
                local_dof_idx,
                (n_elem_dofs,),
            )

            rows = elem_dofs[:, local_rows].reshape(-1)
            cols = elem_dofs[:, local_cols].reshape(-1)

            cached = {
                "rows": rows,
                "cols": cols,
                "n_elem_dofs": n_elem_dofs,
            }
            if StiffnessAssembly.USE_SPARSE_PATTERN_CACHE:
                StiffnessAssembly._sparse_pattern_cache[cache_key] = cached

        rows = cached["rows"]
        cols = cached["cols"]
        n_elem_dofs = int(cached["n_elem_dofs"])

        if isinstance(ke_list, torch.Tensor):
            ke_array = ke_list.to(device=device_t, dtype=dtype)

        elif isinstance(ke_list, (list, tuple)):
            first = ke_list[0]
            if isinstance(first, torch.Tensor):
                same_device = all(
                    isinstance(k, torch.Tensor) and k.device == first.device
                    for k in ke_list
                )
                if same_device:
                    ke_array = torch.stack(ke_list, dim=0).to(device=device_t, dtype=dtype)
                else:
                    ke_array = torch.stack(
                        [
                            k.to(device=device_t, dtype=dtype)
                            if isinstance(k, torch.Tensor)
                            else torch.as_tensor(k, dtype=dtype, device=device_t)
                            for k in ke_list
                        ],
                        dim=0,
                    )
            else:
                ke_array = torch.stack(
                    [
                        torch.as_tensor(k, dtype=dtype, device=device_t)
                        for k in ke_list
                    ],
                    dim=0,
                )

        else:
            ke_array = torch.as_tensor(
                ke_list,
                dtype=dtype,
                device=device_t,
            )

        ke_array = ke_array.reshape(
            -1,
            n_elem_dofs,
            n_elem_dofs,
        )

        vals = ke_array.reshape(-1)

        if sparse_backend == "scipy":
            import scipy.sparse as sp

            # CSR construction reduces duplicate element contributions in SciPy's
            # compiled sparse backend without materializing an intermediate Torch COO.
            return sp.coo_matrix(
                (vals.numpy(), (rows.numpy(), cols.numpy())),
                shape=(ndof, ndof),
            ).tocsr()

        indices = torch.stack([rows, cols], dim=0)

        with torch.sparse.check_sparse_tensor_invariants(False):
            # Keep uncoalesced here for speed; coalesce happens once in downstream global assembly.
            return torch.sparse_coo_tensor(
                indices,
                vals,
                (ndof, ndof),
                dtype=dtype,
                device=device_t,
            )

    @staticmethod
    def assemble_global_system(
        ke_list,
        active_indices,
        grid,
        material=None,
        f_global=None,
        nitsche_bcs=None,
        device="cuda",
        sparse=False,
        poly_order=1,
    ):
        if sparse:
            k = (
                StiffnessAssembly.assemble_global_stiffness_sparse(
                    ke_list,
                    active_indices,
                    grid,
                    device=device,
                    poly_order=poly_order,
                )
            )
        else:
            k = (
                StiffnessAssembly.assemble_global_stiffness(
                    ke_list,
                    active_indices,
                    grid,
                    device=device,
                    poly_order=poly_order,
                )
            )

        f = GlobalAssembly.initialize_force_vector(
            grid=grid,
            f_global=f_global,
            device=device,
            sparse=sparse,
            poly_order=poly_order,
        )

        if not nitsche_bcs:
            return k, f, []

        if material is None:
            raise ValueError(
                "material is required when nitsche_bcs is provided."
            )

        nitsche_sets = NitscheAssembly.compute_terms(
            grid=grid,
            material=material,
            active_indices=active_indices,
            nitsche_bcs=nitsche_bcs,
            device=device,
            poly_order=poly_order,
        )

        k, f = GlobalAssembly.assemble_system(
            k_global=k,
            f_global=f,
            k_add=nitsche_sets,
        )

        return (
            k,
            f,
            [
                nset["boundary_elems"]
                for nset in nitsche_sets
            ],
        )