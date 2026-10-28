import numpy as np
import torch
import trimesh
import time

from geometry import HexGrid


# -----------------------------------------------------------------------------
# CUDA constants
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CUDA constants
# -----------------------------------------------------------------------------

def _make_subcell_node_signs(refined_subcells: int) -> torch.Tensor:
    """
    Builds normalized (-1..1) node offsets for a (refined_subcells+1)^3 regular
    lattice inside a unit cell, replacing the old fixed 8-corner-only table.
    For refined_subcells=1 this reduces exactly to the legacy 8 cube corners.
    """
    n = refined_subcells + 1
    lin = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    signs = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    return signs  # (n^3, 3)


_CORNER_SIGNS_CACHE: dict[int, torch.Tensor] = {}


def _corner_signs_for(refined_subcells: int, device) -> torch.Tensor:
    if refined_subcells not in _CORNER_SIGNS_CACHE:
        _CORNER_SIGNS_CACHE[refined_subcells] = _make_subcell_node_signs(refined_subcells)
    return _CORNER_SIGNS_CACHE[refined_subcells].to(device)


# -----------------------------------------------------------------------------
# Vectorized node generation
# -----------------------------------------------------------------------------

def _cell_nodes_vectorized_torch(centers_t, dx, dy, dz, refined_subcells: int = 1):
    """
    centers_t : (N,3) cuda tensor

    Returns
    -------
    coords : (N, (refined_subcells+1)**3, 3) cuda tensor
    """
    half = torch.tensor(
        [dx / 2.0, dy / 2.0, dz / 2.0],
        dtype=torch.float64,
        device=centers_t.device,
    )  # (3,)

    signs = _corner_signs_for(refined_subcells, centers_t.device)  # (n^3, 3)
    offsets = signs * half  # (n^3, 3)

    return centers_t[:, None, :] + offsets[None, :, :]  # (N, n^3, 3)


def _triangle_box_intersections(triangles, cell_indices, grid, device, batch_size=65536):
    """Return exact triangle-AABB intersections for triangle/cell candidate pairs."""
    triangles_t = torch.as_tensor(triangles, dtype=torch.float64, device=device)
    indices_t = torch.as_tensor(cell_indices, dtype=torch.int64, device=device)
    half = torch.tensor([grid.dx, grid.dy, grid.dz], dtype=torch.float64, device=device) * 0.5
    centers = torch.as_tensor(grid.centers, dtype=torch.float64, device=device).reshape(-1, 3)
    hit_chunks = []
    basis = torch.eye(3, dtype=torch.float64, device=device)
    for start in range(0, indices_t.shape[0], batch_size):
        stop = min(start + batch_size, indices_t.shape[0])
        triangle = triangles_t[start:stop]
        center = centers[indices_t[start:stop, 1]]
        vertex = triangle - center[:, None, :]
        edges = torch.stack((vertex[:, 1] - vertex[:, 0], vertex[:, 2] - vertex[:, 1], vertex[:, 0] - vertex[:, 2]), dim=1)
        overlap = (
            (vertex.amin(dim=1) <= half)
            & (vertex.amax(dim=1) >= -half)
        ).all(dim=1)
        normal = torch.cross(edges[:, 0], edges[:, 1], dim=1)
        overlap &= (vertex[:, 0] * normal).sum(dim=1).abs() <= (normal.abs() * half).sum(dim=1)
        for edge in edges.unbind(dim=1):
            axes = torch.cross(edge[:, None, :], basis[None, :, :], dim=-1)
            projection = (vertex[:, :, None, :] * axes[:, None, :, :]).sum(dim=-1)
            radius = (axes.abs() * half).sum(dim=-1)
            overlap &= (projection.amin(dim=1) <= radius).all(dim=1)
            overlap &= (projection.amax(dim=1) >= -radius).all(dim=1)
        hit_chunks.append(indices_t[start:stop, 1][overlap].cpu())
    return torch.unique(torch.cat(hit_chunks), sorted=True) if hit_chunks else torch.empty(0, dtype=torch.int64)


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------

import numpy as np
import torch
import trimesh
import time

from geometry import HexGrid


class Classification:

    @staticmethod
    def classify_narrow_band_csc(
        volume,
        grid,
        poly_order=1,
        device="cuda",
        band_cells=3,
        candidate_padding_cells=1,
        tri_chunk=128,
        sdf_node_order=None,   # SDF-sampling resolution for NN features, decoupled from classification
        candidate_indices=None,
        candidate_batch_size=None,
        return_sdf=True,
    ):
        """Classify cut cells from mesh triangles with CUDA CSC narrow-band SDF.

        Classification (cut / not-cut) always uses the cheap 8-corner SDF,
        regardless of `sdf_node_order` — the cut/not-cut decision must not
        depend on how finely the NN feature vector samples the cell. The
        refined SDF (if requested) is only ever computed for cells that
        already passed the corner-based cut test, so cost scales with the
        (small) cut-cell count instead of the (large) candidate count.
        """
        from clean_simulation.internal_nn_runtime.geometry import (
            compute_cell_node_sdf_csc,
            extract_volume_triangles,
            points_inside_mesh_ray_parity,
            reorder_refined_sdf_for_nn,
        )

        if candidate_padding_cells < 0:
            raise ValueError("candidate_padding_cells must be non-negative.")

        nx, ny, nz = grid.resolution
        bounds_min = np.asarray(grid.bounds[0], dtype=float)
        cell_size = np.array([grid.dx, grid.dy, grid.dz], dtype=float)
        triangles = np.asarray(volume.mesh.triangles, dtype=float)
        if len(triangles) == 0:
            raise ValueError("Volume mesh contains no triangles.")

        eps = 1e-10

        triangle_min = np.floor(
            (triangles.min(axis=1) - bounds_min - eps) / cell_size
        ).astype(int)
        triangle_max = np.floor(
            (triangles.max(axis=1) - bounds_min + eps) / cell_size
        ).astype(int)

        triangle_min -= candidate_padding_cells
        triangle_max += candidate_padding_cells
        triangle_min = np.maximum(triangle_min, 0)
        triangle_max = np.minimum(triangle_max, np.array([nx - 1, ny - 1, nz - 1]))

        candidate_start = time.perf_counter()
        triangle_cell_pairs = []
        if candidate_indices is None:
            for triangle_id, (lower, upper) in enumerate(zip(triangle_min, triangle_max)):
                ii, jj, kk = np.meshgrid(
                    np.arange(lower[0], upper[0] + 1),
                    np.arange(lower[1], upper[1] + 1),
                    np.arange(lower[2], upper[2] + 1),
                    indexing="ij",
                )
                flat = (ii * ny * nz + jj * nz + kk).reshape(-1)
                triangle_cell_pairs.append(np.column_stack((np.full(flat.shape, triangle_id, dtype=np.int64), flat)))
            pair_array = np.concatenate(triangle_cell_pairs) if triangle_cell_pairs else np.empty((0, 2), dtype=np.int64)
            candidate_indices = torch.from_numpy(np.unique(pair_array[:, 1])).long()
        else:
            candidate_indices = torch.as_tensor(candidate_indices, dtype=torch.long).cpu().unique(sorted=True)
            pair_array = None
        candidate_generation_time = time.perf_counter() - candidate_start

        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
        sdf_start = time.perf_counter()

        # --- Stage 1: exact triangle-cell intersection classification ---
        # Triangle AABBs only generate candidates. SAT rejects the many cells in
        # a long triangle's AABB that the triangle itself does not intersect.
        candidate_band_cells = band_cells + candidate_padding_cells + 2
        effective_sdf_order = poly_order if sdf_node_order is None else sdf_node_order
        batch_size = len(candidate_indices) if candidate_batch_size is None else int(candidate_batch_size)
        if pair_array is None:
            raise ValueError("Exact FNB classification requires automatically generated triangle-AABB candidates.")
        partial_indices = _triangle_box_intersections(
            triangles=triangles[pair_array[:, 0]],
            cell_indices=pair_array,
            grid=grid,
            device=torch.device(device),
        )
        if return_sdf:
            sdf_chunks = []
            for start in range(0, len(partial_indices), batch_size):
                candidate_batch = partial_indices[start:start + batch_size]
                sdf_corners = compute_cell_node_sdf_csc(
                    grid=grid, volume=volume, cell_indices=candidate_batch, poly_order=1,
                    device=device, band_cells=candidate_band_cells, tri_chunk=tri_chunk,
                )
                if effective_sdf_order == 1:
                    sdf_batch = sdf_corners.cpu()
                else:
                    sdf_batch = compute_cell_node_sdf_csc(
                        grid=grid, volume=volume, cell_indices=candidate_batch,
                        poly_order=effective_sdf_order, device=device,
                        band_cells=candidate_band_cells, tri_chunk=tri_chunk,
                    ).cpu()
                    sdf_batch = reorder_refined_sdf_for_nn(sdf_batch, effective_sdf_order)
                sdf_chunks.append(sdf_batch)
            sdf_partial = torch.cat(sdf_chunks) if sdf_chunks else torch.empty(
                (0, (effective_sdf_order + 1) ** 3), dtype=torch.float64
            )
        else:
            sdf_partial = None

        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))

        sdf_classification_time = time.perf_counter() - sdf_start

        print("candidate cells :", len(candidate_indices))
        print("classified cuts :", len(partial_indices))
        print("fraction :", len(partial_indices) / max(len(candidate_indices), 1))

        metadata = {
            "candidate_cell_count": int(candidate_indices.numel()),
            "partial_cell_count": int(partial_indices.numel()),
            "candidate_generation_time": candidate_generation_time,
            "sdf_classification_time": sdf_classification_time,
            "effective_band_cells": int(candidate_band_cells),
        }
        return partial_indices, sdf_partial, metadata

    @staticmethod
    def classify_solid_narrow_band_csc(
        volume,
        grid,
        poly_order=1,
        device="cuda",
        band_cells=3,
        candidate_padding_cells=0,
        tri_chunk=128,
        candidate_batch_size=2048,
        return_sdf=True,
    ):
        """Classify a watertight solid with FNB cuts plus exterior flood fill.

        The narrow-band stage identifies a conservative shell of surface cells.
        A 6-connected flood fill from grid-boundary cells labels the exterior;
        all remaining non-shell cells are full interior cells. This avoids any
        QuESo classification or quadrature generation.
        """
        from scipy.ndimage import binary_propagation

        cut_indices, sdf_cut, metadata = Classification.classify_narrow_band_csc(
            volume=volume,
            grid=grid,
            poly_order=poly_order,
            device=device,
            band_cells=band_cells,
            candidate_padding_cells=candidate_padding_cells,
            tri_chunk=tri_chunk,
            candidate_batch_size=candidate_batch_size,
            return_sdf=return_sdf,
        )
        nx, ny, nz = (int(value) for value in grid.resolution)
        cut_cells = np.zeros((nx, ny, nz), dtype=bool)
        cut_cells.reshape(-1)[cut_indices.numpy()] = True
        traversable = ~cut_cells
        boundary = np.zeros_like(traversable)
        boundary[0, :, :] = True
        boundary[-1, :, :] = True
        boundary[:, 0, :] = True
        boundary[:, -1, :] = True
        boundary[:, :, 0] = True
        boundary[:, :, -1] = True
        flood_fill_start = time.perf_counter()
        exterior = binary_propagation(boundary & traversable, mask=traversable)
        inside_cells = traversable & ~exterior
        metadata["fnb_flood_fill_seconds"] = time.perf_counter() - flood_fill_start
        inside_indices = torch.from_numpy(np.flatnonzero(inside_cells.reshape(-1)).astype(np.int64))
        metadata.update({
            "cut_cell_count": int(cut_indices.numel()),
            "inside_cell_count": int(inside_indices.numel()),
            "active_cell_count": int(cut_indices.numel() + inside_indices.numel()),
        })
        return inside_indices, cut_indices, sdf_cut, metadata

    # classify_sdf and extract_partial_centers unchanged — omitted here for brevity,
    # keep them exactly as in your current file.

    @staticmethod
    def classify_sdf(
        volume,
        grid,
        device="cuda",
        tol=None,
        lower_threshold=None,
        upper_threshold=None,
        query_batch_size=8192,
    ):
        """
        Returns
        -------
        inside  : (Nx,Ny,Nz) bool tensor
        partial : (Nx,Ny,Nz) bool tensor
        sdf     : (Nx,Ny,Nz) float64 tensor
        """

        device = torch.device(device)

        if tol is None:
            tol = 1e-9 * max(grid.dx, grid.dy, grid.dz)

        # ---------------------------------------------------------------------
        # Grid nodes
        # ---------------------------------------------------------------------

        nodes = grid.node_coordinates(poly_order=1)

        if not torch.is_tensor(nodes):
            nodes = torch.as_tensor(
                nodes,
                dtype=torch.float64,
                device=device,
            )
        else:
            nodes = nodes.to(device=device, dtype=torch.float64)

        # ---------------------------------------------------------------------
        # Cell centers
        # ---------------------------------------------------------------------

        centers = grid.centers.reshape(-1, 3)

        if not torch.is_tensor(centers):
            centers = torch.as_tensor(
                centers,
                dtype=torch.float64,
                device=device,
            )
        else:
            centers = centers.to(device=device, dtype=torch.float64)

        N = centers.shape[0]

        # ---------------------------------------------------------------------
        # Grid nodes and Centers calculation using full-field Trimesh queries
        # ---------------------------------------------------------------------
        # Query in bounded batches: Trimesh proximity creates large temporary arrays,
        # so a single query over a high-resolution grid can exhaust host memory.

        query = torch.cat([nodes, centers], dim=0)

        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive.")
        phi_chunks = []
        for start in range(0, len(query), query_batch_size):
            batch = query[start : start + query_batch_size]
            if hasattr(volume, "signed_distance_torch"):
                phi_chunk = volume.signed_distance_torch(batch)
            else:
                phi_np = volume.signed_distance(batch.detach().cpu().numpy())
                phi_chunk = torch.as_tensor(phi_np, dtype=torch.float64)
            phi_chunks.append(phi_chunk.to(device=device, dtype=torch.float64))
        phi_all = torch.cat(phi_chunks)

        Nn = nodes.shape[0]
        phi_nodes = phi_all[:Nn]
        phi_centers_flat = phi_all[Nn:]

        # Collect cell element corners
        elem_to_node = HexGrid.build_node_index(grid, poly_order=1)
        if not torch.is_tensor(elem_to_node):
            elem_to_node = torch.as_tensor(elem_to_node, dtype=torch.long, device=device)
        else:
            elem_to_node = elem_to_node.to(device=device)

        phi_corners = phi_nodes[elem_to_node] # (N, 8)

        # ---------------------------------------------------------------------
        # Classification
        # ---------------------------------------------------------------------

        lower_val = lower_threshold if lower_threshold is not None else -tol
        upper_val = upper_threshold if upper_threshold is not None else tol

        inside = (
            (phi_corners >= lower_val).all(dim=1)
            & (phi_centers_flat >= lower_val)
        )

        outside = (
            (phi_corners <= upper_val).all(dim=1)
            & (phi_centers_flat <= upper_val)
        )

        partial = ~(inside | outside)

        sdf = phi_centers_flat.reshape(grid.shape)

        return (
            inside.reshape(grid.shape),
            partial.reshape(grid.shape),
            sdf,
        )

    @staticmethod
    def extract_partial_centers(grid, partial_mask, device="cuda"):
        """
        Returns
        -------
        pts : (Np,3) cuda tensor
        """

        device = torch.device(device)

        pts = grid.centers.reshape(-1, 3)

        if not torch.is_tensor(pts):
            pts = torch.as_tensor(
                pts,
                dtype=torch.float64,
                device=device,
            )
        else:
            pts = pts.to(device=device, dtype=torch.float64)

        if not torch.is_tensor(partial_mask):
            mask = torch.as_tensor(
                partial_mask.reshape(-1),
                dtype=torch.bool,
                device=device,
            )
        else:
            mask = partial_mask.reshape(-1).to(device=device)

        return pts[mask]