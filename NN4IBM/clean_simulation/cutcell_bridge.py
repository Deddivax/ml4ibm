from __future__ import annotations

import importlib
import math
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..internal_fem_kernel.geometry import HexGrid
from ..internal_fem_kernel.runtime.common import as_index_tensor_cpu, as_tensor


@lru_cache(maxsize=1)
def _cached_symmetry_table() -> dict[str, Any]:
    sym_mod = importlib.import_module("cutcell_ml.symmetries")
    table = getattr(sym_mod, "build_symmetry_table")()
    return {
        "node_perms": as_tensor(table["node_perms"], dtype=torch.int64, device="cpu"),
        "dof_perms": as_tensor(table["dof_perms"], dtype=torch.int64, device="cpu"),
        "dof_signs": as_tensor(table["dof_signs"], dtype=torch.float64, device="cpu"),
    }


def get_symmetry_cache() -> dict[str, Any]:
    return _cached_symmetry_table()


def canonicalize_d_batch(d_batch: torch.Tensor, node_perms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d_orbits = d_batch[:, node_perms]
    key = torch.round(d_orbits * 1.0e12) / 1.0e12
    active = torch.ones((d_batch.shape[0], node_perms.shape[0]), dtype=torch.bool, device=d_batch.device)
    for dim in range(8):
        vals = key[:, :, dim]
        masked = torch.where(active, vals, torch.full_like(vals, float("inf")))
        min_v = masked.min(dim=1, keepdim=True).values
        active &= vals <= (min_v + 1e-15)
        if bool(torch.all(active.sum(dim=1) == 1).item()):
            break
    g_star = torch.argmax(active.to(dtype=torch.int64), dim=1)
    d_can = d_orbits[torch.arange(d_batch.shape[0], device=d_batch.device), g_star]
    return d_can.to(dtype=torch.float64), g_star.to(dtype=torch.int64)


def output_to_k24_batch(out_t: torch.Tensor) -> torch.Tensor:
    if out_t.ndim == 3 and out_t.shape[-2:] == (24, 24):
        return 0.5 * (out_t + out_t.transpose(-1, -2))
    tri = torch.triu_indices(24, 24, device=out_t.device)
    if out_t.ndim == 2 and out_t.shape[1] == 300:
        b = out_t.shape[0]
        k = torch.zeros((b, 24, 24), dtype=out_t.dtype, device=out_t.device)
        k[:, tri[0], tri[1]] = out_t
        k[:, tri[1], tri[0]] = out_t
        return k
    if out_t.ndim == 1 and out_t.shape[0] == 300:
        k = torch.zeros((1, 24, 24), dtype=out_t.dtype, device=out_t.device)
        k[:, tri[0], tri[1]] = out_t[None, :]
        k[:, tri[1], tri[0]] = out_t[None, :]
        return k
    raise RuntimeError(f"Unexpected NN output shape: {tuple(out_t.shape)}")


def inverse_transform_k_batch(k_can: torch.Tensor, g_idx: torch.Tensor, dof_perms: torch.Tensor, dof_signs: torch.Tensor) -> torch.Tensor:
    # Fully vectorized inverse transform: build inverse DOF permutations per sample,
    # then apply matching inverse sign corrections.
    dp = dof_perms[g_idx]   # (B, 24): forward map old_dof -> new_dof
    ds = dof_signs[g_idx]   # (B, 24): forward sign per old_dof
    bsz = k_can.shape[0]
    if bsz == 0:
        return k_can

    ar = torch.arange(dp.shape[1], dtype=torch.int64, device=dp.device).unsqueeze(0).expand_as(dp)
    dp_inv = torch.empty_like(dp)
    dp_inv.scatter_(1, dp, ar)
    ds_inv = torch.gather(ds, 1, dp_inv)

    row_idx = dp_inv[:, :, None].expand(-1, -1, 24)
    col_idx = dp_inv[:, None, :].expand(-1, 24, -1)
    k_perm = torch.gather(k_can, 1, row_idx)
    k_perm = torch.gather(k_perm, 2, col_idx)
    return k_perm * ds_inv[:, :, None] * ds_inv[:, None, :]


"""
Fast narrow-band SDF on CUDA via a CSC-style approach.

Replaces  compute_cell_node_sdf_for_indices  (and the whole
signed_distance call-path inside prepare_nn_inputs) with a fully
GPU-resident kernel that is O(T × band) instead of O(N_nodes × T).

Public API
----------
compute_cell_node_sdf_csc(
    grid, volume, cell_indices, poly_order, *, device, band_cells=3
) -> Tensor  shape (len(cell_indices), (poly_order+1)**3)

The function is a drop-in replacement:

    # old
    d_cut_basic = compute_cell_node_sdf_for_indices(
        grid, volume, partial_indices_t, poly_order)

    # new
    d_cut_basic = compute_cell_node_sdf_csc(
        grid, volume, partial_indices_t, poly_order,
        device=torch.device("cuda"))

Requirements
------------
* volume must expose  .triangles  (Fx3x3 float array/tensor, world coords)
  and  .triangle_normals  (Fx3) pointing outward, OR the helper
  extract_volume_triangles() below will attempt to read them from common
  mesh attributes (trimesh, pyvista, …).
* torch with CUDA.
* No extra C extensions – the kernel is written with torch.compile-friendly
  pure-PyTorch scatter primitives so it also runs on CPU for debugging
  (just slowly).

Algorithm sketch (CSC / Roosing et al. 2019)
--------------------------------------------
For every triangle T_i:
  1. Compute the AABB of the triangle expanded by `band` in each axis.
  2. Every grid node whose world position falls inside that AABB is a
     *candidate* – we compute the signed distance from it to T_i.
  3. We keep the minimum |distance| winner per node (atomic min on the
     unsigned part) and record the sign separately via the dot product
     with the outward face normal.

The AABB culling means we only touch O(band^3) nodes per triangle,
making the total work proportional to  T × band^3  rather than
T × N_nodes.

For the narrow band needed here (cut-cell nodes are at most ~1 cell
away from the surface by definition) band_cells=3 is more than enough.
"""

_SQRT3 = math.sqrt(3.0)


# ---------------------------------------------------------------------------
# Helpers to extract triangle soup from common volume / mesh objects
# ---------------------------------------------------------------------------

def extract_volume_triangles(
    volume: Any,
    device: torch.device,
    debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (triangles, normals) as float64 CUDA tensors.

    triangles : (F, 3, 3)  – vertex positions per face
    normals   : (F, 3)     – outward unit normals per face

    Tries several common attribute conventions in order:
      trimesh  → mesh.triangles, mesh.face_normals
      pyvista  → faces array
      raw dict → {"triangles": ..., "normals": ...}
      direct   → .triangles / .face_normals attributes
    """

    def _t(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float64, device=device)
        return torch.tensor(np.asarray(x), dtype=torch.float64, device=device)

    # clean_simulation geometry wrapper commonly stores the actual mesh under `.mesh`.
    mesh_like = getattr(volume, "mesh", None)
    if mesh_like is not None:
        volume = mesh_like

    # dict-like
    if isinstance(volume, dict):
        tris = _t(volume["triangles"])
        nrms = _t(volume.get("normals") or volume.get("face_normals"))
        tris_out, nrms_out = tris, F.normalize(nrms, dim=-1)
    elif hasattr(volume, "triangles") and hasattr(volume, "face_normals"):
        # trimesh / any object with .triangles + .face_normals
        tris_out = _t(volume.triangles)
        nrms_out = F.normalize(_t(volume.face_normals), dim=-1)
    elif hasattr(volume, "vertices") and hasattr(volume, "faces"):
        # objects that expose vertices + faces separately
        verts = _t(volume.vertices)          # (V, 3)
        faces = torch.tensor(np.asarray(volume.faces), dtype=torch.int64, device=device)  # (F, 3)
        tris_out = verts[faces]              # (F, 3, 3)
        e1 = tris_out[:, 1] - tris_out[:, 0]
        e2 = tris_out[:, 2] - tris_out[:, 0]
        nrms_raw = torch.cross(e1, e2, dim=-1)
        nrms_out = F.normalize(nrms_raw, dim=-1)
    else:
        raise TypeError(
            f"Cannot extract triangles from volume of type {type(volume)}. "
            "Provide a trimesh.Trimesh, a pyvista mesh, or an object with "
            ".triangles and .face_normals attributes."
        )

    if debug:
        F_count = tris_out.shape[0]
        verts_all = tris_out.reshape(-1, 3)          # (F*3, 3)
        bbox_min = verts_all.min(dim=0).values
        bbox_max = verts_all.max(dim=0).values
        centroid = verts_all.mean(dim=0)
        nlen = nrms_out.norm(dim=-1)                 # should be ~1.0
        print("==== TRIANGLE DEBUG ====")
        print(f"Triangles: {F_count}")
        print(f"BBox min  {bbox_min.cpu().tolist()}")
        print(f"BBox max  {bbox_max.cpu().tolist()}")
        print(f"Centroid  {centroid.cpu().tolist()}")
        print(f"Normal lengths: min={nlen.min().item():.6f}  max={nlen.max().item():.6f}  "
              f"(expected 1.0)")
        print(f"BBox coords: x=[{bbox_min[0].item():.4f}, {bbox_max[0].item():.4f}]  "
              f"y=[{bbox_min[1].item():.4f}, {bbox_max[1].item():.4f}]  "
              f"z=[{bbox_min[2].item():.4f}, {bbox_max[2].item():.4f}]")
        n_show = min(5, F_count)
        for fi in range(n_show):
            vv = tris_out[fi].cpu().tolist()
            nn = nrms_out[fi].cpu().tolist()
            print(f"  tri[{fi}] v0={vv[0]}  v1={vv[1]}  v2={vv[2]}")
            print(f"         normal={nn}")
        if nlen.min().item() < 0.99 or nlen.max().item() > 1.01:
            print(f"WARNING: normal lengths deviate from 1.0 — normals may be un-normalized")
        print("========================")

    return tris_out, nrms_out


# ---------------------------------------------------------------------------
# Point-to-triangle signed distance (batched, GPU-friendly)
# ---------------------------------------------------------------------------

def _point_to_triangle_signed_distance(
    points: torch.Tensor,   # (P, 3)
    v0: torch.Tensor,       # (F, 3)
    v1: torch.Tensor,       # (F, 3)
    v2: torch.Tensor,       # (F, 3)
    normals: torch.Tensor,  # (F, 3)  outward unit normals
) -> torch.Tensor:
    """
    For each (point, triangle) pair compute the signed distance.
    Returns (P, F) tensor.

    Sign convention: positive = outside (same side as outward normal).

    The closest point on the triangle is found using the standard
    Eberly / Haines barycentric-region decomposition.
    """
    # Expand for broadcasting: (P, 1, 3) and (1, F, 3)
    p  = points[:, None, :]       # (P, 1, 3)
    a  = v0[None, :, :]           # (1, F, 3)
    b  = v1[None, :, :]
    c  = v2[None, :, :]
    n  = normals[None, :, :]      # (1, F, 3)

    ab = b - a
    ac = c - a
    ap = p - a

    d1 = (ap * ab).sum(-1)   # (P, F)
    d2 = (ap * ac).sum(-1)

    bp = p - b
    d3 = (bp * ab).sum(-1)
    d4 = (bp * ac).sum(-1)

    cp = p - c
    d5 = (cp * ab).sum(-1)
    d6 = (cp * ac).sum(-1)

    # Vertex regions
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # Barycentric coords of clamped closest point
    denom = 1.0 / (va + vb + vc).clamp(min=1e-30)

    # --- region tests ---
    # Vertex A
    in_A = (d1 <= 0) & (d2 <= 0)
    # Vertex B
    in_B = (d3 >= 0) & (d4 <= d3)
    # Vertex C
    in_C = (d6 >= 0) & (d5 <= d6)
    # Edge AB
    in_AB = (~in_A) & (~in_B) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    # Edge AC
    in_AC = (~in_A) & (~in_C) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    # Edge BC
    in_BC = (~in_B) & (~in_C) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    # Closest points in each region
    # -- interior --
    s = (vb * denom).clamp(0, 1)
    t = (vc * denom).clamp(0, 1)
    cp_interior = a + s[..., None] * ab + t[..., None] * ac

    # -- vertex A --
    cp_A = a.expand_as(cp_interior)

    # -- vertex B --
    cp_B = b.expand_as(cp_interior)

    # -- vertex C --
    cp_C = c.expand_as(cp_interior)

    # -- edge AB --
    t_AB = (d1 / (d1 - d3)).clamp(0, 1)
    cp_AB = a + t_AB[..., None] * ab

    # -- edge AC --
    t_AC = (d2 / (d2 - d6)).clamp(0, 1)
    cp_AC = a + t_AC[..., None] * ac

    # -- edge BC --
    denom_BC = ((d4 - d3) + (d5 - d6)).clamp(min=1e-30)
    t_BC = ((d4 - d3) / denom_BC).clamp(0, 1)
    cp_BC = b + t_BC[..., None] * (c - b)

    # Select closest point by region mask (priority order matches Eberly)
    closest = cp_interior.clone()
    closest = torch.where(in_AB[..., None], cp_AB, closest)
    closest = torch.where(in_AC[..., None], cp_AC, closest)
    closest = torch.where(in_BC[..., None], cp_BC, closest)
    closest = torch.where(in_B[..., None],  cp_B,  closest)
    closest = torch.where(in_C[..., None],  cp_C,  closest)
    closest = torch.where(in_A[..., None],  cp_A,  closest)

    # Unsigned distance
    diff = p - closest           # (P, F, 3)
    dist = diff.norm(dim=-1)     # (P, F)

    # Sign: dot product of (point – closest) with triangle normal
    sign = torch.sign((diff * n).sum(-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    return sign * dist


@torch.no_grad()
def points_inside_mesh_ray_parity(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    point_batch_size: int = 2048,
    triangle_batch_size: int = 128,
) -> torch.Tensor:
    """Return watertight-solid containment using deterministic ray parity.

    This is used by FNB classification only. It avoids the incorrect global
    inside/outside inference that results from using a closest triangle's local
    normal as an SDF sign near edges and corners.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    device = triangles.device
    points = points.to(device=device, dtype=torch.float64)
    direction = F.normalize(torch.tensor([1.0, 0.371, 0.193], dtype=torch.float64, device=device), dim=0)
    direction = direction.expand(point_batch_size, -1)
    epsilon = 1.0e-10
    result = torch.empty(points.shape[0], dtype=torch.bool, device=device)

    for point_start in range(0, points.shape[0], point_batch_size):
        origins = points[point_start:point_start + point_batch_size]
        ray_direction = direction[:origins.shape[0]]
        intersections = torch.zeros(origins.shape[0], dtype=torch.int64, device=device)
        for triangle_start in range(0, triangles.shape[0], triangle_batch_size):
            triangle = triangles[triangle_start:triangle_start + triangle_batch_size]
            vertex0, vertex1, vertex2 = triangle[:, 0], triangle[:, 1], triangle[:, 2]
            edge1 = vertex1 - vertex0
            edge2 = vertex2 - vertex0
            h = torch.cross(ray_direction[:, None, :], edge2[None, :, :], dim=-1)
            determinant = (edge1[None, :, :] * h).sum(dim=-1)
            valid = determinant.abs() > epsilon
            inverse_determinant = torch.where(valid, determinant.reciprocal(), torch.zeros_like(determinant))
            offset = origins[:, None, :] - vertex0[None, :, :]
            u = inverse_determinant * (offset * h).sum(dim=-1)
            q = torch.cross(offset, edge1[None, :, :], dim=-1)
            v = inverse_determinant * (ray_direction[:, None, :] * q).sum(dim=-1)
            distance = inverse_determinant * (edge2[None, :, :] * q).sum(dim=-1)
            hit = valid & (u >= -epsilon) & (v >= -epsilon) & (u + v <= 1.0 + epsilon) & (distance > epsilon)
            intersections += hit.sum(dim=1, dtype=torch.int64)
        result[point_start:point_start + origins.shape[0]] = intersections.remainder(2).bool()
    return result.cpu()


@torch.no_grad()
def compute_cell_volume_fractions_ray_parity(
    grid: Any,
    volume: Any,
    cell_indices: torch.Tensor,
    *,
    samples_per_axis: int = 4,
    device: torch.device | str | None = None,
    cell_batch_size: int = 256,
) -> torch.Tensor:
    """Estimate cut-cell solid fractions from watertight-mesh subcell samples."""
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be at least one.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    indices = as_index_tensor_cpu(cell_indices)
    if indices.numel() == 0:
        return torch.empty(0, dtype=torch.float64)

    sample_axis = (torch.arange(samples_per_axis, dtype=torch.float64) + 0.5) / samples_per_axis - 0.5
    sx, sy, sz = torch.meshgrid(sample_axis * grid.dx, sample_axis * grid.dy, sample_axis * grid.dz, indexing="ij")
    offsets = torch.stack((sx.reshape(-1), sy.reshape(-1), sz.reshape(-1)), dim=-1).to(device)
    triangles, _ = extract_volume_triangles(volume, device)
    centers = as_tensor(grid.centers, dtype=torch.float64, device="cpu").reshape(-1, 3)
    fractions = []
    for start in range(0, indices.numel(), cell_batch_size):
        batch_centers = centers[indices[start:start + cell_batch_size]].to(device)
        points = (batch_centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        inside = points_inside_mesh_ray_parity(points, triangles)
        fractions.append(inside.reshape(-1, offsets.shape[0]).to(torch.float64).mean(dim=1))
    return torch.cat(fractions)


# ---------------------------------------------------------------------------
# CSC narrow-band SDF: iterate over triangles, scatter to nearby nodes
# ---------------------------------------------------------------------------

@torch.no_grad()
def _csc_sdf_chunk(
    query_nodes: torch.Tensor,   # (N, 3)  world coords of all query nodes
    tris: torch.Tensor,          # (F, 3, 3)
    normals: torch.Tensor,       # (F, 3)
    band: float,
    tri_chunk: int = 512,
    debug: bool = False,
    allow_unvisited: bool = False,
) -> torch.Tensor:
    """
    Core CSC loop.  Returns (N,) signed distances, initialised to +band.

    Processes triangles in chunks to keep peak memory manageable.
    For each chunk:
      1. Build the AABB of each triangle + band.
      2. Find which query nodes fall inside (vectorised).
      3. Compute exact signed distance for those (node, tri) pairs.
      4. Atomic-style minimum: keep the value with smallest |dist|.

    When debug=True, a per-node visit counter is maintained and extensive
    diagnostics are printed.  A RuntimeError is raised for unvisited nodes
    unless `allow_unvisited` is set for broad FNB candidate screening.
    """
    N = query_nodes.shape[0]
    F_total = tris.shape[0]
    device = query_nodes.device

    # Initialise: large positive value
    best_abs  = torch.full((N,), band * 2.0, dtype=torch.float64, device=device)
    best_sign = torch.ones((N,), dtype=torch.float64, device=device)

    # Visit counter (only in debug mode to avoid overhead in production)
    visit_counter = torch.zeros(N, dtype=torch.int64, device=device) if debug else None

    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]

    # ------------------------------------------------------------------
    # Step 4: pre-loop coordinate-system overlap check
    # ------------------------------------------------------------------
    if debug:
        node_min = query_nodes.min(dim=0).values
        node_max = query_nodes.max(dim=0).values
        tri_verts_all = tris.reshape(-1, 3)
        tri_min = tri_verts_all.min(dim=0).values
        tri_max = tri_verts_all.max(dim=0).values
        # Expand triangle bbox by band on all sides for the overlap check
        tri_min_exp = tri_min - band
        tri_max_exp = tri_max + band
        overlap = (
            (node_min <= tri_max_exp) & (node_max >= tri_min_exp)
        ).all().item()
        print("==== COORDINATE SYSTEM OVERLAP CHECK ====")
        print(f"Node bbox:     min={node_min.cpu().tolist()}")
        print(f"               max={node_max.cpu().tolist()}")
        print(f"Triangle bbox: min={tri_min.cpu().tolist()}")
        print(f"               max={tri_max.cpu().tolist()}")
        print(f"Tri bbox + band ({band:.4f}):")
        print(f"               min={tri_min_exp.cpu().tolist()}")
        print(f"               max={tri_max_exp.cpu().tolist()}")
        print(f"Bounding boxes OVERLAP: {overlap}")
        if not overlap:
            raise RuntimeError(
                "_csc_sdf_chunk: node bbox and (triangle bbox + band) DO NOT OVERLAP.\n"
                f"  node  bbox: {node_min.cpu().tolist()} .. {node_max.cpu().tolist()}\n"
                f"  tri   bbox: {tri_min.cpu().tolist()} .. {tri_max.cpu().tolist()}\n"
                f"  band = {band}\n"
                "This is a coordinate-system mismatch.  Nodes and triangles live in "
                "different coordinate frames.  Check grid.centers vs mesh coordinates."
            )
        print(f"First 10 node coords:")
        for ni in range(min(10, N)):
            print(f"  node[{ni}] = {query_nodes[ni].cpu().tolist()}")
        print("==========================================")

    # ------------------------------------------------------------------
    # Main CSC loop
    # ------------------------------------------------------------------
    for chunk_idx, start in enumerate(range(0, F_total, tri_chunk)):
        end = min(start + tri_chunk, F_total)
        v0c = v0[start:end]        # (C, 3)
        v1c = v1[start:end]
        v2c = v2[start:end]
        nc  = normals[start:end]   # (C, 3)

        # AABB for this chunk – per triangle, not a combined chunk AABB
        vstack = torch.stack([v0c, v1c, v2c], dim=1)   # (C, 3, 3)
        mins = vstack.min(dim=1).values - band          # (C, 3)
        maxs = vstack.max(dim=1).values + band          # (C, 3)

        if debug:
            chunk_tri_min = vstack.reshape(-1, 3).min(dim=0).values
            chunk_tri_max = vstack.reshape(-1, 3).max(dim=0).values

        # Which nodes are inside any of these AABBs?
        # in_aabb[i, j] = True  iff  node i is in the expanded AABB of triangle j
        q       = query_nodes[:, None, :]               # (N, 1, 3)
        in_aabb = (
            (q >= mins[None, :, :]) & (q <= maxs[None, :, :])
        ).all(dim=-1)                                    # (N, C) bool

        active_nodes = in_aabb.any(dim=1)                # (N,)
        n_active = int(active_nodes.sum().item())
        n_inactive = N - n_active

        if debug:
            pct = 100.0 * n_active / max(N, 1)
            print(f"Chunk {chunk_idx:4d} | tris [{start}:{end}] | "
                  f"tri_bbox {chunk_tri_min.cpu().tolist()} .. {chunk_tri_max.cpu().tolist()} | "
                  f"active={n_active} ({pct:.1f}%)  inactive={n_inactive}")

        if not active_nodes.any():
            continue

        active_idx = active_nodes.nonzero(as_tuple=True)[0]   # (A,)
        q_active   = query_nodes[active_idx]                   # (A, 3)

        # Per-triangle mask for active nodes: (A, C)
        in_aabb_active = in_aabb[active_idx]                   # (A, C)

        # Signed distances for ALL (active_node, tri_in_chunk) pairs
        sd = _point_to_triangle_signed_distance(q_active, v0c, v1c, v2c, nc)  # (A, C)

        # Step 5 (debug): spot-check a few active node–triangle distances
        if debug and chunk_idx == 0 and n_active > 0:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(42)
            n_spot = min(5, n_active)
            spot_local = torch.randperm(n_active, generator=rng)[:n_spot]
            print(f"  [spot-check {n_spot} active nodes in chunk 0]")
            for si in spot_local.tolist():
                node_coord = q_active[si].cpu().tolist()
                sd_row = sd[si].cpu()
                has_nan  = bool(torch.isnan(sd_row).any().item())
                has_inf  = bool(torch.isinf(sd_row).any().item())
                best_tri = int(sd_row.abs().argmin().item())
                best_d   = float(sd_row[best_tri].item())
                print(f"    node[{active_idx[si].item()}] coord={node_coord}")
                print(f"      closest tri={start+best_tri}  sd={best_d:.6f}  "
                      f"unsigned={abs(best_d):.6f}  "
                      f"NaN={has_nan}  Inf={has_inf}")
                if has_nan or has_inf:
                    print(f"      WARNING: NaN or Inf detected in signed distances!")

        # Mask out node-triangle pairs whose node is outside this tri's AABB
        large = torch.full_like(sd, band * 2.0)
        sd_masked = torch.where(in_aabb_active, sd, large)

        # Min |distance| across chunk triangles for each active node
        abs_sd = sd_masked.abs()
        best_abs_chunk, best_idx_chunk = abs_sd.min(dim=1)   # (A,)
        best_sign_chunk = sd_masked[
            torch.arange(active_idx.shape[0], device=device),
            best_idx_chunk
        ].sign()
        best_sign_chunk = torch.where(
            best_sign_chunk == 0,
            torch.ones_like(best_sign_chunk),
            best_sign_chunk
        )

        # Update global best for these nodes
        improved = best_abs_chunk < best_abs[active_idx]
        best_abs [active_idx] = torch.where(improved, best_abs_chunk,  best_abs [active_idx])
        best_sign[active_idx] = torch.where(improved, best_sign_chunk, best_sign[active_idx])

        # Update visit counter
        if debug:
            visit_counter.scatter_add_(
                0, active_idx,
                torch.ones(n_active, dtype=torch.int64, device=device)
            )

    # ------------------------------------------------------------------
    # Post-loop diagnostics
    # ------------------------------------------------------------------
    untouched_mask = best_abs >= (band * 2.0 - 1e-12)   # nodes still at init value
    n_untouched = int(untouched_mask.sum().item())

    if debug:
        vc_cpu = visit_counter.cpu()
        print("==== CSC VISIT STATISTICS ====")
        print(f"Nodes total    : {N}")
        print(f"Nodes untouched: {n_untouched}  ({100.0*n_untouched/max(N,1):.2f}%)")
        print(f"Visit counts   : min={vc_cpu.min().item()}  max={vc_cpu.max().item()}  "
              f"mean={vc_cpu.float().mean().item():.2f}")
        counts = torch.bincount(vc_cpu)
        print("Visit histogram (visit_count: num_nodes):")
        for v, cnt in enumerate(counts.tolist()):
            if cnt > 0:
                print(f"  {v:4d} visits: {cnt} nodes")
        zero_visit_idx = (vc_cpu == 0).nonzero(as_tuple=True)[0]
        if len(zero_visit_idx) > 0:
            print(f"Zero-visit node coordinates (first 20):")
            for ni in zero_visit_idx[:20].tolist():
                print(f"  node[{ni}] = {query_nodes[ni].cpu().tolist()}")
        print("================================")
        print(f"SDF (best_abs * best_sign) stats:")
        sdf_preview = best_sign * best_abs
        print(f"  min={sdf_preview.min().item():.6f}  max={sdf_preview.max().item():.6f}")
        print(f"  mean={sdf_preview.mean().item():.6f}  median={sdf_preview.median().item():.6f}")

    if n_untouched > 0 and not allow_unvisited:
        # Print untouched node coordinates before raising
        untouched_idx = untouched_mask.nonzero(as_tuple=True)[0]
        coord_sample = query_nodes[untouched_idx[:20]].cpu().tolist()
        msg = (
            f"_csc_sdf_chunk: {n_untouched} / {N} nodes were NEVER visited by any triangle "
            f"and kept the initialization value band*2={band*2:.6f}.\n"
            f"  band = {band:.6f}\n"
            f"  This means those nodes are farther than band from every triangle AABB.\n"
            f"  Possible causes: coordinate-system mismatch, band too small, or\n"
            f"  cell_indices includes non-cut-cell elements.\n"
            f"  First {len(coord_sample)} untouched node coordinates:\n"
        )
        for c in coord_sample:
            msg += f"    {c}\n"
        raise RuntimeError(msg)

    return best_sign * best_abs


# ---------------------------------------------------------------------------
# Public replacement for compute_cell_node_sdf_for_indices
# ---------------------------------------------------------------------------

def compute_cell_node_sdf_csc(
    grid: Any,
    volume: Any,
    cell_indices: torch.Tensor,
    poly_order: int,
    *,
    device: torch.device | str | None = None,
    band_cells: int = 3,
    tri_chunk: int = 512,
    debug: bool = False,
    allow_unvisited: bool = False,
) -> torch.Tensor:
    """
    GPU narrow-band SDF using the CSC approach (Roosing et al. 2019).

    Drop-in replacement for  compute_cell_node_sdf_for_indices.

    Parameters
    ----------
    grid         : FEM grid object with .centers, .dx, .dy, .dz
    volume       : geometry object – see extract_volume_triangles()
    cell_indices : 1-D integer tensor of cell indices to process
    poly_order   : polynomial order (only 1 fully tested; 2 works)
    device       : CUDA device to use, e.g. torch.device("cuda")
    band_cells   : half-width of the narrow band in units of grid cells
    tri_chunk    : triangles processed per GPU kernel call (memory knob)
    debug        : if True, print extensive diagnostics (no perf impact
                   when False)
    allow_unvisited : retain the positive initialization distance for nodes
                     outside every triangle band; only for broad candidate
                     screening, never for confirmed NN cut cells.

    Returns
    -------
    Tensor of shape  (len(cell_indices), (poly_order+1)**3)  float64,
    matching the layout of  compute_cell_node_sdf_for_indices.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    idx_t = as_index_tensor_cpu(cell_indices)   # always on CPU

    n_cells = int(idx_t.numel())
    n_nodes_per_cell = (poly_order + 1) ** 3

    if n_cells == 0:
        return torch.empty((0, n_nodes_per_cell), dtype=torch.float64)

    # ------- build query node world positions -------
    # Move grid.centers to CPU before indexing so that idx_t (CPU) and
    # centers_all live on the same device — avoids silent cross-device
    # indexing bugs in some PyTorch versions.
    centers_all = as_tensor(grid.centers, dtype=torch.float64, device="cpu").reshape(-1, 3)
    centers_cpu  = centers_all[idx_t]              # (n_cells, 3) on CPU
    centers_t    = centers_cpu.to(device)          # (n_cells, 3) on target device
    offsets = HexGrid._local_node_offsets(
        grid.dx, grid.dy, grid.dz,
        poly_order=poly_order, device=device
    )                                              # (n_nodes_per_cell, 3)
    cell_nodes = centers_t[:, None, :] + offsets[None, :, :]  # (n_cells, n_nodes, 3)

    # Flatten to unique query nodes (adjacent cells share nodes at higher p)
    nodes_flat_all = cell_nodes.reshape(-1, 3)     # (n_cells*n_nodes, 3)
    nodes_flat, inverse = torch.unique(
        nodes_flat_all, dim=0, return_inverse=True
    )                                              # (U, 3), (n_cells*n_nodes,)

    # ------- extract triangle geometry -------
    tris, normals = extract_volume_triangles(volume, device, debug=debug)   # (F,3,3), (F,3)

    # ------- narrow band width in world coords -------
    h    = float((grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0))
    band = band_cells * h

    if debug:
        node_min = nodes_flat.min(dim=0).values
        node_max = nodes_flat.max(dim=0).values
        print("==== compute_cell_node_sdf_csc DIAGNOSTICS ====")
        print(f"Cut cells      : {n_cells}")
        print(f"Nodes per cell : {n_nodes_per_cell}")
        print(f"Total node slots: {n_cells * n_nodes_per_cell}")
        print(f"Unique query nodes: {nodes_flat.shape[0]}")
        print(f"Grid spacing   : dx={grid.dx:.6f}  dy={grid.dy:.6f}  dz={grid.dz:.6f}")
        print(f"h (geom-mean)  : {h:.6f}")
        print(f"band_cells     : {band_cells}")
        print(f"band (world)   : {band:.6f}")
        print(f"Node bbox      : min={node_min.cpu().tolist()}")
        print(f"                 max={node_max.cpu().tolist()}")
        # Verify nodes lie inside mesh bbox + band
        verts_all = tris.reshape(-1, 3)
        mesh_min = verts_all.min(dim=0).values
        mesh_max = verts_all.max(dim=0).values
        print(f"Mesh bbox      : min={mesh_min.cpu().tolist()}")
        print(f"                 max={mesh_max.cpu().tolist()}")
        overlap_lo = torch.max(node_min, mesh_min - band)
        overlap_hi = torch.min(node_max, mesh_max + band)
        overlap_ok = (overlap_lo <= overlap_hi).all().item()
        print(f"Node∩(mesh+band) overlap: {overlap_ok}")
        print(f"First 10 unique node coords:")
        for ni in range(min(10, nodes_flat.shape[0])):
            print(f"  [{ni}] {nodes_flat[ni].cpu().tolist()}")
        print("================================================")

    # ------- run CSC SDF -------
    sdf_unique = _csc_sdf_chunk(
        nodes_flat, tris, normals, band,
        tri_chunk=tri_chunk, debug=debug, allow_unvisited=allow_unvisited,
    )                                              # (U,)

    if debug:
        # Step 6: validate final SDF
        import numpy as np
        sdf_np = sdf_unique.cpu().numpy()
        limit  = _SQRT3 * h
        n_exceed = int((np.abs(sdf_np) > limit).sum())
        pct_exceed = 100.0 * n_exceed / max(len(sdf_np), 1)
        p95 = float(np.percentile(np.abs(sdf_np), 95))
        p99 = float(np.percentile(np.abs(sdf_np), 99))
        print("==== FINAL SDF VALIDATION ====")
        print(f"SDF unique: min={sdf_np.min():.6f}  max={sdf_np.max():.6f}")
        print(f"           mean={sdf_np.mean():.6f}  median={float(np.median(sdf_np)):.6f}")
        print(f"           |sdf| p95={p95:.6f}  p99={p99:.6f}")
        print(f"Expected limit sqrt(3)*h = {limit:.6f}")
        print(f"Nodes exceeding limit: {n_exceed} ({pct_exceed:.2f}%)")
        if pct_exceed > 1.0:
            bad_idx = np.where(np.abs(sdf_np) > limit)[0]
            print(f"First 10 offenders (index, sdf, coord):")
            for bi in bad_idx[:10]:
                print(f"  [{bi}] sdf={sdf_np[bi]:.4f}  coord={nodes_flat[int(bi)].cpu().tolist()}")
        print("==============================")

    # ------- scatter back to (n_cells, n_nodes) layout -------
    sdf_all = sdf_unique[inverse]                  # (n_cells*n_nodes,)
    return sdf_all.reshape(n_cells, n_nodes_per_cell).to(dtype=torch.float64, device="cpu")


def reorder_refined_sdf_for_nn(d_values: torch.Tensor, refined_subcells: int) -> torch.Tensor:
    """Put x-major refined SDF samples in the corner-first NN feature order."""
    if refined_subcells < 1:
        raise ValueError("refined_subcells must be at least 1")
    if refined_subcells == 1:
        return d_values

    node_count = refined_subcells + 1
    expected_width = node_count ** 3
    if d_values.ndim != 2 or d_values.shape[1] != expected_width:
        raise ValueError(
            f"Expected d_values with shape (N, {expected_width}); got {tuple(d_values.shape)}."
        )

    corner_xyz = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=torch.long,
        device=d_values.device,
    )
    corner_indices = (
        corner_xyz[:, 0] * refined_subcells * node_count * node_count
        + corner_xyz[:, 1] * refined_subcells * node_count
        + corner_xyz[:, 2] * refined_subcells
    )
    non_corner = torch.ones(expected_width, dtype=torch.bool, device=d_values.device)
    non_corner[corner_indices] = False
    return torch.cat((d_values[:, corner_indices], d_values[:, non_corner]), dim=1)





def prepare_nn_inputs(
    grid,
    volume,
    partial_indices,
    material,
    poly_order: int,
    d_all: torch.Tensor | None = None,
    clamp_to_synthetic_domain: bool = False,
    verbose: bool = False,
    refined_subcells: int = 1,
    sdf_batch_size: int = 128,
    sdf_device: torch.device | str | None = None,
    global_sign: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns (d_normalized, nu_cut, h_ref).

    d_normalized : (N, 8) float64 – signed distances normalized to [0, 1] via
                   d = -sdf / (2*sqrt(3)*h_ref) + 0.5
                   (0.5 = on surface, 1.0 = one h_ref inside, 0.0 = one h_ref outside)
    nu_cut       : (N,) float64 – Poisson ratio broadcast to one value per cell
    h_ref        : float – grid element edge length (grid.dx, uniform grid assumed)

    compute_nn_cutcell_stiffness undoes the K normalization with ke *= E * h_ref,
    so d must already be in the normalized space when passed to it.
    """
    partial_indices_t = as_index_tensor_cpu(partial_indices)
    n = int(partial_indices_t.numel())
    h_ref = float(grid.dx)

    if n == 0:
        d_empty  = torch.empty((0, 8), dtype=torch.float64)
        nu_empty = torch.empty((0,),   dtype=torch.float64)
        return d_empty, nu_empty, h_ref

    if poly_order != 1:
        raise ValueError("NN cut-cell inference currently expects poly_order=1.")
    if refined_subcells < 1:
        raise ValueError("refined_subcells must be at least 1")
    if sdf_batch_size < 1:
        raise ValueError("sdf_batch_size must be at least 1")

    sign_triangles = None
    sign_device = torch.device(sdf_device) if sdf_device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if global_sign:
        sign_triangles, _ = extract_volume_triangles(volume, sign_device)

    def compute_sdf_batched(nodes_per_axis: int) -> torch.Tensor:
        batches = []
        for start in range(0, n, sdf_batch_size):
            batch_indices = partial_indices_t[start:start + sdf_batch_size]
            d_batch = compute_cell_node_sdf_csc(
                grid,
                volume,
                batch_indices,
                nodes_per_axis,
                device=sdf_device,
                debug=verbose,
            )
            if global_sign:
                centers = as_tensor(grid.centers, dtype=torch.float64, device="cpu").reshape(-1, 3)[batch_indices]
                offsets = HexGrid._local_node_offsets(grid.dx, grid.dy, grid.dz, poly_order=nodes_per_axis, device="cpu")
                nodes = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
                inside = points_inside_mesh_ray_parity(nodes, sign_triangles).reshape_as(d_batch)
                d_batch = d_batch.abs() * torch.where(inside, -1.0, 1.0)
            batches.append(d_batch)
        return torch.cat(batches, dim=0)

    # --- Raw SDF (physical units) ---
    if d_all is not None:
        d_raw = d_all[partial_indices_t].to(dtype=torch.float64)
    else:
        d_raw = compute_sdf_batched(poly_order)  # Only used if refined_subcells == 1

    # Normalize
    scale = 2.0 * math.sqrt(3.0) * h_ref
    d_normalized = -d_raw / scale + 0.5

    # --- Refined SDF (if requested) ---
    if refined_subcells > 1:
        # Compute ALL nodes (corners + non-corners) at refined resolution
        d_refined_raw = compute_sdf_batched(refined_subcells)
        d_refined = -d_refined_raw / scale + 0.5  # Normalize refined SDF

        d_normalized = reorder_refined_sdf_for_nn(d_refined, refined_subcells)
    nu_cut = torch.full((n,), float(material.nu), dtype=torch.float64)

    return d_normalized, nu_cut, h_ref

def write_geometry_cutcell_training_h5(
    *,
    output_h5,
    group_name: str,
    grid,
    volume,
    material,
    poly_order: int = 1,
    order: int = 2,
    device: str = "cuda",
    lower_threshold=None,
    upper_threshold=None,
    batch_size: int = 128,
    export_chunk_size: int = 512,
    replace_existing: bool = True,
    refined_subcells: int = 1,
    queso_elements=None,
    include_sdf: bool = True,
) -> dict[str, Any]:
    """Write training data using compiled QuESo integration points directly.

    ``queso_elements`` may reuse elements from a prior QuESo run on the same
    mesh and grid. Their quadrature still supplies every stiffness entry.
    """
    import h5py
    import json
    import os
    import numpy as np
    from ..internal_fem_kernel.geometry import HexGrid
    from ..internal_fem_kernel.integration.element_stiffness import Stiffness
    from ..internal_fem_kernel.queso_integration import _load_queso_dynamically

    queso_python_path = os.environ.get(
        "QUESO_PYTHON_PATH",
        "c:/Users/z0058p1u/Documents/code/QuESo/out/install/win",
    )
    _load_queso_dynamically(queso_python_path)
    try:
        from QuESoPythonModule.PyQuESo import PyQuESo
    except ImportError:
        import sys
        if queso_python_path not in sys.path:
            sys.path.insert(0, queso_python_path)
        from QuESoPythonModule.PyQuESo import PyQuESo

    if int(poly_order) != 1:
        raise ValueError("poly_order must be 1.")
    if refined_subcells < 1:
        raise ValueError("refined_subcells must be at least 1")
    if refined_subcells > 1 and not output_h5.name.startswith("refined_"):
        output_h5 = output_h5.with_name(f"refined_{output_h5.name}")

    bounds     = grid.bounds
    resolution = grid.resolution
    spacing = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
    if not np.allclose(spacing, spacing[0], rtol=1.0e-12, atol=0.0):
        raise ValueError(f"Uniform grid spacing required; received {tuple(spacing)}.")
    h_ref = float(spacing.mean())   # element edge length (uniform)

    # ── Run QuESo, or reuse a prior run on this exact mesh/grid ───────────────
    stl_path = temp_settings_path = None
    if queso_elements is None:
        stl_path = output_h5.parent / f"temp_{group_name}.stl"
        temp_settings_path = output_h5.parent / f"QuESoSettings_temp_{group_name}.json"
        volume.mesh.export(stl_path)

        settings = {
            "general_settings": {
                "input_filename": str(stl_path),
                "echo_level": 0,
                "write_output_to_file": False,
            },
            "background_grid_settings": {
                "grid_type": "b_spline_grid",
                "lower_bound_xyz": [float(x) for x in bounds[0]],
                "upper_bound_xyz": [float(x) for x in bounds[1]],
                "lower_bound_uvw": [0.0, 0.0, 0.0],
                "upper_bound_uvw": [1.0, 1.0, 1.0],
                "polynomial_order": [2, 2, 2],
                "number_of_elements": [int(x) for x in resolution],
            },
            "trimmed_quadrature_rule_settings": {"moment_fitting_residual": 1e-08},
            "non_trimmed_quadrature_rule_settings": {"integration_method": "Gauss"},
        }
        temp_settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        pq = PyQuESo(str(temp_settings_path))
        pq.Run()
        trimmed_elements = [element for element in pq.GetElements() if element.IsTrimmed()]
    else:
        trimmed_elements = list(queso_elements)
        if any(not element.IsTrimmed() for element in trimmed_elements):
            raise ValueError("queso_elements must contain only trimmed elements")
    n_partial        = len(trimmed_elements)
    print(f"Found {n_partial} cut cells via QuESo.")

    # QuESo e.ID() uses its own internal element numbering that does NOT
    # match the Python HexGrid flat index (i*ny*nz + j*nz + k).
    # Compute the correct flat index from each element's physical centre so
    # that centers_all[flat_idx] returns the right cell centre.
    _b0 = np.array(bounds[0], dtype=np.float64)
    _nx, _ny, _nz = int(resolution[0]), int(resolution[1]), int(resolution[2])
    _dx, _dy, _dz = float(grid.dx), float(grid.dy), float(grid.dz)

    def _queso_to_flat(ele) -> int:
        lo  = np.array(ele.LowerBoundXYZ(), dtype=np.float64)
        hi  = np.array(ele.UpperBoundXYZ(), dtype=np.float64)
        ctr = (lo + hi) * 0.5
        # x[i] = b0[0] + (i + 0.5)*dx  =>  i = (ctr - b0 - dx/2) / dx
        ii = int(round((ctr[0] - _b0[0] - _dx * 0.5) / _dx))
        jj = int(round((ctr[1] - _b0[1] - _dy * 0.5) / _dy))
        kk = int(round((ctr[2] - _b0[2] - _dz * 0.5) / _dz))
        assert 0 <= ii < _nx and 0 <= jj < _ny and 0 <= kk < _nz, (
            f"QuESo element centre {ctr.tolist()} maps to out-of-range index "
            f"({ii}, {jj}, {kk}) for grid {_nx}×{_ny}×{_nz}"
        )
        return ii * _ny * _nz + jj * _nz + kk

    partial_indices_list = [_queso_to_flat(e) for e in trimmed_elements]

    try:
        if temp_settings_path is not None:
            temp_settings_path.unlink(missing_ok=True)
        if stl_path is not None:
            stl_path.unlink(missing_ok=True)
    except Exception:
        pass

    if n_partial == 0:
        return {"output_h5": output_h5, "group_name": group_name,
                "num_cells": 0, "h_ref": h_ref,
                "d_min": float("nan"), "d_max": float("nan"), "diagnostics": {}}

    # ── Compute nodal signed distances for trimmed elements only ──────────────
    # prepare_nn_inputs returns (d_values, h_ref) for the given indices.
    # d_values is in physical coordinates (same units as h_ref).
    # Index order matches partial_indices_list exactly.
    n_distance_nodes = (refined_subcells + 1) ** 3
    if include_sdf:
        partial_indices_t = torch.tensor(partial_indices_list, dtype=torch.int64, device=device)
        d_normalized_t, _, _ = prepare_nn_inputs(
            grid=grid,
            volume=volume,
            partial_indices=partial_indices_t,
            material=material,
            poly_order=poly_order,
            verbose=False,
            refined_subcells=refined_subcells,
            sdf_batch_size=batch_size,
            sdf_device=device,
        )
        d_np_normalized = d_normalized_t.detach().cpu().numpy().astype(np.float64)
        assert d_np_normalized.shape == (n_partial, n_distance_nodes), (
            f"d_values shape {d_np_normalized.shape} does not match n_partial={n_partial}."
        )
    else:
        d_np_normalized = np.empty((n_partial, n_distance_nodes), dtype=np.float64)
    # ── Compute stiffness matrices for trimmed elements ───────────────────────
    tri_upper    = np.triu_indices(24)
    offsets_unit = HexGrid._local_node_offsets(1.0, 1.0, 1.0, 1, 'cpu').numpy()
    v_grid       = float(
        (bounds[1][0] - bounds[0][0]) *
        (bounds[1][1] - bounds[0][1]) *
        (bounds[1][2] - bounds[0][2])
    )

    coords_list, xi_pts_list, w_phys_list, element_sizes = [], [], [], []
    for ele in trimmed_elements:
        c_min   = np.array(ele.LowerBoundXYZ())
        c_max   = np.array(ele.UpperBoundXYZ())
        u_min   = np.array(ele.LowerBoundUVW())
        u_max   = np.array(ele.UpperBoundUVW())

        xi_pts, w_phys = [], []
        for p in ele.GetIntegrationPoints():
            xi_pts.append(2.0 * (np.array([p.X(), p.Y(), p.Z()]) - u_min) / (u_max - u_min) - 1.0)
            w_phys.append(p.Weight() * v_grid)
        coords_list.append((c_min + c_max) / 2.0 + (c_max - c_min) * offsets_unit)
        xi_pts_list.append(torch.as_tensor(np.asarray(xi_pts), dtype=torch.float64))
        w_phys_list.append(torch.as_tensor(np.asarray(w_phys), dtype=torch.float64))
        element_sizes.append(float((c_max - c_min)[0]))

    ke_phys = Stiffness.elements_from_queso_points_batched(
        np.asarray(coords_list), material.C, xi_pts_list, w_phys_list,
        poly_order=1, device='cpu', element_batch_size=export_chunk_size,
    ).cpu().numpy()
    # Normalise by E * h_ele so stored K is dimensionless and O(1).
    ke_phys /= float(material.E) * np.asarray(element_sizes)[:, None, None]
    stiff_ut_np = ke_phys[:, tri_upper[0], tri_upper[1]]
    assert stiff_ut_np.shape == (n_partial, 300)

    # Paranoia: d and K row counts must agree with partial_indices_list
    assert len(d_np_normalized) == len(stiff_ut_np) == len(partial_indices_list), (
        f"Row count mismatch: d={len(d_np_normalized)}, "
        f"K={len(stiff_ut_np)}, idx={len(partial_indices_list)}"
    )

    if include_sdf:
        print(
            f"d_normalized: min={d_np_normalized.min():.4f}  max={d_np_normalized.max():.4f}  "
            f"(expect [0, 1] for cut cells)"
        )

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    with h5py.File(output_h5, "a") as h5:
        if replace_existing and group_name in h5:
            del h5[group_name]

        if group_name not in h5:
            grp = h5.create_group(group_name)
            grp.create_dataset("flat_index",   data=np.array(partial_indices_list, dtype=np.int64), maxshape=(None,))
            if include_sdf:
                grp.create_dataset("d_values", data=d_np_normalized, maxshape=(None, n_distance_nodes))
            grp.create_dataset("stiffness_ut", data=stiff_ut_np,     maxshape=(None, 300))
        else:
            grp      = h5[group_name]
            old_size = len(grp["flat_index"])
            new_size = old_size + n_partial
            grp["flat_index"]  .resize((new_size,))
            if include_sdf:
                grp["d_values"].resize((new_size, n_distance_nodes))
            grp["stiffness_ut"].resize((new_size, 300))
            grp["flat_index"]  [old_size:] = np.array(partial_indices_list, dtype=np.int64)
            if include_sdf:
                grp["d_values"][old_size:] = d_np_normalized
            grp["stiffness_ut"][old_size:] = stiff_ut_np

        grp.attrs.update({
            "integration_order":       int(order),
            "poly_order":              int(poly_order),
            "young_modulus":           1.0,
            "source_young_modulus":    float(material.E),
            "source_h_ref":            float(h_ref),
            "poisson_ratio":           float(material.nu),
            "storage":                 "upper_triangular",
            "matrix_size":             24,
            "source":                  "real_geometry_pipeline_queso",
            "node_order":              "fem_basic_p1",
            "refined_subcells":        int(refined_subcells),
            "n_distance_nodes":        int(n_distance_nodes),
            "refined_node_order":      "eight legacy corners followed by regular-grid non-corner nodes",
            "d_value_convention":      "d = -sdf / (2*sqrt(3)*h_ref) + 0.5  (0.5=surface, 1.0=inside, 0.0=outside)",
            "stiffness_normalization": "ke_partial / (E * h_ele)",
        })

    print(f"Done. Wrote {n_partial} cut-cell entries to {output_h5}:{group_name}")

    return {
        "output_h5":  output_h5,
        "group_name": group_name,
        "num_cells":  n_partial,
        "h_ref":      h_ref,
        "d_min":      float(d_np_normalized.min()),
        "d_max":      float(d_np_normalized.max()),
        "diagnostics": {"queso_elements_count": n_partial},
    }

def compute_nn_cutcell_stiffness(
    d_cut: torch.Tensor,
    nu_cut: torch.Tensor,
    nn_engine: Any,
    material,
    partial_indices,
    h_ref: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    partial_indices_t = as_index_tensor_cpu(partial_indices)
    if int(d_cut.shape[0]) == 0:
        return [], torch.empty((0,), dtype=torch.int64)

    inference_device = torch.device(nn_engine.device)
    model_device = next(nn_engine.model.parameters()).device
    if model_device != inference_device:
        raise RuntimeError(
            f"NN model is on {model_device}, but the inference engine is configured for {inference_device}."
        )
    include_nu = bool(getattr(getattr(nn_engine, "model", None), "include_nu", True))
    d_input = d_cut.to(device=inference_device, dtype=nn_engine.dtype)
    nu_input = nu_cut.to(device=inference_device, dtype=nn_engine.dtype) if include_nu else None
    if inference_device.type == "cuda" and (
        not d_input.is_cuda or (nu_input is not None and not nu_input.is_cuda)
    ):
        raise RuntimeError("CUDA NN inference was requested, but NN feature inputs are not on CUDA.")
    predicted_device = nn_engine.predict_torch(d_input, nu_input)
    if inference_device.type == "cuda" and not predicted_device.is_cuda:
        raise RuntimeError("CUDA NN inference was requested, but the NN prediction is not on CUDA.")
    out_t = predicted_device.to(dtype=torch.float64, device="cpu")

    ke_cut = output_to_k24_batch(out_t)

    ke_cut *= float(h_ref) * float(material.E)

    if ke_cut.ndim == 2:
        ke_cut = ke_cut[None, :, :]

    return [ke_cut[i] for i in range(ke_cut.shape[0])], partial_indices_t

