
#!/usr/bin/env python3
"""Headless, env-var-driven runner for the hanging-sphere cut-cell convergence study.

This reconstructs the canonical (non-exploratory) cell sequence of
05_local_stiffness_convergence.ipynb -- cells 1, 3, 4, 5, 6, 7, 8, 9, 11, 12 --
as a plain script so it can run unattended on a remote GPU box, and adds
per-solve wall-clock timing that is written incrementally to a CSV so you can
watch progress and pull back timing even from a job that is still running or
that failed partway through.

Designed to be launched twice, once per GPU, e.g.:

    CUDA_VISIBLE_DEVICES=0 HANGING_SPHERE_RESOLUTIONS=512 \
        HANGING_SPHERE_REFERENCE_RESOLUTION=512 python3 run_study.py

    CUDA_VISIBLE_DEVICES=1 HANGING_SPHERE_RESOLUTIONS=8,16,32,64,128,256 \
        HANGING_SPHERE_REFERENCE_RESOLUTION=512 python3 run_study.py

Every environment variable from the original notebook is honored
(HANGING_SPHERE_*). A few new ones were added for the remote/512 run --
search for "NEW:" below.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def log(message, started_at=None):
    elapsed = '' if started_at is None else f' [{time.perf_counter() - started_at:.1f}s]'
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] [hanging-sphere] {message}{elapsed}', flush=True)


# --------------------------------------------------------------------------
# Cell 1 + Cell 4: paths, resolution schedule, reference resolution
# --------------------------------------------------------------------------
root = Path.cwd().resolve()
while root != root.parent and not (root / 'clean_simulation').exists():
    root = root.parent
if not (root / 'clean_simulation').exists():
    raise RuntimeError('Run this script from the repository root or a descendant directory.')
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

SUPPORT_OUTPUT_NAMES = {'planar_cap': 'planar_cap_plane_hard_constraints', 'smooth_foundation': 'smooth_foundation'}
OUTPUT_DIR = Path(os.environ.get(
    'HANGING_SPHERE_OUTPUT_DIR',
    root / 'clean_simulation' / 'outputs' / 'sphere_displacement_convergence_aws_512',
)).expanduser().resolve()
SUPPORT_MODE = os.environ.get('HANGING_SPHERE_SUPPORT_MODE', 'smooth_foundation').strip().lower()
if SUPPORT_MODE not in SUPPORT_OUTPUT_NAMES:
    raise ValueError(f'Unknown HANGING_SPHERE_SUPPORT_MODE={SUPPORT_MODE!r}.')
SUPPORT_OUTPUT_DIR = OUTPUT_DIR / SUPPORT_OUTPUT_NAMES[SUPPORT_MODE]
PLANAR_CAP_OUTPUT_DIR = OUTPUT_DIR / SUPPORT_OUTPUT_NAMES['planar_cap']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLANAR_CAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION_SCHEDULE = tuple(int(value) for value in os.environ.get(
    'HANGING_SPHERE_RESOLUTIONS', '4,8,16,32,64,128,256',
).split(','))
REFERENCE_RESOLUTION = int(os.environ.get(
    'HANGING_SPHERE_REFERENCE_RESOLUTION', str(max(RESOLUTION_SCHEDULE)),
))
# NEW: reference resolution need not be part of THIS process's own schedule
# (e.g. the GPU-1 "rest of the schedule" job references 512 without solving it).
DRY_RUN = os.environ.get('HANGING_SPHERE_DRY_RUN', '0') == '1'
CONTINUE_ON_ERROR = os.environ.get('HANGING_SPHERE_CONTINUE_ON_ERROR', '0') == '1'
ELEMENT_BATCH_SIZE = int(os.environ.get('HANGING_SPHERE_ELEMENT_BATCH_SIZE', '32768'))

log(f'Output directory: {OUTPUT_DIR}')
log(f'Resolution schedule for this process: {RESOLUTION_SCHEDULE}; reference: {REFERENCE_RESOLUTION}^3')
log(f'Support mode: {SUPPORT_MODE}; dry run: {DRY_RUN}; element batch size: {ELEMENT_BATCH_SIZE}')

# --------------------------------------------------------------------------
# Cell 3: runtime / solver configuration
# --------------------------------------------------------------------------
SOLVE_DEVICE = os.environ.get('HANGING_SPHERE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
CPU_THREADS = int(os.environ.get('HANGING_SPHERE_CPU_THREADS', min(6, os.cpu_count() or 1)))
CG_MAX_ITER = int(os.environ.get('HANGING_SPHERE_CG_MAX_ITER', '4000'))
SOLVE_DTYPE = getattr(torch, os.environ.get('HANGING_SPHERE_DTYPE', 'float32'))
if SOLVE_DEVICE == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('HANGING_SPHERE_DEVICE=cuda was requested but CUDA is unavailable.')
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

queso_python_path = root / 'QuESo' / 'out' / 'install' / ('win' if os.name == 'nt' else 'linux')
if queso_python_path.exists() and str(queso_python_path) not in sys.path:
    sys.path.insert(0, str(queso_python_path))

from clean_simulation.bootstrap import configure_legacy_imports
configure_legacy_imports(root)
from clean_simulation.case import build_case_from_stl
from clean_simulation.internal_fem_kernel.runtime.assembly_ops import assemble_force_base
from clean_simulation.internal_fem_kernel.runtime.matrix_free import (
    StructuredCompressedMatrixFreeElasticityOperator,
    solve_matrix_free_cg,
)
from clean_simulation.internal_nn_runtime.cutcell_bridge import compute_nn_cutcell_stiffness, prepare_nn_inputs
from clean_simulation.nn_runtime import load_nn_engine
from internal_fem_kernel.assembly.stiffness import StiffnessAssembly
from internal_fem_kernel.core.grid_indexing import active_element_nodes

ALL_FRAMEWORKS = {
    'naive': 'Naive full-cell Gauss',
    'weighted': 'Geometric-volume-fraction weighted Gauss',
    'queso': 'QuESo quadrature',
    'nn': 'NN-cut-cell hybrid',
}
requested_frameworks = tuple(name.strip() for name in os.environ.get(
    'HANGING_SPHERE_FRAMEWORKS', ','.join(ALL_FRAMEWORKS),
).split(',') if name.strip())
unknown_frameworks = set(requested_frameworks) - set(ALL_FRAMEWORKS)
if unknown_frameworks or not requested_frameworks:
    raise ValueError(f'Invalid HANGING_SPHERE_FRAMEWORKS: {requested_frameworks}')
FRAMEWORKS = {name: ALL_FRAMEWORKS[name] for name in requested_frameworks}
NN_CHECKPOINT = Path(os.environ.get(
    'HANGING_SPHERE_NN_CHECKPOINT',
    root / 'clean_simulation' / 'outputs' / 'checkpoints' / 'source_sdf_rebuilt_queso_d10_f1024_all_crystals_10h.pt',
))
SPHERE_OVERRIDES = {
    'grid': {'bounds': [[-9.0, -9.0, -9.0], [9.0, 9.0, 9.0]]},
    'material': {'E': 1.0e6, 'nu': 0.3, 'density': 1.0},
    'integration': {'poly_order': 1, 'gauss_order': 2, 'moment_fitting_residual': 1.0e-8},
    'loads': {'gravity': [0.0, 0.0, -98.1]},
    'bc': {'top_cap_fraction': 0.05, 'constraint_axis': 2, 'constraint_side': 'max'},
}
gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'n/a'
log(f'Device: {SOLVE_DEVICE} ({gpu_name}); CPU threads: {torch.get_num_threads()}')
log(f'Frameworks: {", ".join(FRAMEWORKS)}')
log(f'NN checkpoint: {NN_CHECKPOINT} (exists: {NN_CHECKPOINT.exists()})')


# --------------------------------------------------------------------------
# Cell 5, 6, 7, 8, 9: shared geometry / quadrature / cache helpers
# --------------------------------------------------------------------------
def active_node_ids(active_cells, resolution):
    cells = np.asarray(active_cells, dtype=np.int64)
    node_side = resolution + 1
    i = cells // (resolution * resolution)
    j = (cells // resolution) % resolution
    k = cells % resolution
    base = (i * node_side + j) * node_side + k
    offsets = np.array((0, node_side * node_side, node_side * node_side + node_side, node_side, 1,
                        node_side * node_side + 1, node_side * node_side + node_side + 1, node_side + 1))
    return np.unique((base[:, None] + offsets).reshape(-1))


def strict_sphere_full_cells(case):
    nx, ny, nz = map(int, case.grid.resolution)
    sphere = case.config.get('sphere', {'center': (0.0, 0.0, 0.0), 'radius': 7.5})
    center = np.asarray(sphere['center'], dtype=np.float64)
    radius_squared = float(sphere['radius']) ** 2
    lower = np.asarray(case.grid.bounds[0], dtype=np.float64)
    spacing = np.asarray((case.grid.dx, case.grid.dy, case.grid.dz), dtype=np.float64)
    y = lower[1] + np.arange(ny + 1) * spacing[1]
    z = lower[2] + np.arange(nz + 1) * spacing[2]
    yy, zz = np.meshgrid(y, z, indexing='ij')
    slabs = []
    for i in range(nx):
        x0 = lower[0] + i * spacing[0]
        x1 = x0 + spacing[0]
        distance_squared = np.maximum((x0 - center[0]) ** 2, (x1 - center[0]) ** 2)
        distance_squared += np.maximum((yy[:-1, :-1] - center[1]) ** 2, (yy[1:, 1:] - center[1]) ** 2)
        distance_squared += np.maximum((zz[:-1, :-1] - center[2]) ** 2, (zz[1:, 1:] - center[2]) ** 2)
        local = np.flatnonzero(distance_squared < radius_squared).astype(np.int64)
        if local.size:
            slabs.append(local + i * ny * nz)
    return torch.from_numpy(np.concatenate(slabs) if slabs else np.empty(0, dtype=np.int64))


def point_cell_indices_and_weights(case, point_mask=None):
    points = torch.as_tensor(np.asarray(case.ips), dtype=torch.float64)
    if point_mask is not None:
        points = points[torch.as_tensor(point_mask, dtype=torch.bool)]
    coordinates = points[:, :3]
    weights = points[:, 3]
    lower = torch.as_tensor(case.grid.bounds[0], dtype=torch.float64)
    upper = torch.as_tensor(case.grid.bounds[1], dtype=torch.float64)
    normalized = coordinates.numel() and bool(torch.all((coordinates >= 0.0) & (coordinates <= 1.0)))
    if normalized:
        coordinates = lower + coordinates * (upper - lower)
        weights = weights * torch.prod(upper - lower)
    resolution = torch.as_tensor(case.grid.resolution, dtype=torch.int64)
    spacing = torch.as_tensor((case.grid.dx, case.grid.dy, case.grid.dz), dtype=torch.float64)
    ijk = torch.floor((coordinates - lower) / spacing).to(torch.int64)
    ijk = torch.minimum(torch.maximum(ijk, torch.zeros_like(ijk)), resolution - 1)
    indices = ijk[:, 0] * (resolution[1] * resolution[2]) + ijk[:, 1] * resolution[2] + ijk[:, 2]
    return indices, weights


def physical_cell_weights(case):
    cell_count = int(np.prod(case.grid.resolution))
    indices, weights = point_cell_indices_and_weights(case)
    if not len(indices):
        return torch.zeros(cell_count, dtype=torch.float64)
    volumes = torch.bincount(indices, weights=weights, minlength=cell_count)
    cell_volume = case.grid.dx * case.grid.dy * case.grid.dz
    return (volumes / cell_volume).clamp(0.0, 1.0)


def independent_sphere_volume_fractions(case, cut_indices, quadrature_order=12, chunk_size=16384):
    dtype = torch.float64
    device = cut_indices.device
    sphere = case.config.get('sphere', {'center': (0.0, 0.0, 0.0), 'radius': 7.5})
    center = torch.as_tensor(sphere['center'], dtype=dtype, device=device)
    radius = float(sphere['radius'])
    radius_squared = radius * radius
    spacing = torch.as_tensor((case.grid.dx, case.grid.dy, case.grid.dz), dtype=dtype, device=device)
    lower = torch.as_tensor(case.grid.bounds[0], dtype=dtype, device=device)
    nx, ny, nz = map(int, case.grid.resolution)
    dx, dy, dz = spacing
    cell_volume = dx * dy * dz
    points_np, weights_np = np.polynomial.legendre.leggauss(quadrature_order)
    points = torch.as_tensor(points_np, dtype=dtype, device=device)
    weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
    gx, gy = torch.meshgrid(points, points, indexing='ij')
    gwx, gwy = torch.meshgrid(weights, weights, indexing='ij')
    gx = gx.reshape(1, -1)
    gy = gy.reshape(1, -1)
    gw = (gwx * gwy).reshape(1, -1)
    n_cells = cut_indices.numel()
    fractions = torch.empty(n_cells, dtype=dtype, device=device)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        indices = cut_indices[start:end].to(torch.int64)
        i = indices // (ny * nz)
        j = (indices // nz) % ny
        k = indices % nz
        ijk = torch.stack((i, j, k), dim=1).to(dtype)
        cell_lower = lower + ijk * spacing
        cell_upper = cell_lower + spacing
        xmin, ymin, zmin = cell_lower[:, 0], cell_lower[:, 1], cell_lower[:, 2]
        xmax, ymax, zmax = cell_upper[:, 0], cell_upper[:, 1], cell_upper[:, 2]
        dmin = torch.where(
            center < cell_lower, cell_lower - center,
            torch.where(center > cell_upper, center - cell_upper, torch.zeros_like(cell_lower)),
        )
        min_dist_squared = torch.sum(dmin * dmin, dim=1)
        outside = min_dist_squared > radius_squared
        dmax = torch.maximum(torch.abs(cell_lower - center), torch.abs(cell_upper - center))
        max_dist_squared = torch.sum(dmax * dmax, dim=1)
        inside = max_dist_squared <= radius_squared
        boundary = ~(outside | inside)
        chunk_fractions = torch.zeros(indices.shape[0], dtype=dtype, device=device)
        chunk_fractions[inside] = 1.0
        boundary_ids = torch.nonzero(boundary, as_tuple=False).flatten()
        if boundary_ids.numel() > 0:
            bxmin, bxmax = xmin[boundary_ids], xmax[boundary_ids]
            bymin, bymax = ymin[boundary_ids], ymax[boundary_ids]
            bzmin, bzmax = zmin[boundary_ids], zmax[boundary_ids]
            x = 0.5 * (bxmax - bxmin)[:, None] * gx + 0.5 * (bxmax + bxmin)[:, None]
            y = 0.5 * (bymax - bymin)[:, None] * gy + 0.5 * (bymax + bymin)[:, None]
            sphere_dx = x - center[0]
            sphere_dy = y - center[1]
            remaining = radius_squared - sphere_dx * sphere_dx - sphere_dy * sphere_dy
            valid = remaining > 0.0
            half_height = torch.sqrt(torch.clamp_min(remaining, 0.0))
            sphere_zmin = center[2] - half_height
            sphere_zmax = center[2] + half_height
            intersection_zmin = torch.maximum(sphere_zmin, bzmin[:, None])
            intersection_zmax = torch.minimum(sphere_zmax, bzmax[:, None])
            z_length = torch.clamp_min(intersection_zmax - intersection_zmin, 0.0)
            z_length = torch.where(valid, z_length, torch.zeros_like(z_length))
            volume = torch.sum(z_length * gw, dim=1)
            volume *= 0.25 * (bxmax - bxmin) * (bymax - bymin)
            boundary_fraction = (volume / cell_volume).clamp(0.0, 1.0)
            chunk_fractions[boundary_ids] = boundary_fraction
        fractions[start:end] = chunk_fractions
    return fractions


def ensure_physical_sphere_stl(work_dir):
    import trimesh
    sphere = SPHERE_OVERRIDES.get('sphere', {'center': (0.0, 0.0, 0.0), 'radius': 7.5})
    center = np.asarray(sphere['center'], dtype=np.float64)
    radius = float(sphere['radius'])
    stl_path = work_dir / 'study_geometry_surface.stl'
    expected_bounds = np.vstack((center - radius, center + radius))
    cache_paths = (work_dir / 'queso_cut_points.npy', work_dir / 'queso_cut_stiffness.pt')
    if stl_path.is_file():
        mesh = trimesh.load(stl_path, force='mesh')
        if np.allclose(mesh.bounds, expected_bounds, atol=1.0e-8, rtol=0.0):
            log(f'Reusing cached physical sphere surface and QuESo artifacts in {work_dir.name}.')
            return stl_path
        if any(path.exists() for path in cache_paths):
            raise RuntimeError(f'{work_dir} has QuESo cache files but surface bounds mismatch.')
    if any(path.exists() for path in cache_paths):
        raise RuntimeError(f'{work_dir} has QuESo cache files without a geometrically valid surface.')
    work_dir.mkdir(parents=True, exist_ok=True)
    subdivisions = int(os.environ.get('HANGING_SPHERE_SURFACE_SUBDIVISIONS', '9'))
    trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius).export(stl_path)
    log(f'Created physical radius-{radius:g} sphere surface with {20 * 4 ** subdivisions:,} faces.')
    return stl_path


def build_hard_constraint_problem(resolution):
    overrides = {**SPHERE_OVERRIDES, 'grid': {**SPHERE_OVERRIDES['grid'], 'resolution': [resolution] * 3}}
    work_dir = OUTPUT_DIR / f'queso_matrix_free_{resolution}cubed'
    stl_path = ensure_physical_sphere_stl(work_dir)
    case = build_case_from_stl(
        stl_path=stl_path, overrides=overrides, workspace_root=root, work_dir=work_dir,
        queso_python_path=queso_python_path, queso_basis_order=2, cut_cells_only=False,
        integration_points_npy_path=work_dir / 'queso_all_points.npy', bounds_padding=0.0, device='cpu',
    )
    cache_path = work_dir / 'queso_all_cell_stiffness.pt'
    if cache_path.exists():
        cache = torch.load(cache_path, map_location='cpu', weights_only=True)
        queso_all_ke, queso_all_indices = cache['matrices'], cache['indices']
    else:
        queso_all_ke, queso_all_indices = StiffnessAssembly.compute_element_matrices_from_queso_points(
            grid=case.grid, material=case.material, ips=np.asarray(case.ips), poly_order=case.poly_order,
            device='cpu', element_batch_size=512,
        )
        torch.save({'matrices': queso_all_ke, 'indices': queso_all_indices}, cache_path)

    full_indices = strict_sphere_full_cells(case)
    is_cut_in_queso_all = torch.isin(queso_all_indices, full_indices, invert=True)
    cut_indices = queso_all_indices[is_cut_in_queso_all]
    queso_cut_ke = queso_all_ke[is_cut_in_queso_all]
    is_full_in_queso_all = torch.isin(queso_all_indices, full_indices)
    queso_full_indices = queso_all_indices[is_full_in_queso_all]
    queso_full_ke = queso_all_ke[is_full_in_queso_all]
    full_indices = full_indices[~torch.isin(full_indices, cut_indices)]
    active = torch.sort(torch.cat((full_indices, cut_indices))).values

    full_seed = torch.zeros(np.prod(case.grid.resolution), dtype=torch.bool)
    full_seed[full_indices[0]] = True
    full_ke = StiffnessAssembly.compute_inside_element_matrices(
        grid=case.grid, material=case.material, inside=full_seed, order=case.g_order,
        device='cpu', poly_order=case.poly_order,
    )[0][0]

    cut_force = assemble_force_base(
        grid=case.grid, active_indices=cut_indices, ips=case.ips, config=case.config,
        poly_order=case.poly_order, device='cpu',
    )
    full_force = torch.zeros_like(cut_force).reshape(-1, 3)
    full_nodes = active_element_nodes(full_indices, case.grid.resolution, poly_order=case.poly_order, device='cpu')
    gravity = torch.as_tensor(case.config['loads']['gravity'], dtype=torch.float64) * float(case.config['material']['density'])
    full_force.index_add_(0, full_nodes.reshape(-1), gravity.repeat(len(full_indices) * 8, 1) * (case.grid.dx * case.grid.dy * case.grid.dz / 8.0))

    fractions = independent_sphere_volume_fractions(case, cut_indices)
    log(f'{resolution}^3 subcell volume fractions: min={fractions.min():.6g}; median={fractions.median():.6g}; max={fractions.max():.6g}')
    return case, full_ke, full_indices, queso_cut_ke, cut_indices, active, full_force.reshape(-1) + cut_force, fractions, queso_full_ke, queso_full_indices


# --------------------------------------------------------------------------
# Previously-deleted helper functions (restored by the user)
# --------------------------------------------------------------------------
def hard_constraint_cut_matrices(framework, case, full_ke, queso_cut_ke, cut_indices, fractions, nn_engine=None):
    if framework == 'queso':
        return queso_cut_ke
    if framework == 'naive':
        return full_ke.expand(len(cut_indices), -1, -1).clone()
    if framework == 'weighted':
        return full_ke.expand(len(cut_indices), -1, -1).clone() * fractions[:, None, None]
    if nn_engine is None:
        raise ValueError('The NN framework requires a loaded NN engine.')
    d_cut, nu_cut, h_ref = prepare_nn_inputs(
        grid=case.grid, volume=case.volume, partial_indices=cut_indices,
        material=case.material, poly_order=case.poly_order,
    )
    matrices, nn_indices = compute_nn_cutcell_stiffness(
        d_cut=d_cut, nu_cut=nu_cut, nn_engine=nn_engine, material=case.material,
        partial_indices=cut_indices, h_ref=h_ref,
    )
    result = full_ke.expand(len(cut_indices), -1, -1).clone()
    result[torch.searchsorted(cut_indices, nn_indices.to(torch.int64))] = torch.stack(matrices).cpu()
    return result


def hard_constraint_full_matrices(framework, full_indices, full_ke, queso_full_ke=None, queso_full_indices=None):
    if framework != 'queso':
        return full_ke.expand(len(full_indices), -1, -1).clone()
    if queso_full_ke is None or queso_full_indices is None:
        raise ValueError("QuESo's own inner-cell matrices are required for framework='queso'.")
    positions = torch.searchsorted(queso_full_indices, full_indices)
    if not torch.equal(queso_full_indices[positions], full_indices):
        raise ValueError('QuESo inner-cell index set does not cover all strict full_indices.')
    return queso_full_ke[positions]


def framework_support(framework, case, strict_full_indices, queso_cut_indices):
    full_indices = strict_full_indices
    cut_indices = queso_cut_indices
    active_indices = torch.sort(torch.cat((full_indices, cut_indices))).values
    return full_indices, cut_indices, active_indices


def framework_body_force(case, full_indices, cut_indices):
    cut_force = assemble_force_base(
        grid=case.grid, active_indices=cut_indices, ips=case.ips, config=case.config,
        poly_order=case.poly_order, device='cpu',
    )
    full_force = torch.zeros_like(cut_force).reshape(-1, 3)
    if len(full_indices):
        full_nodes = active_element_nodes(full_indices, case.grid.resolution, poly_order=case.poly_order, device='cpu')
        gravity = torch.as_tensor(case.config['loads']['gravity'], dtype=torch.float64)
        gravity *= float(case.config['material']['density'])
        full_force.index_add_(
            0, full_nodes.reshape(-1),
            gravity.repeat(len(full_indices) * 8, 1) * (case.grid.dx * case.grid.dy * case.grid.dz / 8.0),
        )
    return full_force.reshape(-1) + cut_force


# --------------------------------------------------------------------------
# Cell 11: boundary-condition helpers
# --------------------------------------------------------------------------
PLANAR_CAP_MODE = 'planar_cap_plane_hard_constraints'
SMOOTH_FOUNDATION_MODE = 'smooth_radial_foundation'


def support_solution_path(framework, resolution):
    return SUPPORT_OUTPUT_DIR / f'hanging_sphere_{framework}_displacement_{resolution}cubed.npz'


def checkpoint_matches_shared_support(framework, resolution, expected_active_indices):
    path = support_solution_path(framework, resolution)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            archived_active_indices = np.unique(archive['active_cells'].astype(np.int64, copy=False))
    except (KeyError, OSError, ValueError):
        return False
    return np.array_equal(archived_active_indices, expected_active_indices.cpu().numpy())


def planar_cap_hard_constraint_dofs(case, active_indices):
    axis = int(case.config['bc']['constraint_axis'])
    sphere = case.config.get('sphere', {'center': (0.0, 0.0, 0.0), 'radius': 7.5})
    center = float(sphere['center'][axis])
    radius = float(sphere['radius'])
    cap_start = center + radius - 2.0 * radius * float(case.config['bc']['top_cap_fraction'])
    node_ids = torch.unique(active_element_nodes(
        active_indices, case.grid.resolution, poly_order=case.poly_order, device='cpu',
    ).reshape(-1))
    node_side = int(case.grid.resolution[axis]) + 1
    node_axis = node_ids % node_side if axis == 2 else (node_ids // node_side) % node_side if axis == 1 else node_ids // (node_side * node_side)
    lower = float(case.grid.bounds[0][axis])
    spacing = (case.grid.dx, case.grid.dy, case.grid.dz)[axis]
    coordinates = lower + node_axis.to(torch.float64) * spacing
    cap_nodes = node_ids[torch.isclose(coordinates, torch.as_tensor(cap_start, dtype=coordinates.dtype), atol=1.0e-12, rtol=0.0)]
    if not len(cap_nodes):
        raise RuntimeError(f'Nested planar cap at {cap_start:.6f} selected no active nodes.')
    return (3 * cap_nodes[:, None] + torch.arange(3, dtype=torch.int64)).reshape(-1), cap_start


def smooth_foundation_matrix(case, active_indices, cut_indices):
    resolution = tuple(int(value) for value in case.grid.resolution)
    node_side = resolution[0] + 1
    node_count = node_side ** 3
    cell_count = int(np.prod(resolution))
    cell_fractions = torch.zeros(cell_count, dtype=torch.float64)
    cell_fractions[active_indices] = 1.0
    cut_indices = torch.as_tensor(cut_indices, dtype=torch.int64)
    cell_fractions[cut_indices] = physical_cell_weights(case)[cut_indices]
    cell_volume = case.grid.dx * case.grid.dy * case.grid.dz
    node_ids = torch.as_tensor(active_node_ids(active_indices.numpy(), resolution[0]), dtype=torch.int64)
    node_volumes = torch.zeros(node_count, dtype=torch.float64)
    element_nodes = active_element_nodes(active_indices, case.grid.resolution, poly_order=case.poly_order, device='cpu')
    node_volumes.index_add_(
        0, element_nodes.reshape(-1),
        (cell_fractions[active_indices] * (cell_volume / 8.0)).repeat_interleave(8),
    )
    i = node_ids // (node_side * node_side)
    j = (node_ids // node_side) % node_side
    k = node_ids % node_side
    lower = torch.as_tensor(case.grid.bounds[0], dtype=torch.float64)
    spacing = torch.as_tensor((case.grid.dx, case.grid.dy, case.grid.dz), dtype=torch.float64)
    coordinates = lower + torch.stack((i, j, k), dim=1).to(torch.float64) * spacing
    radius = torch.linalg.vector_norm(coordinates, dim=1)
    taper = ((5.5 - radius) / 1.5).clamp(0.0, 1.0)
    taper = taper * taper * (3.0 - 2.0 * taper)
    diagonal = 40_000.0 * taper * node_volumes[node_ids]
    dofs = (3 * node_ids[:, None] + torch.arange(3, dtype=torch.int64)).reshape(-1)
    values = diagonal.repeat_interleave(3).to(dtype=SOLVE_DTYPE)
    return torch.sparse_coo_tensor(
        torch.stack((dofs, dofs)), values, (3 * node_count, 3 * node_count), dtype=SOLVE_DTYPE,
    ).coalesce()


def support_boundary_terms(case, active_indices, cut_indices):
    if SUPPORT_MODE == 'planar_cap':
        constrained_dofs, cap_start = planar_cap_hard_constraint_dofs(case, active_indices)
        return None, constrained_dofs, {'cap_start': cap_start}
    foundation = smooth_foundation_matrix(case, active_indices, cut_indices)
    active_nodes = int(active_node_ids(active_indices.numpy(), int(case.grid.resolution[0])).size)
    return foundation, None, {'foundation_active_nodes': active_nodes, 'foundation_stiffness': 40_000.0}


# --------------------------------------------------------------------------
# Cell 12: exact per-element operator for QuESo's own inner-cell matrices
# --------------------------------------------------------------------------
class ExactElementMatrixFreeElasticityOperator:
    def __init__(self, element_matrices, active_indices, resolution, nitsche_matrix=None, constrained_dofs=None, element_batch_size=32768):
        self.device = torch.device(SOLVE_DEVICE)
        self.dtype = SOLVE_DTYPE
        self.element_matrices = torch.as_tensor(element_matrices, dtype=self.dtype, device=self.device)
        self.active_indices = torch.as_tensor(active_indices, dtype=torch.int64, device=self.device).reshape(-1)
        self.resolution = tuple(resolution)
        self.node_count = (self.resolution[0] + 1) * (self.resolution[1] + 1) * (self.resolution[2] + 1)
        self.nitsche_matrix = None if nitsche_matrix is None else nitsche_matrix.to(device=self.device, dtype=self.dtype).coalesce()
        self.constrained_dofs = None if constrained_dofs is None else torch.as_tensor(constrained_dofs, dtype=torch.int64, device=self.device).unique()
        self.element_batch_size = element_batch_size
        self.element_nodes = active_element_nodes(self.active_indices, self.resolution, poly_order=1, device=self.device)

    def matvec(self, vector):
        vector = torch.as_tensor(vector, dtype=self.dtype, device=self.device).reshape(-1)
        displacements = vector.reshape(self.node_count, 3)
        result = torch.zeros_like(displacements)
        for start in range(0, len(self.active_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.active_indices))
            nodes = self.element_nodes[start:stop]
            local_u = displacements[nodes].reshape(stop - start, 24)
            local_result = torch.bmm(self.element_matrices[start:stop], local_u.unsqueeze(-1)).squeeze(-1)
            result.index_add_(0, nodes.reshape(-1), local_result.reshape(-1, 3))
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            result += torch.sparse.mm(self.nitsche_matrix, vector[:, None]).reshape(-1)
        if self.constrained_dofs is not None:
            result[self.constrained_dofs] = vector[self.constrained_dofs]
        return result

    def diagonal(self):
        result = torch.zeros((self.node_count, 3), dtype=self.dtype, device=self.device)
        local_diagonal = self.element_matrices.diagonal(dim1=1, dim2=2).reshape(-1, 8, 3)
        for start in range(0, len(self.active_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.active_indices))
            result.index_add_(0, self.element_nodes[start:stop].reshape(-1), local_diagonal[start:stop].reshape(-1, 3))
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            indices = self.nitsche_matrix.indices()
            diagonal = indices[0] == indices[1]
            result.index_add_(0, indices[0, diagonal], self.nitsche_matrix.values()[diagonal])
        if self.constrained_dofs is not None:
            result[self.constrained_dofs] = 1.0
        return torch.where(result.abs() > 1.0e-20, result, torch.ones_like(result))


# --------------------------------------------------------------------------
# NEW: timing instrumentation. Every solved (framework, resolution) pair
# appends one row here and the whole table is rewritten to CSV after every
# solve, so you always have a live, resumable timing record even if a later
# resolution OOMs or the job is killed.
# --------------------------------------------------------------------------
TIMING_CSV = SUPPORT_OUTPUT_DIR / f'timing_{os.environ.get("HANGING_SPHERE_JOB_TAG", "job")}.csv'
TIMING_ROWS = []


def save_timing():
    if not TIMING_ROWS:
        return
    with open(TIMING_CSV, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TIMING_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(TIMING_ROWS)


def solve_support_frameworks(resolution):
    log(f'{resolution}^3: validating requested checkpoints against the shared support')
    problem_started = time.perf_counter()
    (case, full_ke, strict_full_indices, queso_cut_ke, queso_cut_indices, active, shared_force,
     fractions, queso_full_ke, queso_full_indices) = build_hard_constraint_problem(resolution)
    problem_seconds = time.perf_counter() - problem_started
    pending = [framework for framework in FRAMEWORKS if not checkpoint_matches_shared_support(framework, resolution, active)]
    if not pending:
        log(f'{resolution}^3: all {SUPPORT_MODE} checkpoints already match the shared support.')
        del case, full_ke, strict_full_indices, queso_cut_ke, queso_cut_indices, active, shared_force, fractions, queso_full_ke, queso_full_indices
        gc.collect()
        return
    if DRY_RUN:
        log(f'{resolution}^3: DRY RUN -- would solve {pending}; skipping actual solve.')
        del case, full_ke, strict_full_indices, queso_cut_ke, queso_cut_indices, active, shared_force, fractions, queso_full_ke, queso_full_indices
        gc.collect()
        return

    boundary_started = time.perf_counter()
    foundation, constrained_dofs, support_metadata = support_boundary_terms(case, active, queso_cut_indices)
    boundary_seconds = time.perf_counter() - boundary_started
    nn_engine = load_nn_engine(checkpoint_path=NN_CHECKPOINT, device=SOLVE_DEVICE, dtype='float32') if 'nn' in pending else None
    try:
        for framework in pending:
            row = {
                'job_tag': os.environ.get('HANGING_SPHERE_JOB_TAG', 'job'),
                'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', 'n/a'),
                'framework': framework,
                'resolution': resolution,
                'problem_build_seconds': round(problem_seconds, 4),
                'boundary_term_seconds': round(boundary_seconds, 4),
                'status': 'running',
                'cg_iterations': None,
                'converged': None,
                'full_cells': None,
                'cut_cells': None,
                'active_dofs': None,
                'solve_seconds': None,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            started = time.perf_counter()
            try:
                full_indices, cut_indices, active_indices = framework_support(framework, case, strict_full_indices, queso_cut_indices)
                if not torch.equal(active_indices, active):
                    raise AssertionError(f'{framework} does not use the shared full-plus-cut active support.')
                row['full_cells'] = len(full_indices)
                row['cut_cells'] = len(cut_indices)
                cut_matrices = hard_constraint_cut_matrices(framework, case, full_ke, queso_cut_ke, cut_indices, fractions, nn_engine)
                framework_force = framework_body_force(case, full_indices, cut_indices)
                if framework == 'queso':
                    full_matrices = hard_constraint_full_matrices('queso', full_indices, full_ke, queso_full_ke, queso_full_indices)
                    operator = ExactElementMatrixFreeElasticityOperator(
                        torch.cat((full_matrices, cut_matrices), dim=0), torch.cat((full_indices, cut_indices), dim=0),
                        tuple(case.grid.resolution), foundation, constrained_dofs=constrained_dofs,
                        element_batch_size=ELEMENT_BATCH_SIZE,
                    )
                else:
                    operator = StructuredCompressedMatrixFreeElasticityOperator(
                        full_ke, full_indices, cut_matrices, cut_indices, tuple(case.grid.resolution), foundation,
                        element_batch_size=ELEMENT_BATCH_SIZE, slab_cells=1, constrained_dofs=constrained_dofs,
                    )
                row['active_dofs'] = 3 * (int(case.grid.resolution[0]) + 1) ** 3
                solution, info = solve_matrix_free_cg(
                    operator, framework_force, rtol=1.0e-6, atol=1.0e-8, max_iter=CG_MAX_ITER,
                    check_interval=10, report_interval_seconds=30.0, max_seconds_per_iteration=None,
                )
                if not info['converged']:
                    raise RuntimeError(f'{framework} {resolution}^3 solve did not converge: {info}')
                solve_seconds = time.perf_counter() - started
                np.savez_compressed(
                    support_solution_path(framework, resolution), u_nodes=solution.detach().cpu().numpy().reshape(-1, 3),
                    active_cells=active_indices.numpy(), active_nodes=active_node_ids(active_indices.numpy(), resolution),
                    runtime_seconds=solve_seconds, solver_info=np.array([info], dtype=object),
                    boundary_mode=PLANAR_CAP_MODE if SUPPORT_MODE == 'planar_cap' else SMOOTH_FOUNDATION_MODE,
                    support_mode=SUPPORT_MODE, **support_metadata,
                )
                row.update({
                    'status': 'converged', 'converged': True, 'cg_iterations': info.get('iterations'),
                    'solve_seconds': round(solve_seconds, 4),
                })
                log(f'{resolution}^3 {framework} checkpoint saved: {info}', started)
                del full_indices, cut_indices, active_indices, cut_matrices, framework_force, operator, solution
            except Exception as error:
                row['status'] = f'failed: {type(error).__name__}: {error}'
                row['solve_seconds'] = round(time.perf_counter() - started, 4)
                log(f'{resolution}^3 {framework} FAILED after {row["solve_seconds"]}s: {error}')
                log(traceback.format_exc())
                if not CONTINUE_ON_ERROR:
                    TIMING_ROWS.append(row)
                    save_timing()
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            TIMING_ROWS.append(row)
            save_timing()
    finally:
        del nn_engine, case, full_ke, strict_full_indices, queso_cut_ke, queso_cut_indices, active, shared_force, fractions, queso_full_ke, queso_full_indices, foundation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def main():
    log(f'Starting study: resolutions={RESOLUTION_SCHEDULE}; frameworks={list(FRAMEWORKS)}')
    for resolution in RESOLUTION_SCHEDULE:
        resolution_started = time.perf_counter()
        solve_support_frameworks(resolution)
        log(f'Finished all pending frameworks at {resolution}^3', resolution_started)
    save_timing()
    log(f'Study complete. Timing table: {TIMING_CSV}')
    if TIMING_ROWS:
        display_df = pd.DataFrame(TIMING_ROWS)
        log('\n' + display_df.to_string(index=False))


if __name__ == '__main__':
    main()

