from __future__ import annotations

import h5py
from pathlib import Path
from typing import Any
import sys
import torch

from clean_simulation.internal_nn_runtime.geometry import write_geometry_cutcell_training_h5
from clean_simulation.internal_fem_kernel.constants import Material
from clean_simulation.internal_fem_kernel.geometry import HexGrid
from clean_simulation.internal_fem_kernel.volume_factory import VolumeFactory

def generate_geometry_training_dataset(
    *,
    geom_name: str,
    poisson_ratio: float = 0.3,
    integration_order: int = 2,
    poly_order: int = 1,
    resolution: tuple[int, int, int] = (10, 10, 10),
    device: str | None = None,
    workspace_root: str | Path | None = None,
    output_h5: str | Path | None = None,
    replace_existing: bool = True,
    refined_subcells: int = 1,
    sdf_batch_size: int = 128,
    grid_phase: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Generates the exact physical cut-cell configurations and stiffness matrices for a real STL geometry.
    
    This replaces the legacy synthetic marching-cube generators with direct volume integration 
    on actual STL structures using our robust internal FEM kernels.
    """
    root = Path(workspace_root or Path.cwd()).resolve()
    while root != root.parent and not (root / 'clean_simulation').exists():
        root = root.parent
    if not (root / 'clean_simulation').exists():
        raise FileNotFoundError('clean_simulation directory not found from workspace root')

    # Load geometry file
    geom_path = root / 'clean_simulation' / 'geometries' / f'{geom_name}.stl'
    if not geom_path.exists():
        # Fallback to searching
        geom_path = root / 'clean_simulation' / 'geometries' / f'{geom_name.upper()}.stl'
    if not geom_path.exists():
        geom_path = root / 'clean_simulation' / 'geometries' / f'{geom_name.lower()}.stl'
    if not geom_path.exists():
        raise FileNotFoundError(f"STL geometry file not found: {geom_name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    volume = VolumeFactory.from_file(geom_path)
    volume.device = device
    
    # Adapt the bounds to the geometry bounding box + 10% padding
    # new constraint: even if it requires to extend the domain, the cells need to all be cubes
    # This prevents coordinates mismatch where geometry lies entirely outside the grid
    mesh_bounds = volume.mesh.bounds
    import numpy as np # Ensure numpy is imported
    mins = mesh_bounds[0]
    maxs = mesh_bounds[1]
    extents = maxs - mins
    padding = extents * 0.10
    
    # Ensuring minimum padding
    padding = np.maximum(padding, 1e-3)
    
    bounds_lo_raw = mins - padding
    bounds_hi_raw = maxs + padding
    extents_padded = bounds_hi_raw - bounds_lo_raw

    # To guarantee cubic elements (dx == dy == dz), we must select a uniform cell size (h)
    # and adjust the grid bounds so they span exactly an integer number of cells of size h.
    # At resolution (nx, ny, nz) = resolution for standard tests:
    # We find the maximum dynamic padded dimension, divided by nx, as our reference cell size h.
    max_padded_extent = float(np.max(extents_padded))
    h = max_padded_extent / float(resolution[0])

    # Ensure bounds_lo is adjusted centered or extended so that the grid envelopes the CAD and remains cubical
    centers_raw = (bounds_lo_raw + bounds_hi_raw) / 2.0
    half_grid_span = (h * np.array(resolution)) / 2.0
    bounds_lo = tuple((centers_raw - half_grid_span).tolist())
    bounds_hi = tuple((centers_raw + half_grid_span).tolist())

    grid_phase_array = np.asarray(grid_phase, dtype=np.float64)
    if grid_phase_array.shape != (3,) or np.any(np.abs(grid_phase_array) > 0.5):
        raise ValueError("grid_phase must contain three fractions in [-0.5, 0.5]")
    phase_shift = grid_phase_array * h
    bounds_lo = tuple((np.asarray(bounds_lo) + phase_shift).tolist())
    bounds_hi = tuple((np.asarray(bounds_hi) + phase_shift).tolist())

    grid = HexGrid(
        bounds=(bounds_lo, bounds_hi),
        resolution=resolution,
        device=device,
    )
    material = Material(E=1.0, nu=float(poisson_ratio))

    group_name = f"integration_{integration_order}_poly_{poly_order}"
    
    if output_h5 is None:
        output_h5 = root / 'clean_simulation' / 'outputs' / 'data' / f'local_stiffness_matrices_{geom_name}_{integration_order}i_{poly_order}p_hybrid_stats.h5'
    else:
        output_h5 = Path(output_h5)

    if refined_subcells < 1:
        raise ValueError("refined_subcells must be at least 1")
    if refined_subcells > 1 and not output_h5.name.startswith("refined_"):
        output_h5 = output_h5.with_name(f"refined_{output_h5.name}")

    print(f"Generating physical cut-cells dataset for geometry='{geom_name}'...")
    print(f"  Source: {geom_path}")
    print(f"  Output H5: {output_h5}")
    print(f"  Group: {group_name}")
    print(
        f"  Params: poisson={poisson_ratio}, integration_order={integration_order}, "
        f"refined_subcells={refined_subcells}, grid_phase={tuple(grid_phase_array)}"
    )

    result = write_geometry_cutcell_training_h5(
        output_h5=output_h5,
        group_name=group_name,
        grid=grid,
        volume=volume,
        material=material,
        poly_order=poly_order,
        order=integration_order,
        device=device,
        replace_existing=replace_existing,
        refined_subcells=refined_subcells,
        batch_size=sdf_batch_size,
    )
    
    print(f"Generation successful! Created {result['num_cells']} real physical samples.\n")
    return result
