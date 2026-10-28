"""Cache-cold preparation timing study for an arbitrary watertight STL mesh."""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.bootstrap import configure_legacy_imports
from clean_simulation.case import build_case_from_stl
from clean_simulation.internal_fem_kernel.queso_integration import map_integration_points_to_elements
from clean_simulation.internal_nn_runtime.cutcell_bridge import prepare_nn_inputs
from clean_simulation.scripts.studies.blade.geometry import cube_bounds
from clean_simulation.scripts.studies.blade.timing import elapsed


def main() -> None:
    output_dir = Path(os.environ["BLADE_TIMING_OUTPUT_DIR"]).expanduser().resolve()
    stl_path = Path(os.environ["BLADE_STL_PATH"]).expanduser().resolve()
    resolutions = tuple(int(value) for value in os.environ.get("BLADE_RESOLUTIONS", "16,32,64,128").split(","))
    device = os.environ.get("HANGING_SPHERE_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Blade narrow-band SDF timing study.")
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)

    import trimesh

    mesh = trimesh.load(stl_path, file_type="stl", force="mesh")
    if not mesh.is_watertight:
        raise ValueError(f"Blade mesh must be watertight: {stl_path}")
    bounds = cube_bounds(np.asarray(mesh.bounds, dtype=np.float64))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Blade faces={len(mesh.faces):,}; bounds={bounds}; resolutions={resolutions}", flush=True)

    configure_legacy_imports(ROOT)
    from clean_simulation.internal_fem_kernel.assembly.stiffness import StiffnessAssembly

    rows: list[dict[str, object]] = []
    for resolution in resolutions:
        work_dir = output_dir / f"blade_queso_{resolution}cubed"
        if work_dir.exists():
            raise RuntimeError(f"Cache-cold timing requires a new directory, but it exists: {work_dir}")
        overrides = {
            "grid": {"bounds": bounds, "resolution": [resolution] * 3},
            "material": {"E": 1.0e6, "nu": 0.3, "density": 1.0},
            "integration": {"poly_order": 1, "gauss_order": 2, "moment_fitting_residual": 1.0e-8},
        }

        case_started = time.perf_counter()
        case = build_case_from_stl(
            stl_path=stl_path,
            overrides=overrides,
            workspace_root=ROOT,
            work_dir=work_dir,
            queso_python_path=ROOT / "QuESo" / "out" / "install" / "linux",
            queso_basis_order=2,
            cut_cells_only=False,
            bounds_padding=0.0,
            device="cpu",
        )
        case_seconds = elapsed(case_started)
        metadata = case.queso_metadata

        matrix_started = time.perf_counter()
        matrices, indices = StiffnessAssembly.compute_element_matrices_from_queso_points(
            grid=case.grid,
            material=case.material,
            ips=np.asarray(case.ips),
            poly_order=case.poly_order,
            device="cpu",
            element_batch_size=512,
        )
        matrix_seconds = elapsed(matrix_started)

        ip_coordinates = np.asarray(case.ips)[:, :3]
        bounds_array = np.asarray(case.grid.bounds, dtype=np.float64)
        if np.all((ip_coordinates >= -1.0e-12) & (ip_coordinates <= 1.0 + 1.0e-12)) and not np.all(
            (bounds_array >= -1.0e-12) & (bounds_array <= 1.0 + 1.0e-12)
        ):
            ip_coordinates = bounds_array[0] + ip_coordinates * (bounds_array[1] - bounds_array[0])
        mapped_indices, _ = map_integration_points_to_elements(
            ip_coordinates, case.grid.bounds, case.grid.resolution
        )
        trimmed_mask = np.asarray(metadata["trimmed_ip_mask"], dtype=bool)
        cut_indices = torch.from_numpy(np.unique(mapped_indices[trimmed_mask]).astype(np.int64))

        sdf_started = time.perf_counter()
        _, _, _ = prepare_nn_inputs(
            grid=case.grid,
            volume=case.volume,
            partial_indices=cut_indices,
            material=case.material,
            poly_order=case.poly_order,
            sdf_device=device,
            sdf_batch_size=2048,
        )
        sdf_seconds = elapsed(sdf_started)

        rows.append({
            "resolution": resolution,
            "mesh_faces": len(mesh.faces),
            "grid_setup_and_queso_total_seconds": case_seconds,
            "queso_native_classification_quadrature_seconds": metadata["queso_run_time"],
            "queso_point_selection_collection_seconds": metadata["point_extraction_time"],
            "queso_inside_point_collection_seconds": metadata["inside_collect_time"],
            "queso_trimmed_point_collection_seconds": metadata["trimmed_collect_time"],
            "queso_weight_filter_seconds": metadata["weight_filter_time"],
            "queso_points_kept": metadata["total_ip_count_kept"],
            "queso_trimmed_cells": len(cut_indices),
            "queso_element_matrix_integration_seconds": matrix_seconds,
            "nn_csc_narrow_band_sdf_seconds": sdf_seconds,
            "nn_sdf_method": "compute_cell_node_sdf_csc",
            "matrix_count": len(indices),
        })
        output = output_dir / "blade_preparation_timing.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{resolution}^3 complete; saved {output}", flush=True)
        del case, matrices, indices, cut_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()