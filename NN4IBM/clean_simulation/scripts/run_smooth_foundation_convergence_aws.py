from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import torch

from clean_simulation.scripts.studies._notebook import execute_notebook_cell, load_notebook_cells

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "clean_simulation" / "notebooks" / "05_local_stiffness_convergence.ipynb"
OUTPUT_DIR = Path(
    os.environ.get(
        "HANGING_SPHERE_OUTPUT_DIR",
        ROOT / "clean_simulation" / "outputs" / "studies" / "sphere" / "smooth_foundation_aws",
    )
).expanduser().resolve()
FRAMEWORKS = ("naive", "weighted", "queso", "nn")


def smooth_foundation_matrix(
    case, active_indices, cut_indices, node_ids, dtype: torch.dtype, cell_weights, element_nodes_for_cells
) -> torch.Tensor:
    """Create a compact C1 radial elastic foundation on the shared active support."""
    resolution = tuple(int(value) for value in case.grid.resolution)
    node_side = resolution[0] + 1
    node_count = node_side ** 3
    cell_count = int(np.prod(resolution))
    cell_fractions = torch.zeros(cell_count, dtype=torch.float64)
    cell_fractions[active_indices] = 1.0
    cut_indices = torch.as_tensor(cut_indices, dtype=torch.int64)
    cell_fractions[cut_indices] = cell_weights(case)[
        cut_indices
    ]
    cell_volume = case.grid.dx * case.grid.dy * case.grid.dz
    node_volumes = torch.zeros(node_count, dtype=torch.float64)
    element_nodes = element_nodes_for_cells(
        active_indices, case.grid.resolution, poly_order=case.poly_order, device="cpu"
    )
    node_volumes.index_add_(
        0,
        element_nodes.reshape(-1),
        (cell_fractions[active_indices] * (cell_volume / 8.0)).repeat_interleave(8),
    )

    node_ids = torch.as_tensor(node_ids, dtype=torch.int64)
    i = node_ids // (node_side * node_side)
    j = (node_ids // node_side) % node_side
    k = node_ids % node_side
    lower = torch.as_tensor(case.grid.bounds[0], dtype=torch.float64)
    coordinates = lower + torch.stack((i, j, k), dim=1).to(torch.float64) * torch.as_tensor(
        (case.grid.dx, case.grid.dy, case.grid.dz), dtype=torch.float64
    )
    radius = torch.linalg.vector_norm(coordinates, dim=1)
    taper = ((5.5 - radius) / 1.5).clamp(0.0, 1.0)
    taper = taper * taper * (3.0 - 2.0 * taper)
    diagonal = 40_000.0 * taper * node_volumes[node_ids]
    dofs = (3 * node_ids[:, None] + torch.arange(3, dtype=torch.int64)).reshape(-1)
    values = diagonal.repeat_interleave(3).to(dtype=dtype)
    return torch.sparse_coo_tensor(
        torch.stack((dofs, dofs)), values, (3 * node_count, 3 * node_count), dtype=dtype
    ).coalesce()


def solution_path(output_dir: Path, framework: str, resolution: int) -> Path:
    return output_dir / "smooth_foundation" / f"hanging_sphere_{framework}_displacement_{resolution}cubed.npz"


def ensure_seed_surface(output_dir: Path) -> None:
    import trimesh

    subdivisions = int(os.environ.get("HANGING_SPHERE_SURFACE_SUBDIVISIONS", "6"))
    seed_path = output_dir / "queso_matrix_free_128cubed" / "study_geometry_surface.stl"
    expected_faces = 20 * 4 ** subdivisions
    if seed_path.exists() and len(trimesh.load(seed_path, force="mesh").faces) == expected_faces:
        return
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=subdivisions, radius=7.5).export(seed_path)
    print(f"Seeded subdivision-{subdivisions} sphere surface with {expected_faces:,} faces.", flush=True)


def main() -> None:
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Missing canonical notebook: {NOTEBOOK_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_DIR / "smooth_foundation"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HANGING_SPHERE_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ.setdefault("HANGING_SPHERE_RESOLUTIONS", "8,16,32,64,128,256,512")
    os.environ.setdefault("HANGING_SPHERE_FRAMEWORKS", ",".join(FRAMEWORKS))
    os.environ.setdefault("HANGING_SPHERE_DEVICE", "cuda")
    os.environ.setdefault("HANGING_SPHERE_CPU_THREADS", "6")
    os.environ.setdefault("HANGING_SPHERE_CG_MAX_ITER", "12000")
    os.environ.setdefault("HANGING_SPHERE_DTYPE", "float64")
    os.environ["HANGING_SPHERE_RUN_SOLVES"] = "0"
    ensure_seed_surface(OUTPUT_DIR)

    cells = load_notebook_cells(NOTEBOOK_PATH)
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}
    for cell_index in (1, 2, 3, 4, 6):
        execute_notebook_cell(NOTEBOOK_PATH, namespace, cells, cell_index)

    requested_frameworks = tuple(namespace["FRAMEWORKS"])
    resolutions = tuple(namespace["RESOLUTION_SCHEDULE"])
    for resolution in resolutions:
        pending = [
            framework
            for framework in requested_frameworks
            if not solution_path(OUTPUT_DIR, framework, resolution).exists()
        ]
        if not pending:
            print(f"{resolution}^3: all smooth-foundation checkpoints already exist.", flush=True)
            continue
        print(f"[smooth-foundation] {resolution}^3: preparing {', '.join(pending)}", flush=True)
        case, full_ke, full_indices, queso_cut_ke, cut_indices, active, shared_force, fractions = namespace[
            "build_hard_constraint_problem"
        ](resolution)
        node_ids = namespace["active_node_ids"](active.numpy(), resolution)
        foundation = smooth_foundation_matrix(
            case,
            active,
            cut_indices,
            node_ids,
            namespace["SOLVE_DTYPE"],
            namespace["physical_cell_weights"],
            namespace["active_element_nodes"],
        )
        nn_engine = None
        if "nn" in pending:
            nn_engine = namespace["load_nn_engine"](
                checkpoint_path=namespace["NN_CHECKPOINT"], device=namespace["SOLVE_DEVICE"], dtype="float32"
            )
        try:
            for framework in pending:
                cut_matrices = namespace["hard_constraint_cut_matrices"](
                    framework, case, full_ke, queso_cut_ke, cut_indices, fractions, nn_engine
                )
                operator = namespace["StructuredCompressedMatrixFreeElasticityOperator"](
                    full_ke, full_indices, cut_matrices, cut_indices, tuple(case.grid.resolution), foundation,
                    element_batch_size=32768, slab_cells=1,
                )
                solution, info = namespace["solve_matrix_free_cg"](
                    operator, shared_force, rtol=1.0e-6, atol=1.0e-8, max_iter=namespace["CG_MAX_ITER"],
                    check_interval=10, report_interval_seconds=30.0, max_seconds_per_iteration=None,
                )
                if not info["converged"]:
                    raise RuntimeError(f"{framework} {resolution}^3 smooth-foundation solve did not converge: {info}")
                np.savez_compressed(
                    solution_path(OUTPUT_DIR, framework, resolution),
                    u_nodes=solution.detach().cpu().numpy().reshape(-1, 3),
                    active_cells=active.cpu().numpy(),
                    active_nodes=namespace["active_node_ids"](active.cpu().numpy(), resolution),
                    solver_info=np.array([info], dtype=object), boundary_mode="smooth_radial_foundation",
                )
                print(f"[smooth-foundation] {resolution}^3 {framework} checkpoint saved: {info}", flush=True)
                del cut_matrices, operator, solution
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            del nn_engine, case, full_ke, full_indices, queso_cut_ke, cut_indices, active, shared_force, fractions, foundation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()