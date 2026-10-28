"""Independent displacement convergence study for a watertight STL geometry."""

from __future__ import annotations

import csv
import gc
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
from clean_simulation.internal_nn_runtime.cutcell_bridge import compute_nn_cutcell_stiffness, prepare_nn_inputs
from clean_simulation.nn_runtime import load_nn_engine


NN_FRAMEWORKS = {
    "nn": {
        "checkpoint": "source_sdf_rebuilt_queso_d10_f1024_all_crystals_10h.pt",
        "refined_subcells": 1,
        "label": "NN-cut-cell hybrid",
    },
    "nn_27": {
        "checkpoint": "aws_rich_sdf/nn_27_best.pt",
        "refined_subcells": 2,
        "label": "NN-27 cut-cell hybrid",
    },
    "nn_64": {
        "checkpoint": "aws_rich_sdf/nn_64_best.pt",
        "refined_subcells": 3,
        "label": "NN-64 cut-cell hybrid",
    },
}
FRAMEWORKS = ("naive", "weighted", "queso", *NN_FRAMEWORKS)


def cube_bounds(mesh_bounds: np.ndarray, padding_fraction: float = 0.10) -> list[list[float]]:
    center = mesh_bounds.mean(axis=0)
    half_width = float(np.max(mesh_bounds[1] - mesh_bounds[0])) * (0.5 + padding_fraction)
    return [(center - half_width).tolist(), (center + half_width).tolist()]


def physical_points_and_weights(case: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(case.ips, dtype=np.float64)
    coordinates = points[:, :3]
    weights = points[:, 3]
    bounds = np.asarray(case.grid.bounds, dtype=np.float64)
    if np.all((coordinates >= -1.0e-12) & (coordinates <= 1.0 + 1.0e-12)) and not np.all(
        (bounds >= -1.0e-12) & (bounds <= 1.0 + 1.0e-12)
    ):
        coordinates = bounds[0] + coordinates * (bounds[1] - bounds[0])
        weights = weights * np.prod(bounds[1] - bounds[0])
    indices, _ = map_integration_points_to_elements(coordinates, case.grid.bounds, case.grid.resolution)
    return coordinates, weights, indices


def active_node_ids(active_cells: torch.Tensor, resolution: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = resolution
    cells = active_cells.detach().cpu().numpy().astype(np.int64, copy=False)
    i = cells // (ny * nz)
    j = (cells // nz) % ny
    k = cells % nz
    base = (i * (ny + 1) + j) * (nz + 1) + k
    offsets = np.array((0, (ny + 1) * (nz + 1), (ny + 1) * (nz + 1) + (nz + 1), nz + 1, 1,
                        (ny + 1) * (nz + 1) + 1, (ny + 1) * (nz + 1) + (nz + 1) + 1, nz + 2))
    return np.unique((base[:, None] + offsets).reshape(-1))


class ExactElementMatrixFreeElasticityOperator:
    def __init__(self, element_matrices, active_indices, resolution, constrained_dofs, device, element_batch_size=32768):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.element_matrices = torch.as_tensor(element_matrices, dtype=self.dtype, device=self.device)
        self.active_indices = torch.as_tensor(active_indices, dtype=torch.int64, device=self.device)
        self.resolution = tuple(int(value) for value in resolution)
        self.node_count = int(np.prod(np.asarray(self.resolution) + 1))
        self.constrained_dofs = torch.as_tensor(constrained_dofs, dtype=torch.int64, device=self.device).unique()
        self.element_batch_size = element_batch_size
        from internal_fem_kernel.core.grid_indexing import active_element_nodes
        self.element_nodes = active_element_nodes(self.active_indices, self.resolution, poly_order=1, device=self.device)

    def matvec(self, vector):
        vector = torch.as_tensor(vector, dtype=self.dtype, device=self.device).reshape(-1)
        displacement = vector.reshape(self.node_count, 3)
        result = torch.zeros_like(displacement)
        for start in range(0, len(self.active_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.active_indices))
            nodes = self.element_nodes[start:stop]
            local = displacement[nodes].reshape(stop - start, 24)
            local_result = torch.bmm(self.element_matrices[start:stop], local.unsqueeze(-1)).squeeze(-1)
            result.index_add_(0, nodes.reshape(-1), local_result.reshape(-1, 3))
        result = result.reshape(-1)
        result[self.constrained_dofs] = vector[self.constrained_dofs]
        return result

    def diagonal(self):
        result = torch.zeros((self.node_count, 3), dtype=self.dtype, device=self.device)
        local_diagonal = self.element_matrices.diagonal(dim1=1, dim2=2).reshape(-1, 8, 3)
        for start in range(0, len(self.active_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.active_indices))
            result.index_add_(0, self.element_nodes[start:stop].reshape(-1), local_diagonal[start:stop].reshape(-1, 3))
        result = result.reshape(-1)
        result[self.constrained_dofs] = 1.0
        return torch.where(result.abs() > 1.0e-20, result, torch.ones_like(result))


def main() -> None:
    configure_legacy_imports(ROOT)
    from internal_fem_kernel.assembly.stiffness import StiffnessAssembly
    from clean_simulation.internal_fem_kernel.runtime.assembly_ops import assemble_force_base
    from clean_simulation.internal_fem_kernel.runtime.matrix_free import StructuredCompressedMatrixFreeElasticityOperator, solve_matrix_free_cg

    output_dir = Path(os.environ["BLADE_CONVERGENCE_OUTPUT_DIR"]).resolve()
    stl_path = Path(os.environ["BLADE_STL_PATH"]).resolve()
    resolutions = tuple(int(value) for value in os.environ.get("BLADE_RESOLUTIONS", "16,32,64,128").split(","))
    frameworks = tuple(value for value in os.environ.get("BLADE_FRAMEWORKS", ",".join(FRAMEWORKS)).split(",") if value)
    device = os.environ.get("BLADE_DEVICE", "cuda")
    cg_max_iter = int(os.environ.get("BLADE_CG_MAX_ITER", "12000"))
    queso_basis_order = int(os.environ.get("BLADE_QUESO_BASIS_ORDER", "1"))
    if queso_basis_order != 1:
        raise ValueError("Blade Hex8 stiffness requires BLADE_QUESO_BASIS_ORDER=1.")
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Blade convergence study.")

    import trimesh
    mesh = trimesh.load(stl_path, file_type="stl", force="mesh")
    if not mesh.is_watertight:
        raise ValueError("Blade geometry must be watertight.")
    bounds = cube_bounds(np.asarray(mesh.bounds, dtype=np.float64))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []

    for resolution in resolutions:
        work_dir = output_dir / f"queso_{resolution}cubed"
        overrides = {
            "grid": {"bounds": bounds, "resolution": [resolution] * 3},
            "material": {"E": 1.0e6, "nu": 0.3, "density": 1.0},
            "integration": {"poly_order": 1, "gauss_order": 2, "moment_fitting_residual": 1.0e-8},
            "loads": {"gravity": [0.0, 0.0, -98.1]},
            "bc": {"top_cap_fraction": 0.05, "constraint_axis": 2, "constraint_side": "max"},
        }
        case = build_case_from_stl(stl_path=stl_path, overrides=overrides, workspace_root=ROOT, work_dir=work_dir,
                                   queso_python_path=ROOT / "QuESo" / "out" / "install" / "linux", queso_basis_order=queso_basis_order,
                                   cut_cells_only=False, bounds_padding=0.0, device="cpu")
        all_ke, active_indices = StiffnessAssembly.compute_element_matrices_from_queso_points(
            case.grid, case.material, np.asarray(case.ips), poly_order=1, device="cpu", element_batch_size=512
        )
        _, weights, mapped_indices = physical_points_and_weights(case)
        cell_count = resolution ** 3
        cell_volumes = np.bincount(mapped_indices, weights=weights, minlength=cell_count)
        fractions = torch.from_numpy((cell_volumes / (case.grid.dx * case.grid.dy * case.grid.dz)).clip(0.0, 1.0))
        trimmed_mask = np.asarray(case.queso_metadata["trimmed_ip_mask"], dtype=bool)
        cut_indices = torch.from_numpy(np.unique(mapped_indices[trimmed_mask]).astype(np.int64))
        full_mask = ~torch.isin(active_indices, cut_indices)
        full_indices = active_indices[full_mask]
        cut_mask = ~full_mask
        queso_full_ke, queso_cut_ke = all_ke[full_mask], all_ke[cut_mask]
        if not torch.equal(active_indices, torch.sort(torch.cat((full_indices, cut_indices))).values):
            raise RuntimeError("Blade active-cell classification is inconsistent with QuESo matrices.")
        full_seed = torch.zeros(cell_count, dtype=torch.bool)
        full_seed[full_indices[0]] = True
        full_ke = StiffnessAssembly.compute_inside_element_matrices(
            case.grid, case.material, full_seed, order=case.g_order, device="cpu", poly_order=1
        )[0][0]
        force = assemble_force_base(
            grid=case.grid,
            active_indices=active_indices,
            ips=case.ips,
            config=case.config,
            poly_order=1,
            device="cpu",
        )

        nodes = torch.from_numpy(active_node_ids(active_indices, tuple(case.grid.resolution)))
        node_side = resolution + 1
        z_index = nodes % node_side
        z_coordinates = float(case.grid.bounds[0][2]) + z_index.to(torch.float64) * case.grid.dz
        z_limit = float(mesh.bounds[1, 2] - 0.05 * mesh.extents[2])
        constrained_nodes = nodes[z_coordinates >= z_limit]
        if not len(constrained_nodes):
            constrained_nodes = nodes[z_coordinates == z_coordinates.max()]
        constrained_dofs = (3 * constrained_nodes[:, None] + torch.arange(3)).reshape(-1)

        nn_engines = {
            framework: load_nn_engine(
                checkpoint_path=(
                    ROOT
                    / "clean_simulation"
                    / "outputs"
                    / "models"
                    / "archive"
                    / "checkpoints"
                    / config["checkpoint"]
                ),
                device=device,
                dtype="float32",
            )
            for framework, config in NN_FRAMEWORKS.items()
            if framework in frameworks
        }
        try:
            for framework in frameworks:
                target = checkpoint_dir / f"blade_{framework}_displacement_{resolution}cubed.npz"
                if target.exists():
                    continue
                started = time.perf_counter()
                if framework == "queso":
                    operator = ExactElementMatrixFreeElasticityOperator(
                        all_ke, active_indices, case.grid.resolution, constrained_dofs, device
                    )
                else:
                    if framework == "naive":
                        cut_matrices = full_ke.expand(len(cut_indices), -1, -1).clone()
                    elif framework == "weighted":
                        cut_matrices = full_ke.expand(len(cut_indices), -1, -1).clone() * fractions[cut_indices, None, None]
                    elif framework in NN_FRAMEWORKS:
                        nn_config = NN_FRAMEWORKS[framework]
                        d_cut, nu_cut, h_ref = prepare_nn_inputs(
                            case.grid,
                            case.volume,
                            cut_indices,
                            case.material,
                            1,
                            refined_subcells=nn_config["refined_subcells"],
                            sdf_device=device,
                            sdf_batch_size=2048,
                        )
                        predicted, nn_indices = compute_nn_cutcell_stiffness(
                            d_cut,
                            nu_cut,
                            nn_engines[framework],
                            case.material,
                            cut_indices,
                            h_ref,
                        )
                        cut_matrices = full_ke.expand(len(cut_indices), -1, -1).clone()
                        cut_matrices[torch.searchsorted(cut_indices, nn_indices.to(torch.int64))] = torch.stack(predicted).cpu()
                    else:
                        raise ValueError(framework)
                    operator = StructuredCompressedMatrixFreeElasticityOperator(full_ke, full_indices, cut_matrices, cut_indices,
                        tuple(case.grid.resolution), constrained_dofs=constrained_dofs, element_batch_size=32768, slab_cells=1)
                solution, info = solve_matrix_free_cg(operator, force.to(device=device, dtype=torch.float64), rtol=1.0e-9, atol=1.0e-10,
                                                      max_iter=cg_max_iter, check_interval=10, report_interval_seconds=30.0, max_seconds_per_iteration=None)
                if not info["converged"]:
                    raise RuntimeError(f"{framework} {resolution}^3 did not converge: {info}")
                np.savez_compressed(target, u_nodes=solution.cpu().numpy().reshape(-1, 3), active_cells=active_indices.numpy(),
                                    active_nodes=active_node_ids(active_indices, tuple(case.grid.resolution)), runtime_seconds=time.perf_counter() - started,
                                    solver_info=np.array([info], dtype=object), mesh_bounds=mesh.bounds, constrained_nodes=constrained_nodes.numpy())
                records.append({"framework": framework, "resolution": resolution, "status": "converged", "seconds": time.perf_counter() - started})
                print(f"{framework} {resolution}^3 converged", flush=True)
                del operator, solution
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        finally:
            del nn_engines, case, all_ke, active_indices, full_indices, cut_indices, queso_full_ke, queso_cut_ke, force
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if records:
            with (output_dir / "progress.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)


if __name__ == "__main__":
    main()
