"""Run the hanging-sphere convergence study with QuESo element-wise interior stiffness on AWS."""

from __future__ import annotations

import gc
import json
import os
import time
import ctypes
from pathlib import Path

import numpy as np
import torch

from clean_simulation.internal_fem_kernel.core.grid_indexing import active_element_nodes


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "clean_simulation" / "notebooks" / "05_local_stiffness_convergence.ipynb"
OUTPUT_DIR = Path(
    os.environ.get(
        "HANGING_SPHERE_OUTPUT_DIR",
        ROOT / "clean_simulation" / "outputs" / "studies" / "sphere" / "queso_inner_cells_aws",
    )
).expanduser().resolve()
FRAMEWORKS = ("naive", "weighted", "queso", "nn")


def available_system_memory_bytes() -> int:
    """Return physical memory currently available to this Windows process."""
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.available_physical)


def assert_exact_queso_capacity(resolution: int) -> None:
    """Fail before QuESo allocates all per-element float64 stiffness matrices."""
    if os.environ.get("HANGING_SPHERE_ALLOW_UNSAFE_EXACT_QUESO") == "1":
        return

    domain_width = 18.0
    sphere_radius = 7.5
    cell_width = domain_width / resolution
    expanded_radius = sphere_radius + np.sqrt(3.0) * cell_width
    active_upper_bound = min(
        resolution ** 3,
        int(np.ceil((4.0 / 3.0) * np.pi * expanded_radius ** 3 / cell_width ** 3)),
    )
    matrix_bytes = active_upper_bound * 24 * 24 * np.dtype(np.float64).itemsize
    required_cpu_bytes = 3 * matrix_bytes + 4 * 1024 ** 3
    available_cpu_bytes = available_system_memory_bytes()

    device = torch.device(os.environ["HANGING_SPHERE_DEVICE"])
    required_gpu_bytes = int(1.2 * matrix_bytes)
    available_gpu_bytes = None
    if device.type == "cuda":
        free_gpu_bytes, _ = torch.cuda.mem_get_info(device)
        available_gpu_bytes = int(free_gpu_bytes)

    cpu_safe = available_cpu_bytes >= required_cpu_bytes
    gpu_safe = available_gpu_bytes is None or available_gpu_bytes >= required_gpu_bytes
    if cpu_safe and gpu_safe:
        return

    def gibibytes(value: int) -> str:
        return f"{value / 1024 ** 3:.2f} GiB"

    details = [
        f"{resolution}^3 exact all-cell QuESo is unsafe on this machine",
        f"estimated active-cell upper bound: {active_upper_bound:,}",
        f"one float64 element-matrix field: {gibibytes(matrix_bytes)}",
        f"required available CPU memory: {gibibytes(required_cpu_bytes)}; found {gibibytes(available_cpu_bytes)}",
    ]
    if available_gpu_bytes is not None:
        details.append(
            f"required free GPU memory: {gibibytes(required_gpu_bytes)}; found {gibibytes(available_gpu_bytes)}"
        )
    details.append("Use a larger AWS instance or set HANGING_SPHERE_ALLOW_UNSAFE_EXACT_QUESO=1 to bypass this guard.")
    raise RuntimeError(". ".join(details))


def execute_cell(namespace: dict[str, object], cell: dict[str, object]) -> None:
    source = "\n".join(cell["source"])
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)


class ExactElementMatrixFreeElasticityOperator:
    """Matrix-free operator for an element-wise global stiffness field."""

    def __init__(self, element_matrices, active_indices, resolution, nitsche_matrix=None, constrained_dofs=None, element_batch_size=32768):
        self.device = torch.device(os.environ["HANGING_SPHERE_DEVICE"])
        self.dtype = getattr(torch, os.environ["HANGING_SPHERE_DTYPE"])
        self.element_matrices = torch.as_tensor(element_matrices, dtype=self.dtype, device=self.device)
        self.active_indices = torch.as_tensor(active_indices, dtype=torch.int64, device=self.device).reshape(-1)
        self.resolution = tuple(int(value) for value in resolution)
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


def solution_path(namespace: dict[str, object], framework: str, resolution: int) -> Path:
    return namespace["SUPPORT_OUTPUT_DIR"] / f"hanging_sphere_{framework}_displacement_{resolution}cubed.npz"


def load_notebook_setup() -> dict[str, object]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(cell["source"])
        execute_cell(namespace, cell)
        if "def support_boundary_terms" in source:
            break
    required = (
        "build_hard_constraint_problem", "hard_constraint_cut_matrices", "framework_support",
        "framework_body_force", "support_boundary_terms", "solve_matrix_free_cg",
    )
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError("Notebook setup did not define: " + ", ".join(missing))
    return namespace


def main() -> None:
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Missing canonical notebook: {NOTEBOOK_PATH}")
    os.environ.update(
        {
            "HANGING_SPHERE_OUTPUT_DIR": str(OUTPUT_DIR),
            "HANGING_SPHERE_SUPPORT_MODE": "planar_cap",
            "HANGING_SPHERE_RUN_SOLVES": "0",
            "HANGING_SPHERE_RESOLUTIONS": os.environ.get("HANGING_SPHERE_RESOLUTIONS", "8,16,32,64,128,256"),
            "HANGING_SPHERE_FRAMEWORKS": ",".join(FRAMEWORKS),
            "HANGING_SPHERE_DEVICE": os.environ.get("HANGING_SPHERE_DEVICE", "cuda"),
            "HANGING_SPHERE_CPU_THREADS": os.environ.get("HANGING_SPHERE_CPU_THREADS", "6"),
            "HANGING_SPHERE_CG_MAX_ITER": os.environ.get("HANGING_SPHERE_CG_MAX_ITER", "12000"),
            "HANGING_SPHERE_DTYPE": "float64",
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    requested_schedule = os.environ["HANGING_SPHERE_RESOLUTIONS"]
    namespace = load_notebook_setup()
    # The notebook's interactive local-analysis cell can assign a display
    # schedule. The command-line runner remains authoritative for compute.
    os.environ["HANGING_SPHERE_RESOLUTIONS"] = requested_schedule
    namespace["RESOLUTION_SCHEDULE"] = tuple(
        int(value) for value in requested_schedule.split(",")
    )
    namespace["REFERENCE_RESOLUTION"] = max(namespace["RESOLUTION_SCHEDULE"])
    requested_frameworks = tuple(namespace["FRAMEWORKS"])

    for resolution in namespace["RESOLUTION_SCHEDULE"]:
        pending = [framework for framework in requested_frameworks if not solution_path(namespace, framework, resolution).exists()]
        if not pending:
            print(f"{resolution}^3: all corrected checkpoints already exist.", flush=True)
            continue
        assert_exact_queso_capacity(resolution)
        started = time.perf_counter()
        (case, full_ke, strict_full_indices, queso_cut_ke, queso_cut_indices, active, shared_force,
         fractions, queso_full_ke, queso_full_indices) = namespace["build_hard_constraint_problem"](resolution)
        foundation, constrained_dofs, support_metadata = namespace["support_boundary_terms"](case, active, queso_cut_indices)
        nn_engine = None
        if "nn" in pending:
            nn_engine = namespace["load_nn_engine"](checkpoint_path=namespace["NN_CHECKPOINT"], device=namespace["SOLVE_DEVICE"], dtype="float32")
        try:
            for framework in pending:
                framework_started = time.perf_counter()
                full_indices, cut_indices, active_indices = namespace["framework_support"](framework, case, strict_full_indices, queso_cut_indices)
                cut_matrices = namespace["hard_constraint_cut_matrices"](
                    framework, case, full_ke, queso_cut_ke, cut_indices, fractions, nn_engine if framework == "nn" else None,
                )
                force = namespace["framework_body_force"](case, full_indices, cut_indices)
                if framework == "queso":
                    positions = torch.searchsorted(queso_full_indices, full_indices)
                    if not torch.equal(queso_full_indices[positions], full_indices):
                        raise RuntimeError("QuESo full-cell matrix IDs do not cover the strict interior support.")
                    element_matrices = torch.cat((queso_full_ke[positions], cut_matrices), dim=0)
                    operator = ExactElementMatrixFreeElasticityOperator(
                        element_matrices, torch.cat((full_indices, cut_indices)), case.grid.resolution,
                        foundation, constrained_dofs=constrained_dofs,
                    )
                else:
                    operator = namespace["StructuredCompressedMatrixFreeElasticityOperator"](
                        full_ke, full_indices, cut_matrices, cut_indices, tuple(case.grid.resolution), foundation,
                        element_batch_size=32768, slab_cells=1, constrained_dofs=constrained_dofs,
                    )
                solution, info = namespace["solve_matrix_free_cg"](
                    operator, force, rtol=1.0e-6, atol=1.0e-8, max_iter=namespace["CG_MAX_ITER"],
                    check_interval=10, report_interval_seconds=30.0, max_seconds_per_iteration=None,
                )
                if not info["converged"]:
                    raise RuntimeError(f"{framework} {resolution}^3 solve did not converge: {info}")
                np.savez_compressed(
                    solution_path(namespace, framework, resolution),
                    u_nodes=solution.detach().cpu().numpy().reshape(-1, 3),
                    active_cells=active_indices.cpu().numpy(),
                    active_nodes=namespace["active_node_ids"](active_indices.cpu().numpy(), resolution),
                    runtime_seconds=time.perf_counter() - framework_started,
                    solver_info=np.array([info], dtype=object),
                    boundary_mode=namespace["PLANAR_CAP_MODE"], support_mode="planar_cap",
                    queso_inner_cells=(framework == "queso"), **support_metadata,
                )
                print(f"{resolution}^3 {framework}: saved {info} in {time.perf_counter() - framework_started:.1f}s", flush=True)
                del cut_matrices, force, operator, solution
                if framework == "queso":
                    del element_matrices
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            del nn_engine, case, full_ke, strict_full_indices, queso_cut_ke, queso_cut_indices, active, shared_force, fractions, queso_full_ke, queso_full_indices, foundation
            gc.collect()
            torch.cuda.empty_cache()
        print(f"{resolution}^3 completed in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
