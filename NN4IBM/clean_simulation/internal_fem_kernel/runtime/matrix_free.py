from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch


def _element_nodes(active_indices: torch.Tensor, resolution: tuple[int, int, int]) -> torch.Tensor:
    nx, ny, nz = resolution
    node_ny = ny + 1
    node_nz = nz + 1
    elements = active_indices.to(dtype=torch.int64)
    i = elements // (ny * nz)
    remainder = elements % (ny * nz)
    j = remainder // nz
    k = remainder % nz
    base = (i * node_ny + j) * node_nz + k
    return torch.stack((
        base,
        base + node_nz * node_ny,
        base + node_nz * node_ny + node_nz,
        base + node_nz,
        base + 1,
        base + node_nz * node_ny + 1,
        base + node_nz * node_ny + node_nz + 1,
        base + node_nz + 1,
    ), dim=1)


@dataclass
class MatrixFreeElasticityOperator:
    """Apply existing P1 element and Nitsche stiffness without a volume CSR matrix."""

    element_matrices: torch.Tensor
    active_indices: torch.Tensor
    resolution: tuple[int, int, int]
    nitsche_matrix: torch.Tensor | None = None
    element_batch_size: int = 4096

    def __post_init__(self) -> None:
        self.element_matrices = torch.as_tensor(self.element_matrices, dtype=torch.float64, device="cpu")
        self.active_indices = torch.as_tensor(self.active_indices, dtype=torch.int64, device="cpu").reshape(-1)
        if self.element_matrices.shape != (len(self.active_indices), 24, 24):
            raise ValueError("element_matrices must have shape (n_active, 24, 24).")
        if self.element_batch_size < 1:
            raise ValueError("element_batch_size must be positive.")
        self.node_count = (self.resolution[0] + 1) * (self.resolution[1] + 1) * (self.resolution[2] + 1)
        if self.nitsche_matrix is not None:
            self.nitsche_matrix = torch.as_tensor(self.nitsche_matrix, dtype=torch.float64, device="cpu")
            if not self.nitsche_matrix.is_sparse or self.nitsche_matrix.shape != (3 * self.node_count, 3 * self.node_count):
                raise ValueError("nitsche_matrix must be a sparse matrix with the global DOF shape.")
            self.nitsche_matrix = self.nitsche_matrix.coalesce()

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(vector, dtype=torch.float64, device="cpu").reshape(-1)
        if vector.numel() != 3 * self.node_count:
            raise ValueError("vector has the wrong number of displacement DOFs.")
        displacements = vector.reshape(self.node_count, 3)
        result = torch.zeros_like(displacements)
        for start in range(0, len(self.active_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.active_indices))
            nodes = _element_nodes(self.active_indices[start:stop], self.resolution)
            local_u = displacements[nodes].reshape(stop - start, 24)
            local_result = torch.bmm(self.element_matrices[start:stop], local_u.unsqueeze(-1)).squeeze(-1)
            result.index_add_(0, nodes.reshape(-1), local_result.reshape(-1, 3))
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            result += torch.sparse.mm(self.nitsche_matrix, vector[:, None]).reshape(-1)
        return result

    def diagonal(self) -> torch.Tensor:
        result = torch.zeros((self.node_count, 3), dtype=torch.float64)
        local_diagonal = self.element_matrices.diagonal(dim1=1, dim2=2).reshape(-1, 8, 3)
        for start in range(0, len(self.active_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.active_indices))
            nodes = _element_nodes(self.active_indices[start:stop], self.resolution)
            result.index_add_(0, nodes.reshape(-1), local_diagonal[start:stop].reshape(-1, 3))
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            indices = self.nitsche_matrix.indices()
            values = self.nitsche_matrix.values()
            mask = indices[0] == indices[1]
            result.index_add_(0, indices[0, mask], values[mask])
        return torch.where(result.abs() > 1.0e-20, result, torch.ones_like(result))


@dataclass
class CompressedMatrixFreeElasticityOperator:
    """Matrix-free P1 operator with one full-cell matrix and explicit cut cells only."""

    full_cell_matrix: torch.Tensor
    full_indices: torch.Tensor
    cut_matrices: torch.Tensor
    cut_indices: torch.Tensor
    resolution: tuple[int, int, int]
    nitsche_matrix: torch.Tensor | None = None
    element_batch_size: int = 4096

    def __post_init__(self) -> None:
        self.full_cell_matrix = torch.as_tensor(self.full_cell_matrix, dtype=torch.float64, device="cpu")
        self.full_indices = torch.as_tensor(self.full_indices, dtype=torch.int64, device="cpu").reshape(-1)
        self.cut_matrices = torch.as_tensor(self.cut_matrices, dtype=torch.float64, device="cpu")
        self.cut_indices = torch.as_tensor(self.cut_indices, dtype=torch.int64, device="cpu").reshape(-1)
        if self.full_cell_matrix.shape != (24, 24):
            raise ValueError("full_cell_matrix must have shape (24, 24).")
        if self.cut_matrices.shape != (len(self.cut_indices), 24, 24):
            raise ValueError("cut_matrices must have shape (n_cut, 24, 24).")
        self.node_count = (self.resolution[0] + 1) * (self.resolution[1] + 1) * (self.resolution[2] + 1)
        if self.nitsche_matrix is not None:
            self.nitsche_matrix = torch.as_tensor(self.nitsche_matrix, dtype=torch.float64, device="cpu").coalesce()

    def _apply(self, result: torch.Tensor, displacements: torch.Tensor, indices: torch.Tensor, matrices: torch.Tensor, full: bool) -> None:
        for start in range(0, len(indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(indices))
            nodes = _element_nodes(indices[start:stop], self.resolution)
            local_u = displacements[nodes].reshape(stop - start, 24)
            local_matrices = self.full_cell_matrix.expand(stop - start, -1, -1) if full else matrices[start:stop]
            local_result = torch.bmm(local_matrices, local_u.unsqueeze(-1)).squeeze(-1)
            result.index_add_(0, nodes.reshape(-1), local_result.reshape(-1, 3))

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(vector, dtype=torch.float64, device="cpu").reshape(-1)
        if vector.numel() != 3 * self.node_count:
            raise ValueError("vector has the wrong number of displacement DOFs.")
        displacements = vector.reshape(self.node_count, 3)
        result = torch.zeros_like(displacements)
        self._apply(result, displacements, self.full_indices, self.full_cell_matrix, True)
        self._apply(result, displacements, self.cut_indices, self.cut_matrices, False)
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            result += torch.sparse.mm(self.nitsche_matrix, vector[:, None]).reshape(-1)
        return result

    def diagonal(self) -> torch.Tensor:
        result = torch.zeros((self.node_count, 3), dtype=torch.float64)
        for indices, matrices, full in ((self.full_indices, self.full_cell_matrix, True), (self.cut_indices, self.cut_matrices, False)):
            local_diagonal = matrices.diagonal().reshape(8, 3) if full else matrices.diagonal(dim1=1, dim2=2).reshape(-1, 8, 3)
            for start in range(0, len(indices), self.element_batch_size):
                stop = min(start + self.element_batch_size, len(indices))
                nodes = _element_nodes(indices[start:stop], self.resolution)
                values = local_diagonal.expand(stop - start, -1, -1) if full else local_diagonal[start:stop]
                result.index_add_(0, nodes.reshape(-1), values.reshape(-1, 3))
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            indices = self.nitsche_matrix.indices()
            values = self.nitsche_matrix.values()
            mask = indices[0] == indices[1]
            result.index_add_(0, indices[0, mask], values[mask])
        return torch.where(result.abs() > 1.0e-20, result, torch.ones_like(result))


@dataclass
class StructuredCompressedMatrixFreeElasticityOperator(CompressedMatrixFreeElasticityOperator):
    """Compressed operator with structured masked application for repeated full cells.

    The full-cell term is evaluated on node-grid slices, avoiding element-node
    gathers and expanded local matrices in every Krylov iteration. Trimmed
    cells retain their exact individual matrices and the Nitsche term stays
    unchanged.
    """

    slab_cells: int = 16
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    constrained_dofs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.full_cell_matrix = self.full_cell_matrix.to(device=self.device, dtype=self.dtype)
        self.full_indices = self.full_indices.to(device=self.device)
        self.cut_matrices = self.cut_matrices.to(device=self.device, dtype=self.dtype)
        self.cut_indices = self.cut_indices.to(device=self.device)
        if self.nitsche_matrix is not None:
            self.nitsche_matrix = self.nitsche_matrix.to(device=self.device, dtype=self.dtype).coalesce()
        nx, ny, nz = self.resolution
        if self.slab_cells < 1:
            raise ValueError("slab_cells must be positive.")
        self.full_mask = torch.zeros((nx, ny, nz), dtype=self.dtype, device=self.device)
        self.full_mask.reshape(-1)[self.full_indices] = 1.0
        self.local_offsets = (
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        )
        self.local_matrix = self.full_cell_matrix.reshape(8, 3, 8, 3)
        self.local_matrix_input_to_output = self.local_matrix.permute(2, 3, 0, 1).reshape(24, 24).contiguous()
        self.cut_nodes = _element_nodes(self.cut_indices, self.resolution)
        if self.constrained_dofs is not None:
            self.constrained_dofs = torch.as_tensor(
                self.constrained_dofs, dtype=torch.int64, device=self.device
            ).reshape(-1).unique()

    def _apply_structured_full(self, result: torch.Tensor, displacements: torch.Tensor) -> None:
        nx, ny, nz = self.resolution
        for start in range(0, nx, self.slab_cells):
            stop = min(start + self.slab_cells, nx)
            mask = self.full_mask[start:stop]
            for output_node, (ox, oy, oz) in enumerate(self.local_offsets):
                contribution = torch.zeros((*mask.shape, 3), dtype=self.dtype, device=self.device)
                for input_node, (ix, iy, iz) in enumerate(self.local_offsets):
                    block = self.local_matrix[output_node, :, input_node, :]
                    contribution += displacements[
                        start + ix:stop + ix,
                        iy:iy + ny,
                        iz:iz + nz,
                    ] @ block.transpose(0, 1)
                result[
                    start + ox:stop + ox,
                    oy:oy + ny,
                    oz:oz + nz,
                ] += contribution * mask[..., None]

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(vector, dtype=self.dtype, device=self.device).reshape(-1)
        if vector.numel() != 3 * self.node_count:
            raise ValueError("vector has the wrong number of displacement DOFs.")
        nx, ny, nz = self.resolution
        displacements = vector.reshape(nx + 1, ny + 1, nz + 1, 3)
        result = torch.zeros_like(displacements)
        self._apply_structured_full(result, displacements)
        result_nodes = result.reshape(-1, 3)
        displacement_nodes = displacements.reshape(-1, 3)
        for start in range(0, len(self.cut_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.cut_indices))
            nodes = self.cut_nodes[start:stop]
            local_u = displacement_nodes[nodes].reshape(stop - start, 24)
            local_result = torch.bmm(self.cut_matrices[start:stop], local_u.unsqueeze(-1)).squeeze(-1)
            result_nodes.index_add_(0, nodes.reshape(-1), local_result.reshape(-1, 3))
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            result += torch.sparse.mm(self.nitsche_matrix, vector[:, None]).reshape(-1)
        if self.constrained_dofs is not None:
            result[self.constrained_dofs] = vector[self.constrained_dofs]
        return result

    def diagonal(self) -> torch.Tensor:
        nx, ny, nz = self.resolution
        result = torch.zeros((nx + 1, ny + 1, nz + 1, 3), dtype=self.dtype, device=self.device)
        for local_node, (ox, oy, oz) in enumerate(self.local_offsets):
            diagonal = self.local_matrix[local_node, :, local_node, :].diagonal()
            result[ox:ox + nx, oy:oy + ny, oz:oz + nz] += self.full_mask[..., None] * diagonal
        self._apply_diagonal_cuts(result.reshape(-1, 3))
        result = result.reshape(-1)
        if self.nitsche_matrix is not None:
            indices = self.nitsche_matrix.indices()
            values = self.nitsche_matrix.values()
            mask = indices[0] == indices[1]
            result.index_add_(0, indices[0, mask], values[mask])
        if self.constrained_dofs is not None:
            result[self.constrained_dofs] = 1.0
        return torch.where(result.abs() > 1.0e-20, result, torch.ones_like(result))

    def _apply_diagonal_cuts(self, result: torch.Tensor) -> None:
        local_diagonal = self.cut_matrices.diagonal(dim1=1, dim2=2).reshape(-1, 8, 3)
        for start in range(0, len(self.cut_indices), self.element_batch_size):
            stop = min(start + self.element_batch_size, len(self.cut_indices))
            nodes = _element_nodes(self.cut_indices[start:stop], self.resolution)
            result.index_add_(0, nodes.reshape(-1), local_diagonal[start:stop].reshape(-1, 3))


_DOT_CHUNK_SIZE = 1_000_000_000


def _vector_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute a dot product without exceeding the backend's int32 index limit."""
    if left.numel() != right.numel():
        raise ValueError("dot-product vectors must have the same number of entries.")
    if left.numel() <= _DOT_CHUNK_SIZE:
        return torch.dot(left, right)
    total = torch.zeros((), dtype=left.dtype, device=left.device)
    for start in range(0, left.numel(), _DOT_CHUNK_SIZE):
        stop = min(start + _DOT_CHUNK_SIZE, left.numel())
        total += torch.dot(left[start:stop], right[start:stop])
    return total


def _vector_norm(vector: torch.Tensor) -> torch.Tensor:
    return _vector_dot(vector, vector).clamp_min(0.0).sqrt()


def solve_matrix_free_cg(operator: MatrixFreeElasticityOperator, rhs: torch.Tensor, *, rtol: float = 1.0e-7, atol: float = 1.0e-9, max_iter: int = 4000, check_interval: int = 25, report_interval_seconds: float | None = None, max_seconds_per_iteration: float | None = None, performance_warmup_iterations: int = 5, preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None, initial_solution: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    operator_values = operator.full_cell_matrix if hasattr(operator, "full_cell_matrix") else operator.element_matrices
    rhs = torch.as_tensor(rhs, dtype=operator_values.dtype, device=operator_values.device).reshape(-1)
    constrained_dofs = getattr(operator, "constrained_dofs", None)
    if constrained_dofs is not None:
        rhs = rhs.clone()
        rhs[constrained_dofs] = 0.0
    rhs_is_cuda = rhs.is_cuda
    tolerance = max(float(atol), float(rtol) * max(float(_vector_norm(rhs)), 1.0))
    diagonal = operator.diagonal()
    solution = torch.zeros_like(rhs) if initial_solution is None else torch.as_tensor(
        initial_solution, dtype=rhs.dtype, device=rhs.device
    ).reshape(-1).clone()
    if solution.shape != rhs.shape:
        raise ValueError("initial_solution has the wrong number of displacement DOFs.")
    if constrained_dofs is not None:
        solution[constrained_dofs] = 0.0
    if initial_solution is None:
        residual = rhs.clone()
    else:
        residual = operator.matvec(solution)
        residual.neg_().add_(rhs)
    del rhs
    if preconditioner is None:
        direction = residual / diagonal
        preconditioned = None
    else:
        preconditioned = preconditioner(residual)
        direction = preconditioned.clone()
    rz_old = _vector_dot(residual, direction)
    initial_residual = float(_vector_norm(residual))
    if initial_residual <= tolerance:
        print(f"Matrix-free CG: initial solution satisfies tolerance={tolerance:.3e}.", flush=True)
        return solution, {"converged": True, "iters": 0, "residual": initial_residual}
    started_at = time.perf_counter()
    last_report_at = started_at
    print(f"Matrix-free CG: {solution.numel():,} DOFs, tolerance={tolerance:.3e}, max_iter={max_iter:,}.", flush=True)

    for iteration in range(1, max_iter + 1):
        if rhs_is_cuda:
            torch.cuda.synchronize(solution.device)
        iteration_started_at = time.perf_counter()
        if iteration > 1 and preconditioner is None:
            del applied
        applied = operator.matvec(direction)
        if rhs_is_cuda:
            torch.cuda.synchronize(solution.device)
        if max_seconds_per_iteration is not None and iteration >= performance_warmup_iterations:
            seconds_per_iteration = time.perf_counter() - iteration_started_at
            if seconds_per_iteration > max_seconds_per_iteration:
                raise RuntimeError(
                    f"Matrix-free CG performance budget exceeded at iteration {iteration:,}: "
                    f"{seconds_per_iteration:.3f}s/iteration exceeds {max_seconds_per_iteration:.3f}s/iteration."
                )
        alpha = rz_old / _vector_dot(direction, applied)
        # Full-size CUDA vectors can exceed available memory if alpha * vector
        # materializes a temporary, so perform the Krylov AXPY updates in place.
        alpha_value = float(alpha)
        solution.add_(direction, alpha=alpha_value)
        residual.add_(applied, alpha=-alpha_value)
        if iteration % check_interval == 0 or iteration == max_iter:
            residual_norm = float(_vector_norm(residual))
            now = time.perf_counter()
            if report_interval_seconds is not None and now - last_report_at >= report_interval_seconds:
                print(
                    f"Matrix-free CG: iteration {iteration:,}/{max_iter:,}; residual={residual_norm:.3e}; "
                    f"elapsed={now - started_at:.1f}s.",
                    flush=True,
                )
                last_report_at = now
            if residual_norm <= tolerance:
                print(f"Matrix-free CG: converged in {iteration:,} iterations after {now - started_at:.1f}s.", flush=True)
                return solution, {"converged": True, "iters": iteration, "residual": residual_norm}
        if preconditioner is None:
            # Ap is no longer needed after the residual update; reuse its storage for M^-1 r.
            applied.copy_(residual).div_(diagonal)
            rz_new = _vector_dot(residual, applied)
            direction.mul_(rz_new / rz_old).add_(applied)
        else:
            preconditioned = preconditioner(residual)
            rz_new = _vector_dot(residual, preconditioned)
            direction = preconditioned + (rz_new / rz_old) * direction
        rz_old = rz_new

    final_residual = float(_vector_norm(residual))
    print(f"Matrix-free CG: stopped at {max_iter:,} iterations after {time.perf_counter() - started_at:.1f}s; residual={final_residual:.3e}.", flush=True)
    return solution, {"converged": False, "iters": max_iter, "residual": final_residual}