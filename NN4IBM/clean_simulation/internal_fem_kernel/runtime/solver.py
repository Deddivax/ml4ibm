from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..fem import FEM


def solve_sparse_cg(
    a_sparse: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_iter: int = 4000,
    check_interval: int = 25,
) -> tuple[torch.Tensor, Dict[str, float | int | bool]]:
    if not a_sparse.is_sparse:
        raise TypeError("solve_sparse_cg expects a sparse COO tensor.")
    if check_interval < 1:
        raise ValueError("check_interval must be positive.")

    a = a_sparse.coalesce()
    x = torch.zeros_like(b)

    def mv(v: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(a, v[:, None]).reshape(-1)

    idx = a.indices()
    vals = a.values()
    diag = torch.zeros(a.shape[0], dtype=vals.dtype, device=vals.device)
    diag_mask = idx[0] == idx[1]
    if bool(torch.any(diag_mask).item()):
        diag.index_add_(0, idx[0, diag_mask], vals[diag_mask])
    diag = torch.where(torch.abs(diag) > 1e-20, diag, torch.ones_like(diag))

    r = b - mv(x)
    z = r / diag
    p = z.clone()
    rz_old = torch.dot(r, z)
    b_norm = torch.linalg.norm(b)
    tol = max(float(atol), float(rtol) * max(float(b_norm.item()), 1.0))

    converged = False
    last_res = float(torch.linalg.norm(r).item())
    iters = 0

    for k in range(int(max_iter)):
        ap = mv(p)
        denom = torch.dot(p, ap)
        alpha = rz_old / denom
        x = x + alpha * p
        r = r - alpha * ap

        iters = k + 1
        z = r / diag
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

        if iters % check_interval == 0 or iters == max_iter:
            last_res = float(torch.linalg.norm(r).item())
            if last_res <= tol:
                converged = True
                break

    if iters % check_interval:
        last_res = float(torch.linalg.norm(r).item())
        converged = last_res <= tol

    return x, {
        "converged": converged,
        "iters": int(iters),
        "residual": float(last_res),
    }


def torch_coo_to_scipy_csr(a_sparse: torch.Tensor):
    """Convert CPU PyTorch COO to CSR without PyTorch's expensive coalesce."""
    if a_sparse.device.type != "cpu" or not a_sparse.is_sparse:
        raise ValueError("torch_coo_to_scipy_csr requires a CPU sparse COO tensor.")

    import scipy.sparse as sp

    indices = a_sparse._indices().numpy()
    values = a_sparse._values().numpy()
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=a_sparse.shape).tocsr()


def solve_sparse_cg_scipy(
    csr,
    b: torch.Tensor,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_iter: int = 4000,
    progress_interval_seconds: float = 30.0,
    preconditioner_name: str = "amg",
) -> tuple[torch.Tensor, Dict[str, float | int | bool]]:
    """Solve a CPU COO system through compiled SciPy CSR CG.

    PyTorch CPU sparse operations are dispatched from a Python CG loop. For
    multi-million-DOF displacement cases, the loop overhead dominates runtime.
    SciPy keeps the same Jacobi-preconditioned CG algorithm in compiled code.
    """
    import scipy.sparse.linalg as spla

    started_at = time.perf_counter()
    print(
        f"SciPy CG: preparing {csr.shape[0]:,} DOFs / {csr.nnz:,} CSR entries "
        f"(rtol={rtol:.1e}, max_iter={max_iter:,}).",
        flush=True,
    )
    diagonal = csr.diagonal()
    diagonal[np.abs(diagonal) <= 1e-20] = 1.0
    jacobi_preconditioner = spla.LinearOperator(
        csr.shape,
        matvec=lambda vector: vector / diagonal,
        dtype=csr.dtype,
    )
    preconditioner = jacobi_preconditioner
    if preconditioner_name in {"amg", "elasticity_amg"}:
        try:
            import pyamg

            if preconditioner_name == "elasticity_amg":
                if csr.shape[0] % 3:
                    raise ValueError("Elasticity AMG requires three displacement DOFs per node.")
                candidates = np.zeros((csr.shape[0], 3), dtype=csr.dtype)
                candidates[0::3, 0] = 1.0
                candidates[1::3, 1] = 1.0
                candidates[2::3, 2] = 1.0
                print(
                    "SciPy CG: building elasticity-aware AMG preconditioner with translation modes.",
                    flush=True,
                )
            else:
                candidates = None
                print("SciPy CG: building smoothed-aggregation AMG preconditioner.", flush=True)
            preconditioner_started_at = time.perf_counter()
            multigrid = pyamg.smoothed_aggregation_solver(
                csr,
                B=candidates,
                symmetry="symmetric",
                max_coarse=500,
            )
            preconditioner = multigrid.aspreconditioner(cycle="V")
            print(
                f"SciPy CG: AMG setup completed in {time.perf_counter() - preconditioner_started_at:.1f}s "
                f"with {len(multigrid.levels)} levels and {multigrid.operator_complexity():.2f} operator complexity.",
                flush=True,
            )
        except Exception as error:
            print(
                f"SciPy CG: AMG setup failed ({type(error).__name__}: {error}); falling back to Jacobi.",
                flush=True,
            )
    elif preconditioner_name != "jacobi":
        raise ValueError(f"Unsupported CPU preconditioner: {preconditioner_name!r}.")

    rhs = b.detach().numpy()
    iterations = 0
    solve_started_at = time.perf_counter()
    last_report_at = solve_started_at

    def count_iteration(solution: np.ndarray) -> None:
        nonlocal iterations, last_report_at
        iterations += 1
        now = time.perf_counter()
        if now - last_report_at >= progress_interval_seconds:
            residual = np.linalg.norm(csr @ solution - rhs)
            elapsed = now - solve_started_at
            print(
                f"SciPy CG: iteration {iterations:,}/{max_iter:,}; residual={residual:.3e}; "
                f"{elapsed:.1f}s elapsed; {iterations / elapsed:.2f} iterations/s.",
                flush=True,
            )
            last_report_at = now

    print(f"SciPy CG: starting {preconditioner_name}-preconditioned solve.", flush=True)
    solution, info = spla.cg(
        csr,
        rhs,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=int(max_iter),
        M=preconditioner,
        callback=count_iteration,
    )
    residual = float(np.linalg.norm(csr @ solution - rhs))
    print(
        f"SciPy CG: {'converged' if info == 0 else 'stopped'} after {iterations:,} iterations "
        f"in {time.perf_counter() - solve_started_at:.1f}s; residual={residual:.3e}; info={info}.",
        flush=True,
    )
    return torch.from_numpy(solution).to(dtype=b.dtype), {
        "converged": bool(info == 0),
        "iters": int(iterations if info == 0 else info),
        "residual": residual,
    }


def solve_with_nitsche(
    grid,
    material,
    config: dict,
    top_mesh,
    active_indices,
    poly_order: int,
    g_order: int,
    k_base,
    f_base,
    precomputed_nitsche_sets=None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, float], Any]:
    timings: Dict[str, float] = {}
    dtype = torch.float64
    device_t = torch.device(device)

    def sync() -> None:
        if device_t.type == "cuda":
            torch.cuda.synchronize(device=device_t)

    if not torch.is_tensor(f_base):
        f_base = torch.as_tensor(f_base, dtype=dtype, device=device_t)
    else:
        f_base = f_base.to(dtype=dtype, device=device_t)

    if precomputed_nitsche_sets is None:
        print("Nitsche solve: assembling boundary terms.", flush=True)
        sync()
        t0 = time.perf_counter()
        triangles = top_mesh.triangles
        if not torch.is_tensor(triangles):
            triangles = torch.tensor(triangles, dtype=dtype, device=device_t)
        else:
            triangles = triangles.to(dtype=dtype, device=device_t)

        nitsche_bcs = [{
            "triangles": triangles,
            "prescribed_displacement": None,
            "penalty": float(config["bc"]["nitsche_penalty"]),
            "order": g_order,
        }]
        nitsche_sets = FEM.NitscheAssembly.compute_terms(
            grid=grid,
            material=material,
            active_indices=active_indices,
            nitsche_bcs=nitsche_bcs,
            device=device_t,
            poly_order=poly_order,
        )
        sync()
        timings["nitsche_terms"] = time.perf_counter() - t0
    else:
        nitsche_sets = precomputed_nitsche_sets
        timings["nitsche_terms"] = 0.0
        print("Nitsche solve: reusing previously assembled boundary terms.", flush=True)

    sync()
    print("Nitsche solve: assembling constrained global sparse system.", flush=True)
    t0 = time.perf_counter()
    if device_t.type == "cpu" and not isinstance(k_base, torch.Tensor):
        k_global = k_base.copy()
        for nset in nitsche_sets:
            k_add = nset.get("k_add")
            if k_add is not None:
                k_global += torch_coo_to_scipy_csr(k_add)
        f_global = f_base.clone()
        for nset in nitsche_sets:
            f_add = nset.get("f_add")
            if f_add is not None:
                f_global += torch.as_tensor(f_add, dtype=f_global.dtype, device="cpu")
        timings["global_assembly"] = time.perf_counter() - t0
        print(
            f"Nitsche solve: global system ready in {timings['global_assembly']:.1f}s; "
            f"{k_global.shape[0]:,} DOFs / {k_global.nnz:,} CSR entries.",
            flush=True,
        )
    elif device_t.type == "cpu" and isinstance(k_base, torch.Tensor) and k_base.is_sparse:
        print(
            f"Nitsche solve: converting base COO ({k_base._nnz():,} entries) directly to SciPy CSR.",
            flush=True,
        )
        k_global = torch_coo_to_scipy_csr(k_base)
        for nset in nitsche_sets:
            k_add = nset.get("k_add")
            if k_add is not None:
                k_global = k_global + torch_coo_to_scipy_csr(k_add)
        f_global = f_base.clone()
        for nset in nitsche_sets:
            f_add = nset.get("f_add")
            if f_add is not None:
                f_global += torch.as_tensor(f_add, dtype=f_global.dtype, device="cpu")
        timings["global_assembly"] = time.perf_counter() - t0
        print(
            f"Nitsche solve: global system ready in {timings['global_assembly']:.1f}s; "
            f"{k_global.shape[0]:,} DOFs / {k_global.nnz:,} CSR entries.",
            flush=True,
        )
    else:
        k_global, f_global = FEM.GlobalAssembly.assemble_system(
            k_global=k_base,
            f_global=f_base,
            k_add=nitsche_sets,
            verbose=True,
        )
        if isinstance(k_global, torch.Tensor) and k_global.is_sparse and not k_global.is_coalesced():
            k_global = k_global.coalesce()
        sync()
        timings["global_assembly"] = time.perf_counter() - t0

    sync()
    print("Nitsche solve: starting linear solve.", flush=True)
    t0 = time.perf_counter()
    if device_t.type == "cpu" and not isinstance(k_global, torch.Tensor):
        solver_cfg = dict(config.get("solver", {})) if isinstance(config, dict) else {}
        u_vec, cg_info = solve_sparse_cg_scipy(
            k_global,
            f_global,
            rtol=float(solver_cfg.get("cg_rtol", 1e-7)),
            atol=float(solver_cfg.get("cg_atol", 1e-9)),
            max_iter=int(solver_cfg.get("cg_max_iter", max(600, min(4000, 2 * int(f_global.numel()))))),
            preconditioner_name=str(solver_cfg.get("cpu_preconditioner", "jacobi")),
        )
        timings["cg_iterations_count"] = float(cg_info["iters"])
        timings["cg_nonconverged_count"] = 0.0 if bool(cg_info["converged"]) else 1.0
    elif isinstance(k_global, torch.Tensor) and k_global.is_sparse:
        solver_cfg = dict(config.get("solver", {})) if isinstance(config, dict) else {}
        cg_rtol = float(solver_cfg.get("cg_rtol", 1e-7))
        cg_atol = float(solver_cfg.get("cg_atol", 1e-9))
        cg_max_iter = int(solver_cfg.get("cg_max_iter", max(600, min(4000, 2 * int(f_global.numel())))))
        cg_check_interval = int(solver_cfg.get("cg_check_interval", 25))
        cpu_preconditioner = str(solver_cfg.get("cpu_preconditioner", "jacobi"))
        allow_dense_fallback = bool(solver_cfg.get("allow_dense_fallback", False))
        dense_fallback_max_dofs = int(solver_cfg.get("dense_fallback_max_dofs", 30000))

        if device_t.type == "cpu":
            u_vec, cg_info = solve_sparse_cg_scipy(
                k_global,
                f_global,
                rtol=cg_rtol,
                atol=cg_atol,
                max_iter=cg_max_iter,
                preconditioner_name=cpu_preconditioner,
            )
        else:
            u_vec, cg_info = solve_sparse_cg(
                k_global,
                f_global,
                rtol=cg_rtol,
                atol=cg_atol,
                max_iter=cg_max_iter,
                check_interval=cg_check_interval,
            )
        timings["cg_iterations_count"] = float(cg_info["iters"])
        timings["cg_nonconverged_count"] = 0.0 if bool(cg_info["converged"]) else 1.0

        if (not cg_info["converged"] and allow_dense_fallback and int(f_global.numel()) <= dense_fallback_max_dofs):
            u_vec = torch.linalg.solve(k_global.to_dense(), f_global)
    else:
        u_vec = torch.linalg.solve(k_global, f_global)

    n_nodes_total = grid.total_nodes(poly_order=poly_order)
    u_nodes = u_vec.reshape(-1, 3)
    sync()
    timings["linear_solve"] = time.perf_counter() - t0

    return u_nodes, u_vec, n_nodes_total, timings, nitsche_sets, k_global
