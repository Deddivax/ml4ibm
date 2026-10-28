import time

import torch

from core.grid_indexing import structured_total_nodes


def initialize_force_vector(grid, f_global=None, device="cpu", sparse=False, poly_order=1):
    if f_global is not None:
        return f_global
    ndof = 3 * structured_total_nodes(grid.resolution, poly_order=poly_order)
    return torch.zeros(ndof, dtype=torch.float64) if sparse else torch.zeros(ndof, dtype=torch.float64, device=device)


def assemble_system(k_global, f_global, k_add=None, f_add=None, verbose=False):
    k_out = k_global
    f_out = f_global
    if isinstance(k_add, list):
        for nset in k_add:
            if not isinstance(nset, dict):
                raise TypeError("Each nitsche set must be a dict.")
            k_out, f_out = assemble_system(
                k_global=k_out,
                f_global=f_out,
                k_add=nset.get("k_add"),
                f_add=nset.get("f_add"),
                verbose=verbose,
            )
        return k_out, f_out
    if k_add is not None:
        k_out_sparse = isinstance(k_out, torch.Tensor) and k_out.is_sparse
        k_add_sparse = isinstance(k_add, torch.Tensor) and k_add.is_sparse
        if k_out_sparse:
            if not k_add_sparse:
                k_add = k_add.to_sparse()
            if verbose:
                print(
                    f"Global assembly: base={k_out._nnz():,} nnz, "
                    f"Nitsche={k_add._nnz():,} nnz; concatenating COO entries.",
                    flush=True,
                )
            started_at = time.perf_counter()
            combined = k_out + k_add
            if verbose:
                print(
                    f"Global assembly: sparse addition took {time.perf_counter() - started_at:.1f}s; "
                    f"coalescing {combined._nnz():,} entries.",
                    flush=True,
                )
            started_at = time.perf_counter()
            k_out = combined.coalesce()
            if verbose:
                print(
                    f"Global assembly: COO coalescing took {time.perf_counter() - started_at:.1f}s; "
                    f"result={k_out._nnz():,} nnz.",
                    flush=True,
                )
        else:
            if k_add_sparse:
                k_add = k_add.to_dense()
            k_out = k_out + k_add
    if f_add is not None:
        if not isinstance(f_add, torch.Tensor):
            f_add = torch.as_tensor(f_add, dtype=f_out.dtype, device=f_out.device)
        f_out = f_out + f_add
    return k_out, f_out


class GlobalAssembly:
    initialize_force_vector = staticmethod(initialize_force_vector)
    assemble_system = staticmethod(assemble_system)
