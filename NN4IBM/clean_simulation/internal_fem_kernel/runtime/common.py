from __future__ import annotations

from typing import Any, Dict

import torch

from ..core.grid_indexing import active_element_nodes


CANONICAL_STAGE_KEYS: tuple[str, ...] = (
    "classification",
    "stiffness_inner",
    "stiffness_cut",
    "stiffness_assembly",
    "force_assembly",
    "nitsche_terms",
    "global_assembly",
    "linear_solve",
)


def as_tensor(
    x: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype if dtype is not None else x.dtype, device=device if device is not None else x.device)
    return torch.as_tensor(x, dtype=dtype, device=device)


def as_index_tensor_cpu(x: Any) -> torch.Tensor:
    return as_tensor(x, dtype=torch.int64, device="cpu").reshape(-1)


def as_bool_mask(mask: Any, *, device: torch.device | str | None = None) -> torch.Tensor:
    return as_tensor(mask, dtype=torch.bool, device=device)


def sync_cuda_if_needed(device: torch.device | str | None) -> None:
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize(device=dev)


def sum_duration_fields(timings: Dict[str, float]) -> float:
    return float(sum(float(timings.get(k, 0.0)) for k in CANONICAL_STAGE_KEYS))


def canonical_timings(timings: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {k: float(timings.get(k, 0.0)) for k in CANONICAL_STAGE_KEYS}
    out["total"] = sum_duration_fields(out)
    return out


def finalize_pipeline_timings(timings: Dict[str, float]) -> Dict[str, float]:
    merged = dict(timings)
    merged.update(canonical_timings(timings))
    return merged


def reset_pipeline_runtime_caches(device: torch.device | str | None = None) -> None:
    if device is not None:
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device=dev)


def l2_relative_difference_common_active(
    u_a,
    u_b,
    *,
    grid,
    active_indices_a,
    active_indices_b,
    poly_order: int,
    eps: float = 1e-30,
) -> float:
    ta = as_tensor(u_a, dtype=torch.float64).reshape(-1, 3)
    tb = as_tensor(u_b, dtype=torch.float64, device=ta.device).reshape(-1, 3)

    if ta.shape != tb.shape:
        raise ValueError(f"Displacement shape mismatch: {tuple(ta.shape)} vs {tuple(tb.shape)}")

    n_nodes_total = int(grid.total_nodes(poly_order=poly_order))
    nodes_a = active_element_nodes(as_index_tensor_cpu(active_indices_a), grid.resolution, poly_order=poly_order, device="cpu").reshape(-1)
    nodes_b = active_element_nodes(as_index_tensor_cpu(active_indices_b), grid.resolution, poly_order=poly_order, device="cpu").reshape(-1)

    mask_a = torch.zeros((n_nodes_total,), dtype=torch.bool)
    mask_b = torch.zeros((n_nodes_total,), dtype=torch.bool)
    mask_a[nodes_a] = True
    mask_b[nodes_b] = True
    mask_common = (mask_a & mask_b).to(device=ta.device)

    if not bool(torch.any(mask_common).item()):
        raise ValueError("No common active nodes between the compared methods.")

    ta_sel = ta[mask_common].reshape(-1)
    tb_sel = tb[mask_common].reshape(-1)
    diff = torch.linalg.norm(ta_sel - tb_sel)
    ref = torch.clamp(torch.linalg.norm(tb_sel), min=float(eps))
    return float((diff / ref).item())