from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pipeline_api import (
    relative_l2_on_common_active,
    run_classic_pipeline,
    run_hybrid_pipeline,
    run_queso_pipeline,
)


@dataclass(slots=True)
class DisplacementResult:
    pipeline: str
    u_nodes: Any
    u_vec: Any
    active_indices: Any
    n_nodes_total: int
    timings: dict[str, float]


def run_pipeline(
    *,
    problem: Any,
    pipeline: str,
    device: str = "cuda",
    nn_engine: Any | None = None,
    classic_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = pipeline.strip().lower()
    if name == "classic":
        return run_classic_pipeline(problem=problem, device=device)
    if name == "queso":
        return run_queso_pipeline(problem=problem, device=device)
    if name == "hybrid":
        if nn_engine is None:
            raise ValueError("hybrid pipeline requires nn_engine")
        return run_hybrid_pipeline(
            problem=problem,
            device=device,
            nn_engine=nn_engine,
            classic_result=classic_result,
        )


def solve_global_displacement(
    *,
    problem: Any,
    pipeline: str,
    device: str = "cuda",
    nn_engine: Any | None = None,
    classic_result: dict[str, Any] | None = None,
) -> DisplacementResult:
    result = run_pipeline(
        problem=problem,
        pipeline=pipeline,
        device=device,
        nn_engine=nn_engine,
        classic_result=classic_result,
    )
    return DisplacementResult(
        pipeline=pipeline,
        u_nodes=result.get("u_nodes"),
        u_vec=result.get("u_vec"),
        active_indices=result.get("active_indices"),
        n_nodes_total=int(result.get("n_nodes_total", 0)),
        timings=dict(result.get("timings", {})),
    )


def compare_displacements(
    *,
    problem: Any,
    a_result: dict[str, Any],
    b_result: dict[str, Any],
) -> float:
    return relative_l2_on_common_active(
        u_a=a_result["u_nodes"],
        u_b=b_result["u_nodes"],
        problem=problem,
        active_indices_a=a_result["active_indices"],
        active_indices_b=b_result["active_indices"],
    )
