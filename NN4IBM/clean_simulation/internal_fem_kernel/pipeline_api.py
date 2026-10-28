from __future__ import annotations

from typing import Any

# Clear, local, geometry-agnostic pipeline API over the internal FEM kernel.
from .pipeline_runtime import (
    l2_relative_difference_common_active,
    run_classic_pipeline_timed,
    run_hybrid_pipeline_timed,
    run_queso_pipeline_timed,
)
from .runtime.problem_options import threshold_bounds


def run_queso_pipeline(*, problem: Any, device: str) -> dict[str, Any]:
    return run_queso_pipeline_timed(
        grid=problem.grid,
        material=problem.material,
        config=problem.config,
        ips=problem.ips,
        top_mesh=problem.top_mesh,
        poly_order=problem.poly_order,
        g_order=problem.g_order,
        device=device,
    )


def run_classic_pipeline(*, problem: Any, device: str) -> dict[str, Any]:
    lower_threshold, upper_threshold = threshold_bounds(problem)

    return run_classic_pipeline_timed(
        grid=problem.grid,
        volume=problem.volume,
        material=problem.material,
        config=problem.config,
        ips_for_body_force=problem.ips,
        top_mesh=problem.top_mesh,
        poly_order=problem.poly_order,
        g_order=problem.g_order,
        order=2,
        cut_max_depth=3,
        device=device,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )


def run_hybrid_pipeline(
    *,
    problem: Any,
    device: str,
    nn_engine: Any,
    classic_result: dict[str, Any] | None,
) -> dict[str, Any]:
    lower_threshold, upper_threshold = threshold_bounds(problem)

    return run_hybrid_pipeline_timed(
        grid=problem.grid,
        volume=problem.volume,
        material=problem.material,
        config=problem.config,
        ips_for_body_force=problem.ips,
        top_mesh=problem.top_mesh,
        nn_engine=nn_engine,
        poly_order=problem.poly_order,
        g_order=problem.g_order,
        order=2,
        classic_nitsche_sets=None if classic_result is None else classic_result.get("nitsche_sets"),
        active_indices_reference=None if classic_result is None else classic_result.get("active_indices"),
        device=device,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )


def relative_l2_on_common_active(
    *,
    u_a: Any,
    u_b: Any,
    problem: Any,
    active_indices_a: Any,
    active_indices_b: Any,
) -> float:
    return float(
        l2_relative_difference_common_active(
            u_a,
            u_b,
            grid=problem.grid,
            active_indices_a=active_indices_a,
            active_indices_b=active_indices_b,
            poly_order=problem.poly_order,
        )
    )
