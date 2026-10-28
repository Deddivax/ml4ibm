from __future__ import annotations

from typing import Any

from .common import (
    CANONICAL_STAGE_KEYS,
    as_bool_mask,
    as_index_tensor_cpu,
    as_tensor,
    canonical_timings,
    finalize_pipeline_timings,
    l2_relative_difference_common_active,
    reset_pipeline_runtime_caches,
    sum_duration_fields,
    sync_cuda_if_needed,
)


def run_classic_pipeline_timed(*args: Any, **kwargs: Any) -> Any:
    from .pipelines import run_classic_pipeline_timed as _run_classic_pipeline_timed

    return _run_classic_pipeline_timed(*args, **kwargs)


def run_hybrid_pipeline_timed(*args: Any, **kwargs: Any) -> Any:
    from .pipelines import run_hybrid_pipeline_timed as _run_hybrid_pipeline_timed

    return _run_hybrid_pipeline_timed(*args, **kwargs)


def run_queso_pipeline_timed(*args: Any, **kwargs: Any) -> Any:
    from .pipelines import run_queso_pipeline_timed as _run_queso_pipeline_timed

    return _run_queso_pipeline_timed(*args, **kwargs)

__all__ = [
    "CANONICAL_STAGE_KEYS",
    "as_bool_mask",
    "as_index_tensor_cpu",
    "as_tensor",
    "canonical_timings",
    "finalize_pipeline_timings",
    "l2_relative_difference_common_active",
    "reset_pipeline_runtime_caches",
    "run_classic_pipeline_timed",
    "run_full_nn_pipeline_timed",
    "run_hybrid_pipeline_timed",
    "run_queso_nn_pipeline_timed",
    "run_queso_nn_pipeline_timed_from_ips",
    "run_queso_pipeline_timed",
    "sum_duration_fields",
    "sync_cuda_if_needed",
]
