"""Compatibility exports for the internal FEM kernel runtime."""

from __future__ import annotations

from .runtime.common import l2_relative_difference_common_active
from .runtime.pipelines import (
    run_classic_pipeline_timed,
    run_hybrid_pipeline_timed,
    run_queso_pipeline_timed,
)

__all__ = [
    "l2_relative_difference_common_active",
    "run_classic_pipeline_timed",
    "run_hybrid_pipeline_timed",
    "run_queso_pipeline_timed",
]

