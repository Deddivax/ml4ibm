from __future__ import annotations

from typing import Any

from clean_simulation.compare import compare_pipelines
from clean_simulation.config import CaseConfig, ComparisonRequest, NNRuntimeConfig


def compare_with_queso_defaults(
    *,
    device: str = "cpu",
    reference_pipeline: str = "classic",
    case: CaseConfig | None = None,
    nn: NNRuntimeConfig | None = None,
    pipelines: tuple[str, ...] = ("classic", "queso"),
) -> dict[str, Any]:
    request = ComparisonRequest(
        pipelines=pipelines,
        reference_pipeline=reference_pipeline,
        device=device,
        case=CaseConfig() if case is None else case,
        nn=NNRuntimeConfig() if nn is None else nn,
    )
    return compare_pipelines(request)


__all__ = ["compare_with_queso_defaults"]
