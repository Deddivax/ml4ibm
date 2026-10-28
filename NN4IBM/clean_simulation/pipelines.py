from __future__ import annotations

from typing import Any

from .core.legacy_imports import configure_legacy_imports


VALID_PIPELINES = {
    "queso",
    "classic",
    "hybrid",
}


def _require_nn(name: str, nn_engine: Any | None) -> None:
    if nn_engine is None:
        raise ValueError(f"Pipeline '{name}' needs nn_engine. Provide a checkpoint path.")


def run_pipeline(
    *,
    name: str,
    case: Any,
    device: str,
    nn_engine: Any | None = None,
    classic_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configure_legacy_imports(case.workspace_root)
    from internal_fem_kernel.pipeline_api import (
        run_classic_pipeline,
        run_hybrid_pipeline,
        run_queso_pipeline,
    )
    pipeline = name.strip().lower()
    if pipeline not in VALID_PIPELINES:
        known = ", ".join(sorted(VALID_PIPELINES))
        raise ValueError(f"Unknown pipeline '{name}'. Expected one of: {known}")
    if pipeline == "queso":
        return run_queso_pipeline(problem=case, device=device)

    if pipeline == "classic":
        return run_classic_pipeline(problem=case, device=device)

    if pipeline == "hybrid":
        _require_nn("hybrid", nn_engine)
        return run_hybrid_pipeline(problem=case, device=device, nn_engine=nn_engine, classic_result=classic_result)

