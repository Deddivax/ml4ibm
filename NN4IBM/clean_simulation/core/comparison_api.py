"""Convenience API for one configured pipeline comparison."""

from __future__ import annotations

from typing import Any

from ..compare import compare_pipelines
from ..config import make_comparison_request


def run_comparison_once(
    *,
    pipelines: str | tuple[str, ...] = ("classic", "queso"),
    reference_pipeline: str = "classic",
    device: str = "cpu",
    nn_checkpoint_path=None,
    nn_device: str = "cpu",
    nn_dtype: str = "float32",
    case_builder: str | None = None,
    case_args: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    workspace_root=None,
    work_dir=None,
    queso_python_path=None,
    queso_basis_order: int = 2,
    keep_negative_weights: bool = True,
    weight_zero_tol: float = 0.0,
) -> dict[str, Any]:
    """Build a comparison request and run it once."""
    request = make_comparison_request(
        pipelines=pipelines,
        reference_pipeline=reference_pipeline,
        device=device,
        nn_checkpoint_path=nn_checkpoint_path,
        nn_device=nn_device,
        nn_dtype=nn_dtype,
        case_builder=case_builder,
        case_args=case_args,
        overrides=overrides,
        workspace_root=workspace_root,
        work_dir=work_dir,
        queso_python_path=queso_python_path,
        queso_basis_order=queso_basis_order,
        keep_negative_weights=keep_negative_weights,
        weight_zero_tol=weight_zero_tol,
    )
    return compare_pipelines(request)


__all__ = ["run_comparison_once"]
