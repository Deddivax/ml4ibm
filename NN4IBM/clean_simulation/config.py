from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class NNRuntimeConfig:
    checkpoint_path: Path | None = None
    device: str = "cpu"
    dtype: str = "float32"


@dataclass(slots=True)
class CaseConfig:
    case_builder: str | None = None
    case_args: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    workspace_root: Path | None = None
    work_dir: Path | None = None
    queso_python_path: Path | None = None
    queso_basis_order: int = 2
    keep_negative_weights: bool = True
    weight_zero_tol: float = 0.0


@dataclass(slots=True)
class ComparisonRequest:
    pipelines: tuple[str, ...] = ("classic", "queso")
    reference_pipeline: str = "classic"
    device: str = "cpu"
    nn: NNRuntimeConfig = field(default_factory=NNRuntimeConfig)
    case: CaseConfig = field(default_factory=CaseConfig)


def make_comparison_request(
    *,
    pipelines: str | tuple[str, ...] = ("classic", "queso"),
    reference_pipeline: str = "classic",
    device: str = "cpu",
    nn_checkpoint_path: Path | str | None = None,
    nn_device: str = "cpu",
    nn_dtype: str = "float32",
    case_builder: str | None = None,
    case_args: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    workspace_root: Path | str | None = None,
    work_dir: Path | str | None = None,
    queso_python_path: Path | str | None = None,
    queso_basis_order: int = 2,
    keep_negative_weights: bool = True,
    weight_zero_tol: float = 0.0,
) -> ComparisonRequest:
    if isinstance(pipelines, str):
        pipeline_names = tuple(part.strip() for part in pipelines.split(",") if part.strip())
    else:
        pipeline_names = tuple(str(name).strip() for name in pipelines if str(name).strip())

    if not pipeline_names:
        pipeline_names = ("classic", "queso")

    ref = reference_pipeline.strip() or pipeline_names[0]

    return ComparisonRequest(
        pipelines=pipeline_names,
        reference_pipeline=ref,
        device=device,
        nn=NNRuntimeConfig(
            checkpoint_path=None if nn_checkpoint_path is None else Path(nn_checkpoint_path),
            device=nn_device,
            dtype=nn_dtype,
        ),
        case=CaseConfig(
            case_builder=case_builder,
            case_args={} if case_args is None else dict(case_args),
            overrides={} if overrides is None else dict(overrides),
            workspace_root=None if workspace_root is None else Path(workspace_root),
            work_dir=None if work_dir is None else Path(work_dir),
            queso_python_path=None if queso_python_path is None else Path(queso_python_path),
            queso_basis_order=int(queso_basis_order),
            keep_negative_weights=bool(keep_negative_weights),
            weight_zero_tol=float(weight_zero_tol),
        ),
    )
