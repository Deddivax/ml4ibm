from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .core.legacy_imports import configure_legacy_imports
from .case import build_case
from .config import ComparisonRequest
from .nn_runtime import load_nn_engine
from .pipelines import run_pipeline


def _resolve_latest_checkpoint(workspace_root: Path | None) -> Path | None:
    if workspace_root is None:
        return None

    roots = [
        workspace_root / "clean_simulation" / "outputs" / "checkpoints",
        workspace_root / "NN-IBM" / "outputs" / "training_checkpoints",
    ]

    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob("*.pt")))

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _compute_precision(
    *,
    pipeline_result: dict[str, Any],
    reference_result: dict[str, Any],
    case: Any,
) -> float:
    configure_legacy_imports(case.workspace_root)
    from internal_fem_kernel.pipeline_api import relative_l2_on_common_active

    return relative_l2_on_common_active(
        u_a=pipeline_result["u_nodes"],
        u_b=reference_result["u_nodes"],
        problem=case,
        active_indices_a=pipeline_result["active_indices"],
        active_indices_b=reference_result["active_indices"],
    )


def _build_case_and_engine(request: ComparisonRequest) -> tuple[Any, Any | None]:
    case = build_case(
        case_builder=request.case.case_builder,
        case_args=request.case.case_args,
        overrides=request.case.overrides,
        workspace_root=request.case.workspace_root,
        work_dir=request.case.work_dir,
        queso_python_path=request.case.queso_python_path,
        queso_basis_order=request.case.queso_basis_order,
        keep_negative_weights=request.case.keep_negative_weights,
        weight_zero_tol=request.case.weight_zero_tol,
        device=request.device,
    )

    nn_engine = None
    need_nn = any(name in {"hybrid"} for name in request.pipelines)
    if need_nn:
        ckpt_path = request.nn.checkpoint_path
        if ckpt_path is None:
            ckpt_path = _resolve_latest_checkpoint(case.workspace_root)
        if ckpt_path is None:
            raise ValueError("At least one requested pipeline needs a NN checkpoint.")
        nn_engine = load_nn_engine(
            checkpoint_path=ckpt_path,
            device=request.nn.device,
            dtype=request.nn.dtype,
        )
    return case, nn_engine


def run_pipelines(
    request: ComparisonRequest,
    *,
    case: Any | None = None,
    nn_engine: Any | None = None,
) -> dict[str, dict[str, Any]]:
    if case is None:
        case, built_nn_engine = _build_case_and_engine(request)
        if nn_engine is None:
            nn_engine = built_nn_engine

    results: dict[str, dict[str, Any]] = {}

    # Shared dependency for alignment and nitsche re-use in mixed pipelines.
    if any(name in {"hybrid", request.reference_pipeline} for name in request.pipelines):
        if "classic" not in results:
            results["classic"] = run_pipeline(
                name="classic",
                case=case,
                device=request.device,
                nn_engine=nn_engine,
                classic_result=None,
            )

    classic_result = results.get("classic")

    for name in request.pipelines:
        if name in results:
            continue
        results[name] = run_pipeline(
            name=name,
            case=case,
            device=request.device,
            nn_engine=nn_engine,
            classic_result=classic_result,
        )

    return results


def compare_pipelines(request: ComparisonRequest) -> dict[str, Any]:
    case, nn_engine = _build_case_and_engine(request)
    results = run_pipelines(request, case=case, nn_engine=nn_engine)

    ref_name = request.reference_pipeline
    if ref_name not in results:
        classic_result = results.get("classic")
        results[ref_name] = run_pipeline(
            name=ref_name,
            case=case,
            device=request.device,
            nn_engine=nn_engine,
            classic_result=classic_result,
        )

    ref_result = results[ref_name]

    rows: list[dict[str, Any]] = []
    for name in request.pipelines:
        item = results[name]
        timings = dict(item.get("timings", {}))
        rows.append(
            {
                "pipeline": name,
                "total_time_s": float(timings.get("total", 0.0)),
                "timings": timings,
                "relative_l2_error_vs_reference": 0.0
                if name == ref_name
                else _compute_precision(
                    pipeline_result=item,
                    reference_result=ref_result,
                    case=case,
                ),
                "active_elements": int(len(item.get("active_indices", []))),
            }
        )

    return {
        "request": asdict(request),
        "reference_pipeline": ref_name,
        "rows": rows,
        "queso_metadata": case.queso_metadata,
        "case": {
            "poly_order": case.poly_order,
            "gauss_order": case.g_order,
            "work_dir": str(case.work_dir),
            "workspace_root": str(case.workspace_root),
        },
    }


def collect_global_displacements(request: ComparisonRequest) -> dict[str, Any]:
    case, nn_engine = _build_case_and_engine(request)
    results = run_pipelines(request, case=case, nn_engine=nn_engine)

    displacements: dict[str, Any] = {}
    for name in request.pipelines:
        item = results[name]
        displacements[name] = {
            "u_nodes": item.get("u_nodes"),
            "u_vec": item.get("u_vec"),
            "active_indices": item.get("active_indices"),
            "n_nodes_total": item.get("n_nodes_total"),
            "timings": dict(item.get("timings", {})),
            "k_global": item.get("k_global"),
        }

    return {
        "request": asdict(request),
        "displacements": displacements,
        "case": {
            "poly_order": case.poly_order,
            "gauss_order": case.g_order,
            "work_dir": str(case.work_dir),
            "workspace_root": str(case.workspace_root),
        },
    }
