from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "SimulationCase": (".case", "SimulationCase"),
    "build_example_case": (".case", "build_example_case"),
    "collect_global_displacements": (".compare", "collect_global_displacements"),
    "compare_pipelines": (".compare", "compare_pipelines"),
    "run_pipelines": (".compare", "run_pipelines"),
    "CaseConfig": (".config", "CaseConfig"),
    "ComparisonRequest": (".config", "ComparisonRequest"),
    "NNRuntimeConfig": (".config", "NNRuntimeConfig"),
    "make_comparison_request": (".config", "make_comparison_request"),
    "compare_with_queso_defaults": (".queso", "compare_with_queso_defaults"),
    "run_comparison_once": (".core.comparison_api", "run_comparison_once"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))

__all__ = [
    "build_example_case",
    "SimulationCase",
    "run_pipelines",
    "compare_pipelines",
    "collect_global_displacements",
    "compare_with_queso_defaults",
    "run_comparison_once",
    "make_comparison_request",
    "CaseConfig",
    "ComparisonRequest",
    "NNRuntimeConfig",
    "internal_nn_runtime",
]
