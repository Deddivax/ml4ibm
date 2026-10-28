from pathlib import Path
import sys
from importlib import import_module

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "run_classic_pipeline": (".pipeline_api", "run_classic_pipeline"),
    "run_queso_pipeline": (".pipeline_api", "run_queso_pipeline"),
    "run_hybrid_pipeline": (".pipeline_api", "run_hybrid_pipeline"),
    "relative_l2_on_common_active": (".pipeline_api", "relative_l2_on_common_active"),
    "run_pipeline": (".workflows", "run_pipeline"),
    "solve_global_displacement": (".workflows", "solve_global_displacement"),
    "compare_displacements": (".workflows", "compare_displacements"),
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
    "run_classic_pipeline",
    "run_queso_pipeline",
    "run_hybrid_pipeline",
    "relative_l2_on_common_active",
    "run_pipeline",
    "solve_global_displacement",
    "compare_displacements",
]
