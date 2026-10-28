from .integration import map_integration_points_to_elements, run_queso_get_integration_points


def compare_with_queso_defaults(*args, **kwargs):
    from .performance import compare_with_queso_defaults as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "run_queso_get_integration_points",
    "map_integration_points_to_elements",
    "compare_with_queso_defaults",
]
