from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def make_queso_settings(
    *,
    bounds: Sequence[Sequence[float]],
    resolution: Sequence[int],
    input_filename: str | Path,
    basis_order: int = 2,
    moment_fitting_residual: float = 1.0e-8,
) -> dict[str, Any]:
    """Build the shared QuESo input schema for a geometry and background grid."""
    return {
        "general_settings": {
            "input_filename": str(input_filename),
            "echo_level": 0,
            "write_output_to_file": False,
        },
        "background_grid_settings": {
            "grid_type": "b_spline_grid",
            "lower_bound_xyz": [float(value) for value in bounds[0]],
            "upper_bound_xyz": [float(value) for value in bounds[1]],
            "lower_bound_uvw": [0.0, 0.0, 0.0],
            "upper_bound_uvw": [1.0, 1.0, 1.0],
            "polynomial_order": [int(basis_order)] * 3,
            "number_of_elements": [int(value) for value in resolution],
        },
        "trimmed_quadrature_rule_settings": {
            "moment_fitting_residual": float(moment_fitting_residual),
        },
        "non_trimmed_quadrature_rule_settings": {
            "integration_method": "Gauss",
        },
    }