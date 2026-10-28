"""Geometry helpers for Blade STL study commands."""

import numpy as np


def cube_bounds(mesh_bounds: np.ndarray, padding_fraction: float = 0.10) -> list[list[float]]:
    center = mesh_bounds.mean(axis=0)
    half_width = float(np.max(mesh_bounds[1] - mesh_bounds[0])) * (0.5 + padding_fraction)
    return [(center - half_width).tolist(), (center + half_width).tolist()]