"""Archive loading and active-node support calculations for displacement studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def active_node_ids(active_cells: np.ndarray, resolution: int) -> np.ndarray:
    """Return unique grid nodes incident to the supplied flat cell indices."""
    cells = np.unique(np.asarray(active_cells, dtype=np.int64))
    node_side = resolution + 1
    i = cells // (resolution * resolution)
    j = (cells // resolution) % resolution
    k = cells % resolution
    base = (i * node_side + j) * node_side + k
    offsets = np.array((
        0, node_side * node_side, node_side * node_side + node_side, node_side,
        1, node_side * node_side + 1, node_side * node_side + node_side + 1, node_side + 1,
    ))
    return np.unique((base[:, None] + offsets).reshape(-1))


def load_archived_displacement_field(archive_directory: Path, framework: str, resolution: int) -> dict[str, np.ndarray]:
    """Load one archived displacement field using the established NPZ schema."""
    archive_path = archive_directory / f"hanging_sphere_{framework}_displacement_{resolution}cubed.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing archived displacement field: {archive_path}")
    with np.load(archive_path, allow_pickle=True) as archive:
        return {
            "u_nodes": np.asarray(archive["u_nodes"], dtype=np.float64),
            "active_cells": np.asarray(archive["active_cells"], dtype=np.int64),
            "active_nodes": np.asarray(archive["active_nodes"], dtype=np.int64),
        }


def common_active_support(
    reference: dict[str, np.ndarray], predicted: dict[str, np.ndarray], resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate archived active-node arrays and return shared and exclusive supports."""
    reference_nodes = active_node_ids(reference["active_cells"], resolution)
    predicted_nodes = active_node_ids(predicted["active_cells"], resolution)
    if not np.array_equal(np.unique(reference["active_nodes"]), reference_nodes):
        raise RuntimeError("Reference archive active_nodes does not match nodes incident to its active cells.")
    if not np.array_equal(np.unique(predicted["active_nodes"]), predicted_nodes):
        raise RuntimeError("Predicted archive active_nodes does not match nodes incident to its active cells.")
    common = np.intersect1d(reference_nodes, predicted_nodes, assume_unique=True)
    if not len(common):
        raise RuntimeError("The archived fields have no common physical active-node support.")
    return common, np.setdiff1d(reference_nodes, common, assume_unique=True), np.setdiff1d(predicted_nodes, common, assume_unique=True)
