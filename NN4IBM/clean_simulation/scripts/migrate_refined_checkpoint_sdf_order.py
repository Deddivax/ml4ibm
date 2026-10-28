"""Migrate a refined CutCellNet checkpoint to the corrected SDF feature order."""

from __future__ import annotations

import argparse
from pathlib import Path
import pathlib
import sys
import types

import torch


ORDER_MARKER = "vtk_corners_then_x_major_non_corners_v2"
OLD_ORDER_MARKER = "vtk_corners_then_z_major_non_corners_v1"


def _install_pathlib_local_shim() -> None:
    """Allow loading checkpoints written by Python versions with pathlib._local."""
    if "pathlib._local" in sys.modules:
        return

    shim = types.ModuleType("pathlib._local")
    shim.Path = pathlib.Path
    shim.PosixPath = pathlib.PosixPath
    shim.WindowsPath = pathlib.WindowsPath
    shim.PurePath = pathlib.PurePath
    shim.PurePosixPath = pathlib.PurePosixPath
    shim.PureWindowsPath = pathlib.PureWindowsPath
    sys.modules["pathlib._local"] = shim


def _source_columns(refined_subcells: int) -> torch.Tensor:
    """Return old feature positions for each corrected feature position."""
    node_count = refined_subcells + 1
    width = node_count ** 3
    corner_xyz = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=torch.long,
    )
    old_corner_raw = (
        corner_xyz[:, 0] * refined_subcells
        + corner_xyz[:, 1] * refined_subcells * node_count
        + corner_xyz[:, 2] * refined_subcells * node_count * node_count
    )
    new_corner_raw = (
        corner_xyz[:, 0] * refined_subcells * node_count * node_count
        + corner_xyz[:, 1] * refined_subcells * node_count
        + corner_xyz[:, 2] * refined_subcells
    )
    raw_indices = torch.arange(width, dtype=torch.long)
    old_raw_order = torch.cat((old_corner_raw, raw_indices[~torch.isin(raw_indices, old_corner_raw)]))
    new_raw_order = torch.cat((new_corner_raw, raw_indices[~torch.isin(raw_indices, new_corner_raw)]))
    old_position_for_raw = torch.empty(width, dtype=torch.long)
    old_position_for_raw[old_raw_order] = torch.arange(width, dtype=torch.long)
    return old_position_for_raw[new_raw_order]


def migrate_checkpoint(input_path: Path, output_path: Path) -> None:
    _install_pathlib_local_shim()
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if checkpoint.get("sdf_feature_order") == ORDER_MARKER:
        raise ValueError(f"{input_path} is already marked as {ORDER_MARKER}.")

    model_cfg = checkpoint.get("model_cfg", checkpoint.get("model_config", {})) or {}
    width = int(model_cfg.get("n_distance_nodes", 0))
    node_count = round(width ** (1.0 / 3.0))
    if node_count ** 3 != width or node_count <= 2:
        raise ValueError(f"Expected a refined cubic SDF feature width; got {width}.")

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None or "input_proj.weight" not in state_dict:
        raise KeyError("Checkpoint does not contain CutCellNet input_proj.weight.")

    source_columns = _source_columns(node_count - 1)
    input_weight = state_dict["input_proj.weight"]
    if input_weight.shape[1] != width:
        raise ValueError(
            f"Expected {width} input columns with no nu/class inputs; got {input_weight.shape[1]}."
        )
    state_dict["input_proj.weight"] = input_weight[:, source_columns].clone()

    for name in ("in_mean", "in_std"):
        if name in state_dict:
            if state_dict[name].numel() != width:
                raise ValueError(f"State-dict {name} has unexpected width {state_dict[name].numel()}.")
            state_dict[name] = state_dict[name][source_columns].clone()

    normalizers = checkpoint.get("normalizers", {}) or {}
    for name in ("in_mean", "in_std"):
        if name in normalizers:
            values = torch.as_tensor(normalizers[name])
            if values.numel() != width:
                raise ValueError(f"Checkpoint normalizer {name} has unexpected width {values.numel()}.")
            normalizers[name] = values[source_columns].tolist()
    checkpoint["normalizers"] = normalizers
    checkpoint["sdf_feature_order"] = ORDER_MARKER
    checkpoint["sdf_feature_order_migrated_from"] = OLD_ORDER_MARKER
    checkpoint["sdf_feature_order_source_checkpoint"] = input_path.name

    torch.save(checkpoint, output_path)


def mark_current_order_checkpoint(input_path: Path, output_path: Path) -> None:
    """Mark an unlabelled checkpoint known to use the current feature order.

    This does not reorder any tensors. Use it only for checkpoints trained from
    a current-order dataset or initialized from an already migrated checkpoint.
    """
    _install_pathlib_local_shim()
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if checkpoint.get("sdf_feature_order") is not None:
        raise ValueError(f"{input_path} already declares an SDF feature order.")

    model_cfg = checkpoint.get("model_cfg", checkpoint.get("model_config", {})) or {}
    if int(model_cfg.get("n_distance_nodes", 0)) <= 8:
        raise ValueError("Only refined checkpoints need an SDF feature-order marker.")

    checkpoint["sdf_feature_order"] = ORDER_MARKER
    checkpoint["sdf_feature_order_marked_from"] = input_path.name
    torch.save(checkpoint, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Checkpoint trained with the legacy refined SDF order.")
    parser.add_argument("output", type=Path, help="New checkpoint path for the corrected SDF order.")
    parser.add_argument(
        "--mark-current-order",
        action="store_true",
        help="Only add the current-order marker; do not permute tensors.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)

    if args.mark_current_order:
        mark_current_order_checkpoint(args.input, args.output)
        print(f"Marked current-order checkpoint: {args.output}")
    else:
        migrate_checkpoint(args.input, args.output)
        print(f"Migrated checkpoint: {args.output}")


if __name__ == "__main__":
    main()