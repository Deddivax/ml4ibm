"""Fail-closed integrity audit for exact original-SDF lattice cut-cell data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


GROUP_NAME = "integration_2_poly_1"
FEATURE_SOURCE = "exact_original_sdf_marching_cubes_lattice"
REQUIRED_DATASETS = ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name")
RANGE_TOLERANCE = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    return parser.parse_args()


def decode_names(values: np.ndarray) -> np.ndarray:
    return np.char.decode(values, "utf-8") if values.dtype.kind == "S" else values.astype(str)


def packed_diagonal(values: np.ndarray) -> np.ndarray:
    upper_i, upper_j = np.triu_indices(24)
    diagonal_positions = np.flatnonzero(upper_i == upper_j)
    return values[:, diagonal_positions]


def audit(data_path: Path, chunk_rows: int) -> tuple[pd.DataFrame, dict[str, object]]:
    if chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    with h5py.File(data_path, "r") as handle:
        if GROUP_NAME not in handle:
            raise ValueError(f"Missing HDF5 group {GROUP_NAME!r}")
        group = handle[GROUP_NAME]
        missing = [name for name in REQUIRED_DATASETS if name not in group]
        if missing:
            raise ValueError(f"Missing required datasets: {missing}")
        if group.attrs.get("feature_source") != FEATURE_SOURCE:
            raise ValueError(f"Expected feature_source={FEATURE_SOURCE!r}, got {group.attrs.get('feature_source')!r}")
        if int(group.attrs.get("feature_count", -1)) != 8:
            raise ValueError("Dataset does not declare the strict eight-feature contract")

        rows = len(group["flat_index"])
        expected_shapes = {"d_values": (rows, 8), "stiffness_ut": (rows, 300), "cut_volume_fraction": (rows,), "geometry_name": (rows,)}
        actual_shapes = {name: group[name].shape for name in expected_shapes}
        if actual_shapes != expected_shapes:
            raise ValueError(f"Dataset shapes do not match the exact-lattice schema: {actual_shapes}")

        aggregates: dict[str, dict[str, object]] = defaultdict(lambda: {
            "rows": 0, "finite_rows": 0, "feature_range_rows": 0, "volume_range_rows": 0,
            "crossing_rows": 0, "nonnegative_diagonal_rows": 0, "flat_indices": [],
        })
        totals = {"finite_rows": 0, "feature_range_rows": 0, "volume_range_rows": 0, "crossing_rows": 0, "nonnegative_diagonal_rows": 0}
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            flat_indices = group["flat_index"][start:stop]
            d_values = group["d_values"][start:stop]
            stiffness = group["stiffness_ut"][start:stop]
            fractions = group["cut_volume_fraction"][start:stop]
            names = decode_names(group["geometry_name"][start:stop])
            finite = np.isfinite(d_values).all(axis=1) & np.isfinite(stiffness).all(axis=1) & np.isfinite(fractions)
            feature_range = (d_values >= -RANGE_TOLERANCE).all(axis=1) & (d_values <= 1.0 + RANGE_TOLERANCE).all(axis=1)
            volume_range = (fractions >= -RANGE_TOLERANCE) & (fractions <= 1.0 + RANGE_TOLERANCE)
            crossing = (d_values.min(axis=1) <= 0.5) & (d_values.max(axis=1) >= 0.5)
            nonnegative_diagonal = (packed_diagonal(stiffness) >= -RANGE_TOLERANCE).all(axis=1)
            for name in np.unique(names):
                mask = names == name
                stats = aggregates[str(name)]
                stats["rows"] += int(mask.sum())
                for key, values in (("finite_rows", finite), ("feature_range_rows", feature_range), ("volume_range_rows", volume_range), ("crossing_rows", crossing), ("nonnegative_diagonal_rows", nonnegative_diagonal)):
                    count = int(values[mask].sum())
                    stats[key] += count
                    totals[key] += count
                stats["flat_indices"].append(flat_indices[mask])

        records = []
        for name, stats in sorted(aggregates.items()):
            flat_indices = np.concatenate(stats.pop("flat_indices"))
            unique_indices = np.unique(flat_indices)
            records.append({
                "geometry_name": name,
                **stats,
                "unique_flat_indices": int(len(unique_indices)),
                "duplicate_flat_indices": int(len(flat_indices) - len(unique_indices)),
                "out_of_grid_flat_indices": int(((flat_indices < 0) | (flat_indices >= 100 ** 3)).sum()),
            })
        per_geometry = pd.DataFrame(records)
        rejected = json.loads(group.attrs.get("rejected_geometries", "[]"))
        summary = {
            "data": str(data_path), "rows": rows, "geometries": len(per_geometry), "feature_source": FEATURE_SOURCE,
            "feature_convention": str(group.attrs.get("feature_convention", "")), "pipeline": str(group.attrs.get("pipeline", "")),
            "mesh_validation": str(group.attrs.get("mesh_validation", "")), "rejected_geometries": rejected, **totals,
        }
        summary["eight_corner_crossing_fraction"] = totals["crossing_rows"] / rows

    failures = [
        summary["finite_rows"] != rows,
        summary["feature_range_rows"] != rows,
        summary["volume_range_rows"] != rows,
        summary["nonnegative_diagonal_rows"] != rows,
        bool((per_geometry["duplicate_flat_indices"] != 0).any()),
        bool((per_geometry["out_of_grid_flat_indices"] != 0).any()),
    ]
    if any(failures):
        raise ValueError(json.dumps({"summary": summary, "failed_geometries": per_geometry.to_dict("records")}, indent=2))
    return per_geometry, summary


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_geometry, summary = audit(args.data, args.chunk_rows)
    per_geometry.to_csv(args.output_dir / "exact_lattice_health_by_geometry.csv", index=False)
    (args.output_dir / "exact_lattice_health_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Exact-lattice health gate passed. Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()