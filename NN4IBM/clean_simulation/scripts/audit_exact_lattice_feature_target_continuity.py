"""Measure the local relationship between exact-lattice eight-SDF inputs and K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


GROUP_NAME = "integration_2_poly_1"
FEATURE_SOURCE = "exact_original_sdf_marching_cubes_lattice"
THRESHOLDS = np.asarray((1.0e-12, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1))
UPPER_I, UPPER_J = np.triu_indices(24)
FROBENIUS_WEIGHTS = np.where(UPPER_I == UPPER_J, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worst-pairs", type=int, default=100)
    return parser.parse_args()


def decode_names(values: np.ndarray) -> np.ndarray:
    return np.char.decode(values, "utf-8") if values.dtype.kind == "S" else values.astype(str)


def symmetric_frobenius(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(FROBENIUS_WEIGHTS * values * values, axis=1))


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.worst_pairs < 1:
        raise ValueError("--worst-pairs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, "r") as handle:
        group = handle[GROUP_NAME]
        if group.attrs.get("feature_source") != FEATURE_SOURCE:
            raise ValueError("This audit requires the exact original-SDF lattice corpus")
        dataset_rows = len(group["flat_index"])
        names = decode_names(group["geometry_name"][...])
        all_records, worst_records = [], []
        for geometry_name in np.unique(names):
            global_indices = np.flatnonzero(names == geometry_name)
            features = np.asarray(group["d_values"][global_indices], dtype=np.float64)
            targets = np.asarray(group["stiffness_ut"][global_indices], dtype=np.float64)
            distances, neighbors = cKDTree(features).query(features, k=2)
            nearest_distance = distances[:, 1]
            nearest_local = neighbors[:, 1]
            neighbor_targets = targets[nearest_local]
            target_norm = symmetric_frobenius(targets)
            neighbor_norm = symmetric_frobenius(neighbor_targets)
            target_difference = symmetric_frobenius(targets - neighbor_targets)
            relative_difference = target_difference / np.maximum(0.5 * (target_norm + neighbor_norm), 1.0e-12)
            correlation = spearmanr(nearest_distance, relative_difference).statistic
            for threshold in THRESHOLDS:
                selected = nearest_distance <= threshold
                if not selected.any():
                    continue
                all_records.append({
                    "geometry_name": geometry_name,
                    "sdf_l2_distance_at_most": float(threshold),
                    "pairs": int(selected.sum()),
                    "median_relative_K_difference": float(np.median(relative_difference[selected])),
                    "p95_relative_K_difference": float(np.quantile(relative_difference[selected], 0.95)),
                    "max_relative_K_difference": float(np.max(relative_difference[selected])),
                    "median_absolute_K_difference": float(np.median(target_difference[selected])),
                    "spearman_nearest_d_to_relative_K_gap": float(correlation),
                })
            candidate_count = min(args.worst_pairs, len(global_indices))
            for local_index in np.argpartition(relative_difference, -candidate_count)[-candidate_count:]:
                partner_local = int(nearest_local[local_index])
                worst_records.append({
                    "geometry_name": geometry_name,
                    "row_index": int(global_indices[local_index]),
                    "neighbor_row_index": int(global_indices[partner_local]),
                    "sdf_l2_distance": float(nearest_distance[local_index]),
                    "sdf_max_corner_difference": float(np.abs(features[local_index] - features[partner_local]).max()),
                    "K_difference_frobenius": float(target_difference[local_index]),
                    "relative_K_difference": float(relative_difference[local_index]),
                    "case_code": int(np.sum((features[local_index] > 0.5) * (1 << np.arange(8)))),
                    "neighbor_case_code": int(np.sum((features[partner_local] > 0.5) * (1 << np.arange(8)))),
                })

    summaries = pd.DataFrame(all_records)
    worst_pairs = pd.DataFrame(worst_records).nlargest(args.worst_pairs, "relative_K_difference")
    aggregate = summaries.groupby("sdf_l2_distance_at_most", as_index=False).agg(
        geometries=("geometry_name", "nunique"), pairs=("pairs", "sum"),
        median_of_geometry_medians=("median_relative_K_difference", "median"),
        worst_geometry_p95_relative_K_difference=("p95_relative_K_difference", "max"),
        worst_geometry_max_relative_K_difference=("max_relative_K_difference", "max"),
        median_spearman_correlation=("spearman_nearest_d_to_relative_K_gap", "median"),
    )
    summaries.to_csv(args.output_dir / "nearest_neighbor_continuity_by_geometry.csv", index=False)
    aggregate.to_csv(args.output_dir / "nearest_neighbor_continuity_summary.csv", index=False)
    worst_pairs.to_csv(args.output_dir / "nearest_neighbor_continuity_worst_pairs.csv", index=False)
    exact = aggregate.loc[aggregate["sdf_l2_distance_at_most"] == THRESHOLDS[0]]
    report = {
        "data": str(args.data), "rows": dataset_rows,
        "thresholds": THRESHOLDS.tolist(), "exact_duplicate_summary": exact.to_dict("records"),
        "worst_pair": worst_pairs.iloc[0].to_dict() if len(worst_pairs) else None,
    }
    (args.output_dir / "nearest_neighbor_continuity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()