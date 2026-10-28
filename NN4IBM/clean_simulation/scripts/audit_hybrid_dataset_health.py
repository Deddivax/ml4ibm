"""Fail-closed health audit for hybrid parity-aware cut-cell training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GROUP_NAME = "integration_2_poly_1"
LOW_CORNER_THRESHOLD = 0.25
HIGH_VOLUME_THRESHOLD = 0.75
MIN_PARITY_VOLUME_CORRELATION = 0.90
RANGE_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=("cutcellnet", "transformer"), default="transformer")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--residual-scaling", default="rezero")
    parser.add_argument("--split-seed", type=int, default=227344)
    return parser.parse_args()


def decode_names(values: np.ndarray) -> np.ndarray:
    return np.char.decode(values, "utf-8") if values.dtype.kind == "S" else values.astype(str)


def audit_data(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    with h5py.File(path, "r") as handle:
        if GROUP_NAME not in handle:
            raise ValueError(f"Missing HDF5 group {GROUP_NAME!r}.")
        group = handle[GROUP_NAME]
        required = ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name")
        missing = [name for name in required if name not in group]
        if missing:
            raise ValueError(f"Missing required datasets: {missing}")
        flat_index = group["flat_index"][...]
        distances = group["d_values"][...]
        stiffness = group["stiffness_ut"][...]
        volume_fraction = group["cut_volume_fraction"][...]
        geometry_name = decode_names(group["geometry_name"][...])
        attributes = {key: group.attrs[key] for key in group.attrs}

    rows = len(flat_index)
    if rows == 0 or distances.shape != (rows, 8) or stiffness.shape != (rows, 300):
        raise ValueError(
            f"Expected non-empty (N, 8) distances and (N, 300) stiffness targets; "
            f"received {distances.shape} and {stiffness.shape}."
        )
    if volume_fraction.shape != (rows,) or geometry_name.shape != (rows,):
        raise ValueError("Cut-volume fraction or geometry-name rows do not match the feature rows.")

    finite = np.isfinite(distances).all(axis=1) & np.isfinite(stiffness).all(axis=1) & np.isfinite(volume_fraction)
    in_distance_range = (
        (distances >= -RANGE_TOLERANCE).all(axis=1)
        & (distances <= 1.0 + RANGE_TOLERANCE).all(axis=1)
    )
    in_volume_range = (
        (volume_fraction >= -RANGE_TOLERANCE)
        & (volume_fraction <= 1.0 + RANGE_TOLERANCE)
    )
    corner_fraction = (distances > 0.5).mean(axis=1)
    contradictory = (
        ((corner_fraction <= LOW_CORNER_THRESHOLD) & (volume_fraction >= HIGH_VOLUME_THRESHOLD))
        | ((corner_fraction >= 1.0 - LOW_CORNER_THRESHOLD) & (volume_fraction <= 1.0 - HIGH_VOLUME_THRESHOLD))
    )

    geometry_rows = []
    for name in np.unique(geometry_name):
        mask = geometry_name == name
        correlation = float(np.corrcoef(corner_fraction[mask] - 0.5, volume_fraction[mask])[0, 1])
        geometry_rows.append(
            {
                "geometry_name": name,
                "rows": int(mask.sum()),
                "parity_volume_correlation": correlation,
                "contradictory_rows": int(contradictory[mask].sum()),
                "finite_rows": int(finite[mask].sum()),
                "distance_range_rows": int(in_distance_range[mask].sum()),
                "volume_range_rows": int(in_volume_range[mask].sum()),
            }
        )
    per_geometry = pd.DataFrame(geometry_rows).sort_values("geometry_name").reset_index(drop=True)
    summary = {
        "data": str(path),
        "rows": rows,
        "geometries": int(len(per_geometry)),
        "feature_shape": list(distances.shape),
        "target_shape": list(stiffness.shape),
        "finite_rows": int(finite.sum()),
        "distance_range_rows": int(in_distance_range.sum()),
        "volume_range_rows": int(in_volume_range.sum()),
        "contradictory_rows": int(contradictory.sum()),
        "minimum_parity_volume_correlation": float(per_geometry["parity_volume_correlation"].min()),
        "source_geometries": attributes.get("source_geometries", ""),
        "reconstruction_audit": attributes.get("reconstruction_audit", ""),
    }
    failed = (
        summary["finite_rows"] != rows
        or summary["distance_range_rows"] != rows
        or summary["volume_range_rows"] != rows
        or summary["contradictory_rows"] != 0
        or summary["minimum_parity_volume_correlation"] < MIN_PARITY_VOLUME_CORRELATION
    )
    if failed:
        raise ValueError(json.dumps(summary, indent=2, default=str))
    return per_geometry, summary


def run_checkpoint_audit(args: argparse.Namespace, output_dir: Path) -> Path:
    distribution_path = output_dir / "previous_stage_2_validation_loss_distribution.csv"
    command = [
        sys.executable,
        str(ROOT / "clean_simulation" / "scripts" / "audit_validation_loss_distribution.py"),
        "--checkpoint", str(args.checkpoint),
        "--data", str(args.data),
        "--model-type", args.model_type,
        "--width", str(args.width),
        "--blocks", str(args.blocks),
        "--residual-scaling", args.residual_scaling,
        "--seed", str(args.split_seed),
        "--output", str(distribution_path),
    ]
    subprocess.run(command, check=True)
    return distribution_path


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not args.data.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("The requested HDF5 data or previous Stage 2 checkpoint is missing.")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_geometry, summary = audit_data(args.data)
    per_geometry.to_csv(output_dir / "dataset_health_by_geometry.csv", index=False)
    distribution_path = run_checkpoint_audit(args, output_dir)
    distribution = pd.read_csv(distribution_path)
    error_by_geometry = distribution.groupby("geometry_name", as_index=False).agg(
        validation_rows=("floored_relative_frobenius_error", "size"),
        mean_error=("floored_relative_frobenius_error", "mean"),
        median_error=("floored_relative_frobenius_error", "median"),
        p95_error=("floored_relative_frobenius_error", lambda values: values.quantile(0.95)),
        p99_error=("floored_relative_frobenius_error", lambda values: values.quantile(0.99)),
    )
    error_by_geometry.to_csv(output_dir / "previous_stage_2_error_by_geometry.csv", index=False)
    summary["previous_stage_2_checkpoint"] = str(args.checkpoint)
    summary["validation_distribution"] = str(distribution_path)
    summary["validation_rows"] = int(len(distribution))
    summary["validation_error_quantiles"] = {
        str(level): float(distribution["floored_relative_frobenius_error"].quantile(level))
        for level in (0.5, 0.9, 0.95, 0.99, 0.999)
    }
    (output_dir / "dataset_health_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Health gate passed. Reports written to {output_dir}")


if __name__ == "__main__":
    main()