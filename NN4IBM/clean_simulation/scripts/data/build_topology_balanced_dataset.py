"""Create an HDF5 training corpus with excluded geometries and rare topology oversampling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from clean_simulation.internal_nn_runtime.cutcell_ml.classification_utils import RAW_MASK_TO_CLASS


def _class_labels(transformed_distances: np.ndarray) -> np.ndarray:
    corner_distances = transformed_distances[:, :8]
    powers_of_two = 1 << np.arange(8, dtype=np.uint16)
    raw_masks = ((corner_distances > 0.5).astype(np.uint16) * powers_of_two).sum(axis=1)
    return np.asarray(RAW_MASK_TO_CLASS, dtype=np.int16)[raw_masks]


def _copy_rows(
    source_group: h5py.Group,
    destination_group: h5py.Group,
    source_indices: np.ndarray,
    destination_start: int,
) -> int:
    destination_stop = destination_start + len(source_indices)
    for name, source_dataset in source_group.items():
        if not isinstance(source_dataset, h5py.Dataset):
            continue
        destination_group[name][destination_start:destination_stop] = source_dataset[source_indices]
    destination_group["cut_class"][destination_start:destination_stop] = _class_labels(
        source_group["d_values"][source_indices]
    )
    return destination_stop


def _filtered_geometry_metadata(value: object, excluded_prefixes: tuple[str, ...]) -> object:
    if not isinstance(value, str):
        return value
    try:
        geometry_names = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(geometry_names, list) or not all(isinstance(name, str) for name in geometry_names):
        return value
    return json.dumps([name for name in geometry_names if not name.startswith(excluded_prefixes)])


def build_dataset(
    source_path: Path,
    output_path: Path,
    group_name: str,
    excluded_prefixes: tuple[str, ...],
    rare_share_threshold: float,
    rare_multiplier: int,
    chunk_rows: int,
) -> dict[str, object]:
    if rare_multiplier < 1:
        raise ValueError("rare_multiplier must be at least one")

    class_counts: Counter[int] = Counter()
    retained_rows = 0
    excluded_rows = 0
    lookup = np.asarray(RAW_MASK_TO_CLASS, dtype=np.int16)
    powers_of_two = 1 << np.arange(8, dtype=np.uint16)

    with h5py.File(source_path, "r") as source_file:
        source_group = source_file[group_name]
        row_count = len(source_group["d_values"])
        for start in range(0, row_count, chunk_rows):
            stop = min(start + chunk_rows, row_count)
            names = np.char.decode(source_group["geometry_name"][start:stop], "utf-8")
            keep = ~np.char.startswith(names, excluded_prefixes)
            transformed_distances = source_group["d_values"][start:stop, :8][keep]
            raw_masks = ((transformed_distances > 0.5).astype(np.uint16) * powers_of_two).sum(axis=1)
            class_counts.update(lookup[raw_masks].tolist())
            retained_rows += int(keep.sum())
            excluded_rows += int((~keep).sum())

        rare_classes = tuple(
            class_id
            for class_id, count in sorted(class_counts.items())
            if count / retained_rows < rare_share_threshold
        )
        rare_rows = sum(class_counts[class_id] for class_id in rare_classes)
        output_rows = retained_rows + (rare_multiplier - 1) * rare_rows

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as output_file:
            destination_group = output_file.create_group(group_name)
            for key, value in source_group.attrs.items():
                destination_group.attrs[key] = _filtered_geometry_metadata(value, excluded_prefixes)
            for name, source_dataset in source_group.items():
                if not isinstance(source_dataset, h5py.Dataset):
                    continue
                destination_dataset = destination_group.create_dataset(
                    name,
                    shape=(output_rows, *source_dataset.shape[1:]),
                    dtype=source_dataset.dtype,
                    chunks=(min(chunk_rows, output_rows), *source_dataset.shape[1:]),
                )
                for key, value in source_dataset.attrs.items():
                    destination_dataset.attrs[key] = value
            destination_group.create_dataset(
                "cut_class", shape=(output_rows,), dtype=np.int16, chunks=(min(chunk_rows, output_rows),)
            )

            destination_offset = 0
            for start in range(0, row_count, chunk_rows):
                stop = min(start + chunk_rows, row_count)
                names = np.char.decode(source_group["geometry_name"][start:stop], "utf-8")
                indices = np.flatnonzero(~np.char.startswith(names, excluded_prefixes)) + start
                destination_offset = _copy_rows(source_group, destination_group, indices, destination_offset)

            for repeat in range(rare_multiplier - 1):
                for start in range(0, row_count, chunk_rows):
                    stop = min(start + chunk_rows, row_count)
                    names = np.char.decode(source_group["geometry_name"][start:stop], "utf-8")
                    indices = np.flatnonzero(~np.char.startswith(names, excluded_prefixes)) + start
                    labels = _class_labels(source_group["d_values"][indices])
                    indices = indices[np.isin(labels, rare_classes)]
                    destination_offset = _copy_rows(source_group, destination_group, indices, destination_offset)

            if destination_offset != output_rows:
                raise RuntimeError(f"Wrote {destination_offset} rows, expected {output_rows}")
            destination_group.attrs["excluded_geometry_prefixes"] = json.dumps(excluded_prefixes)
            destination_group.attrs["topology_class_rule"] = "canonical class from transformed d_values[:8] > 0.5"
            destination_group.attrs["rare_class_share_threshold"] = rare_share_threshold
            destination_group.attrs["rare_classes"] = json.dumps(rare_classes)
            destination_group.attrs["rare_class_multiplier"] = rare_multiplier
            destination_group.attrs["source_h5"] = str(source_path)

    return {
        "retained_rows": retained_rows,
        "excluded_rows": excluded_rows,
        "output_rows": output_rows,
        "rare_classes": rare_classes,
        "class_counts": dict(class_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_h5", type=Path)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument("--group", default="integration_2_poly_1")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--rare-share-threshold", type=float, default=0.01)
    parser.add_argument("--rare-multiplier", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=65_536)
    args = parser.parse_args()

    summary = build_dataset(
        args.source_h5,
        args.output_h5,
        args.group,
        tuple(args.exclude_prefix),
        args.rare_share_threshold,
        args.rare_multiplier,
        args.chunk_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()