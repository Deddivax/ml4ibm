"""Regenerate richer SDF features while preserving an established target corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "clean_simulation" / "notebooks" / "01_data_generation.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queso-python-path", type=Path, required=True)
    parser.add_argument(
        "--source-h5",
        type=Path,
        required=True,
        help="Existing compatible corpus whose target-side datasets are preserved.",
    )
    parser.add_argument(
        "--exclude-geometry",
        action="append",
        default=[],
        help="Geometry basename to omit. This changes the corpus and is rejected by default.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated: retained for command-line compatibility.",
    )
    parser.add_argument(
        "--geometry",
        action="append",
        default=[],
        help="Geometry basename to retain; may be supplied more than once.",
    )
    parser.add_argument(
        "--sdf-subcells",
        type=int,
        help="Override the marching-cubes samples per FEM cell for this invocation.",
    )
    parser.add_argument(
        "--mc-subcells",
        type=int,
        help="Override only the marching-cubes samples per FEM cell.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        help="Write the assembled dataset to this path instead of the canonical default.",
    )
    parser.add_argument(
        "--material-mesh-dir",
        type=Path,
        help="Optional directory of retained grid-locked MC meshes used for material parity.",
    )
    return parser.parse_args()


def source_for_marker(notebook: dict, marker: str) -> str:
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if marker in source:
            return source
    raise RuntimeError(f"Notebook cell with marker {marker!r} was not found.")


def main() -> None:
    args = parse_args()
    if not args.queso_python_path.is_dir():
        raise FileNotFoundError(f"Missing QuESo Python install: {args.queso_python_path}")
    if not args.source_h5.is_file():
        raise FileNotFoundError(f"Missing source dataset: {args.source_h5}")
    if args.output_h5 is None:
        raise ValueError("--output-h5 is required for SDF-only regeneration")
    if args.output_h5.resolve() == args.source_h5.resolve():
        raise ValueError("--output-h5 must differ from --source-h5")
    if args.exclude_geometry or args.geometry:
        raise ValueError(
            "SDF-only regeneration preserves the existing corpus exactly; "
            "--exclude-geometry and --geometry are not supported."
        )
    if args.resume:
        raise ValueError("--resume is not supported for SDF-only regeneration")
    os.environ["QUESO_PYTHON_PATH"] = str(args.queso_python_path.resolve())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = source_for_marker(notebook, "# %% Grid-locked reconstruction setup")
    windows_path = "c:/Users/z0058p1u/Documents/code/QuESo/out/install/win"
    setup = setup.replace(windows_path, str(args.queso_python_path.resolve()))
    if args.sdf_subcells is None or args.sdf_subcells < 1:
        raise ValueError("--sdf-subcells must be at least one")
    if args.mc_subcells is not None:
        if args.mc_subcells < 1:
            raise ValueError("--mc-subcells must be at least one")
        setup = setup.replace(
            "MC_SDF_BATCH_SIZE = 2048",
            "\n".join((
                "MC_SDF_BATCH_SIZE = 2048",
                f"MC_RECONSTRUCTION_SUBCELLS = {args.mc_subcells}",
            )),
            1,
        ).replace(
            "np.asarray(grid.resolution, dtype=np.int64) * SDF_SUBCELLS + 1",
            "np.asarray(grid.resolution, dtype=np.int64) * MC_RECONSTRUCTION_SUBCELLS + 1",
            1,
        )
    namespace: dict[str, object] = {"__name__": "hybrid_parity_dataset_generation"}
    exec(compile(setup, str(NOTEBOOK), "exec"), namespace)

    import numpy as np
    import torch

    source_grid = namespace["source_grid"]
    volume_factory = namespace["VolumeFactory"]
    root = namespace["root"]
    group_name = "integration_2_poly_1"
    baseline_atol = 1.0e-6

    def lattice_sdf_features(source_path: Path, grid, indices: np.ndarray, refined_subcells: int) -> np.ndarray:
        nx, ny, nz = grid.resolution
        i = indices // (ny * nz)
        remainder = indices % (ny * nz)
        j, k = remainder // nz, remainder % nz
        node_count = refined_subcells + 1
        offsets = np.stack(np.meshgrid(
            np.arange(node_count), np.arange(node_count), np.arange(node_count), indexing="ij",
        ), axis=-1).reshape(-1, 3) / refined_subcells
        corners = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))) * refined_subcells
        corner_indices = corners[:, 0] * node_count * node_count + corners[:, 1] * node_count + corners[:, 2]
        feature_order = np.concatenate((corner_indices, np.setdiff1d(np.arange(node_count ** 3), corner_indices)))
        lower = np.asarray(grid.bounds[0], dtype=np.float64)
        spacing = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
        cells = np.column_stack((i, j, k))
        values = np.empty((len(indices), len(offsets)), dtype=np.float64)
        surface = namespace["pv"].read(str(source_path))
        batch_size = 65_536
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            points = lower + (cells[start:stop, None, :] + offsets[None, :, :]) * spacing
            values[start:stop] = -namespace["pv"].PolyData(points.reshape(-1, 3)).compute_implicit_distance(surface)["implicit_distance"].reshape(stop - start, -1)
        return (0.5 + values / (2.0 * np.sqrt(3.0) * float(grid.dx)))[:, feature_order]

    with h5py.File(args.source_h5, "r") as source_file, h5py.File(args.output_h5, "w") as output_file:
        source_group = source_file[group_name]
        if not all(name in source_group for name in ("flat_index", "geometry_name", "d_values")):
            raise RuntimeError("Source dataset must contain flat_index, geometry_name, and d_values.")
        if source_group["d_values"].shape[1] != 8:
            raise RuntimeError("Source dataset must contain the established eight-node SDF representation.")

        output_group = output_file.create_group(group_name)
        for name, dataset in source_group.items():
            if name != "d_values":
                source_file.copy(dataset, output_group, name=name)
        for key, value in source_group.attrs.items():
            output_group.attrs[key] = value

        flat_index = source_group["flat_index"][...]
        geometry_names = np.char.decode(source_group["geometry_name"][...], "utf-8")
        d_values = np.empty((len(flat_index), (args.sdf_subcells + 1) ** 3), dtype=np.float64)
        for geometry_name in np.unique(geometry_names):
            rows = np.flatnonzero(geometry_names == geometry_name)
            source_path = root / "clean_simulation" / "geometries" / f"{geometry_name}.stl"
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source STL: {source_path}")
            volume = volume_factory.from_file(source_path)
            grid = source_grid(volume)
            baseline = lattice_sdf_features(source_path, grid, flat_index[rows], 1)
            max_baseline_difference = float(np.max(np.abs(baseline - source_group["d_values"][rows])))
            if max_baseline_difference > baseline_atol:
                raise RuntimeError(
                    f"{geometry_name}: source SDF differs from the existing corpus by "
                    f"{max_baseline_difference:.3e} (limit {baseline_atol:.1e}); refusing to reuse targets."
                )
            d_values[rows] = lattice_sdf_features(source_path, grid, flat_index[rows], args.sdf_subcells)
            print(
                f"{geometry_name}: verified {len(rows):,} existing rows "
                f"(max baseline drift {max_baseline_difference:.3e}) and regenerated SDF only",
                flush=True,
            )

        output_group.create_dataset("d_values", data=d_values)
        output_group.attrs["refined_subcells"] = int(args.sdf_subcells)
        output_group.attrs["n_distance_nodes"] = int(d_values.shape[1])
        output_group.attrs["sdf_regeneration"] = "exact_original_source_sdf_lattice_at_existing_flat_indices; targets copied without marching-cubes or QuESo assembly"
        output_group.attrs["sdf_source_h5"] = str(args.source_h5)
        print(f"Done: wrote {len(flat_index):,} rows with {d_values.shape[1]} SDF inputs to {args.output_h5}")


if __name__ == "__main__":
    main()