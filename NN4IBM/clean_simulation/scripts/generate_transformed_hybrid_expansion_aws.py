"""Double the hybrid cut-cell corpus with validated transformed STL variants.

Each candidate follows the canonical source-SDF -> locked-grid marching-cubes
-> parity-oriented rebuilt mesh -> QuESo target path. Candidates whose rebuilt
mesh fails the marching-cubes watertightness audit are recorded and excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.queso.settings import make_queso_settings

NOTEBOOK = ROOT / "clean_simulation" / "notebooks" / "01_data_generation.ipynb"
DATA_DIR = ROOT / "clean_simulation" / "outputs" / "data"
GROUP_NAME = "integration_2_poly_1"
LOW_CORNER_THRESHOLD = 0.25
HIGH_VOLUME_THRESHOLD = 0.75
TRANSFORM_PATTERN = re.compile(r"^(?P<family>.+)_trans_(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queso-python-path", type=Path, required=True)
    parser.add_argument(
        "--source-h5",
        type=Path,
        default=DATA_DIR / "grid_locked_source_sdf_rebuilt_queso_i2_p1_simple_harsh_crystals30_filtered.h5",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=DATA_DIR / "grid_locked_source_magnitude_material_parity_rebuilt_queso_i2_p1_simple_harsh_crystals30_transformed_filtered.h5",
    )
    parser.add_argument("--target-extra-rows", type=int, help="Defaults to the source corpus row count.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed candidate checkpoints.")
    return parser.parse_args()


def load_canonical_helpers(queso_python_path: Path) -> dict[str, object]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "Grid-locked reconstruction setup" in "".join(cell["source"])
    )
    source = source.replace(
        "c:/Users/z0058p1u/Documents/code/QuESo/out/install/win",
        str(queso_python_path),
    )
    namespace: dict[str, object] = {"__name__": "canonical_transformed_hybrid_helpers"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    namespace["root"] = ROOT
    return namespace


def transformed_candidates() -> list[str]:
    candidates: list[tuple[int, str, str]] = []
    for path in (ROOT / "clean_simulation" / "geometries").glob("*_trans_*.stl"):
        match = TRANSFORM_PATTERN.fullmatch(path.stem)
        if match:
            candidates.append((int(match["index"]), match["family"], path.stem))
    return [name for _, _, name in sorted(candidates)]


def valid_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as handle:
        group = handle.get("samples")
        return group is not None and all(
            name in group for name in ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name")
        )


def run_queso(helpers: dict[str, object], mesh_path: Path, grid) -> tuple[np.ndarray, list[object]]:
    helpers["_load_queso_dynamically"](str(helpers["QUESO_PYTHON_PATH"]))
    from QuESoPythonModule.PyQuESo import PyQuESo

    settings_path = mesh_path.with_suffix(".queso.json")
    settings = make_queso_settings(
        bounds=grid.bounds,
        resolution=grid.resolution,
        input_filename=mesh_path,
    )
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    try:
        queso = PyQuESo(str(settings_path))
        queso.Run()
        elements = [element for element in queso.GetElements() if element.IsTrimmed()]
    finally:
        settings_path.unlink(missing_ok=True)
    lower = np.asarray(grid.bounds[0], dtype=np.float64)
    spacing = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
    shape = np.asarray(grid.resolution, dtype=np.int64)
    pairs = []
    for element in elements:
        center = (np.asarray(element.LowerBoundXYZ()) + np.asarray(element.UpperBoundXYZ())) * 0.5
        ijk = np.rint((center - lower - spacing * 0.5) / spacing).astype(np.int64)
        if np.any(ijk < 0) or np.any(ijk >= shape):
            raise RuntimeError(f"QuESo produced an element outside the locked grid: {center}")
        pairs.append((int(ijk[0] * shape[1] * shape[2] + ijk[1] * shape[2] + ijk[2]), element))
    pairs.sort(key=lambda pair: pair[0])
    return np.asarray([flat for flat, _ in pairs], dtype=np.int64), [element for _, element in pairs]


def build_checkpoint(helpers: dict[str, object], geometry_name: str, checkpoint_path: Path, mesh_dir: Path) -> int:
    source_path = ROOT / "clean_simulation" / "geometries" / f"{geometry_name}.stl"
    source_volume = helpers["VolumeFactory"].from_file(source_path)
    source_volume.device = helpers["DEVICE"]
    grid = helpers["source_grid"](source_volume)
    rebuilt_mesh = helpers["reconstructed_mesh"](source_volume, source_path, grid)
    rebuilt_path = mesh_dir / f"{geometry_name}_locked_mc.stl"
    rebuilt_mesh.export(rebuilt_path)
    rebuilt_volume = helpers["VolumeFactory"].from_file(rebuilt_path)
    rebuilt_volume.device = helpers["DEVICE"]
    rebuilt_indices, rebuilt_elements = run_queso(helpers, rebuilt_path, grid)
    if not len(rebuilt_indices):
        raise RuntimeError("QuESo found no rebuilt-geometry trimmed cells")
    material = helpers["Material"](E=1.0, nu=helpers["POISSON_RATIO"])
    result = helpers["write_geometry_cutcell_training_h5"](
        output_h5=checkpoint_path,
        group_name="samples",
        grid=grid,
        volume=rebuilt_volume,
        material=material,
        poly_order=helpers["POLY_ORDER"],
        order=helpers["INTEGRATION_ORDER"],
        device=helpers["DEVICE"],
        batch_size=helpers["SDF_BATCH_SIZE"],
        replace_existing=True,
        refined_subcells=helpers["SDF_SUBCELLS"],
        queso_elements=rebuilt_elements,
        include_sdf=False,
    )
    if result["num_cells"] != len(rebuilt_indices):
        raise RuntimeError("Writer rows do not match rebuilt QuESo cells")
    rebuilt_fractions = helpers["cut_volume_fractions"](rebuilt_elements, grid)
    with h5py.File(checkpoint_path, "a") as handle:
        group = handle["samples"]
        flat_index = group["flat_index"][...]
        rebuilt_positions = np.searchsorted(rebuilt_indices, flat_index)
        if not np.array_equal(rebuilt_indices[rebuilt_positions], flat_index):
            raise RuntimeError("Writer and rebuilt QuESo selected different cells")
        source_distances = helpers["nodal_sdf"](source_volume, grid, flat_index)
        material_distances = helpers["nodal_sdf"](rebuilt_volume, grid, flat_index)
        fractions = rebuilt_fractions[rebuilt_positions]
        correlation = float(np.corrcoef(material_distances.mean(axis=1) - 0.5, fractions)[0, 1])
        if not np.isfinite(correlation) or abs(correlation) < 0.90:
            raise RuntimeError(f"Ambiguous rebuilt-SDF/QuESo material orientation ({correlation:.3f})")
        complemented = correlation < 0.0
        if complemented:
            material_distances = 1.0 - material_distances
        hybrid_distances = 0.5 + np.where(material_distances >= 0.5, 1.0, -1.0) * np.abs(source_distances - 0.5)
        if not np.isfinite(hybrid_distances).all() or not np.isfinite(group["stiffness_ut"][...]).all():
            raise RuntimeError("Non-finite hybrid features or rebuilt stiffness targets")
        group.create_dataset("d_values", data=hybrid_distances)
        group.create_dataset("cut_volume_fraction", data=fractions)
        group.create_dataset("geometry_name", data=np.full(len(flat_index), geometry_name.encode("utf-8"), dtype="S96"))
        group.attrs["reconstruction_audit"] = json.dumps(rebuilt_mesh.metadata["reconstruction_audit"])
        group.attrs["material_sdf_volume_correlation_before_alignment"] = correlation
        group.attrs["material_sdf_was_complemented"] = complemented
    return len(flat_index)


def append_filtered(output_group, checkpoint_path: Path) -> tuple[int, int]:
    with h5py.File(checkpoint_path, "r") as handle:
        group = handle["samples"]
        distances = group["d_values"][...]
        fractions = group["cut_volume_fraction"][...]
        corner_fraction = (distances > 0.5).mean(axis=1)
        keep = (
            np.isfinite(distances).all(axis=1)
            & np.isfinite(fractions)
            & ~(((corner_fraction <= LOW_CORNER_THRESHOLD) & (fractions >= HIGH_VOLUME_THRESHOLD))
                | ((corner_fraction >= 1.0 - LOW_CORNER_THRESHOLD) & (fractions <= 1.0 - HIGH_VOLUME_THRESHOLD)))
        )
        for name in ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name"):
            values = group[name][...][keep]
            start = len(output_group[name])
            output_group[name].resize((start + len(values),) + output_group[name].shape[1:])
            output_group[name][start:] = values
        return int(keep.sum()), int((~keep).sum())


def main() -> None:
    args = parse_args()
    if not args.queso_python_path.is_dir() or not args.source_h5.is_file():
        raise FileNotFoundError("Missing QuESo Python install or source HDF5.")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    with h5py.File(args.source_h5, "r") as source:
        source_rows = len(source[GROUP_NAME]["flat_index"])
    target_extra_rows = args.target_extra_rows or source_rows
    expansion_name = args.output_h5.stem.removesuffix("_filtered")
    checkpoint_dir = DATA_DIR / f"{expansion_name}_checkpoints"
    mesh_dir = DATA_DIR / f"{expansion_name}_marching_cubes_meshes"
    manifest_path = DATA_DIR / f"{expansion_name}_manifest.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_canonical_helpers(args.queso_python_path.resolve())
    helpers["QUESO_PYTHON_PATH"] = str(args.queso_python_path.resolve())

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    estimated_rows = 0
    started = time.perf_counter()
    for geometry_name in transformed_candidates():
        checkpoint_path = checkpoint_dir / f"{geometry_name}.h5"
        try:
            if not (args.resume and valid_checkpoint(checkpoint_path)):
                rows = build_checkpoint(helpers, geometry_name, checkpoint_path, mesh_dir)
            else:
                with h5py.File(checkpoint_path, "r") as handle:
                    rows = len(handle["samples"]["flat_index"])
            accepted.append({"geometry_name": geometry_name, "checkpoint": checkpoint_path.name, "raw_rows": rows})
            estimated_rows += rows
            print(f"Accepted {geometry_name}: {rows:,} raw rows; candidate total={estimated_rows:,}", flush=True)
        except Exception as error:
            checkpoint_path.unlink(missing_ok=True)
            rejected.append({"geometry_name": geometry_name, "reason": f"{type(error).__name__}: {error}"})
            print(f"Rejected {geometry_name}: {rejected[-1]['reason']}", flush=True)
        if estimated_rows >= target_extra_rows:
            break
    if estimated_rows < target_extra_rows:
        raise RuntimeError(f"Only {estimated_rows:,} validated transformed rows are available; requested {target_extra_rows:,}.")

    temporary_output = args.output_h5.with_suffix(".tmp.h5")
    temporary_output.unlink(missing_ok=True)
    merged: list[dict[str, int | str]] = []
    with h5py.File(args.source_h5, "r") as source, h5py.File(temporary_output, "w") as output:
        source_group = source[GROUP_NAME]
        output_group = output.create_group(GROUP_NAME)
        for name in ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name"):
            shape = source_group[name].shape
            output_group.create_dataset(name, data=source_group[name][...], maxshape=(None,) + shape[1:])
        output_group.attrs.update(source_group.attrs)
        for entry in accepted:
            retained, removed = append_filtered(output_group, checkpoint_dir / str(entry["checkpoint"]))
            merged.append({"geometry_name": str(entry["geometry_name"]), "retained_rows": retained, "removed_rows": removed})
        output_group.attrs["expanded_from"] = args.source_h5.name
        output_group.attrs["added_geometries"] = json.dumps([entry["geometry_name"] for entry in accepted])
        output_group.attrs["rejected_geometries"] = json.dumps(rejected)
        output_group.attrs["pipeline"] = "source_sdf_to_locked_grid_marching_cubes_to_parity_oriented_rebuilt_queso"
    temporary_output.replace(args.output_h5)
    manifest_path.write_text(json.dumps({
        "source_h5": str(args.source_h5), "output_h5": str(args.output_h5), "target_extra_rows": target_extra_rows,
        "accepted": accepted, "rejected": rejected, "merged": merged,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_h5} in {time.perf_counter() - started:.1f}s; manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()