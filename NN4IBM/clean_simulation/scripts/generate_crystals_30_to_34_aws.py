from __future__ import annotations

import json
import os
from pathlib import Path
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
EXPANSION_NAME = os.environ.get("CUT_CELL_EXPANSION_NAME", "crystals30_to_34")
SOURCE_H5 = Path(os.environ.get(
    "CUT_CELL_SOURCE_H5",
    DATA_DIR / "grid_locked_source_sdf_rebuilt_queso_i2_p1_simple_harsh_crystals30_filtered.h5",
)).expanduser().resolve()
OUTPUT_H5 = Path(os.environ.get(
    "CUT_CELL_OUTPUT_H5",
    DATA_DIR / f"grid_locked_source_sdf_rebuilt_queso_i2_p1_simple_harsh_{EXPANSION_NAME}_filtered.h5",
)).expanduser().resolve()
CHECKPOINT_DIR = DATA_DIR / f"grid_locked_source_sdf_rebuilt_queso_{EXPANSION_NAME}_checkpoints"
MESH_DIR = DATA_DIR / f"grid_locked_marching_cubes_stl_source_features_{EXPANSION_NAME}"
GROUP_NAME = "integration_2_poly_1"
NEW_GEOMETRIES = tuple(
    name.strip()
    for name in os.environ.get(
        "CUT_CELL_EXTRA_GEOMETRIES",
        ",".join(f"crystal_{index:04d}" for index in range(30, 34)),
    ).split(",")
    if name.strip()
)
LOW_CORNER_THRESHOLD = 0.25
HIGH_CVF_THRESHOLD = 0.75


def load_canonical_helpers() -> dict[str, object]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "Grid-locked reconstruction setup" in "".join(cell["source"])
    )
    namespace: dict[str, object] = {"__name__": "canonical_data_generation_helpers"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def existing_checkpoint_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as checkpoint:
        group = checkpoint.get("samples")
        return group is not None and all(
            name in group
            for name in ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name")
        )


def append_filtered_rows(source_group, output_group, geometry_h5: Path) -> tuple[int, int, int]:
    with h5py.File(geometry_h5, "r") as checkpoint:
        group = checkpoint["samples"]
        d_values = group["d_values"][...]
        fractions = group["cut_volume_fraction"][...]
        finite = np.isfinite(fractions) & np.isfinite(d_values).all(axis=1)
        corner_fraction = (d_values > 0.5).mean(axis=1)
        contradictory = (
            (corner_fraction <= LOW_CORNER_THRESHOLD) & (fractions >= HIGH_CVF_THRESHOLD)
        ) | (
            (corner_fraction >= 1.0 - LOW_CORNER_THRESHOLD) & (fractions <= 1.0 - HIGH_CVF_THRESHOLD)
        )
        keep = finite & ~contradictory
        for name in ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name"):
            values = group[name][...][keep]
            output_group[name].resize((len(output_group[name]) + len(values),) + output_group[name].shape[1:])
            output_group[name][-len(values):] = values
        return len(keep), int(keep.sum()), int((~keep).sum())


def main() -> None:
    if not SOURCE_H5.is_file():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_H5}")
    if not NEW_GEOMETRIES:
        raise ValueError("CUT_CELL_EXTRA_GEOMETRIES must provide at least one geometry.")
    missing = [name for name in NEW_GEOMETRIES if not (ROOT / "clean_simulation" / "geometries" / f"{name}.stl").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing new source STLs: {missing}")

    helpers = load_canonical_helpers()
    helpers["root"] = ROOT
    queso_path = Path(os.environ.get("QUESO_PYTHON_PATH", ROOT / "QuESo" / "out" / "install" / "linux"))
    if not queso_path.is_dir():
        raise FileNotFoundError(f"Missing QuESo Python install: {queso_path}")

    def run_queso(mesh_path: Path, grid):
        helpers["_load_queso_dynamically"](str(queso_path))
        from QuESoPythonModule.PyQuESo import PyQuESo

        settings_path = mesh_path.with_suffix(".queso.json")
        settings = make_queso_settings(
            bounds=grid.bounds,
            resolution=grid.resolution,
            input_filename=mesh_path,
        )
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
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
        return np.asarray([index for index, _ in pairs], dtype=np.int64), [element for _, element in pairs]

    helpers["run_queso"] = run_queso
    checkpoint_dir = CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    material = helpers["Material"](E=1.0, nu=helpers["POISSON_RATIO"])
    started = time.perf_counter()

    for position, geometry_name in enumerate(NEW_GEOMETRIES, start=1):
        geometry_h5 = checkpoint_dir / f"{geometry_name}.h5"
        if existing_checkpoint_is_valid(geometry_h5):
            print(f"[{position}/{len(NEW_GEOMETRIES)}] {geometry_name}: validated checkpoint exists; skipping.", flush=True)
            continue
        geometry_started = time.perf_counter()
        source_path = ROOT / "clean_simulation" / "geometries" / f"{geometry_name}.stl"
        source_volume = helpers["VolumeFactory"].from_file(source_path)
        source_volume.device = helpers["DEVICE"]
        grid = helpers["source_grid"](source_volume)
        rebuilt_mesh = helpers["reconstructed_mesh"](source_volume, source_path, grid)
        rebuilt_path = MESH_DIR / f"{geometry_name}_gridlocked_sdf{helpers['SDF_SUBCELLS']}.stl"
        rebuilt_mesh.export(rebuilt_path)
        rebuilt_volume = helpers["VolumeFactory"].from_file(rebuilt_path)
        rebuilt_volume.device = helpers["DEVICE"]
        rebuilt_indices, rebuilt_elements = run_queso(rebuilt_path, grid)
        if not len(rebuilt_indices):
            raise RuntimeError(f"{geometry_name}: QuESo found no rebuilt-geometry trimmed cells")
        result = helpers["write_geometry_cutcell_training_h5"](
            output_h5=geometry_h5, group_name="samples", grid=grid, volume=rebuilt_volume, material=material,
            poly_order=helpers["POLY_ORDER"], order=helpers["INTEGRATION_ORDER"], device=helpers["DEVICE"],
            batch_size=helpers["SDF_BATCH_SIZE"], replace_existing=True, refined_subcells=helpers["SDF_SUBCELLS"],
            queso_elements=rebuilt_elements, include_sdf=False,
        )
        if result["num_cells"] != len(rebuilt_indices):
            raise RuntimeError(f"{geometry_name}: writer rows do not match rebuilt QuESo cells")
        rebuilt_fractions = helpers["cut_volume_fractions"](rebuilt_elements, grid)
        with h5py.File(geometry_h5, "a") as checkpoint:
            group = checkpoint["samples"]
            flat_index = group["flat_index"][...]
            order = np.argsort(flat_index)
            if not np.array_equal(flat_index[order], rebuilt_indices):
                raise RuntimeError(f"{geometry_name}: writer and rebuilt QuESo selected different cells")
            source_d_values = helpers["nodal_sdf"](source_volume, grid, flat_index)
            fractions = rebuilt_fractions[order]
            correlation = float(np.corrcoef(source_d_values.mean(axis=1) - 0.5, fractions)[0, 1])
            if not np.isfinite(correlation) or abs(correlation) < 0.9:
                raise RuntimeError(f"{geometry_name}: ambiguous source-SDF/QuESo material orientation ({correlation:.3f})")
            complemented = correlation < 0.0
            if complemented:
                source_d_values = 1.0 - source_d_values
            if not np.isfinite(source_d_values).all() or not np.isfinite(group["stiffness_ut"][...]).all():
                raise RuntimeError(f"{geometry_name}: non-finite initial SDF or rebuilt stiffness target")
            group.create_dataset("d_values", data=source_d_values)
            group.create_dataset("cut_volume_fraction", data=fractions)
            group.create_dataset("geometry_name", data=np.full(len(flat_index), geometry_name.encode("utf-8"), dtype="S64"))
            group.attrs["source_sdf_volume_correlation_before_alignment"] = correlation
            group.attrs["source_sdf_was_complemented"] = complemented
            group.attrs["reconstruction_audit"] = json.dumps(rebuilt_mesh.metadata["reconstruction_audit"])
        print(f"[{position}/{len(NEW_GEOMETRIES)}] {geometry_name}: {len(rebuilt_indices):,} rows; correlation={correlation:+.3f} [{time.perf_counter() - geometry_started:.1f}s]", flush=True)

    temporary = OUTPUT_H5.with_suffix(".tmp.h5")
    temporary.unlink(missing_ok=True)
    with h5py.File(SOURCE_H5, "r") as source, h5py.File(temporary, "w") as output:
        source_group = source[GROUP_NAME]
        output_group = output.create_group(GROUP_NAME)
        for name in ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name"):
            shape = source_group[name].shape
            output_group.create_dataset(name, data=source_group[name][...], maxshape=(None,) + shape[1:])
        output_group.attrs.update(source_group.attrs)
        output_group.attrs["expanded_geometry_count"] = len(NEW_GEOMETRIES)
        output_group.attrs["expanded_from"] = SOURCE_H5.name
        output_group.attrs["added_geometries"] = json.dumps(NEW_GEOMETRIES)
        output_group.attrs["orientation_outlier_filter"] = json.dumps({"low_corner_threshold": LOW_CORNER_THRESHOLD, "high_cvf_threshold": HIGH_CVF_THRESHOLD})
        for geometry_name in NEW_GEOMETRIES:
            total, retained, removed = append_filtered_rows(source_group, output_group, checkpoint_dir / f"{geometry_name}.h5")
            print(f"Merged {geometry_name}: retained {retained:,}/{total:,}; removed {removed:,}", flush=True)
    temporary.replace(OUTPUT_H5)
    print(f"Wrote {OUTPUT_H5.name} with {len(NEW_GEOMETRIES)} added crystals [{time.perf_counter() - started:.1f}s]", flush=True)


if __name__ == "__main__":
    main()