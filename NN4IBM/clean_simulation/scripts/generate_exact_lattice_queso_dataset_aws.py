"""Regenerate cut-cell targets from one original SDF lattice per geometry.

The stored eight-node values are indexed directly from the exact scalar lattice
used by Marching Cubes to build the mesh given to QuESo. Mesh validation is
fail-closed: non-watertight components and invalid nesting orientation reject a
geometry instead of modifying its features to match the rebuilt mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
GROUP_NAME = "integration_2_poly_1"
CORNER_OFFSETS = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queso-python-path", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--geometries", nargs="+", required=True)
    parser.add_argument("--device", default="cuda", help="Torch device for stiffness assembly; QuESo remains CPU-side.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-rejections",
        action="store_true",
        help="Assemble validated geometries and record rejected ones instead of failing before output creation.",
    )
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
        queso_python_path.resolve().as_posix(),
    )
    namespace: dict[str, object] = {"__name__": "exact_lattice_generation_helpers"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    namespace["root"] = ROOT
    return namespace


def source_lattice(helpers: dict[str, object], source_path: Path, grid) -> np.ndarray:
    pv = helpers["pv"]
    counts = np.asarray(grid.resolution, dtype=np.int64) + 1
    lower = np.asarray(grid.bounds[0], dtype=np.float64)
    spacing = (np.asarray(grid.bounds[1], dtype=np.float64) - lower) / (counts - 1)
    total_points = int(np.prod(counts))
    yz_count = int(counts[1] * counts[2])
    values = np.empty(total_points, dtype=np.float64)
    surface = pv.read(str(source_path))
    for start in range(0, total_points, int(helpers["MC_SDF_BATCH_SIZE"])):
        stop = min(start + int(helpers["MC_SDF_BATCH_SIZE"]), total_points)
        flat = np.arange(start, stop, dtype=np.int64)
        i = flat // yz_count
        remainder = flat % yz_count
        j = remainder // counts[2]
        k = remainder % counts[2]
        points = lower + np.column_stack((i, j, k)) * spacing
        values[start:stop] = -pv.PolyData(points).compute_implicit_distance(surface)["implicit_distance"]
    lattice = values.reshape(tuple(counts))
    if not np.isfinite(lattice).all() or not (lattice.min() < 0.0 < lattice.max()):
        raise RuntimeError("Original source SDF lattice is non-finite or has no zero crossing")
    return lattice


def reconstructed_mesh(helpers: dict[str, object], lattice: np.ndarray, grid):
    lower = np.asarray(grid.bounds[0], dtype=np.float64)
    spacing = (np.asarray(grid.bounds[1], dtype=np.float64) - lower) / (np.asarray(lattice.shape) - 1)
    vertices, faces, _, _ = helpers["marching_cubes"](lattice, level=0.0, spacing=tuple(spacing), allow_degenerate=False)
    mesh = helpers["trimesh"].Trimesh(vertices=vertices + lower, faces=faces, process=False)
    if len(mesh.faces) == 0:
        raise RuntimeError("Marching Cubes produced no faces")
    mesh, audit = helpers["orient_nested_shells"](mesh)
    if not mesh.is_watertight:
        raise RuntimeError("Validated reconstructed mesh is not watertight")
    mesh.metadata["reconstruction_audit"] = audit
    return mesh


def run_queso(helpers: dict[str, object], mesh_path: Path, grid, queso_python_path: Path) -> tuple[np.ndarray, list[object]]:
    helpers["_load_queso_dynamically"](str(queso_python_path))
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
            raise RuntimeError(f"QuESo element mapped outside the original SDF lattice: {center}")
        pairs.append((int(ijk[0] * shape[1] * shape[2] + ijk[1] * shape[2] + ijk[2]), element))
    pairs.sort(key=lambda pair: pair[0])
    return np.asarray([index for index, _ in pairs], dtype=np.int64), [element for _, element in pairs]


def lattice_features(lattice: np.ndarray, flat_indices: np.ndarray, grid) -> np.ndarray:
    nx, ny, nz = grid.resolution
    i = flat_indices // (ny * nz)
    remainder = flat_indices % (ny * nz)
    j = remainder // nz
    k = remainder % nz
    raw_values = np.stack([lattice[i + ox, j + oy, k + oz] for ox, oy, oz in CORNER_OFFSETS], axis=1)
    return 0.5 + raw_values / (2.0 * np.sqrt(3.0) * float(grid.dx))


def build_checkpoint(helpers: dict[str, object], geometry_name: str, checkpoint_path: Path, mesh_dir: Path, queso_python_path: Path) -> int:
    source_path = ROOT / "clean_simulation" / "geometries" / f"{geometry_name}.stl"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_volume = helpers["VolumeFactory"].from_file(source_path)
    source_volume.device = helpers["DEVICE"]
    grid = helpers["source_grid"](source_volume)
    lattice = source_lattice(helpers, source_path, grid)
    rebuilt_mesh = reconstructed_mesh(helpers, lattice, grid)
    rebuilt_path = mesh_dir / f"{geometry_name}_exact_lattice_mc.stl"
    rebuilt_mesh.export(rebuilt_path)
    flat_indices, elements = run_queso(helpers, rebuilt_path, grid, queso_python_path)
    if not len(flat_indices):
        raise RuntimeError("QuESo found no trimmed cells")
    material = helpers["Material"](E=1.0, nu=helpers["POISSON_RATIO"])
    result = helpers["write_geometry_cutcell_training_h5"](
        output_h5=checkpoint_path, group_name="samples", grid=grid, volume=helpers["VolumeFactory"].from_file(rebuilt_path),
        material=material, poly_order=helpers["POLY_ORDER"], order=helpers["INTEGRATION_ORDER"], device=helpers["DEVICE"],
        batch_size=helpers["SDF_BATCH_SIZE"], replace_existing=True, refined_subcells=helpers["SDF_SUBCELLS"],
        queso_elements=elements, include_sdf=False,
    )
    if result["num_cells"] != len(flat_indices):
        raise RuntimeError("Writer and QuESo row counts differ")
    direct_features = lattice_features(lattice, flat_indices, grid)
    if not np.isfinite(direct_features).all() or not np.isfinite(h5py.File(checkpoint_path, "r")["samples"]["stiffness_ut"][...]).all():
        raise RuntimeError("Non-finite exact-lattice features or stiffness target")
    fractions = helpers["cut_volume_fractions"](elements, grid)
    with h5py.File(checkpoint_path, "a") as handle:
        group = handle["samples"]
        written_indices = group["flat_index"][...]
        positions = np.searchsorted(flat_indices, written_indices)
        if not np.array_equal(flat_indices[positions], written_indices):
            raise RuntimeError("Writer and QuESo selected different flat indices")
        group.create_dataset("d_values", data=direct_features[positions])
        group.create_dataset("cut_volume_fraction", data=fractions[positions])
        group.create_dataset("geometry_name", data=np.full(len(written_indices), geometry_name.encode("utf-8"), dtype="S96"))
        group.attrs["feature_source"] = "exact_original_sdf_marching_cubes_lattice"
        group.attrs["feature_convention"] = "0.5 + phi / (2*sqrt(3)*grid_dx)"
        group.attrs["source_sdf_lattice_shape"] = lattice.shape
        group.attrs["source_mesh_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        group.attrs["reconstruction_audit"] = json.dumps(rebuilt_mesh.metadata["reconstruction_audit"])
    return len(flat_indices)


def valid_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as handle:
        group = handle.get("samples")
        return group is not None and group.attrs.get("feature_source") == "exact_original_sdf_marching_cubes_lattice"


def main() -> None:
    args = parse_args()
    if not args.queso_python_path.is_dir():
        raise FileNotFoundError(args.queso_python_path)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    output_name = args.output_h5.stem.removesuffix("_filtered")
    checkpoint_dir = DATA_DIR / f"{output_name}_checkpoints"
    mesh_dir = DATA_DIR / f"{output_name}_marching_cubes_meshes"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_canonical_helpers(args.queso_python_path.resolve())
    helpers["DEVICE"] = args.device
    accepted, rejected = [], []
    started = time.perf_counter()
    for geometry_name in args.geometries:
        checkpoint_path = checkpoint_dir / f"{geometry_name}.h5"
        try:
            rows = len(h5py.File(checkpoint_path, "r")["samples"]["flat_index"]) if args.resume and valid_checkpoint(checkpoint_path) else build_checkpoint(helpers, geometry_name, checkpoint_path, mesh_dir, args.queso_python_path.resolve())
            accepted.append({"geometry_name": geometry_name, "checkpoint": checkpoint_path.name, "rows": rows})
            print(f"Accepted {geometry_name}: {rows:,} rows", flush=True)
        except Exception as error:
            checkpoint_path.unlink(missing_ok=True)
            rejected.append({"geometry_name": geometry_name, "reason": f"{type(error).__name__}: {error}"})
            print(f"Rejected {geometry_name}: {rejected[-1]['reason']}", flush=True)
    if rejected and not args.allow_rejections:
        raise RuntimeError(f"Fail-closed regeneration rejected {len(rejected)} geometry(s): {rejected}")
    if not accepted:
        raise RuntimeError("No geometries passed fail-closed regeneration")
    temporary = args.output_h5.with_suffix(".tmp.h5")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as output:
        group = output.create_group(GROUP_NAME)
        for entry in accepted:
            with h5py.File(checkpoint_dir / str(entry["checkpoint"]), "r") as checkpoint:
                source = checkpoint["samples"]
                for name in ("flat_index", "d_values", "stiffness_ut", "cut_volume_fraction", "geometry_name"):
                    if name not in group:
                        group.create_dataset(name, data=source[name][...], maxshape=(None,) + source[name].shape[1:])
                    else:
                        start = len(group[name])
                        group[name].resize((start + len(source[name]),) + group[name].shape[1:])
                        group[name][start:] = source[name][...]
        group.attrs["pipeline"] = "exact_original_sdf_lattice_to_marching_cubes_to_validated_mesh_to_queso"
        group.attrs["feature_source"] = "exact_original_sdf_marching_cubes_lattice"
        group.attrs["feature_convention"] = "0.5 + phi / (2*sqrt(3)*grid_dx)"
        group.attrs["feature_count"] = 8
        group.attrs["mesh_validation"] = "watertight_components_and_nested_shell_orientation_fail_closed"
        group.attrs["geometries"] = json.dumps(args.geometries)
        group.attrs["accepted_geometries"] = json.dumps([entry["geometry_name"] for entry in accepted])
        group.attrs["rejected_geometries"] = json.dumps(rejected)
    temporary.replace(args.output_h5)
    manifest = args.output_h5.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"output_h5": str(args.output_h5), "accepted": accepted, "rejected": rejected}, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_h5} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()