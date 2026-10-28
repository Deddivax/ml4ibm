"""Regenerate eggcarton cut cells from an auditable marching-cubes lattice.

This is a non-production probe. It removes disconnected marching-cubes islands,
stores the exact scalar lattice used to extract the mesh as the eight model
features, and compares protected checkpoints against original and rebuilt rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pyvista as pv
import torch
import trimesh
from skimage.measure import marching_cubes


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.internal_fem_kernel.constants import Material
from clean_simulation.internal_fem_kernel.geometry import HexGrid
from clean_simulation.internal_fem_kernel.integration.element_stiffness import Stiffness
from clean_simulation.internal_fem_kernel.queso_integration import _load_queso_dynamically
from clean_simulation.queso.settings import make_queso_settings
from clean_simulation.internal_fem_kernel.volume_factory import VolumeFactory
from clean_simulation.internal_nn_runtime.cutcell_ml.data.datasets import reconstruct_matrices_from_ut
from clean_simulation.internal_nn_runtime.cutcell_ml.model import CutCellNet, CutCellNetTransformer


GROUP = "integration_2_poly_1"
GEOMETRY = "eggcarton_sphere"
RESOLUTION = (100, 100, 100)
MC_BATCH_SIZE = 2048
MODEL_BATCH_SIZE = 2048
NORMALIZER_BATCHES = 128
TRAIN_SPLIT = 0.9
MODEL_SEEDS = {"resnet": 545642, "transformer": 227344}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queso-python-path", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--output-stl", type=Path, required=True)
    parser.add_argument("--original-h5", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def make_grid(source_path: Path) -> HexGrid:
    source_volume = VolumeFactory.from_file(source_path)
    minimum, maximum = np.asarray(source_volume.mesh.bounds, dtype=np.float64)
    padding = np.maximum((maximum - minimum) * 0.10, 1.0e-3)
    raw_lower, raw_upper = minimum - padding, maximum + padding
    h = float(np.max(raw_upper - raw_lower) / RESOLUTION[0])
    center = (raw_lower + raw_upper) * 0.5
    half_span = h * np.asarray(RESOLUTION, dtype=np.float64) * 0.5
    return HexGrid(bounds=(tuple(center - half_span), tuple(center + half_span)), resolution=RESOLUTION, device="cpu")


def sample_marching_cubes_lattice(source_path: Path, grid: HexGrid) -> np.ndarray:
    counts = np.asarray(grid.resolution, dtype=np.int64) + 1
    lower = np.asarray(grid.bounds[0], dtype=np.float64)
    spacing = (np.asarray(grid.bounds[1], dtype=np.float64) - lower) / (counts - 1)
    total_points = int(np.prod(counts))
    yz_count = int(counts[1] * counts[2])
    signed_distance = np.empty(total_points, dtype=np.float64)
    surface = pv.read(str(source_path))

    for start in range(0, total_points, MC_BATCH_SIZE):
        stop = min(start + MC_BATCH_SIZE, total_points)
        flat = np.arange(start, stop, dtype=np.int64)
        i = flat // yz_count
        remainder = flat % yz_count
        j = remainder // counts[2]
        k = remainder % counts[2]
        points = lower + np.column_stack((i, j, k)) * spacing
        signed_distance[start:stop] = -pv.PolyData(points).compute_implicit_distance(surface)["implicit_distance"]

    lattice = signed_distance.reshape(tuple(counts))
    if not np.isfinite(lattice).all() or not (lattice.min() < 0.0 < lattice.max()):
        raise RuntimeError("The sampled source SDF does not contain a finite zero surface.")
    return lattice


def largest_component_mesh(lattice: np.ndarray, grid: HexGrid) -> tuple[trimesh.Trimesh, dict[str, object]]:
    lower = np.asarray(grid.bounds[0], dtype=np.float64)
    spacing = (np.asarray(grid.bounds[1], dtype=np.float64) - lower) / (np.asarray(lattice.shape) - 1)
    vertices, faces, _, _ = marching_cubes(lattice, level=0.0, spacing=tuple(spacing), allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=vertices + lower, faces=faces, process=True)
    components = list(mesh.split(only_watertight=False))
    if not components:
        raise RuntimeError("Marching cubes produced no components.")
    components.sort(key=lambda component: abs(float(component.volume)), reverse=True)
    retained = components[0].copy()
    trimesh.repair.fix_normals(retained, multibody=False)
    if not retained.is_watertight:
        raise RuntimeError("The retained marching-cubes component is not watertight.")
    audit = {
        "component_count_before_filter": len(components),
        "discarded_component_count": len(components) - 1,
        "retained_faces": len(retained.faces),
        "retained_volume": abs(float(retained.volume)),
        "discarded_volume": sum(abs(float(component.volume)) for component in components[1:]),
    }
    return retained, audit


def run_queso(mesh_path: Path, grid: HexGrid, queso_python_path: Path) -> tuple[np.ndarray, list]:
    _load_queso_dynamically(str(queso_python_path))
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
            raise RuntimeError(f"QuESo element mapped outside the locked grid: {center}")
        flat_index = int(ijk[0] * shape[1] * shape[2] + ijk[1] * shape[2] + ijk[2])
        pairs.append((flat_index, element))
    pairs.sort(key=lambda pair: pair[0])
    return np.asarray([index for index, _ in pairs], dtype=np.int64), [element for _, element in pairs]


def lattice_features(lattice: np.ndarray, flat_indices: np.ndarray, grid: HexGrid) -> np.ndarray:
    nx, ny, nz = grid.resolution
    i = flat_indices // (ny * nz)
    remainder = flat_indices % (ny * nz)
    j = remainder // nz
    k = remainder % nz
    offsets = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
    values = np.stack([lattice[i + ox, j + oy, k + oz] for ox, oy, oz in offsets], axis=1)
    return 0.5 + values / (2.0 * np.sqrt(3.0) * float(grid.dx))


def stiffness_targets(elements: list, grid: HexGrid) -> tuple[np.ndarray, np.ndarray]:
    material = Material(E=1.0, nu=0.3)
    offsets = HexGrid._local_node_offsets(1.0, 1.0, 1.0, 1, "cpu").numpy()
    domain_volume = float(np.prod(np.asarray(grid.bounds[1]) - np.asarray(grid.bounds[0])))
    coordinates, xi_points, weights, element_sizes, fractions = [], [], [], [], []
    for element in elements:
        lower = np.asarray(element.LowerBoundXYZ())
        upper = np.asarray(element.UpperBoundXYZ())
        uvw_lower = np.asarray(element.LowerBoundUVW())
        uvw_upper = np.asarray(element.UpperBoundUVW())
        integration_points = element.GetIntegrationPoints()
        xi_points.append(torch.as_tensor([
            2.0 * (np.asarray((point.X(), point.Y(), point.Z())) - uvw_lower) / (uvw_upper - uvw_lower) - 1.0
            for point in integration_points
        ], dtype=torch.float64))
        weights.append(torch.as_tensor([point.Weight() * domain_volume for point in integration_points], dtype=torch.float64))
        coordinates.append((lower + upper) * 0.5 + (upper - lower) * offsets)
        element_sizes.append(float((upper - lower)[0]))
        fractions.append(sum(point.Weight() for point in integration_points) * domain_volume / float(grid.dx * grid.dy * grid.dz))
    matrices = Stiffness.elements_from_queso_points_batched(
        np.asarray(coordinates), material.C, xi_points, weights, poly_order=1, device="cpu", element_batch_size=512,
    ).cpu().numpy()
    matrices /= np.asarray(element_sizes)[:, None, None]
    upper = np.triu_indices(24)
    return matrices[:, upper[0], upper[1]], np.clip(np.asarray(fractions, dtype=np.float64), 0.0, 1.0)


def load_model(kind: str, checkpoint_path: Path) -> torch.nn.Module:
    if kind == "resnet":
        model = CutCellNet(
            fusion_dim=512, n_residual_blocks=4, include_nu=False, trunk_type="resnet",
            use_class_conditioning=False, include_class_inputs=False, residual_scaling="fixed",
            spd_mode="cholesky", dropout_p=0.0, use_reduced_null_space=True, matrix_size=24, n_distance_nodes=8,
        )
    else:
        model = CutCellNetTransformer(embed_dim=256, n_heads=8, n_layers=8, ffn_dim=1024, dropout=0.0, reduced_matrix_size=18)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.cuda().eval()


def checkpoint_normalizer(data_path: Path, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(data_path, "r") as handle:
        distances = np.asarray(handle[GROUP]["d_values"], dtype=np.float32)
    generator = torch.Generator().manual_seed(seed)
    train_size = int(TRAIN_SPLIT * len(distances))
    train_indices = torch.randperm(len(distances), generator=generator)[:train_size]
    shuffled = torch.randperm(train_size, generator=generator)
    sampled = train_indices[shuffled[: NORMALIZER_BATCHES * MODEL_BATCH_SIZE]].numpy()
    values = torch.from_numpy(distances[sampled])
    return values.mean(dim=0), values.std(dim=0, correction=0).clamp(min=1.0e-4)


def relative_errors(model: torch.nn.Module, features: np.ndarray, packed_stiffness: np.ndarray) -> np.ndarray:
    targets = reconstruct_matrices_from_ut(packed_stiffness).astype(np.float32)
    target_norm = np.linalg.norm(targets, axis=(1, 2))
    floor = max(float(np.quantile(target_norm, 0.05)), 1.0e-4)
    errors = []
    with torch.no_grad():
        for start in range(0, len(features), MODEL_BATCH_SIZE):
            stop = start + MODEL_BATCH_SIZE
            inputs = torch.from_numpy(features[start:stop].astype(np.float32, copy=False)).cuda()
            predicted_unit, log_norm = model(inputs, nu=None, cut_class=None)
            predicted = predicted_unit.float() * torch.exp(log_norm.float())[:, None, None]
            error = torch.linalg.matrix_norm(predicted - torch.from_numpy(targets[start:stop]).cuda(), ord="fro", dim=(-2, -1))
            errors.append(error.cpu().numpy())
    return np.concatenate(errors) / np.maximum(target_norm, floor)


def main() -> None:
    args = parse_args()
    source_path = ROOT / "clean_simulation" / "geometries" / f"{GEOMETRY}.stl"
    grid = make_grid(source_path)
    lattice = sample_marching_cubes_lattice(source_path, grid)
    mesh, mesh_audit = largest_component_mesh(lattice, grid)
    args.output_stl.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output_stl)
    flat_indices, elements = run_queso(args.output_stl, grid, args.queso_python_path)
    features = lattice_features(lattice, flat_indices, grid)
    stiffness_ut, fractions = stiffness_targets(elements, grid)

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as handle:
        group = handle.create_group(GROUP)
        group.create_dataset("flat_index", data=flat_indices)
        group.create_dataset("d_values", data=features)
        group.create_dataset("stiffness_ut", data=stiffness_ut)
        group.create_dataset("cut_volume_fraction", data=fractions)
        group.create_dataset("geometry_name", data=np.full(len(flat_indices), GEOMETRY.encode("utf-8"), dtype="S64"))
        group.attrs["feature_source"] = "exact_marching_cubes_scalar_lattice"
        group.attrs["feature_convention"] = "0.5 + lattice_phi / (2*sqrt(3)*h)"
        group.attrs["mesh_audit"] = json.dumps(mesh_audit)

    with h5py.File(args.original_h5, "r") as handle:
        original = handle["samples"] if "samples" in handle else handle[GROUP]
        original_indices = np.asarray(original["flat_index"])
        original_features = np.asarray(original["d_values"], dtype=np.float32)
        original_stiffness = np.asarray(original["stiffness_ut"], dtype=np.float64)
        if "geometry_name" in original:
            names = np.char.decode(np.asarray(original["geometry_name"]), "utf-8")
            geometry_rows = names == GEOMETRY
            original_indices = original_indices[geometry_rows]
            original_features = original_features[geometry_rows]
            original_stiffness = original_stiffness[geometry_rows]
    original_positions = {int(index): position for position, index in enumerate(original_indices)}
    common = np.asarray([index for index in flat_indices if int(index) in original_positions], dtype=np.int64)
    fixed_positions = np.searchsorted(flat_indices, common)
    original_positions = np.asarray([original_positions[int(index)] for index in common], dtype=np.int64)
    target_delta = reconstruct_matrices_from_ut(stiffness_ut[fixed_positions]) - reconstruct_matrices_from_ut(original_stiffness[original_positions])
    target_reference = reconstruct_matrices_from_ut(original_stiffness[original_positions])
    target_relative = np.linalg.norm(target_delta, axis=(1, 2)) / np.maximum(np.linalg.norm(target_reference, axis=(1, 2)), 1.0e-12)
    feature_delta = np.abs(features[fixed_positions] - original_features[original_positions])

    report: dict[str, object] = {
        "mesh_audit": mesh_audit,
        "fixed_rows": int(len(flat_indices)),
        "original_rows": int(len(original_indices)),
        "common_rows": int(len(common)),
        "feature_max_abs_delta_common": float(feature_delta.max()),
        "feature_p99_abs_delta_common": float(np.quantile(feature_delta, 0.99)),
        "target_relative_delta_common_mean": float(target_relative.mean()),
        "target_relative_delta_common_p99": float(np.quantile(target_relative, 0.99)),
    }
    for kind, checkpoint in (("resnet", args.resnet_checkpoint), ("transformer", args.transformer_checkpoint)):
        model = load_model(kind, checkpoint)
        input_mean, input_std = checkpoint_normalizer(args.original_h5, MODEL_SEEDS[kind])
        model.set_normalizers(input_mean.cuda(), input_std.cuda())
        original_error = relative_errors(model, original_features[original_positions], original_stiffness[original_positions])
        fixed_error = relative_errors(model, features[fixed_positions], stiffness_ut[fixed_positions])
        report[kind] = {
            "original_mean_error": float(original_error.mean()),
            "fixed_mean_error": float(fixed_error.mean()),
            "mean_error_change": float(fixed_error.mean() - original_error.mean()),
            "original_p95_error": float(np.quantile(original_error, 0.95)),
            "fixed_p95_error": float(np.quantile(fixed_error, 0.95)),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()