"""Evaluate Blade displacement convergence on a fixed physical interior sample."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import trimesh

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_blade_displacement_convergence import FRAMEWORKS, checkpoint_path, load_checkpoint
from clean_simulation.internal_nn_runtime.cutcell_bridge import (
    _point_to_triangle_signed_distance,
    extract_volume_triangles,
)


def load_grid_bounds(root: Path, resolution: int) -> np.ndarray:
    with np.load(checkpoint_path(root, "queso", resolution), allow_pickle=False) as archive:
        mesh_bounds = np.asarray(archive["mesh_bounds"], dtype=np.float64)
    center = mesh_bounds.mean(axis=0)
    half_width = float(np.max(mesh_bounds[1] - mesh_bounds[0])) * 0.6
    return np.stack((center - half_width, center + half_width))


def physical_interior_points(
    stl_path: Path, bounds: np.ndarray, count: int, seed: int, device: str, classification_batch_size: int
) -> np.ndarray:
    with stl_path.open("rb") as stream:
        mesh = trimesh.load(stream, file_type="stl", force="mesh")
    if not mesh.is_watertight:
        raise ValueError("Blade geometry must be watertight.")
    triangles, normals = extract_volume_triangles(mesh, torch.device(device))
    engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
    inside_batches = []
    requested = 0
    while requested < count:
        candidates = engine.draw(max(8192, 2 * (count - requested))).numpy()
        points = bounds[0] + candidates * (bounds[1] - bounds[0])
        inside_mask = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), classification_batch_size):
            stop = min(start + classification_batch_size, len(points))
            point_tensor = torch.from_numpy(points[start:stop]).to(device=device, dtype=torch.float64)
            signed_distance = _point_to_triangle_signed_distance(
                point_tensor, triangles[:, 0], triangles[:, 1], triangles[:, 2], normals
            )
            nearest = signed_distance.gather(1, signed_distance.abs().argmin(dim=1, keepdim=True)).squeeze(1)
            inside_mask[start:stop] = nearest.cpu().numpy() < 0.0
        inside = points[inside_mask]
        inside_batches.append(inside)
        requested += len(inside)
    return np.concatenate(inside_batches, axis=0)[:count]


def interpolate_at_points(field: np.ndarray, points: np.ndarray, bounds: np.ndarray, device: str) -> np.ndarray:
    normalized = 2.0 * (points - bounds[0]) / (bounds[1] - bounds[0]) - 1.0
    values = torch.from_numpy(field).to(device=device).permute(3, 0, 1, 2).unsqueeze(0)
    grid = torch.from_numpy(normalized[:, [2, 1, 0]]).to(device=device, dtype=values.dtype).reshape(1, 1, 1, -1, 3)
    sampled = functional.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(0).squeeze(1).squeeze(1).transpose(0, 1).cpu().numpy()


def active_cell_point_mask(points: np.ndarray, bounds: np.ndarray, resolution: int, active_cells: np.ndarray) -> np.ndarray:
    coordinates = np.floor((points - bounds[0]) * resolution / (bounds[1] - bounds[0])).astype(np.int64)
    coordinates = np.clip(coordinates, 0, resolution - 1)
    cell_ids = (coordinates[:, 0] * resolution + coordinates[:, 1]) * resolution + coordinates[:, 2]
    mask = np.zeros(resolution**3, dtype=bool)
    mask[active_cells] = True
    return mask[cell_ids]


def main() -> None:
    root = Path(os.environ["BLADE_CONVERGENCE_OUTPUT_DIR"]).expanduser().resolve()
    reference_root = Path(os.environ.get("BLADE_REFERENCE_OUTPUT_DIR", root)).expanduser().resolve()
    coarse_root = Path(os.environ.get("BLADE_COARSE_OUTPUT_DIR", root)).expanduser().resolve()
    extended_root = Path(os.environ.get("BLADE_EXTENDED_OUTPUT_DIR", root)).expanduser().resolve()
    reference_resolution = int(os.environ.get("BLADE_REFERENCE_RESOLUTION", "512"))
    extended_min_resolution = int(os.environ.get("BLADE_EXTENDED_MIN_RESOLUTION", "256"))
    resolutions = tuple(int(value) for value in os.environ.get("BLADE_EVALUATION_RESOLUTIONS", "16,32,64,128,256").split(","))
    sample_count = int(os.environ.get("BLADE_PHYSICAL_SAMPLE_COUNT", "262144"))
    sample_seed = int(os.environ.get("BLADE_PHYSICAL_SAMPLE_SEED", "227344"))
    classification_batch_size = int(os.environ.get("BLADE_PHYSICAL_CLASSIFICATION_BATCH_SIZE", "64"))
    device = os.environ.get("BLADE_EVALUATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BLADE_EVALUATION_DEVICE=cuda was requested but CUDA is unavailable.")

    stl_path = Path(os.environ["BLADE_STL_PATH"]).expanduser().resolve()
    bounds = load_grid_bounds(reference_root, reference_resolution)
    points = physical_interior_points(stl_path, bounds, sample_count, sample_seed, device, classification_batch_size)
    common_active_mask = np.ones(len(points), dtype=bool)
    for resolution in resolutions:
        candidate_root = extended_root if resolution >= extended_min_resolution else coarse_root
        _, active_cells, _ = load_checkpoint(candidate_root, "queso", resolution)
        common_active_mask &= active_cell_point_mask(points, bounds, resolution, active_cells)
    points = points[common_active_mask]
    if not len(points):
        raise RuntimeError("No physical sample points are active at every requested resolution.")
    print(f"Using {len(points):,} fixed interior points active at every resolution.", flush=True)
    reference, _, _ = load_checkpoint(reference_root, "queso", reference_resolution)
    reference_values = interpolate_at_points(reference, points, bounds, device)
    reference_norm = float(np.linalg.norm(reference_values))
    if reference_norm <= np.finfo(np.float64).tiny:
        raise RuntimeError("The QuESo reference displacement has zero norm on the physical sample.")

    domain_length = float(np.max(bounds[1] - bounds[0]) * 1.2)
    rows = []
    for framework, label in FRAMEWORKS.items():
        for resolution in resolutions:
            candidate_root = extended_root if resolution >= extended_min_resolution else coarse_root
            field, _, runtime = load_checkpoint(candidate_root, framework, resolution)
            values = interpolate_at_points(field, points, bounds, device)
            difference = values - reference_values
            component_error = np.linalg.norm(difference, axis=0) / np.linalg.norm(reference_values, axis=0)
            relative_error = float(np.linalg.norm(difference) / reference_norm)
            rows.append({
                "framework": framework, "framework_label": label, "resolution": resolution,
                "h": domain_length / resolution, "reference_resolution": reference_resolution,
                "physical_sample_count": sample_count, "physical_sample_seed": sample_seed,
                "relative_l2_error": relative_error,
                "relative_l2_error_x": float(component_error[0]),
                "relative_l2_error_y": float(component_error[1]),
                "relative_l2_error_z": float(component_error[2]), "solve_seconds": runtime,
            })
            print(f"{framework:8s} {resolution:3d}^3: relative physical L2 = {relative_error:.6e}", flush=True)
            del field, values, difference
            if device == "cuda":
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows).sort_values(["framework", "resolution"]).reset_index(drop=True)
    results["local_order"] = np.nan
    fit_rows = []
    for framework, curve in results.groupby("framework", sort=False):
        curve = curve.sort_values("resolution")
        errors, mesh_sizes = curve.relative_l2_error.to_numpy(), curve.h.to_numpy()
        results.loc[curve.index[1:], "local_order"] = np.log(errors[:-1] / errors[1:]) / np.log(mesh_sizes[:-1] / mesh_sizes[1:])
        slope, intercept = np.polyfit(np.log(mesh_sizes), np.log(errors), 1)
        prediction = slope * np.log(mesh_sizes) + intercept
        residual = np.sum((np.log(errors) - prediction) ** 2)
        total = np.sum((np.log(errors) - np.log(errors).mean()) ** 2)
        fit_rows.append({"framework": framework, "framework_label": curve.framework_label.iloc[0], "global_order": slope, "r_squared": 1.0 - residual / total})

    stem = f"blade_displacement_physical_l2_vs_queso_{reference_resolution}"
    results.to_csv(root / f"{stem}.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(root / f"{stem}_global_fits.csv", index=False)
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for framework, label in FRAMEWORKS.items():
        curve = results[results.framework == framework].sort_values("h", ascending=False)
        start_resolution, end_resolution = (int(curve.iloc[-2].resolution), int(curve.iloc[-1].resolution))
        axis.loglog(curve.h, curve.relative_l2_error, marker="o", linewidth=2, label=f"{label} ($p_{{{start_resolution}\\to{end_resolution}}}={curve.iloc[-1].local_order:.2f}$)")
    axis.invert_xaxis()
    axis.set(xlabel="Cubic grid spacing h", ylabel="Relative displacement L2 error on fixed Blade interior sample")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(root / f"{stem}.png", dpi=180, bbox_inches="tight")
    print("\n" + results[["framework", "resolution", "relative_l2_error", "local_order"]].to_string(index=False))
    print("\n" + pd.DataFrame(fit_rows).to_string(index=False))


if __name__ == "__main__":
    main()