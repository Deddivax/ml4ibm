"""Cache-cold Blade STL-to-local-K timing with independent FNB and QuESo paths."""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.bootstrap import configure_legacy_imports
from clean_simulation.case import build_case_from_stl
from clean_simulation.internal_fem_kernel.classification import Classification
from clean_simulation.internal_fem_kernel.queso_integration import map_integration_points_to_elements
from clean_simulation.internal_nn_runtime.cutcell_bridge import (
    compute_cell_volume_fractions_ray_parity,
    compute_nn_cutcell_stiffness,
    prepare_nn_inputs,
)
from clean_simulation.nn_runtime import load_nn_engine


NN_VARIANTS = {
    "nn_fnb": {
        "checkpoint": "source_sdf_rebuilt_queso_d10_f1024_all_crystals_10h.pt",
        "refined_subcells": 1,
    },
    "nn_27_fnb": {
        "checkpoint": "aws_rich_sdf/nn_27_best.pt",
        "refined_subcells": 2,
    },
    "nn_64_fnb": {
        "checkpoint": "aws_rich_sdf/nn_64_best.pt",
        "refined_subcells": 3,
    },
}
VARIANTS = ("naive_fnb", "weighted_fnb", *NN_VARIANTS, "queso_p2", "queso_p3")


def cube_bounds(mesh_bounds: np.ndarray, padding_fraction: float = 0.10) -> list[list[float]]:
    center = mesh_bounds.mean(axis=0)
    half_width = float(np.max(mesh_bounds[1] - mesh_bounds[0])) * (0.5 + padding_fraction)
    return [(center - half_width).tolist(), (center + half_width).tolist()]


def elapsed(started_at: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started_at


def merge_k_representation(indices_a, matrices_a, indices_b, matrices_b):
    indices = torch.cat((indices_a.to(torch.int64), indices_b.to(torch.int64)))
    matrices = torch.cat((matrices_a.cpu(), matrices_b.cpu()))
    order = torch.argsort(indices)
    return indices[order], matrices[order]


def main() -> None:
    output_dir = Path(os.environ["BLADE_E2E_K_TIMING_OUTPUT_DIR"]).expanduser().resolve()
    stl_path = Path(os.environ["BLADE_STL_PATH"]).expanduser().resolve()
    resolutions = tuple(int(value) for value in os.environ.get("BLADE_RESOLUTIONS", "16,32,64,128").split(","))
    variants = tuple(value for value in os.environ.get("BLADE_E2E_K_VARIANTS", ",".join(VARIANTS)).split(",") if value)
    device = os.environ.get("BLADE_DEVICE", "cuda")
    fnb_band_cells = int(os.environ.get("BLADE_FNB_BAND_CELLS", "3"))
    fraction_samples = int(os.environ.get("BLADE_WEIGHTED_SAMPLES_PER_AXIS", "4"))
    if set(variants) - set(VARIANTS):
        raise ValueError(f"Unknown timing variants: {set(variants) - set(VARIANTS)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BLADE_DEVICE=cuda was requested but CUDA is unavailable.")
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)

    configure_legacy_imports(ROOT)
    from constants import Material
    from geometry import HexGrid, Volume
    from internal_fem_kernel.assembly.stiffness import StiffnessAssembly
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=False)
    if torch.device(device).type == "cuda":
        torch.empty(1, device=device).fill_(0.0)
        torch.cuda.synchronize(torch.device(device))
    if torch.device(device).type == "cuda" and any(variant.endswith("_fnb") for variant in variants):
        warmup_mesh = trimesh.load(stl_path, file_type="stl", force="mesh")
        warmup_bounds = cube_bounds(np.asarray(warmup_mesh.bounds, dtype=np.float64))
        warmup_volume = Volume(warmup_mesh, device="cpu")
        warmup_grid = HexGrid(bounds=warmup_bounds, resolution=[8, 8, 8], device="cpu")
        Classification.classify_solid_narrow_band_csc(
            volume=warmup_volume, grid=warmup_grid, poly_order=1, device=device,
            band_cells=fnb_band_cells, candidate_padding_cells=0, tri_chunk=128,
            candidate_batch_size=2048, return_sdf=False,
        )
        torch.cuda.synchronize(torch.device(device))
        del warmup_mesh, warmup_volume, warmup_grid
        gc.collect()
        torch.cuda.empty_cache()
    rows: list[dict[str, object]] = []
    for resolution in resolutions:
        for variant in variants:
            trial_dir = output_dir / f"{variant}_{resolution}cubed"
            trial_dir.mkdir(parents=True, exist_ok=False)
            nn_config = NN_VARIANTS.get(variant)
            nn_engine = (
                load_nn_engine(
                    checkpoint_path=(
                        ROOT
                        / "clean_simulation"
                        / "outputs"
                        / "models"
                        / "archive"
                        / "checkpoints"
                        / nn_config["checkpoint"]
                    ),
                    device=device,
                    dtype="float32",
                )
                if nn_config is not None
                else None
            )
            if nn_engine is not None and torch.device(nn_engine.device).type != "cuda":
                raise RuntimeError(f"NN timing requires CUDA inference, received engine device {nn_engine.device!r}.")
            total_started = time.perf_counter()
            mesh_started = time.perf_counter()
            mesh = trimesh.load(stl_path, file_type="stl", force="mesh")
            if not mesh.is_watertight:
                raise ValueError("Blade mesh must be watertight.")
            stl_load_seconds = elapsed(mesh_started)
            setup_started = time.perf_counter()
            bounds = cube_bounds(np.asarray(mesh.bounds, dtype=np.float64))
            volume = Volume(mesh, device="cpu")
            grid = HexGrid(bounds=bounds, resolution=[resolution] * 3, device="cpu")
            material = Material(E=1.0e6, nu=0.3)
            grid_geometry_setup_seconds = elapsed(setup_started)
            stages = {name: 0.0 for name in (
                "fnb_candidate_generation_seconds", "fnb_intersection_seconds", "fnb_flood_fill_seconds",
                "weighted_volume_fraction_seconds", "nn_sdf_seconds", "nn_inference_seconds",
                "queso_classification_ips_seconds", "queso_stiffness_integration_seconds",
                "full_ke_reference_seconds", "full_ke_broadcast_seconds", "k_representation_assembly_seconds",
            )}

            if variant.endswith("_fnb"):
                fnb_started = time.perf_counter()
                inside_indices, cut_indices, _, fnb_meta = Classification.classify_solid_narrow_band_csc(
                    volume=volume, grid=grid, poly_order=1, device=device, band_cells=fnb_band_cells,
                    candidate_padding_cells=0, tri_chunk=128, candidate_batch_size=2048, return_sdf=False,
                )
                fnb_total_seconds = elapsed(fnb_started)
                stages["fnb_candidate_generation_seconds"] = float(fnb_meta["candidate_generation_time"])
                stages["fnb_intersection_seconds"] = float(fnb_meta["sdf_classification_time"])
                stages["fnb_flood_fill_seconds"] = float(fnb_meta["fnb_flood_fill_seconds"])
                active_indices = torch.sort(torch.cat((inside_indices, cut_indices))).values
                full_mask = torch.zeros(resolution ** 3, dtype=torch.bool)
                full_mask[inside_indices] = True
                full_started = time.perf_counter()
                inside_k, inside_k_indices, full_diag = StiffnessAssembly.compute_inside_element_matrices(
                    grid, material, full_mask, order=2, device="cpu", poly_order=1, return_diagnostics=True,
                )
                stages["full_ke_reference_seconds"] = float(full_diag["inside_ref_compute"])
                stages["full_ke_broadcast_seconds"] = elapsed(full_started) - stages["full_ke_reference_seconds"]
                full_ke = inside_k[0] if len(inside_k) else torch.empty((24, 24), dtype=torch.float64)
                assembly_started = time.perf_counter()
                if variant == "weighted_fnb":
                    fraction_started = time.perf_counter()
                    fractions = compute_cell_volume_fractions_ray_parity(
                        grid, volume, cut_indices, samples_per_axis=fraction_samples, device=device,
                    )
                    stages["weighted_volume_fraction_seconds"] = elapsed(fraction_started)
                    cut_k = full_ke.expand(len(cut_indices), -1, -1).clone() * fractions[:, None, None]
                elif nn_config is not None:
                    sdf_started = time.perf_counter()
                    d_cut, nu_cut, h_ref = prepare_nn_inputs(
                        grid,
                        volume,
                        cut_indices,
                        material,
                        1,
                        refined_subcells=nn_config["refined_subcells"],
                        sdf_device=device,
                        sdf_batch_size=2048,
                        global_sign=True,
                    )
                    stages["nn_sdf_seconds"] = elapsed(sdf_started)
                    inference_started = time.perf_counter()
                    predicted, predicted_indices = compute_nn_cutcell_stiffness(d_cut, nu_cut, nn_engine, material, cut_indices, h_ref)
                    stages["nn_inference_seconds"] = elapsed(inference_started)
                    cut_indices, cut_k = predicted_indices.to(torch.int64), torch.stack(predicted).cpu()
                else:
                    cut_k = full_ke.expand(len(cut_indices), -1, -1).clone()
                k_indices, k_representation = merge_k_representation(inside_k_indices, inside_k, cut_indices, cut_k)
                stages["k_representation_assembly_seconds"] = elapsed(assembly_started) - stages["weighted_volume_fraction_seconds"] - stages["nn_sdf_seconds"] - stages["nn_inference_seconds"]
                cut_cells = len(cut_indices)
            else:
                queso_started = time.perf_counter()
                case = build_case_from_stl(
                    stl_path=stl_path,
                    volume=volume,
                    overrides={"grid": {"bounds": bounds, "resolution": [resolution] * 3}, "material": {"E": 1.0e6, "nu": 0.3, "density": 1.0}, "integration": {"poly_order": 1, "gauss_order": 2, "moment_fitting_residual": 1.0e-8}},
                    workspace_root=ROOT, work_dir=trial_dir, queso_python_path=ROOT / "QuESo" / "out" / "install" / "linux",
                    queso_basis_order=3 if variant == "queso_p3" else 2, cut_cells_only=False, bounds_padding=0.0, device="cpu",
                )
                stages["queso_classification_ips_seconds"] = elapsed(queso_started)
                integration_started = time.perf_counter()
                k_representation, k_indices = StiffnessAssembly.compute_element_matrices_from_queso_points(
                    case.grid, case.material, np.asarray(case.ips), poly_order=1, device="cpu", element_batch_size=512,
                )
                stages["queso_stiffness_integration_seconds"] = elapsed(integration_started)
                coords = np.asarray(case.ips, dtype=np.float64)[:, :3]
                if np.all((coords >= 0.0) & (coords <= 1.0)):
                    coords = np.asarray(case.grid.bounds[0]) + coords * (np.asarray(case.grid.bounds[1]) - np.asarray(case.grid.bounds[0]))
                mapped, _ = map_integration_points_to_elements(coords, case.grid.bounds, case.grid.resolution)
                cut_cells = len(np.unique(mapped[np.asarray(case.queso_metadata["trimmed_ip_mask"], dtype=bool)]))
                active_indices = k_indices
                fnb_total_seconds = 0.0

            total_seconds = elapsed(total_started)
            named_total = stl_load_seconds + grid_geometry_setup_seconds + sum(stages.values())
            row = {
                "variant": variant, "resolution": resolution, "stl_load_seconds": stl_load_seconds,
                "grid_geometry_setup_seconds": grid_geometry_setup_seconds, "fnb_total_seconds": fnb_total_seconds,
                **stages, "end_to_end_k_seconds": total_seconds, "unattributed_seconds": total_seconds - named_total,
                "active_cells": len(active_indices), "cut_cells": cut_cells,
                "weighted_samples_per_axis": fraction_samples if variant == "weighted_fnb" else 0,
                "nn_global_watertight_sign": nn_config is not None, "nn_refined_subcells": nn_config["refined_subcells"] if nn_config is not None else 0,
                "queso_basis_order": 3 if variant == "queso_p3" else (2 if variant == "queso_p2" else 0),
                "nn_inference_device": str(nn_engine.device) if nn_engine is not None else "not_applicable",
            }
            rows.append(row)
            with (output_dir / "blade_end_to_end_k_timing.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{variant} {resolution}^3: {total_seconds:.3f}s STL-to-local-K; residual={row['unattributed_seconds']:.3f}s", flush=True)
            del mesh, volume, grid, material, k_representation, k_indices, nn_engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()