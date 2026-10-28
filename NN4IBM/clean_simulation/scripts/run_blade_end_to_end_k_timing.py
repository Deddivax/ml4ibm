"""Cache-cold Blade timing from STL load through framework-specific K construction."""

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
from clean_simulation.internal_fem_kernel.queso_integration import map_integration_points_to_elements
from clean_simulation.internal_nn_runtime.cutcell_bridge import compute_nn_cutcell_stiffness, prepare_nn_inputs
from clean_simulation.nn_runtime import load_nn_engine
from clean_simulation.scripts.studies.blade.geometry import cube_bounds
from clean_simulation.scripts.studies.blade.timing import elapsed


VARIANTS = ("naive", "weighted", "queso_p2", "queso_p3", "nn_queso_classifier")


def queso_basis_order(variant: str) -> int:
    return 3 if variant == "queso_p3" else 2


def main() -> None:
    output_dir = Path(os.environ["BLADE_E2E_K_TIMING_OUTPUT_DIR"]).expanduser().resolve()
    stl_path = Path(os.environ["BLADE_STL_PATH"]).expanduser().resolve()
    resolutions = tuple(int(value) for value in os.environ.get("BLADE_RESOLUTIONS", "16,32,64,128").split(","))
    variants = tuple(value for value in os.environ.get("BLADE_E2E_K_VARIANTS", ",".join(VARIANTS)).split(",") if value)
    device = os.environ.get("BLADE_DEVICE", "cuda")
    if set(variants) - set(VARIANTS):
        raise ValueError(f"Unknown timing variants: {set(variants) - set(VARIANTS)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BLADE_DEVICE=cuda was requested but CUDA is unavailable.")
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)

    configure_legacy_imports(ROOT)
    from internal_fem_kernel.assembly.stiffness import StiffnessAssembly
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "clean_simulation" / "outputs" / "models" / "archive" / "checkpoints" / "source_sdf_rebuilt_queso_d10_f1024_all_crystals_10h.pt"
    rows: list[dict[str, object]] = []
    for resolution in resolutions:
        for variant in variants:
            trial_dir = output_dir / f"{variant}_{resolution}cubed"
            if trial_dir.exists():
                raise RuntimeError(f"Cache-cold timing requires an unused trial directory: {trial_dir}")
            nn_engine = (
                load_nn_engine(checkpoint_path=checkpoint, device=device, dtype="float32")
                if variant == "nn_queso_classifier" else None
            )
            total_started = time.perf_counter()
            load_started = time.perf_counter()
            with stl_path.open("rb") as stream:
                mesh = trimesh.load(stream, file_type="stl", force="mesh")
            if not mesh.is_watertight:
                raise ValueError("Blade mesh must be watertight.")
            bounds = cube_bounds(np.asarray(mesh.bounds, dtype=np.float64))
            stl_load_seconds = elapsed(load_started)
            overrides = {
                "grid": {"bounds": bounds, "resolution": [resolution] * 3},
                "material": {"E": 1.0e6, "nu": 0.3, "density": 1.0},
                "integration": {"poly_order": 1, "gauss_order": 2, "moment_fitting_residual": 1.0e-8},
            }
            case_started = time.perf_counter()
            case = build_case_from_stl(
                stl_path=stl_path, overrides=overrides, workspace_root=ROOT, work_dir=trial_dir,
                queso_python_path=ROOT / "QuESo" / "out" / "install" / "linux",
                queso_basis_order=queso_basis_order(variant), cut_cells_only=False, bounds_padding=0.0, device="cpu",
            )
            queso_pipeline_seconds = elapsed(case_started)
            coordinates = np.asarray(case.ips, dtype=np.float64)[:, :3]
            weights = np.asarray(case.ips, dtype=np.float64)[:, 3]
            if np.all((coordinates >= 0.0) & (coordinates <= 1.0)):
                coordinates = np.asarray(case.grid.bounds[0]) + coordinates * (np.asarray(case.grid.bounds[1]) - np.asarray(case.grid.bounds[0]))
                weights = weights * np.prod(np.asarray(case.grid.bounds[1]) - np.asarray(case.grid.bounds[0]))
            mapped_indices, _ = map_integration_points_to_elements(coordinates, case.grid.bounds, case.grid.resolution)
            active_indices = torch.from_numpy(np.unique(mapped_indices).astype(np.int64))
            cut_indices = torch.from_numpy(np.unique(mapped_indices[np.asarray(case.queso_metadata["trimmed_ip_mask"], dtype=bool)]).astype(np.int64))
            queso_matrices = None
            queso_integration_seconds = 0.0
            if variant.startswith("queso_"):
                matrix_started = time.perf_counter()
                queso_matrices, queso_active_indices = StiffnessAssembly.compute_element_matrices_from_queso_points(
                    case.grid, case.material, np.asarray(case.ips), poly_order=1, device="cpu", element_batch_size=512
                )
                queso_integration_seconds = elapsed(matrix_started)
                if not torch.equal(active_indices, queso_active_indices):
                    raise RuntimeError("QuESo stiffness integration returned a different active-cell set than the classified IPs.")
            full_seed = torch.zeros(resolution ** 3, dtype=torch.bool)
            full_seed[active_indices[0]] = True
            full_ke = StiffnessAssembly.compute_inside_element_matrices(case.grid, case.material, full_seed, order=case.g_order, device="cpu", poly_order=1)[0][0]
            fractions_seconds = 0.0
            nn_sdf_seconds = 0.0
            inference_seconds = 0.0
            assembly_started = time.perf_counter()
            if variant == "naive":
                k_representation = full_ke.expand(len(active_indices), -1, -1).clone()
            elif variant == "weighted":
                fractions_started = time.perf_counter()
                cell_volumes = np.bincount(mapped_indices, weights=weights, minlength=resolution ** 3)
                fractions = torch.from_numpy((cell_volumes / (case.grid.dx * case.grid.dy * case.grid.dz)).clip(0.0, 1.0))
                fractions_seconds = elapsed(fractions_started)
                k_representation = full_ke.expand(len(active_indices), -1, -1).clone()
                k_representation[torch.searchsorted(active_indices, cut_indices)] *= fractions[cut_indices, None, None]
            elif variant.startswith("queso_"):
                k_representation = queso_matrices
            else:
                sdf_started = time.perf_counter()
                d_cut, nu_cut, h_ref = prepare_nn_inputs(case.grid, case.volume, cut_indices, case.material, 1, sdf_device=device, sdf_batch_size=2048)
                nn_sdf_seconds = elapsed(sdf_started)
                inference_started = time.perf_counter()
                predicted, predicted_indices = compute_nn_cutcell_stiffness(d_cut, nu_cut, nn_engine, case.material, cut_indices, h_ref)
                inference_seconds = elapsed(inference_started)
                k_representation = full_ke.expand(len(active_indices), -1, -1).clone()
                k_representation[torch.searchsorted(active_indices, predicted_indices.to(torch.int64))] = torch.stack(predicted).cpu()
                del d_cut, nu_cut, predicted, predicted_indices
            framework_assembly_seconds = elapsed(assembly_started) - fractions_seconds - nn_sdf_seconds - inference_seconds
            total_seconds = elapsed(total_started)
            row = {
                "variant": variant, "resolution": resolution, "stl_load_seconds": stl_load_seconds,
                "queso_basis_order": queso_basis_order(variant),
                "queso_classification_ips_seconds": queso_pipeline_seconds,
                "queso_stiffness_integration_seconds": queso_integration_seconds,
                "volume_fraction_seconds": fractions_seconds, "nn_sdf_seconds": nn_sdf_seconds,
                "nn_inference_seconds": inference_seconds, "framework_assembly_seconds": framework_assembly_seconds,
                "end_to_end_k_seconds": total_seconds, "active_cells": len(active_indices), "cut_cells": len(cut_indices),
                "nn_classifier": "queso" if variant == "nn_queso_classifier" else "not_applicable",
            }
            rows.append(row)
            csv_path = output_dir / "blade_end_to_end_k_timing.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{variant} {resolution}^3: {total_seconds:.3f}s STL-to-K", flush=True)
            del mesh, case, queso_matrices, active_indices, cut_indices, full_ke, k_representation, nn_engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()