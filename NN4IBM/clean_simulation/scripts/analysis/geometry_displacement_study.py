"""Portable classic-versus-NN displacement studies for arbitrary STL geometries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from clean_simulation.compare import collect_global_displacements
from clean_simulation.config import CaseConfig, ComparisonRequest, NNRuntimeConfig
from clean_simulation.scripts.analysis.displacement_fields import active_node_ids


@dataclass(frozen=True, slots=True)
class GeometryStudyCase:
    """One named STL input in a displacement comparison study."""

    name: str
    stl_path: Path


def default_geometry_cases(workspace_root: str | Path) -> tuple[GeometryStudyCase, ...]:
    """Return the portable sphere, eggcarton, and Blade study inputs."""
    root = Path(workspace_root).resolve()
    return (
        GeometryStudyCase("sphere", root / "clean_simulation/geometries/hanging_sphere_surface.stl"),
        GeometryStudyCase("eggcarton", root / "clean_simulation/geometries/eggcarton_sphere.stl"),
        GeometryStudyCase(
            "blade",
            root / "clean_simulation/outputs/studies/blade/grid_locked_end_to_end/"
            "prepared_grid_locked_geometry/Blade_grid_locked_mc_16cubed.stl",
        ),
    )


def run_geometry_displacement_study(
    cases: Iterable[GeometryStudyCase],
    *,
    checkpoint_path: str | Path,
    workspace_root: str | Path,
    output_dir: str | Path,
    resolution: int = 8,
    device: str = "cpu",
    queso_python_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run classic and hybrid solves and return per-case and per-node error tables."""
    root = Path(workspace_root).resolve()
    artifacts = Path(output_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    queso_path = (
        Path(queso_python_path)
        if queso_python_path is not None
        else root / "QuESo/out/install/win"
    )

    summary_rows: list[dict[str, object]] = []
    node_tables: list[pd.DataFrame] = []
    for study_case in cases:
        stl_path = Path(study_case.stl_path)
        if not stl_path.is_file():
            raise FileNotFoundError(f"Study STL is missing: {stl_path}")

        request = ComparisonRequest(
            pipelines=("classic", "hybrid"),
            reference_pipeline="classic",
            device=device,
            nn=NNRuntimeConfig(checkpoint_path=Path(checkpoint_path), device=device),
            case=CaseConfig(
                case_builder="clean_simulation.scripts.analysis.geometry_cases:build_small_stl_case",
                case_args={
                    "stl_path": str(stl_path),
                    "resolution": int(resolution),
                    "workspace_root": str(root),
                    "work_dir": str(artifacts / study_case.name),
                    "queso_python_path": str(queso_path),
                },
            ),
        )
        collected = collect_global_displacements(request)
        reference = collected["displacements"]["classic"]
        predicted = collected["displacements"]["hybrid"]
        reference_nodes = active_node_ids(reference["active_indices"], resolution)
        predicted_nodes = active_node_ids(predicted["active_indices"], resolution)
        common_nodes = np.intersect1d(reference_nodes, predicted_nodes, assume_unique=True)
        if not len(common_nodes):
            raise RuntimeError(f"{study_case.name} has no common active-node support.")

        reference_u = np.asarray(reference["u_nodes"], dtype=np.float64)[common_nodes]
        predicted_u = np.asarray(predicted["u_nodes"], dtype=np.float64)[common_nodes]
        difference = predicted_u - reference_u
        reference_norm = float(np.linalg.norm(reference_u))
        relative_l2 = float(np.linalg.norm(difference) / max(reference_norm, np.finfo(np.float64).tiny))
        node_error = np.linalg.norm(difference, axis=1)
        node_reference = np.linalg.norm(reference_u, axis=1)
        node_tables.append(pd.DataFrame({
            "geometry": study_case.name,
            "node_id": common_nodes,
            "absolute_displacement_error": node_error,
            "reference_displacement_magnitude": node_reference,
            "relative_error_to_reference_peak": node_error / max(float(node_reference.max()), np.finfo(np.float64).tiny),
        }))
        summary_rows.append({
            "geometry": study_case.name,
            "stl_path": str(stl_path),
            "resolution": int(resolution),
            "reference_pipeline": "classic",
            "candidate_pipeline": "hybrid",
            "relative_l2_displacement_error": relative_l2,
            "common_active_nodes": int(len(common_nodes)),
            "classic_only_nodes": int(len(np.setdiff1d(reference_nodes, common_nodes, assume_unique=True))),
            "hybrid_only_nodes": int(len(np.setdiff1d(predicted_nodes, common_nodes, assume_unique=True))),
            "classic_active_elements": int(len(reference["active_indices"])),
            "hybrid_active_elements": int(len(predicted["active_indices"])),
            "classic_total_time_s": float(reference["timings"].get("total", 0.0)),
            "hybrid_total_time_s": float(predicted["timings"].get("total", 0.0)),
        })

    summary = pd.DataFrame(summary_rows)
    node_errors = pd.concat(node_tables, ignore_index=True) if node_tables else pd.DataFrame()
    summary.to_csv(artifacts / "displacement_summary.csv", index=False)
    node_errors.to_csv(artifacts / "nodewise_displacement_errors.csv", index=False)
    return summary, node_errors