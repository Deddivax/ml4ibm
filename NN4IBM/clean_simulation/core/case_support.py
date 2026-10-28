"""Configuration, geometry, and QuESo setup helpers for case construction."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from clean_simulation.queso.settings import make_queso_settings


def load_internal_case_builder(builder_path: str):
    module_name, symbol_name = builder_path.split(":", maxsplit=1)
    if not module_name.startswith("clean_simulation."):
        raise ValueError(
            "External case_builder modules are disabled. "
            "Use an internal module path starting with 'clean_simulation.'"
        )
    builder = getattr(importlib.import_module(module_name), symbol_name)
    if not callable(builder):
        raise TypeError(f"Case builder is not callable: {builder_path}")
    return builder


def write_queso_settings(
    *, config: dict[str, Any], stl_path: Path, settings_path: Path, basis_order: int
) -> None:
    settings = make_queso_settings(
        bounds=config["grid"]["bounds"],
        resolution=config["grid"]["resolution"],
        input_filename=stl_path,
        basis_order=basis_order,
        moment_fitting_residual=config["integration"]["moment_fitting_residual"],
    )
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def build_volume_from_geometry_spec(volume_factory, geometry_spec, root: Path):
    kind = str(geometry_spec.get("kind", "sphere")).lower()
    center = tuple(geometry_spec.get("center", (0.0, 0.0, 0.0)))
    if kind == "sphere":
        return volume_factory.sphere(radius=float(geometry_spec["radius"]), center=center)
    if kind == "box":
        return volume_factory.box(extents=tuple(geometry_spec["extents"]), center=center)
    if kind == "cylinder":
        return volume_factory.cylinder(
            radius=float(geometry_spec["radius"]),
            height=float(geometry_spec["height"]),
            segments=int(geometry_spec.get("segments", 64)),
        )
    if kind == "file":
        mesh_path = Path(geometry_spec["path"])
        print(f"[CASE BUILD] Loading geometry from file: {mesh_path}")
        mesh_path = mesh_path if mesh_path.is_absolute() else root / mesh_path
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Geometry file not found: {mesh_path}")
        return volume_factory.from_file(mesh_path)
    raise ValueError("geometry.kind must be one of: sphere, box, cylinder, file.")


def extract_boundary_mesh(mesh, top_fraction, sphere_radius=None, axis=2, side="max"):
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("bc.constraint_axis must be 0, 1, or 2.")
    if side not in ("min", "max"):
        raise ValueError("bc.constraint_side must be 'min' or 'max'.")

    face_centers = mesh.triangles_center
    axis_min = float(mesh.vertices[:, axis].min())
    axis_max = float(mesh.vertices[:, axis].max())
    cap_height = float(top_fraction) * (
        2.0 * float(sphere_radius) if sphere_radius is not None else axis_max - axis_min
    )
    cap_height = max(cap_height, 1e-6)
    mask = face_centers[:, axis] >= axis_max - cap_height if side == "max" else face_centers[:, axis] <= axis_min + cap_height
    top_faces = mesh.faces[np.where(mask)[0]]
    if len(top_faces) == 0:
        raise ValueError(
            "Constraint selection produced no faces; adjust bc.constraint_axis, "
            "bc.constraint_side, or bc.top_cap_fraction."
        )
    return trimesh.Trimesh(vertices=mesh.vertices, faces=top_faces, process=False)


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config_for_mesh(mesh: trimesh.Trimesh, *, bounds_padding: float) -> dict[str, Any]:
    lower, upper = mesh.bounds
    padding = (upper - lower) * float(bounds_padding)
    return {
        "grid": {"bounds": [(lower - padding).tolist(), (upper + padding).tolist()], "resolution": [8, 8, 8]},
        "material": {"E": 1.0e5, "nu": 0.3},
        "integration": {"poly_order": 2, "gauss_order": 3, "moment_fitting_residual": 1.0e-10},
        "bc": {"top_cap_fraction": 0.05, "constraint_axis": 2, "constraint_side": "max", "nitsche_penalty": 100.0},
    }