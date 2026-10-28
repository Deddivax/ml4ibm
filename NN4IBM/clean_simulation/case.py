from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .core.legacy_imports import configure_legacy_imports
from .core.case_support import (
    build_volume_from_geometry_spec,
    deep_merge,
    default_config_for_mesh,
    extract_boundary_mesh,
    load_internal_case_builder,
    write_queso_settings,
)
from .queso.integration import run_queso_get_integration_points, run_queso_write_integration_points_hdf5


@dataclass(slots=True)
class SimulationCase:
    grid: Any
    volume: Any
    material: Any
    config: dict[str, Any]
    poly_order: int
    g_order: int
    ips: Any
    queso_metadata: dict[str, Any]
    top_mesh: trimesh.Trimesh
    workspace_root: Path
    work_dir: Path




def build_case_from_stl(
    *,
    stl_path: str | Path,
    overrides: dict[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    queso_python_path: str | Path | None = None,
    queso_basis_order: int = 2,
    keep_negative_weights: bool = True,
    weight_zero_tol: float = 0.0,
    cut_cells_only: bool = False,
    integration_points_npy_path: str | Path | None = None,
    integration_points_hdf5_path: str | Path | None = None,
    point_batch_size: int = 100_000,
    bounds_padding: float = 0.1,
    device: str = "cpu",
    volume: Any | None = None,
) -> SimulationCase:
    """
    Build a SimulationCase for arbitrary geometry supplied as an STL file.

    Unlike `build_example_case`, this does not depend on any example-specific
    config (e.g. the hanging-sphere config) — it works for any watertight or
    non-watertight STL mesh. Grid bounds are auto-derived from the mesh's
    bounding box (padded by `bounds_padding` fraction of the extent) unless
    explicitly provided via `overrides["grid"]["bounds"]`.

    Parameters
    ----------
    stl_path:
        Path to the input STL geometry file.
    overrides:
        Dict merged (recursively) on top of the generic default config.
        Supports the same keys as the hanging-sphere config: grid, material,
        integration, bc.
    bounds_padding:
        Fraction of the mesh's bounding-box extent to pad on each side when
        auto-deriving grid bounds. Ignored if bounds are explicitly overridden.
    volume:
        Optional already-loaded volume. When supplied, its mesh is used rather
        than reading `stl_path` again.
    cut_cells_only:
        Return QuESo integration points from trimmed elements only. Use this
        for local cut-cell studies that do not require interior-cell assembly.
    integration_points_hdf5_path:
        Write retained integration points to a chunked HDF5 file instead of
        materializing them in memory. The returned case has an empty `ips` array.
    integration_points_npy_path:
        Reuse retained integration points from a NumPy file when it exists;
        otherwise write the newly extracted point array there for later runs.
    """
    import time
    t_start = time.perf_counter()
    print("[CASE BUILD] Configuring legacy imports and loading libraries...")
    root = configure_legacy_imports(workspace_root)

    t_imp_start = time.perf_counter()
    from .internal_fem_kernel.constants import Material
    from .internal_fem_kernel.geometry import HexGrid
    from .internal_fem_kernel.volume_factory import VolumeFactory
    print(f"[CASE BUILD] Imports took {time.perf_counter() - t_imp_start:.4f} seconds.")

    stl_path = Path(stl_path)
    if not stl_path.is_absolute():
        stl_path = (root / stl_path).resolve()
    if not stl_path.is_file():
        raise FileNotFoundError(f"Geometry STL file not found: {stl_path}")

    t_geo = time.perf_counter()
    if volume is None:
        volume = VolumeFactory.from_file(stl_path)

    default_cfg = default_config_for_mesh(volume.mesh, bounds_padding=bounds_padding)
    cfg = deep_merge(default_cfg, overrides or {})

    poly_order = int(cfg["integration"]["poly_order"])
    g_order = int(cfg["integration"]["gauss_order"])

    grid = HexGrid(
        bounds=cfg["grid"]["bounds"],
        resolution=cfg["grid"]["resolution"],
        device=device,
    )
    material = Material(
        E=float(cfg["material"]["E"]),
        nu=float(cfg["material"]["nu"]),
    )

    top_mesh = extract_boundary_mesh(
        volume.mesh,
        top_fraction=float(cfg["bc"]["top_cap_fraction"]),
        sphere_radius=None,
        axis=cfg["bc"].get("constraint_axis", 2),
        side=cfg["bc"].get("constraint_side", "max"),
    )
    print(f"[CASE BUILD] Geometry + mesh setup took {time.perf_counter() - t_geo:.4f} seconds.")

    if work_dir is None:
        work_path = root / "outputs" / "clean_simulation"
    else:
        work_path = Path(work_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    # We use the user-supplied STL directly rather than re-exporting one from
    # a procedurally generated volume.
    settings_path = work_path / f"QuESoSettings_{stl_path.stem}.json"

    t_settings = time.perf_counter()
    write_queso_settings(
        config=cfg,
        stl_path=stl_path,
        settings_path=settings_path,
        basis_order=int(queso_basis_order),
    )
    print(f"[CASE BUILD] QuESo settings write took {time.perf_counter() - t_settings:.4f} seconds.")

    t_queso = time.perf_counter()
    # Fail fast if QuESo fails to run: do not fall back or catch exceptions silently
    extraction_args = {
        'settings_json_path': settings_path,
        'work_dir': work_path,
        'queso_python_path': None if queso_python_path is None else str(Path(queso_python_path).resolve()),
        'keep_negative_weights': bool(keep_negative_weights),
        'weight_zero_tol': float(weight_zero_tol),
        'cut_cells_only': bool(cut_cells_only),
    }
    if integration_points_npy_path is not None and integration_points_hdf5_path is not None:
        raise ValueError('Specify at most one integration-point cache path.')
    if integration_points_npy_path is not None:
        points_path = Path(integration_points_npy_path)
        if points_path.exists():
            ips = np.load(points_path, mmap_mode='r')
            queso_metadata = {
                'queso_run_time': 0.0,
                'point_extraction_time': 0.0,
                'inside_collect_time': 0.0,
                'trimmed_collect_time': 0.0,
                'weight_filter_time': 0.0,
                'reused_point_cache': True,
            }
            print(f"[CASE BUILD] Reusing {len(ips):,} retained QuESo points from {points_path.name}.")
        else:
            ips, queso_metadata = run_queso_get_integration_points(return_metadata=True, **extraction_args)
            points_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(points_path, ips)
            print(f"[CASE BUILD] Saved {len(ips):,} retained QuESo points to {points_path.name}.")
    elif integration_points_hdf5_path is None:
        ips, queso_metadata = run_queso_get_integration_points(return_metadata=True, **extraction_args)
    else:
        queso_metadata = run_queso_write_integration_points_hdf5(
            output_path=integration_points_hdf5_path,
            point_batch_size=int(point_batch_size),
            physical_bounds=np.asarray(grid.bounds, dtype=np.float64),
            **extraction_args,
        )
        ips = np.zeros((0, 4), dtype=np.float64)
    print(f"[CASE BUILD] QuESo total took {time.perf_counter() - t_queso:.4f} seconds.")
    print(f"[CASE BUILD] QuESo Run() took {queso_metadata['queso_run_time']:.4f} seconds.")
    print(f"[CASE BUILD] QuESo IP extraction took {queso_metadata['point_extraction_time']:.4f} seconds.")
    if integration_points_npy_path is not None or integration_points_hdf5_path is None:
        print(f"[CASE BUILD] QuESo inside/trimmed IP extraction: {queso_metadata['inside_collect_time']:.4f}s / {queso_metadata['trimmed_collect_time']:.4f}s.")
        print(f"[CASE BUILD] QuESo weight filtering took {queso_metadata['weight_filter_time']:.4f} seconds.")
    else:
        print(f"[CASE BUILD] QuESo wrote {queso_metadata['point_count']:,} retained points to {integration_points_hdf5_path}.")
    print(f"[CASE BUILD] Total case build time: {time.perf_counter() - t_start:.4f} seconds.")

    return SimulationCase(
        grid=grid,
        volume=volume,
        material=material,
        config=cfg,
        poly_order=poly_order,
        g_order=g_order,
        ips=ips,
        queso_metadata=queso_metadata,
        top_mesh=top_mesh,
        workspace_root=root,
        work_dir=work_path,
    )


def build_example_case(
    *,
    overrides: dict[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    queso_python_path: str | Path | None = None,
    queso_basis_order: int = 2,
    keep_negative_weights: bool = True,
    weight_zero_tol: float = 0.0,
    device: str = "cpu",
) -> SimulationCase:
    import time
    t_start = time.perf_counter()
    print("[CASE BUILD] Configuring legacy imports and loading libraries...")
    root = configure_legacy_imports(workspace_root)

    t_imp_start = time.perf_counter()
    from constants import Material
    from geometry import HexGrid
    from clean_simulation.examples.hanging_sphere.config import get_hanging_sphere_config
    from volume_factory import VolumeFactory
    print(f"[CASE BUILD] Imports took {time.perf_counter() - t_imp_start:.4f} seconds.")

    cfg = get_hanging_sphere_config(overrides or {})
    poly_order = int(cfg["integration"]["poly_order"])
    g_order = int(cfg["integration"]["gauss_order"])

    t_geo = time.perf_counter()
    geometry_spec = cfg.get("geometry")
    if geometry_spec is None:
        geometry_spec = {"kind": "sphere", **cfg["sphere"]}
    volume = build_volume_from_geometry_spec(VolumeFactory, geometry_spec, root)
    grid = HexGrid(
        bounds=cfg["grid"]["bounds"],
        resolution=cfg["grid"]["resolution"],
        device=device,
    )
    material = Material(
        E=float(cfg["material"]["E"]),
        nu=float(cfg["material"]["nu"]),
    )
    if geometry_spec.get("kind", "sphere") == "sphere":
        sphere_radius = geometry_spec.get("radius")
    else:
        sphere_radius = None
    top_mesh = extract_boundary_mesh(
        volume.mesh,
        top_fraction=float(cfg["bc"]["top_cap_fraction"]),
        sphere_radius=sphere_radius,
        axis=cfg["bc"].get("constraint_axis", 2),
        side=cfg["bc"].get("constraint_side", "max"),
    )
    print(f"[CASE BUILD] Geometry + mesh setup took {time.perf_counter() - t_geo:.4f} seconds.")

    if work_dir is None:
        work_path = root / "outputs" / "clean_simulation"
    else:
        work_path = Path(work_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    stl_path = work_path / "study_geometry_surface.stl"
    settings_path = work_path / "QuESoSettings_study_geometry.json"
    t_export = time.perf_counter()
    volume.mesh.export(stl_path)
    print(f"[CASE BUILD] STL export took {time.perf_counter() - t_export:.4f} seconds.")

    t_settings = time.perf_counter()
    write_queso_settings(
        config=cfg,
        stl_path=stl_path,
        settings_path=settings_path,
        basis_order=int(queso_basis_order),
    )
    print(f"[CASE BUILD] QuESo settings write took {time.perf_counter() - t_settings:.4f} seconds.")

    t_queso = time.perf_counter()
    # Fail fast if QuESo fails to run: do not fall back or catch exceptions silently
    ips, queso_metadata = run_queso_get_integration_points(
        settings_json_path=settings_path,
        work_dir=work_path,
        queso_python_path=None if queso_python_path is None else str(Path(queso_python_path).resolve()),
        keep_negative_weights=bool(keep_negative_weights),
        weight_zero_tol=float(weight_zero_tol),
        return_metadata=True,
    )
    print(f"[CASE BUILD] QuESo total took {time.perf_counter() - t_queso:.4f} seconds.")
    print(f"[CASE BUILD] QuESo Run() took {queso_metadata['queso_run_time']:.4f} seconds.")
    print(f"[CASE BUILD] QuESo IP extraction took {queso_metadata['point_extraction_time']:.4f} seconds.")
    print(f"[CASE BUILD] QuESo inside/trimmed IP extraction: {queso_metadata['inside_collect_time']:.4f}s / {queso_metadata['trimmed_collect_time']:.4f}s.")
    print(f"[CASE BUILD] QuESo weight filtering took {queso_metadata['weight_filter_time']:.4f} seconds.")
    print(f"[CASE BUILD] Total case build time: {time.perf_counter() - t_start:.4f} seconds.")

    return SimulationCase(
        grid=grid,
        volume=volume,
        material=material,
        config=cfg,
        poly_order=poly_order,
        g_order=g_order,
        ips=ips,
        queso_metadata=queso_metadata,
        top_mesh=top_mesh,
        workspace_root=root,
        work_dir=work_path,
    )


# Backward-compatible alias while internal API moves to generic naming.
build_hanging_sphere_case = build_example_case


def build_case(
    *,
    case_builder: str | None = None,
    case_args: dict[str, Any] | None = None,
    stl_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    queso_python_path: str | Path | None = None,
    queso_basis_order: int = 2,
    keep_negative_weights: bool = True,
    weight_zero_tol: float = 0.0,
    device: str = "cpu",
) -> Any:
    if case_builder:
        builder = load_internal_case_builder(case_builder)
        kwargs = dict(case_args or {})
        kwargs.setdefault("device", device)
        return builder(**kwargs)

    if stl_path is not None:
        return build_case_from_stl(
            stl_path=stl_path,
            overrides=overrides,
            workspace_root=workspace_root,
            work_dir=work_dir,
            queso_python_path=queso_python_path,
            queso_basis_order=queso_basis_order,
            keep_negative_weights=keep_negative_weights,
            weight_zero_tol=weight_zero_tol,
            device=device,
        )

    return build_example_case(
        overrides=overrides,
        workspace_root=workspace_root,
        work_dir=work_dir,
        queso_python_path=queso_python_path,
        queso_basis_order=queso_basis_order,
        keep_negative_weights=keep_negative_weights,
        weight_zero_tol=weight_zero_tol,
        device=device,
    )