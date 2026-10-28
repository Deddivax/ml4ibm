"""Compare FNB cut-cell classification against QuESo for a watertight Blade STL."""

from __future__ import annotations

import csv
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
from clean_simulation.scripts.studies.blade.geometry import cube_bounds


def main() -> None:
    stl_path = Path(os.environ["BLADE_STL_PATH"]).expanduser().resolve()
    output_dir = Path(os.environ["BLADE_FNB_CHECK_OUTPUT_DIR"]).expanduser().resolve()
    resolutions = tuple(int(value) for value in os.environ.get("BLADE_RESOLUTIONS", "16,32,64,128").split(","))
    device = os.environ.get("BLADE_DEVICE", "cuda")
    band_cells = int(os.environ.get("BLADE_FNB_BAND_CELLS", "3"))
    candidate_padding_cells = int(os.environ.get("BLADE_FNB_CANDIDATE_PADDING_CELLS", "1"))
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)

    import trimesh

    mesh = trimesh.load(stl_path, file_type="stl", force="mesh")
    if not mesh.is_watertight:
        raise ValueError("Blade mesh must be watertight.")
    bounds = cube_bounds(np.asarray(mesh.bounds, dtype=np.float64))
    configure_legacy_imports(ROOT)
    from constants import Material
    from geometry import HexGrid
    from volume_factory import VolumeFactory

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for resolution in resolutions:
        grid = HexGrid(bounds=bounds, resolution=[resolution] * 3, device="cpu")
        volume = VolumeFactory.from_file(stl_path)
        material = Material(E=1.0e6, nu=0.3)
        started = time.perf_counter()
        fnb_inside_indices, fnb_indices, _, fnb_metadata = Classification.classify_solid_narrow_band_csc(
            volume=volume,
            grid=grid,
            poly_order=1,
            device=device,
            band_cells=band_cells,
            candidate_padding_cells=candidate_padding_cells,
            tri_chunk=128,
            candidate_batch_size=2048,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fnb_seconds = time.perf_counter() - started
        fnb_indices = torch.unique(fnb_indices.to(dtype=torch.int64), sorted=True)
        fnb_active_indices = torch.sort(torch.cat((fnb_inside_indices, fnb_indices))).values

        queso_case = build_case_from_stl(
            stl_path=stl_path,
            overrides={
                "grid": {"bounds": bounds, "resolution": [resolution] * 3},
                "material": {"E": 1.0e6, "nu": 0.3, "density": 1.0},
                "integration": {"poly_order": 1, "gauss_order": 2, "moment_fitting_residual": 1.0e-8},
            },
            workspace_root=ROOT,
            work_dir=output_dir / f"queso_{resolution}cubed",
            queso_python_path=ROOT / "QuESo" / "out" / "install" / "linux",
            queso_basis_order=2,
            cut_cells_only=False,
            bounds_padding=0.0,
            device="cpu",
        )
        coordinates = np.asarray(queso_case.ips, dtype=np.float64)[:, :3]
        if np.all((coordinates >= 0.0) & (coordinates <= 1.0)):
            grid_bounds = np.asarray(queso_case.grid.bounds, dtype=np.float64)
            coordinates = grid_bounds[0] + coordinates * (grid_bounds[1] - grid_bounds[0])
        mapped_indices, _ = map_integration_points_to_elements(coordinates, queso_case.grid.bounds, queso_case.grid.resolution)
        queso_indices = torch.from_numpy(np.unique(
            mapped_indices[np.asarray(queso_case.queso_metadata["trimmed_ip_mask"], dtype=bool)]
        ).astype(np.int64))
        queso_active_indices = torch.from_numpy(np.unique(mapped_indices).astype(np.int64))
        false_positive = torch.isin(fnb_indices, queso_indices, invert=True)
        false_negative = torch.isin(queso_indices, fnb_indices, invert=True)
        active_false_positive = torch.isin(fnb_active_indices, queso_active_indices, invert=True)
        active_false_negative = torch.isin(queso_active_indices, fnb_active_indices, invert=True)
        row = {
            "resolution": resolution,
            "fnb_band_cells": band_cells,
            "fnb_candidate_padding_cells": candidate_padding_cells,
            "fnb_candidate_cells": fnb_metadata["candidate_cell_count"],
            "fnb_cut_cells": int(fnb_indices.numel()),
            "fnb_seconds": fnb_seconds,
            "queso_cut_cells": int(queso_indices.numel()),
            "fnb_false_positive_cells": int(false_positive.sum()),
            "fnb_false_negative_cells": int(false_negative.sum()),
            "fnb_active_cells": int(fnb_active_indices.numel()),
            "queso_active_cells": int(queso_active_indices.numel()),
            "fnb_active_false_positive_cells": int(active_false_positive.sum()),
            "fnb_active_false_negative_cells": int(active_false_negative.sum()),
        }
        rows.append(row)
        print(row, flush=True)

    output_path = output_dir / "blade_fnb_vs_queso_cut_cells.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()