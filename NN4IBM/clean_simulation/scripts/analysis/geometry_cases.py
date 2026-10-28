"""Generic STL case builders for portable multi-geometry analysis studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clean_simulation.case import build_case_from_stl


def build_small_stl_case(
    *,
    stl_path: str | Path,
    resolution: int = 8,
    workspace_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    queso_python_path: str | Path | None = None,
    device: str = "cpu",
    overrides: dict[str, Any] | None = None,
):
    """Build a low-complexity arbitrary-STL case suitable for validation studies."""
    base_overrides: dict[str, Any] = {
        "grid": {"resolution": [int(resolution)] * 3},
        "material": {"density": 1.0},
        "loads": {"gravity": [0.0, 0.0, -9.81]},
        "integration": {"poly_order": 1, "gauss_order": 2},
    }
    if overrides:
        for section, values in overrides.items():
            if isinstance(values, dict) and isinstance(base_overrides.get(section), dict):
                base_overrides[section].update(values)
            else:
                base_overrides[section] = values
    return build_case_from_stl(
        stl_path=stl_path,
        overrides=base_overrides,
        workspace_root=workspace_root,
        work_dir=work_dir,
        queso_python_path=queso_python_path,
        device=device,
    )
