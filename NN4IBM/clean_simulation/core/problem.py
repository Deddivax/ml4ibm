from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BoundaryConditionSpec:
    name: str
    triangles: Any
    prescribed_displacement: Any | None = None
    penalty: float = 1.0e7
    order: int = 2


@dataclass(slots=True)
class SimulationProblem:
    grid: Any
    volume: Any
    material: Any
    config: dict[str, Any]
    poly_order: int
    g_order: int
    integration_points: Any
    boundary_conditions: list[BoundaryConditionSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    workspace_root: Path | None = None
    work_dir: Path | None = None
