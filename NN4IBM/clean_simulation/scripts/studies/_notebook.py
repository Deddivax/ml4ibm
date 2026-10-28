"""Execute selected cells from canonical notebooks used by study commands."""

from __future__ import annotations

import json
from pathlib import Path


def load_notebook_cells(notebook_path: Path) -> list[dict[str, object]]:
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Missing canonical notebook: {notebook_path}")
    return json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]


def execute_notebook_cell(
    notebook_path: Path,
    namespace: dict[str, object],
    cells: list[dict[str, object]],
    cell_index: int,
) -> None:
    source = "\n".join(cells[cell_index]["source"])
    exec(compile(source, f"{notebook_path.name}: cell {cell_index + 1}", "exec"), namespace)