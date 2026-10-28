"""Compatibility imports for historical direct-script execution.

New package-qualified code must not require this module. It remains for existing
entry points that need vendored engine modules and legacy path cleanup.
"""

from __future__ import annotations

import sys
from pathlib import Path


_BOOTSTRAPPED = False


def _remove_external_legacy_paths(root: Path) -> None:
    forbidden = {
        str((root / "basic_simulation").resolve()),
        str((root / "NN-IBM").resolve()),
    }
    sys.path[:] = [path for path in sys.path if str(Path(path).resolve()) not in forbidden]


def configure_legacy_imports(workspace_root: str | Path | None = None) -> Path:
    """Configure imports to use vendored internal kernels only."""
    global _BOOTSTRAPPED

    root = Path(workspace_root).resolve() if workspace_root is not None else Path(__file__).resolve().parents[2]
    if not _BOOTSTRAPPED:
        _remove_external_legacy_paths(root)
        clean_package = root / "clean_simulation"
        for candidate in (clean_package / "internal_fem_kernel", clean_package, root):
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
        _BOOTSTRAPPED = True
    return root
