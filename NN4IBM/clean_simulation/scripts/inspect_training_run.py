"""Compatibility entry point for the flat training-run inspector."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean_simulation.scripts.training._inspect import main


if __name__ == "__main__":
    main()