"""Geometry-specific study command entry points."""

from pathlib import Path
from runpy import run_path


def run_legacy(filename: str) -> None:
    run_path(Path(__file__).resolve().parent.parent / filename, run_name="__main__")
