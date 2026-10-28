"""Run the guarded hanging-sphere solve portion of the convergence notebook."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--frameworks", default="naive,weighted,queso,nn")
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    notebook_path = repository_root / "clean_simulation" / "notebooks" / "05_local_stiffness_convergence.ipynb"
    output_dir = arguments.output_dir.expanduser().resolve()
    os.chdir(repository_root)
    os.environ.update(
        {
            "HANGING_SPHERE_OUTPUT_DIR": str(output_dir),
            "HANGING_SPHERE_RESOLUTIONS": str(arguments.resolution),
            "HANGING_SPHERE_RUN_SOLVES": "1",
            "HANGING_SPHERE_DEVICE": arguments.device,
            "HANGING_SPHERE_CPU_THREADS": str(arguments.cpu_threads),
            "HANGING_SPHERE_FRAMEWORKS": arguments.frameworks,
            "HANGING_SPHERE_DTYPE": "float32",
        }
    )

    notebook = nbformat.read(notebook_path, as_version=4)
    namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.source.startswith("# Editor-safe local smooth-foundation study"):
            continue
        if "inspect.getsource(framework_body_force)" in cell.source:
            continue
        source = cell.source.replace(
            "framework_force = framework_body_force(case, full_indices, cut_indices)",
            "framework_force = shared_force",
        )
        exec(compile(source, str(notebook_path), "exec"), namespace)
        if "if RUN_SOLVES:" in source:
            break

    print(f"Completed {arguments.resolution}^3 hanging-sphere solve section.", flush=True)


if __name__ == "__main__":
    main()