from __future__ import annotations

import os
import shutil
from pathlib import Path

from clean_simulation.scripts.studies._notebook import execute_notebook_cell, load_notebook_cells

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "clean_simulation" / "notebooks" / "05_local_stiffness_convergence.ipynb"
OUTPUT_DIR = Path(
    os.environ.get(
        "HANGING_SPHERE_OUTPUT_DIR",
        ROOT / "clean_simulation" / "outputs" / "studies" / "sphere" / "corrected",
    )
).expanduser().resolve()
CHECKPOINT_PATH = Path(
    os.environ.get(
        "HANGING_SPHERE_NN_CHECKPOINT",
        ROOT / "clean_simulation" / "outputs" / "models" / "archive" / "checkpoints" / "source_sdf_rebuilt_queso_d10_f1024_all_crystals_10h.pt",
    )
).expanduser().resolve()
RESOLUTIONS = "8,16,32,64,128,256,512"
FRAMEWORK_NAMES = ("naive", "weighted", "queso", "nn")


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing NN checkpoint: {CHECKPOINT_PATH}")

    sphere_stl = ROOT / "clean_simulation" / "geometries" / "hanging_sphere_surface.stl"
    if not sphere_stl.is_file():
        raise FileNotFoundError(f"Missing hanging sphere STL: {sphere_stl}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_dir = OUTPUT_DIR / "queso_matrix_free_128cubed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sphere_stl, seed_dir / "study_geometry_surface.stl")

    os.environ["HANGING_SPHERE_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["HANGING_SPHERE_RUN_SOLVES"] = "1"
    resolutions = os.environ.get("HANGING_SPHERE_RESOLUTIONS", RESOLUTIONS)
    framework_names = tuple(
        name.strip()
        for name in os.environ.get("HANGING_SPHERE_FRAMEWORKS", ",".join(FRAMEWORK_NAMES)).split(",")
        if name.strip()
    )
    unknown_frameworks = set(framework_names) - set(FRAMEWORK_NAMES)
    if not framework_names or unknown_frameworks:
        raise ValueError(f"Unsupported framework selection: {framework_names}")

    os.environ["HANGING_SPHERE_RESOLUTIONS"] = resolutions
    os.environ["HANGING_SPHERE_REFERENCE_RESOLUTION"] = "512"
    os.environ["HANGING_SPHERE_COMPARISON_RESOLUTIONS"] = resolutions.replace(",512", "")
    os.environ["HANGING_SPHERE_QUESO_REFERENCE_RESOLUTION"] = "512"
    os.environ["HANGING_SPHERE_QUESO_REFERENCE_COMPARISON_RESOLUTIONS"] = resolutions
    os.environ.setdefault("HANGING_SPHERE_DEVICE", "cuda")
    os.environ.setdefault("HANGING_SPHERE_CPU_THREADS", "6")
    os.environ.setdefault("HANGING_SPHERE_CG_MAX_ITER", "12000")
    os.environ.setdefault("HANGING_SPHERE_DTYPE", "float64")

    cells = load_notebook_cells(NOTEBOOK_PATH)
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}
    for cell_index in (1, 2, 3, 4):
        execute_notebook_cell(NOTEBOOK_PATH, namespace, cells, cell_index)

    framework_labels = namespace["FRAMEWORKS"]
    framework_colors = namespace["FRAMEWORK_COLORS"]
    namespace["FRAMEWORKS"] = {name: framework_labels[name] for name in framework_names}
    namespace["FRAMEWORK_COLORS"] = {name: framework_colors[name] for name in framework_names}

    for cell_index in (6, 8, 9):
        execute_notebook_cell(NOTEBOOK_PATH, namespace, cells, cell_index)

    print(f"Completed corrected QuESo-reference study for {framework_names} at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()