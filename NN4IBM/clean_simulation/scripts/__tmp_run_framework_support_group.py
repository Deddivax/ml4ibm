import json
import os
import shutil
import sys
from pathlib import Path


resolutions = sys.argv[1]
device = sys.argv[2]
root = Path.cwd()
notebook_path = Path("/tmp/05_local_stiffness_convergence_framework_support.ipynb")
output_dir = root / "clean_simulation" / "outputs" / "studies" / "sphere" / "framework_support"
seed_dir = output_dir / "queso_matrix_free_128cubed"
seed_dir.mkdir(parents=True, exist_ok=True)
seed_path = seed_dir / "study_geometry_surface.stl"
if not seed_path.exists():
    shutil.copy2(root / "clean_simulation" / "geometries" / "hanging_sphere_surface.stl", seed_path)

os.environ.update({
    "HANGING_SPHERE_OUTPUT_DIR": str(output_dir),
    "HANGING_SPHERE_RUN_SOLVES": "1",
    "HANGING_SPHERE_RESOLUTIONS": resolutions,
    "HANGING_SPHERE_FRAMEWORKS": "naive,weighted,queso,nn",
    "HANGING_SPHERE_DEVICE": device,
    "HANGING_SPHERE_DTYPE": "float64",
    "HANGING_SPHERE_CPU_THREADS": "16",
    "HANGING_SPHERE_CG_MAX_ITER": "12000",
})

notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
for index in (1, 2, 3, 4, 6):
    source = notebook["cells"][index]["source"]
    exec(compile("\n".join(source), f"{notebook_path.name}: cell {index + 1}", "exec"), namespace)
