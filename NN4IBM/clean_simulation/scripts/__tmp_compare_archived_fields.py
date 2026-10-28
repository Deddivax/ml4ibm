import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np


data_dir = Path(
    "clean_simulation/outputs/studies/sphere/corrected/planar_cap_plane_hard_constraints"
)
pairs = (("naive", "weighted"), ("naive", "queso"), ("naive", "nn"), ("queso", "nn"))


def load_field(framework, resolution):
    path = data_dir / f"hanging_sphere_{framework}_displacement_{resolution}cubed.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["u_nodes"]


for resolution in (8, 256):
    fields = {framework: load_field(framework, resolution) for framework in ("naive", "weighted", "queso", "nn")}
    reference_norm = np.linalg.norm(fields["queso"])
    print(f"N={resolution}")
    for left, right in pairs:
        relative = np.linalg.norm(fields[left] - fields[right]) / reference_norm
        print(f"  {left:8s} vs {right:8s}: {relative:.6e}")
