from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(
    os.environ.get(
        "HANGING_SPHERE_OUTPUT_DIR",
        ROOT / "clean_simulation" / "outputs" / "studies" / "sphere" / "corrected",
    )
).expanduser().resolve() / "planar_cap_plane_hard_constraints"
REFERENCE_RESOLUTION = 512
NODEWISE_REFERENCE_FLOOR_FRACTION = float(os.environ.get("HANGING_SPHERE_NODEWISE_REFERENCE_FLOOR_FRACTION", "1.0e-4"))
RESOLUTIONS = (8, 16, 32, 64, 128, 256, 512)
FRAMEWORKS = {
    "naive": "Naive full-cell Gauss",
    "weighted": "Geometric-volume-fraction weighted Gauss",
    "nn": "NN-cut-cell hybrid",
}
COLORS = {"naive": "#b94732", "weighted": "#c98b1b", "nn": "#315c99"}


def solution_path(framework: str, resolution: int) -> Path:
    return OUTPUT_DIR / f"hanging_sphere_{framework}_displacement_{resolution}cubed.npz"


def reference_node_mask(active_cells: np.ndarray, resolution: int) -> np.ndarray:
    scale = REFERENCE_RESOLUTION // resolution
    cells = np.zeros((resolution,) * 3, dtype=bool)
    cells.reshape(-1)[active_cells] = True
    fine_cells = np.repeat(np.repeat(np.repeat(cells, scale, axis=0), scale, axis=1), scale, axis=2)
    nodes = np.zeros((REFERENCE_RESOLUTION + 1,) * 3, dtype=bool)
    for offset_x, offset_y, offset_z in (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ):
        nodes[
            offset_x:offset_x + REFERENCE_RESOLUTION,
            offset_y:offset_y + REFERENCE_RESOLUTION,
            offset_z:offset_z + REFERENCE_RESOLUTION,
        ] |= fine_cells
    return nodes.reshape(-1)


def interpolate_to_reference(solution: np.ndarray, resolution: int, device: str) -> np.ndarray:
    field = torch.as_tensor(solution, dtype=torch.float64, device=device).reshape((resolution + 1,) * 3 + (3,))
    interpolated = torch.nn.functional.interpolate(
        field.permute(3, 0, 1, 2).unsqueeze(0),
        size=(REFERENCE_RESOLUTION + 1,) * 3,
        mode="trilinear",
        align_corners=True,
    )
    return interpolated.squeeze(0).permute(1, 2, 3, 0).reshape(-1, 3).cpu().numpy()


def relative_l2_error(values: np.ndarray, reference_values: np.ndarray, reference_norm: float) -> float:
    return float(np.linalg.norm(values - reference_values) / reference_norm)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(solution_path("queso", REFERENCE_RESOLUTION), allow_pickle=True) as reference_file:
        reference_u = reference_file["u_nodes"]
        reference_nodes = reference_node_mask(reference_file["active_cells"], REFERENCE_RESOLUTION)

    rows = []
    reference_values = reference_u[reference_nodes]
    reference_norm = max(np.linalg.norm(reference_values), np.finfo(np.float64).tiny)
    reference_magnitudes = np.linalg.norm(reference_values, axis=1)
    reference_floor = max(
        float(reference_magnitudes.max()) * NODEWISE_REFERENCE_FLOOR_FRACTION,
        np.finfo(np.float64).tiny,
    )
    valid_reference = reference_magnitudes > reference_floor

    for framework, label in FRAMEWORKS.items():
        for resolution in RESOLUTIONS:
            with np.load(solution_path(framework, resolution), allow_pickle=True) as candidate_file:
                interpolated = interpolate_to_reference(candidate_file["u_nodes"], resolution, device)
                scale = REFERENCE_RESOLUTION // resolution
                aligned_values = interpolated.reshape((REFERENCE_RESOLUTION + 1,) * 3 + (3,))[::scale, ::scale, ::scale]
                aligned_node_max_abs_error = float(np.max(np.abs(
                    aligned_values - candidate_file["u_nodes"].reshape((resolution + 1,) * 3 + (3,))
                )))
                if aligned_node_max_abs_error > 1.0e-11:
                    raise RuntimeError(
                        f"{framework} {resolution}^3 interpolation does not preserve aligned nodal values: "
                        f"{aligned_node_max_abs_error:.3e}"
                    )
                candidate_nodes = reference_node_mask(candidate_file["active_cells"], resolution)
                embedded_values = interpolated[reference_nodes]
                embedded_difference = embedded_values - reference_values
                embedded_error = relative_l2_error(embedded_values, reference_values, reference_norm)
                embedded_rms_nodewise_error = np.sqrt(np.mean((np.linalg.norm(embedded_difference[valid_reference], axis=1) / reference_magnitudes[valid_reference]) ** 2))

                # This is intentionally a separate support-loss diagnostic. It must
                # not replace the trilinearly embedded field used for convergence.
                zero_filled_values = embedded_values.copy()
                zero_filled_values[~candidate_nodes[reference_nodes]] = 0.0
                zero_filled_error = relative_l2_error(zero_filled_values, reference_values, reference_norm)
                support_coverage = float((candidate_nodes & reference_nodes).sum() / reference_nodes.sum())

                if resolution == REFERENCE_RESOLUTION:
                    identity_error = float(np.max(np.abs(embedded_values - candidate_file["u_nodes"][reference_nodes])))
                    if identity_error > 1.0e-11:
                        raise RuntimeError(
                            f"{framework} {resolution}^3 interpolation is not identity-preserving: {identity_error:.3e}"
                        )
                rows.append({
                    "framework": framework,
                    "framework_label": label,
                    "resolution": resolution,
                    "reference_resolution": REFERENCE_RESOLUTION,
                    "reference_nodes": int(reference_nodes.sum()),
                    "nodewise_reference_floor_fraction": NODEWISE_REFERENCE_FLOOR_FRACTION,
                    "nodewise_relative_reference_nodes": int(valid_reference.sum()),
                    "candidate_nodes_on_reference_support": int((candidate_nodes & reference_nodes).sum()),
                    "support_coverage": support_coverage,
                    "aligned_node_max_abs_error": aligned_node_max_abs_error,
                    "embedded_relative_l2_error": embedded_error,
                    "embedded_rms_nodewise_relative_error": embedded_rms_nodewise_error,
                    "zero_filled_full_reference_relative_l2_error": zero_filled_error,
                })
                print(
                    f"{framework:8s} {resolution:3d}^3: RMS nodewise relative displacement error={embedded_rms_nodewise_error:.6e}; "
                    f"zero-filled support error={zero_filled_error:.6e}; coverage={support_coverage:.6%}"
                )
                del interpolated, embedded_difference, zero_filled_values
                if device == "cuda":
                    torch.cuda.empty_cache()

    results = pd.DataFrame(rows).sort_values(["framework", "resolution"])
    results_path = OUTPUT_DIR / "three_framework_plane_cap_queso_512_full_reference_support.csv"
    figure_path = OUTPUT_DIR / "three_framework_plane_cap_queso_512_full_reference_support.png"
    results.to_csv(results_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for framework, label in FRAMEWORKS.items():
        curve = results[results["framework"] == framework]
        axes[0].semilogy(curve["resolution"], curve["embedded_rms_nodewise_relative_error"], marker="o", linewidth=2, color=COLORS[framework], label=label)
        axes[1].semilogy(curve["resolution"], curve["zero_filled_full_reference_relative_l2_error"], marker="o", linewidth=2, color=COLORS[framework], label=label)
    axes[0].set(xlabel="Grid resolution N", ylabel="RMS node-to-node relative displacement error")
    axes[1].set(xlabel="Grid resolution N", ylabel="Relative L2 after zeroing missing candidate support")
    for axis in axes:
        axis.set_xticks(RESOLUTIONS)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    print(f"Saved {results_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()