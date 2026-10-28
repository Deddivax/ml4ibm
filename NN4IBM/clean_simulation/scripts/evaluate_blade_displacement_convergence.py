"""Evaluate Blade displacement convergence against a QuESo reference checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional


FRAMEWORKS = {
    "naive": "Naive full-cell Gauss",
    "weighted": "Volume-fraction weighted Gauss",
    "queso": "QuESo quadrature",
    "nn": "NN-cut-cell hybrid",
    "nn_27": "NN-27 cut-cell hybrid",
    "nn_64": "NN-64 cut-cell hybrid",
}


def checkpoint_path(root: Path, framework: str, resolution: int) -> Path:
    return root / "checkpoints" / f"blade_{framework}_displacement_{resolution}cubed.npz"


def load_checkpoint(root: Path, framework: str, resolution: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = checkpoint_path(root, framework, resolution)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        field = np.asarray(archive["u_nodes"], dtype=np.float64)
        active_cells = np.unique(np.asarray(archive["active_cells"], dtype=np.int64))
        runtime = float(archive["runtime_seconds"])
    expected_shape = ((resolution + 1) ** 3, 3)
    if field.shape != expected_shape:
        raise ValueError(f"{path.name}: expected {expected_shape}, received {field.shape}.")
    return field.reshape((resolution + 1,) * 3 + (3,)), active_cells, runtime


def active_node_mask(active_cells: np.ndarray, resolution: int) -> np.ndarray:
    cells = np.zeros((resolution, resolution, resolution), dtype=bool)
    cells.reshape(-1)[active_cells] = True
    nodes = np.zeros((resolution + 1, resolution + 1, resolution + 1), dtype=bool)
    for offset_x, offset_y, offset_z in (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ):
        nodes[
            offset_x:offset_x + resolution,
            offset_y:offset_y + resolution,
            offset_z:offset_z + resolution,
        ] |= cells
    return nodes


def interpolate_to_reference(field: np.ndarray, reference_resolution: int, device: str) -> np.ndarray:
    resolution = field.shape[0] - 1
    if resolution == reference_resolution:
        return field
    tensor = torch.from_numpy(field).to(device=device).permute(3, 0, 1, 2).unsqueeze(0)
    interpolated = functional.interpolate(
        tensor,
        size=(reference_resolution + 1,) * 3,
        mode="trilinear",
        align_corners=True,
    )
    return interpolated.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()


def main() -> None:
    root = Path(os.environ["BLADE_CONVERGENCE_OUTPUT_DIR"]).expanduser().resolve()
    reference_root = Path(os.environ.get("BLADE_REFERENCE_OUTPUT_DIR", root)).expanduser().resolve()
    coarse_root = Path(os.environ.get("BLADE_COARSE_OUTPUT_DIR", root)).expanduser().resolve()
    extended_root = Path(os.environ.get("BLADE_EXTENDED_OUTPUT_DIR", root)).expanduser().resolve()
    extended_min_resolution = int(os.environ.get("BLADE_EXTENDED_MIN_RESOLUTION", "256"))
    reference_resolution = int(os.environ.get("BLADE_REFERENCE_RESOLUTION", "256"))
    resolutions = tuple(int(value) for value in os.environ.get("BLADE_EVALUATION_RESOLUTIONS", "16,32,64,128").split(","))
    device = os.environ.get("BLADE_EVALUATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BLADE_EVALUATION_DEVICE=cuda was requested but CUDA is unavailable.")

    reference, reference_cells, _ = load_checkpoint(reference_root, "queso", reference_resolution)
    reference_mask = active_node_mask(reference_cells, reference_resolution)
    reference_values = reference[reference_mask]
    reference_norm = float(np.linalg.norm(reference_values))
    if reference_norm <= np.finfo(np.float64).tiny:
        raise RuntimeError("The QuESo reference displacement has zero norm on its active-node support.")

    bounds = np.asarray(np.load(checkpoint_path(reference_root, "queso", reference_resolution), allow_pickle=False)["mesh_bounds"], dtype=np.float64)
    domain_length = float(np.max(bounds[1] - bounds[0]) * 1.2)
    rows = []
    for framework, label in FRAMEWORKS.items():
        for resolution in resolutions:
            candidate_root = extended_root if resolution >= extended_min_resolution else coarse_root
            field, _, runtime = load_checkpoint(candidate_root, framework, resolution)
            field_on_reference = interpolate_to_reference(field, reference_resolution, device)
            values = field_on_reference[reference_mask]
            difference = values - reference_values
            component_error = np.linalg.norm(difference, axis=0) / np.linalg.norm(reference_values, axis=0)
            relative_error = float(np.linalg.norm(difference) / reference_norm)
            aligned = field_on_reference[::reference_resolution // resolution, ::reference_resolution // resolution, ::reference_resolution // resolution]
            interpolation_error = float(np.max(np.abs(aligned - field)))
            if interpolation_error > 1.0e-10:
                raise RuntimeError(f"{framework} {resolution}^3 interpolation alignment error: {interpolation_error:.3e}")
            rows.append({
                "framework": framework,
                "framework_label": label,
                "resolution": resolution,
                "h": domain_length / resolution,
                "reference_resolution": reference_resolution,
                "reference_active_nodes": int(reference_mask.sum()),
                "relative_l2_error": relative_error,
                "relative_l2_error_x": float(component_error[0]),
                "relative_l2_error_y": float(component_error[1]),
                "relative_l2_error_z": float(component_error[2]),
                "interpolation_alignment_max_abs_error": interpolation_error,
                "solve_seconds": runtime,
            })
            print(f"{framework:8s} {resolution:3d}^3: relative L2 = {relative_error:.6e}", flush=True)
            del field, field_on_reference, values, difference
            if device == "cuda":
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows).sort_values(["framework", "resolution"]).reset_index(drop=True)
    results["local_order"] = np.nan
    fit_rows = []
    for framework, curve in results.groupby("framework", sort=False):
        curve = curve.sort_values("resolution")
        errors = curve["relative_l2_error"].to_numpy()
        mesh_sizes = curve["h"].to_numpy()
        results.loc[curve.index[1:], "local_order"] = np.log(errors[:-1] / errors[1:]) / np.log(mesh_sizes[:-1] / mesh_sizes[1:])
        slope, intercept = np.polyfit(np.log(mesh_sizes), np.log(errors), 1)
        prediction = slope * np.log(mesh_sizes) + intercept
        residual = np.sum((np.log(errors) - prediction) ** 2)
        total = np.sum((np.log(errors) - np.log(errors).mean()) ** 2)
        fit_rows.append({"framework": framework, "framework_label": curve.framework_label.iloc[0], "global_order": slope, "r_squared": 1.0 - residual / total})

    csv_path = root / f"blade_displacement_l2_vs_queso_{reference_resolution}.csv"
    fit_path = root / f"blade_displacement_l2_vs_queso_{reference_resolution}_global_fits.csv"
    png_path = root / f"blade_displacement_l2_vs_queso_{reference_resolution}.png"
    results.to_csv(csv_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)

    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for framework, label in FRAMEWORKS.items():
        curve = results[results["framework"] == framework].sort_values("h", ascending=False)
        start_resolution = int(curve.iloc[-2]["resolution"])
        end_resolution = int(curve.iloc[-1]["resolution"])
        final_order = curve.iloc[-1]["local_order"]
        axis.loglog(
            curve["h"],
            curve["relative_l2_error"],
            marker="o",
            linewidth=2,
            label=f"{label} ($p_{{{start_resolution}\\to{end_resolution}}}={final_order:.2f}$)",
        )
    axis.invert_xaxis()
    axis.set(xlabel="Cubic grid spacing h", ylabel=f"Relative displacement L2 error on QuESo-{reference_resolution} active nodes")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(png_path, dpi=180, bbox_inches="tight")

    print("\n" + results[["framework", "resolution", "relative_l2_error", "local_order"]].to_string(index=False))
    print("\n" + pd.DataFrame(fit_rows).to_string(index=False))
    print(f"Saved {csv_path}")
    print(f"Saved {fit_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
