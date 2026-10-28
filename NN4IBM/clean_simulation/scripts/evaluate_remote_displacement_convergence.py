"""Compute fixed-reference-support physical-volume-weighted convergence from archived checkpoints."""

import gc
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt


REFERENCE_RESOLUTION = int(os.environ.get("REFERENCE_RESOLUTION", "512"))
DOMAIN_LENGTH = 18.0
SPHERE_RADIUS = 7.5
EVALUATION_RESOLUTIONS = tuple(
    int(value) for value in os.environ.get("EVALUATION_RESOLUTIONS", "8,16,32,64,128,256").split(",")
)
FIT_RESOLUTIONS = tuple(
    int(value) for value in os.environ.get("FIT_RESOLUTIONS", "16,32,64,128,256").split(",")
)
FRAMEWORKS = {
    "naive": "Naive full-cell Gauss",
    "weighted": "Geometric-volume-fraction weighted Gauss",
    "queso": "QuESo quadrature",
    "nn": "NN-cut-cell hybrid",
}
SUPPORT_MODE = os.environ.get("HANGING_SPHERE_SUPPORT_MODE", "planar_cap").strip().lower()
SUPPORT_DIRECTORIES = {
    "planar_cap": "planar_cap_plane_hard_constraints",
    "smooth_foundation": "smooth_foundation",
}
if SUPPORT_MODE not in SUPPORT_DIRECTORIES:
    raise ValueError(f"Unknown HANGING_SPHERE_SUPPORT_MODE={SUPPORT_MODE!r}.")


def load_checkpoint(data_dir, framework, resolution):
    path = data_dir / f"hanging_sphere_{framework}_displacement_{resolution}cubed.npz"
    if not path.is_file() and SUPPORT_MODE == "smooth_foundation":
        path = data_dir / f"hanging_sphere_{framework}_smooth_foundation_{resolution}cubed.npz"
    with np.load(path, allow_pickle=False) as archive:
        field = archive["u_nodes"]
        active_cells = np.unique(archive["active_cells"].astype(np.int64, copy=False))
    expected_shape = ((resolution + 1) ** 3, 3)
    if field.shape != expected_shape:
        raise ValueError(f"{path}: expected {expected_shape}, got {field.shape}")
    return field.reshape((resolution + 1,) * 3 + (3,)), active_cells


def active_node_mask(active_cells, resolution):
    node_side = resolution + 1
    cells = np.zeros((resolution,) * 3, dtype=bool)
    cells.reshape(-1)[active_cells] = True
    nodes = np.zeros((node_side,) * 3, dtype=bool)
    for offset in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)):
        x, y, z = offset
        nodes[x:x + resolution, y:y + resolution, z:z + resolution] |= cells
    return nodes.reshape(-1)

def physical_nodal_weights(root, active_cells, resolution):
    node_side = resolution + 1
    cache_dir = root / f"queso_matrix_free_{resolution}cubed"
    cut_points_path = cache_dir / "queso_cut_points.npy"
    cut_stiffness_path = cache_dir / "queso_cut_stiffness.pt"
    if cut_points_path.is_file() and cut_stiffness_path.is_file():
        points = np.load(cut_points_path)
        cut_cells = np.asarray(
            torch.load(cut_stiffness_path, map_location="cpu", weights_only=True)["indices"],
            dtype=np.int64,
        )
    else:
        points = np.load(cache_dir / "queso_all_points.npy")
        cut_cells = np.asarray(active_cells, dtype=np.int64)
    coordinates, point_weights = points[:, :3], points[:, 3].copy()
    if np.all((coordinates >= 0.0) & (coordinates <= 1.0)):
        coordinates = -9.0 + DOMAIN_LENGTH * coordinates
        point_weights *= DOMAIN_LENGTH ** 3
    spacing = DOMAIN_LENGTH / resolution
    ijk = np.clip(np.floor((coordinates + 9.0) / spacing).astype(np.int64), 0, resolution - 1)
    point_cells = (ijk[:, 0] * resolution + ijk[:, 1]) * resolution + ijk[:, 2]
    cut_volumes = np.zeros(resolution ** 3, dtype=np.float32)
    np.add.at(cut_volumes, point_cells, point_weights.astype(np.float32))
    weights = np.zeros(node_side ** 3, dtype=np.float32)
    offsets = np.asarray((0, node_side ** 2, node_side ** 2 + node_side, node_side, 1,
                          node_side ** 2 + 1, node_side ** 2 + node_side + 1, node_side + 1))
    for chunk in np.array_split(active_cells, max(1, len(active_cells) // 1_000_000 + 1)):
        i = chunk // resolution ** 2
        j = (chunk // resolution) % resolution
        k = chunk % resolution
        base = (i * node_side + j) * node_side + k
        locations = np.searchsorted(cut_cells, chunk)
        is_cut = locations < len(cut_cells)
        is_cut[is_cut] &= cut_cells[locations[is_cut]] == chunk[is_cut]
        volumes = np.full(len(chunk), spacing ** 3, dtype=np.float32)
        volumes[is_cut] = cut_volumes[chunk[is_cut]]
        for offset in offsets:
            np.add.at(weights, base + offset, volumes / 8.0)
    return weights

def reference_on_grid(reference, resolution):
    if resolution == REFERENCE_RESOLUTION:
        return reference
    return functional.interpolate(
        torch.from_numpy(reference).permute(3, 0, 1, 2).unsqueeze(0),
        size=(resolution + 1,) * 3, mode="trilinear", align_corners=True,
    ).squeeze(0).permute(1, 2, 3, 0).numpy()


def field_on_reference_grid(field):
    if field.shape[:3] == (REFERENCE_RESOLUTION + 1,) * 3:
        return field
    return functional.interpolate(
        torch.from_numpy(field).permute(3, 0, 1, 2).unsqueeze(0),
        size=(REFERENCE_RESOLUTION + 1,) * 3, mode="trilinear", align_corners=True,
    ).squeeze(0).permute(1, 2, 3, 0).numpy()


def relative_errors(field, evaluation_mask, reference, weights):
    if field.shape != reference.shape:
        raise ValueError(f"Field and reference grids differ: {field.shape} versus {reference.shape}.")
    difference = field.reshape(-1, 3)[evaluation_mask].astype(np.float64) - reference.reshape(-1, 3)[evaluation_mask].astype(np.float64)
    reference_values = reference.reshape(-1, 3)[evaluation_mask].astype(np.float64)
    local_weights = weights[evaluation_mask].astype(np.float64)
    component = np.sqrt(np.sum(local_weights[:, None] * difference ** 2, axis=0) /
                        np.sum(local_weights[:, None] * reference_values ** 2, axis=0))
    vector = np.sqrt(np.sum(local_weights * np.sum(difference ** 2, axis=1)) /
                     np.sum(local_weights * np.sum(reference_values ** 2, axis=1)))
    return int(evaluation_mask.sum()), vector, component

def evaluate_fixed_reference_support(data_dir, reference, reference_volume):
    rows = []
    for framework, label in FRAMEWORKS.items():
        for resolution in EVALUATION_RESOLUTIONS:
            print(f"Evaluating fixed reference support: {framework} {resolution}^3", flush=True)
            field, active_cells = load_checkpoint(data_dir, framework, resolution)
            evaluation_mask = reference_mask
            count, vector, component = relative_errors(
                field_on_reference_grid(field), evaluation_mask, reference, reference_weights,
            )
            support_volume = reference_volume
            rows.append({
                "framework": framework, "framework_label": label, "reference_framework": "queso",
                "region": "fixed_queso_reference_support", "resolution": resolution, "h": DOMAIN_LENGTH / resolution,
                "comparison_nodes": count, "shared_support_volume": support_volume,
                "shared_support_volume_fraction": support_volume / reference_volume,
                "relative_l2_error": vector, "relative_l2_error_x": component[0],
                "relative_l2_error_y": component[1], "relative_l2_error_z": component[2],
                "relative_h1_seminorm_error": np.nan,
                "h1_status": "Unavailable: archived checkpoints contain displacement only.",
            })
            print(
                f"  active_nodes={count:,}; support_volume_fraction={support_volume / reference_volume:.6%}",
                flush=True,
            )
            del field, active_cells
            gc.collect()
    results = pd.DataFrame(rows).sort_values(["framework", "resolution"]).reset_index(drop=True)
    results["local_order"] = np.nan
    for _, curve in results.groupby("framework", sort=False):
        errors, mesh_sizes = curve.relative_l2_error.to_numpy(), curve.h.to_numpy()
        local = np.full(len(curve), np.nan)
        local[:-1] = np.log(errors[:-1] / errors[1:]) / np.log(mesh_sizes[:-1] / mesh_sizes[1:])
        results.loc[curve.index, "local_order"] = local
    return results


def main():
    root = Path(os.environ.get("HANGING_SPHERE_OUTPUT_ROOT", ".")).expanduser().resolve()
    data_dir = root / SUPPORT_DIRECTORIES[SUPPORT_MODE]
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing support archive directory: {data_dir}")
    reference, reference_active = load_checkpoint(data_dir, "queso", REFERENCE_RESOLUTION)
    global reference_mask, reference_weights
    reference_mask = active_node_mask(reference_active, REFERENCE_RESOLUTION)
    reference_weights = physical_nodal_weights(root, reference_active, REFERENCE_RESOLUTION)
    reference_mask &= reference_weights > 0.0
    reference_volume = float(reference_weights[reference_mask].sum())
    results = evaluate_fixed_reference_support(data_dir, reference, reference_volume)
    fits = []
    for framework, curve in results.groupby("framework", sort=False):
        curve = curve[curve["resolution"].isin(FIT_RESOLUTIONS)].sort_values("resolution")
        errors, mesh_sizes = curve.relative_l2_error.to_numpy(), curve.h.to_numpy()
        valid = errors > np.finfo(float).tiny
        slope, intercept = np.polyfit(np.log(mesh_sizes[valid]), np.log(errors[valid]), 1)
        prediction = slope * np.log(mesh_sizes[valid]) + intercept
        total = np.sum((np.log(errors[valid]) - np.log(errors[valid]).mean()) ** 2)
        local = np.full(len(curve), np.nan)
        local[:-1] = np.log(errors[:-1] / errors[1:]) / np.log(mesh_sizes[:-1] / mesh_sizes[1:])
        results.loc[curve.index, "local_order"] = local
        fits.append({"framework": framework, "framework_label": curve.framework_label.iloc[0], "global_order": slope,
                     "global_r_squared": 1.0 - np.sum((np.log(errors[valid]) - prediction) ** 2) / total,
                     "fit_points": int(valid.sum())})
    output = data_dir / f"four_framework_physical_volume_weighted_l2_vs_queso_{REFERENCE_RESOLUTION}.csv"
    results.to_csv(output, index=False)
    pd.DataFrame(fits).to_csv(output.with_name(output.stem + "_global_fits.csv"), index=False)
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for framework, label in FRAMEWORKS.items():
        curve = results[results["framework"] == framework].sort_values("h", ascending=False)
        local_order = curve.iloc[-2]["local_order"]
        order_label = "" if not np.isfinite(local_order) else f"; p(32->64)={local_order:.2f}"
        axis.loglog(
            curve["h"], curve["relative_l2_error"], marker="o", linewidth=2,
            label=f"{label}{order_label}",
        )
    queso_curve = results[results["framework"] == "queso"].sort_values("h", ascending=False)
    if len(queso_curve) >= 2:
        guide_h = queso_curve["h"].to_numpy()[-2:]
        guide_error = queso_curve["relative_l2_error"].to_numpy()[-2]
        axis.loglog(
            guide_h, guide_error * (guide_h / guide_h[0]) ** 2,
            "--", color="0.35", linewidth=1.5, label="second-order guide",
        )
    axis.invert_xaxis()
    axis.set(xlabel="Mesh size h = 18 / N (refinement to the right)", ylabel="Relative displacement L2 error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure_path = output.with_name(f"four_framework_fixed_reference_support_l2_vs_queso_{REFERENCE_RESOLUTION}.png")
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    print(results[["framework", "resolution", "comparison_nodes", "shared_support_volume_fraction", "relative_l2_error", "local_order"]].to_string(index=False))
    print(pd.DataFrame(fits).to_string(index=False))
    print(f"Saved {output}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()