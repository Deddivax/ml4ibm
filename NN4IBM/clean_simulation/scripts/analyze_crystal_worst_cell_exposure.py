from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "clean_simulation" / "outputs" / "local_stiffness_convergence"
ERROR_FILE = OUTPUT_DIR / "medium_transformer_0p245952_matched_mc_cell_errors_128cubed.csv"
RESOLUTION = 128
ERROR_COLUMN = "relative_frobenius_error_vs_matched_queso"


def cell_node_ids(cell_ids: np.ndarray) -> np.ndarray:
    side = RESOLUTION + 1
    i = cell_ids // (RESOLUTION * RESOLUTION)
    j = (cell_ids // RESOLUTION) % RESOLUTION
    k = cell_ids % RESOLUTION
    base = (i * side + j) * side + k
    offsets = np.array((
        0, side * side, side * side + side, side,
        1, side * side + 1, side * side + side + 1, side + 1,
    ))
    return base[:, None] + offsets


def main() -> None:
    errors = pd.read_csv(ERROR_FILE)
    errors["relative_error_percent"] = 100.0 * errors[ERROR_COLUMN]
    errors["error_percentile"] = errors[ERROR_COLUMN].rank(pct=True, method="max")
    worst_cells = errors.nlargest(20, ERROR_COLUMN).copy()
    correlations = errors[[
        "inside_corner_count",
        "minimum_absolute_lattice_sdf",
        "lattice_sdf_span",
        "mean_lattice_sdf",
        "center_x",
        "center_y",
        "center_z",
        "matched_queso_frobenius_norm",
        ERROR_COLUMN,
    ]].corr(method="spearman")[ERROR_COLUMN].drop(ERROR_COLUMN).sort_values(key=np.abs, ascending=False)
    topology = errors.groupby("inside_corner_count").agg(
        cells=("cell_id", "size"),
        median_error_percent=("relative_error_percent", "median"),
        p95_error_percent=("relative_error_percent", lambda values: values.quantile(0.95)),
        top_5_percent_share=("error_percentile", lambda values: (values >= 0.95).mean()),
    ).reset_index()

    nodes = cell_node_ids(errors["cell_id"].to_numpy(dtype=np.int64))
    node_count = (RESOLUTION + 1) ** 3
    exposure_count = np.zeros(node_count, dtype=np.int64)
    maximum_incident_error = np.zeros(node_count, dtype=np.float64)
    np.add.at(exposure_count, nodes.ravel(), 1)
    np.maximum.at(maximum_incident_error, nodes.ravel(), np.repeat(errors["relative_error_percent"].to_numpy(), 8))
    top_5_cutoff = errors["relative_error_percent"].quantile(0.95)
    top_cells = errors["relative_error_percent"].to_numpy() >= top_5_cutoff
    top_incidence = np.zeros(node_count, dtype=np.int64)
    np.add.at(top_incidence, nodes[top_cells].ravel(), 1)
    exposed_nodes = exposure_count > 0
    exposure = pd.DataFrame({
        "node_id": np.flatnonzero(exposed_nodes),
        "incident_trimmed_cells": exposure_count[exposed_nodes],
        "incident_top_5_percent_cells": top_incidence[exposed_nodes],
        "maximum_incident_cell_error_percent": maximum_incident_error[exposed_nodes],
    })
    exposure["incident_to_top_5_percent_cell"] = exposure["incident_top_5_percent_cells"] > 0
    exposure.to_csv(OUTPUT_DIR / "crystal_cell_to_node_exposure_128cubed.csv", index=False)
    worst_cells.to_csv(OUTPUT_DIR / "crystal_worst_20_cells_128cubed.csv", index=False)
    topology.to_csv(OUTPUT_DIR / "crystal_error_by_inside_corner_count_128cubed.csv", index=False)
    correlations.rename("spearman_rank_correlation").to_csv(OUTPUT_DIR / "crystal_worst_cell_spearman_128cubed.csv")

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    axes[0].barh(correlations.index, correlations.values, color="#287271")
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Spearman rank correlation", title="What predicts crystal cell error?")
    axes[1].boxplot([errors.loc[errors.inside_corner_count == count, "relative_error_percent"] for count in topology.inside_corner_count], tick_labels=topology.inside_corner_count)
    axes[1].set(xlabel="Inside lattice corners", ylabel="Relative local stiffness error (%)", title="Error by cut topology")
    axes[2].hist(exposure["maximum_incident_cell_error_percent"], bins=np.logspace(-3, 3, 70), color="#e76f51", alpha=0.9)
    axes[2].axvline(top_5_cutoff, color="#264653", linestyle="--", label=f"Cell p95 = {top_5_cutoff:.3f}%")
    axes[2].set_xscale("log")
    axes[2].set(xlabel="Worst incident cell error (%)", ylabel="Crystal grid-node count", title="Cell-to-node error exposure")
    axes[2].legend(frameon=False)
    figure.savefig(OUTPUT_DIR / "crystal_worst_cell_exposure_128cubed.png", dpi=180)

    lines = [
        "# Crystal Worst-Cell Error Study",
        "",
        f"Matched source-lattice to marching-cubes to QuESo comparison, {len(errors):,} trimmed cells at {RESOLUTION} cubed.",
        "",
        "## Error Tail",
        f"- Median: {errors.relative_error_percent.median():.4f}%",
        f"- p95: {errors.relative_error_percent.quantile(0.95):.4f}%",
        f"- p99: {errors.relative_error_percent.quantile(0.99):.4f}%",
        f"- Maximum: {errors.relative_error_percent.max():.4f}%",
        "",
        "## Strongest Associations",
    ]
    lines.extend(f"- {name}: Spearman rho = {value:.4f}" for name, value in correlations.items())
    lines.extend([
        "",
        "## Cell-to-Node Exposure",
        f"- Crystal nodes incident to at least one trimmed cell: {len(exposure):,}",
        f"- Nodes incident to at least one top-5% error cell: {exposure.incident_to_top_5_percent_cell.sum():,} ({exposure.incident_to_top_5_percent_cell.mean():.2%})",
        "- This is a structural exposure measure. It is not a displacement-error correlation because no crystal displacement field is available in the benchmark artifacts.",
        "",
        "## Interpretation",
        "The extreme tail is concentrated in low-stiffness, low-inside-corner cuts, rather than in a particular crystal location. Such cells have small reference matrices, so a comparable absolute prediction discrepancy becomes a much larger relative Frobenius error. A same-geometry matrix-free displacement solve is required before classifying independent bad displacement nodes and measuring their association with these cells.",
    ])
    (OUTPUT_DIR / "crystal_worst_cell_error_study.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
