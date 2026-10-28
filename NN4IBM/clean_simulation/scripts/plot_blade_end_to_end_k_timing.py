"""Plot cache-cold Blade STL-to-K timings using retained displacement errors."""

from __future__ import annotations

import os
from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter


ROOT = Path(__file__).resolve().parents[2]
TIMING_VARIANT_LABELS = {
    "naive_fnb": "Naive full-cell Gauss (FNB CUDA; K CPU)",
    "weighted_fnb": "Volume-fraction weighted Gauss (FNB + ray parity CUDA; K CPU)",
    "nn_fnb": "NN cut-cell K (FNB + CSC SDF + NN CUDA; K CPU)",
    "queso_p2": "QuESo quadrature (basis 2; CPU)",
}
FRAMEWORK_COLORS = {
    "naive_fnb": "#1f77b4",
    "nn_fnb": "#ff7f0e",
    "queso_p2": "#2ca02c",
    "weighted_fnb": "#d62728",
}
ERROR_FRAMEWORKS = {
    "naive_fnb": "naive",
    "weighted_fnb": "weighted",
    "nn_fnb": "nn",
    "queso_p2": "queso",
}
ACCURACY_MARKERS = {
    "naive_fnb": "o",
    "weighted_fnb": "s",
    "nn_fnb": "^",
    "queso_p2": "D",
}


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def main() -> None:
    timing_path = Path(os.environ.get(
        "BLADE_E2E_K_TIMING_CSV",
        ROOT / "clean_simulation" / "outputs" / "aws_blade_end_to_end_k_timing_v2" / "blade_end_to_end_k_timing.csv",
    )).expanduser().resolve()
    error_path = Path(os.environ.get(
        "BLADE_RETAINED_V3_ERROR_CSV",
        ROOT / "clean_simulation" / "notebooks" / "aws_blade_convergence_v3" / "blade_displacement_l2_vs_queso_512.csv",
    )).expanduser().resolve()
    output_dir = Path(os.environ.get(
        "BLADE_E2E_K_TIMING_PLOT_DIR",
        timing_path.parent,
    )).expanduser().resolve()
    if not timing_path.is_file():
        raise FileNotFoundError(timing_path)
    if not error_path.is_file():
        raise FileNotFoundError(error_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    timings = pd.read_csv(timing_path)
    errors = pd.read_csv(error_path)
    require_columns(timings, {"variant", "resolution", "end_to_end_k_seconds"}, timing_path)
    require_columns(errors, {"framework", "resolution", "relative_l2_error"}, error_path)
    if errors.duplicated(["framework", "resolution"]).any():
        raise ValueError("Error CSV has duplicate framework/resolution rows.")
    timings = timings.copy()
    timings["variant"] = timings["variant"].astype(str)
    timings["resolution"] = timings["resolution"].astype(int)
    timings["label"] = timings["variant"].map(TIMING_VARIANT_LABELS)
    timings = timings[timings["variant"].isin(TIMING_VARIANT_LABELS)].copy()
    if timings.duplicated(["variant", "resolution"]).any():
        raise ValueError("Timing CSV has duplicate variant/resolution rows.")

    figure, axis = plt.subplots(figsize=(9.4, 6.2), constrained_layout=True)
    for variant, curve in timings.groupby("variant", sort=False):
        curve = curve.sort_values("resolution")
        axis.loglog(
            curve["resolution"], curve["end_to_end_k_seconds"], "o-",
            color=FRAMEWORK_COLORS[variant], label=TIMING_VARIANT_LABELS[variant],
        )
    axis.set(
        xlabel="Grid resolution N",
        ylabel="STL-to-K time (s)",
        title="Blade end-to-end K construction time",
    )
    axis.xaxis.set_major_locator(FixedLocator([16, 32, 64, 128]))
    axis.xaxis.set_major_formatter(FixedFormatter([r"$2^4$", r"$2^5$", r"$2^6$", r"$2^7$"]))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    time_resolution_path = output_dir / "blade_end_to_end_k_time_vs_resolution.png"
    figure.savefig(time_resolution_path, dpi=180)
    plt.close(figure)

    accuracy_variants = pd.DataFrame({
        "framework": list(ERROR_FRAMEWORKS.values()),
        "variant": list(ERROR_FRAMEWORKS.keys()),
    })
    accuracy_by_resolution = errors.merge(
        accuracy_variants,
        on="framework",
        how="inner",
        validate="many_to_one",
    )
    accuracy_order_offsets = {
        "naive_fnb": (-8, 10),
        "weighted_fnb": (-8, 10),
        "nn_fnb": (-8, 12),
        "queso_p2": (-8, -10),
    }
    figure, axis = plt.subplots(figsize=(9.4, 6.2), constrained_layout=True)
    for variant, curve in accuracy_by_resolution.groupby("variant", sort=False):
        curve = curve.sort_values("resolution")
        axis.loglog(
            curve["resolution"], curve["relative_l2_error"],
            linestyle="-", marker=ACCURACY_MARKERS[variant], markersize=7,
            color=FRAMEWORK_COLORS[variant], label=TIMING_VARIANT_LABELS[variant],
        )
        if len(curve) >= 2:
            previous = curve.iloc[-2]
            final = curve.iloc[-1]
            observed_order = log(previous["relative_l2_error"] / final["relative_l2_error"]) / log(
                final["resolution"] / previous["resolution"]
            )
            axis.annotate(
                fr"$p={observed_order:.2f}$",
                (final["resolution"], final["relative_l2_error"]),
                xytext=accuracy_order_offsets[variant], textcoords="offset points",
                ha="right", va="center", fontsize=9, color=FRAMEWORK_COLORS[variant],
            )
    axis.set(
        xlabel="Grid resolution N",
        ylabel="Relative L2 error vs QuESo 512",
        title="Blade displacement accuracy versus resolution",
    )
    accuracy_resolutions = sorted(accuracy_by_resolution["resolution"].unique())
    axis.xaxis.set_major_locator(FixedLocator(accuracy_resolutions))
    axis.xaxis.set_major_formatter(FixedFormatter([fr"$2^{{{int(value).bit_length() - 1}}}$" for value in accuracy_resolutions]))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    accuracy_resolution_path = output_dir / "blade_displacement_l2_vs_resolution.png"
    figure.savefig(accuracy_resolution_path, dpi=180)
    plt.close(figure)

    accuracy_timings = timings[timings["variant"].isin(ERROR_FRAMEWORKS)].copy()
    accuracy_timings["framework"] = accuracy_timings["variant"].map(ERROR_FRAMEWORKS)
    joined = accuracy_timings.merge(
        errors[["framework", "resolution", "relative_l2_error"]],
        on=["framework", "resolution"],
        how="left",
        validate="one_to_one",
    )
    if joined["relative_l2_error"].isna().any():
        missing = joined.loc[joined["relative_l2_error"].isna(), ["variant", "resolution"]]
        raise ValueError(f"Retained V3 error values are missing for:\n{missing.to_string(index=False)}")
    joined["error_source"] = error_path.name
    joined["timing_scope"] = "Cache-cold STL load through K representation; excludes operator construction and CG."
    joined["accuracy_note"] = (
        "L2 errors versus the QuESo 512 reference are paired by method and resolution."
    )
    joined = joined.sort_values(["variant", "resolution"]).reset_index(drop=True)
    joined_path = output_dir / "blade_end_to_end_k_time_vs_retained_v3_l2_error.csv"
    joined.to_csv(joined_path, index=False)

    figure, axis = plt.subplots(figsize=(9.4, 6.2), constrained_layout=True)
    for variant, curve in joined.groupby("variant", sort=False):
        curve = curve.sort_values("relative_l2_error", ascending=False)
        axis.loglog(
            curve["relative_l2_error"], curve["end_to_end_k_seconds"],
            linestyle="None", marker=ACCURACY_MARKERS[variant], markersize=8,
            color=FRAMEWORK_COLORS[variant], label=TIMING_VARIANT_LABELS[variant],
        )
        for point in curve.itertuples(index=False):
            axis.annotate(
                f"{point.resolution}",
                (point.relative_l2_error, point.end_to_end_k_seconds),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
            )
    axis.invert_xaxis()
    axis.set(
        xlabel="Relative L2 error vs QuESo 512",
        ylabel="STL-to-K time (s)",
        title="Blade K construction time versus displacement accuracy",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    time_accuracy_path = output_dir / "blade_end_to_end_k_time_vs_retained_v3_l2_error.png"
    figure.savefig(time_accuracy_path, dpi=180)
    plt.close(figure)

    print(f"Saved: {time_resolution_path}")
    print(f"Saved: {accuracy_resolution_path}")
    print(f"Saved: {joined_path}")
    print(f"Saved: {time_accuracy_path}")


if __name__ == "__main__":
    main()