"""Filter non-finite and corner/volume orientation-conflict dataset rows."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def filter_orientation_outliers(
    source_h5: Path,
    *,
    low_corner_threshold: float = 0.25,
    high_volume_threshold: float = 0.75,
) -> dict[str, object]:
    """Write a sibling cleaned HDF5 dataset and return its filtering summary."""
    with h5py.File(source_h5, "r") as source:
        group = source["integration_2_poly_1"]
        values = {
            "flat_index": group["flat_index"][...],
            "d_values": group["d_values"][...],
            "stiffness_ut": group["stiffness_ut"][...],
            "cut_volume_fraction": group["cut_volume_fraction"][...],
            "geometry_name": group["geometry_name"][...],
        }
        attributes = dict(group.attrs)

    distances = values["d_values"]
    fractions = values["cut_volume_fraction"]
    finite = np.isfinite(fractions) & np.isfinite(distances).all(axis=1)
    corner_fraction = (distances > 0.5).mean(axis=1)
    void_but_material = (
        finite
        & (corner_fraction <= low_corner_threshold)
        & (fractions >= high_volume_threshold)
    )
    material_but_void = (
        finite
        & (corner_fraction >= 1.0 - low_corner_threshold)
        & (fractions <= 1.0 - high_volume_threshold)
    )
    outliers = void_but_material | material_but_void
    nonfinite = ~finite
    keep = ~(outliers | nonfinite)
    output_h5 = source_h5.with_name(source_h5.stem + "_clean.h5")
    output_h5.unlink(missing_ok=True)

    with h5py.File(output_h5, "w") as output:
        group = output.create_group("integration_2_poly_1")
        for name, value in values.items():
            group.create_dataset(name, data=value[keep])
        group.attrs.update(attributes)
        group.attrs["orientation_outlier_filter"] = json.dumps(
            {
                "low_corner_threshold": low_corner_threshold,
                "high_cvf_threshold": high_volume_threshold,
                "n_removed_outliers": int(outliers.sum()),
                "n_removed_nonfinite": int(nonfinite.sum()),
                "n_retained": int(keep.sum()),
            }
        )

    geometry_names = np.asarray(
        [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in values["geometry_name"]]
    )
    removed_by_geometry = {
        name: int((~keep & (geometry_names == name)).sum())
        for name in np.unique(geometry_names[~keep])
    }
    return {
        "output_h5": output_h5,
        "total_rows": int(len(keep)),
        "removed_nonfinite": int(nonfinite.sum()),
        "removed_outliers": int(outliers.sum()),
        "retained_rows": int(keep.sum()),
        "removed_by_geometry": removed_by_geometry,
    }
