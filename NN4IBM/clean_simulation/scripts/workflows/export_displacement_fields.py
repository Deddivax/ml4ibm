"""Export global displacement fields for configured simulation pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from clean_simulation import collect_global_displacements
from clean_simulation.scripts.workflows._request import add_common_request_args, build_request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and export global displacement fields for selected pipelines."
    )
    add_common_request_args(parser, include_reference=False)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clean_simulation/displacements"),
        help="Directory where .npy displacement arrays are written.",
    )

    args = parser.parse_args()
    request = build_request(
        pipelines=args.pipelines,
        reference="",
        device=args.device,
        nn_device=args.nn_device,
        nn_dtype=args.nn_dtype,
        nn_checkpoint=args.nn_checkpoint,
        config_overrides=args.config_overrides,
        work_dir=args.work_dir,
        queso_python_path=args.queso_python_path,
        case_builder=args.case_builder,
        case_args=args.case_args,
    )
    result = collect_global_displacements(request)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Displacement Export ===")
    for pipeline_name, data in result["displacements"].items():
        u_nodes = np.asarray(data["u_nodes"])
        u_vec = np.asarray(data["u_vec"])
        active_indices = np.asarray(data["active_indices"])
        np.save(output_dir / f"{pipeline_name}_u_nodes.npy", u_nodes)
        np.save(output_dir / f"{pipeline_name}_u_vec.npy", u_vec)
        np.save(output_dir / f"{pipeline_name}_active_indices.npy", active_indices)
        print(f"- {pipeline_name:9s} u_nodes={u_nodes.shape} u_vec={u_vec.shape} active={active_indices.shape}")
    print(f"\nSaved arrays to: {output_dir}")


if __name__ == "__main__":
    main()