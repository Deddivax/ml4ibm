"""Compare configured classic, QuESo, hybrid, and NN pipelines."""

from __future__ import annotations

import argparse
import json

from clean_simulation import compare_pipelines
from clean_simulation.scripts.workflows._request import add_common_request_args, build_request


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean timing and precision comparison across FEM/NN pipelines.")
    add_common_request_args(parser, include_reference=True)
    args = parser.parse_args()
    request = build_request(
        pipelines=args.pipelines, reference=args.reference, device=args.device,
        nn_device=args.nn_device, nn_dtype=args.nn_dtype, nn_checkpoint=args.nn_checkpoint,
        config_overrides=args.config_overrides, work_dir=args.work_dir,
        queso_python_path=args.queso_python_path, case_builder=args.case_builder,
        case_args=args.case_args,
    )
    result = compare_pipelines(request)
    print("\n=== Pipeline Comparison Summary ===")
    for row in result["rows"]:
        print(f"- {row['pipeline']:9s} time={row['total_time_s']:.6f}s rel_l2={row['relative_l2_error_vs_reference']:.6e} active_elems={row['active_elements']}")
    print("\n=== Raw JSON ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()