"""Compare the configured QuESo pipeline against other supported pipelines."""

from __future__ import annotations

import argparse
import json

from clean_simulation import compare_with_queso_defaults
from clean_simulation.config import CaseConfig, NNRuntimeConfig
from clean_simulation.scripts.workflows._request import parse_json_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classic vs queso timing and precision with queso-enabled defaults.")
    parser.add_argument("--device", default="cpu", help="Execution device for FEM stages.")
    parser.add_argument("--reference", default="classic", help="Reference pipeline for relative precision.")
    parser.add_argument("--pipelines", default="classic,queso", help="Comma-separated list. Defaults to classic,queso.")
    parser.add_argument("--config-overrides", default="", help="JSON string overriding hanging sphere config.")
    parser.add_argument("--work-dir", default=None, help="Output working directory for generated QuESo files.")
    parser.add_argument("--queso-python-path", default=None, help="Path containing QuESoPythonModule.")
    parser.add_argument("--nn-checkpoint", default=None, help="Optional NN checkpoint when including hybrid/queso_nn/full_nn.")
    parser.add_argument("--nn-device", default="cpu", help="NN device.")
    parser.add_argument("--nn-dtype", default="float32", help="NN dtype.")
    args = parser.parse_args()
    case_cfg = CaseConfig(overrides=parse_json_arg(args.config_overrides), work_dir=None if args.work_dir in {None, ""} else args.work_dir, queso_python_path=None if args.queso_python_path in {None, ""} else args.queso_python_path)
    nn_cfg = NNRuntimeConfig(checkpoint_path=None if args.nn_checkpoint in {None, ""} else args.nn_checkpoint, device=args.nn_device, dtype=args.nn_dtype)
    pipelines = tuple(part.strip() for part in args.pipelines.split(",") if part.strip())
    result = compare_with_queso_defaults(device=args.device, reference_pipeline=args.reference, case=case_cfg, nn=nn_cfg, pipelines=pipelines)
    print("\n=== Queso Performance Comparison ===")
    for row in result["rows"]:
        print(f"- {row['pipeline']:9s} time={row['total_time_s']:.6f}s rel_l2={row['relative_l2_error_vs_reference']:.6e} active_elems={row['active_elements']}")
    print("\n=== Raw JSON ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()