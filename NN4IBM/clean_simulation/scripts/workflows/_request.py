"""Shared argument parsing and request construction for workflow commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clean_simulation import CaseConfig, ComparisonRequest, NNRuntimeConfig


def parse_json_arg(value: str) -> dict:
    text = value.strip()
    return {} if not text else json.loads(text)


def add_common_request_args(parser: argparse.ArgumentParser, *, include_reference: bool) -> None:
    parser.add_argument("--pipelines", default="classic,queso", help="Comma-separated list: classic,queso,hybrid")
    if include_reference:
        parser.add_argument("--reference", default="classic", help="Reference pipeline used for relative precision metric.")
    parser.add_argument("--device", default="cpu", help="Execution device for FEM stages.")
    parser.add_argument("--nn-device", default="cpu", help="Device for NN inference engine.")
    parser.add_argument("--nn-dtype", default="float32", help="NN dtype: float32|float16|bfloat16|float64")
    parser.add_argument("--nn-checkpoint", type=Path, default=None, help="Path to NN checkpoint (.pt)")
    parser.add_argument("--config-overrides", default="", help="JSON string overriding hanging sphere config.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Output working directory for generated QuESo files.")
    parser.add_argument("--queso-python-path", type=Path, default=None, help="Path containing QuESoPythonModule.")
    parser.add_argument("--case-builder", default="", help="Optional case builder path module:symbol. Example: package.module:create_problem")
    parser.add_argument("--case-args", default="", help="JSON args for case builder callable.")


def build_request(
    *,
    pipelines: str,
    reference: str,
    device: str,
    nn_device: str,
    nn_dtype: str,
    nn_checkpoint: Path | None,
    config_overrides: str,
    work_dir: Path | None,
    queso_python_path: Path | None,
    case_builder: str,
    case_args: str,
) -> ComparisonRequest:
    pipeline_names = tuple(part.strip() for part in pipelines.split(",") if part.strip())
    return ComparisonRequest(
        pipelines=pipeline_names,
        reference_pipeline=reference.strip() or (pipeline_names[0] if pipeline_names else "classic"),
        device=device,
        nn=NNRuntimeConfig(checkpoint_path=nn_checkpoint, device=nn_device, dtype=nn_dtype),
        case=CaseConfig(
            case_builder=case_builder.strip() or None,
            case_args=parse_json_arg(case_args),
            overrides=parse_json_arg(config_overrides),
            work_dir=work_dir,
            queso_python_path=queso_python_path,
        ),
    )