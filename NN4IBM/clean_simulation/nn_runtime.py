from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .core.legacy_imports import configure_legacy_imports


def _dtype_from_name(name: str) -> torch.dtype:
    value = name.strip().lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float64", "fp64", "double"}:
        return torch.float64
    return torch.float32


def load_nn_engine(
    *,
    checkpoint_path: str | Path,
    device: str = "cpu",
    dtype: str = "float32",
    num_threads: int = 1,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    warmup_iters: int = 8,
    warmup_batch_size: int = 128,
) -> Any:
    configure_legacy_imports()

    from .internal_nn_runtime import FastCutCellInference, InferenceOptions

    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    options = InferenceOptions(
        device=device,
        dtype=_dtype_from_name(dtype),
        use_jit=True,
        use_compile=use_compile,
        compile_mode=compile_mode,
        num_threads=num_threads,
        warmup_iters=warmup_iters,
        warmup_batch_size=warmup_batch_size,
    )
    return FastCutCellInference.from_checkpoint(ckpt, options=options)
