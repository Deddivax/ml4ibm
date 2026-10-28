"""Ready-to-use loader for the promoted best cut-cell model checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .nn_runtime import load_nn_engine


BEST_MODEL_CHECKPOINT = Path(__file__).resolve().parent / "outputs" / "best_checkpoint" / "best_model.pt"


def load_best_model(
    *,
    device: str = "cpu",
    dtype: str = "float32",
    num_threads: int | None = None,
    use_compile: bool | None = None,
) -> Any:
    """Load the promoted checkpoint with optimized CPU or CUDA inference settings."""
    if not BEST_MODEL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "The promoted best checkpoint is missing: "
            f"{BEST_MODEL_CHECKPOINT}. Restore clean_simulation/outputs/best_checkpoint."
        )
    is_cuda = device.startswith("cuda")
    return load_nn_engine(
        checkpoint_path=BEST_MODEL_CHECKPOINT,
        device=device,
        dtype=dtype,
        num_threads=max(1, num_threads if num_threads is not None else (os.cpu_count() or 1)),
        use_compile=is_cuda if use_compile is None else use_compile,
        warmup_iters=16,
        warmup_batch_size=1024 if is_cuda else 256,
    )


__all__ = ["BEST_MODEL_CHECKPOINT", "load_best_model"]
