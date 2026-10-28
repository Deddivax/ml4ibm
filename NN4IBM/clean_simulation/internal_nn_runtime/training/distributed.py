"""Distributed metric and model-state checks for DDP training."""

from __future__ import annotations

import torch
import torch.distributed as dist


def average_metrics(metrics: dict, device: str) -> dict:
    """Return global sample-weighted metrics when DDP is initialized."""
    if not dist.is_initialized():
        return metrics
    local_samples = float(metrics["n_samples"])
    sample_count = torch.tensor(local_samples, device=device)
    dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    averaged = {"n_samples": int(sample_count.item())}
    for name, value in metrics.items():
        if name == "n_samples":
            continue
        total = torch.tensor(float(value) * local_samples, device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        averaged[name] = float((total / sample_count).item())
    return averaged


def assert_model_parity(model: torch.nn.Module, device: str) -> None:
    """Fail when a DDP optimizer step leaves ranks with different weights."""
    if not dist.is_initialized():
        return
    checksum = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in model.parameters():
        checksum.add_(parameter.detach().double().sum())
    checksum_min = checksum.clone()
    checksum_max = checksum.clone()
    dist.all_reduce(checksum_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(checksum_max, op=dist.ReduceOp.MAX)
    tolerance = torch.finfo(checksum.dtype).eps * max(1.0, abs(checksum.item())) * 64
    if (checksum_max - checksum_min).abs().item() > tolerance:
        raise RuntimeError(
            "DDP model-state parity check failed after the optimizer step: "
            f"checksum range [{checksum_min.item():.12e}, {checksum_max.item():.12e}]."
        )