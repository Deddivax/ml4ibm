"""Timing helpers for Blade study commands."""

import time

import torch


def elapsed(started_at: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started_at