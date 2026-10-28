"""Compatibility facade for active-training optimization setup."""

from .training.optimization import build_optimizer_and_scheduler

__all__ = ["build_optimizer_and_scheduler"]
