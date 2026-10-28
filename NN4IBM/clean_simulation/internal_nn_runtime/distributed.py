"""Compatibility facade for active-training distributed utilities."""

from .training.distributed import assert_model_parity, average_metrics

__all__ = ["assert_model_parity", "average_metrics"]
