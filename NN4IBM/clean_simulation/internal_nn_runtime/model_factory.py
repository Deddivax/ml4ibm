"""Compatibility facade for active-training model construction."""

from .training.model_factory import build_training_model

__all__ = ["build_training_model"]
