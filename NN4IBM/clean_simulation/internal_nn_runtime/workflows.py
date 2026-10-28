"""Compatibility facade for the active training pipeline.

New code should import ``clean_simulation.internal_nn_runtime.training.pipeline``.
"""

from .training.pipeline import PipelineResult, run_training_pipeline

__all__ = ["PipelineResult", "run_training_pipeline"]
