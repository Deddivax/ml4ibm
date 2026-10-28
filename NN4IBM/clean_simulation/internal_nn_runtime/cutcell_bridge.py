"""Compatibility facade for the geometry-to-NN runtime bridge.

New code should import ``clean_simulation.internal_nn_runtime.geometry`` or
``clean_simulation.internal_nn_runtime.symmetry`` for its named concern.
"""

from .geometry.cutcell_runtime import *  # noqa: F403
