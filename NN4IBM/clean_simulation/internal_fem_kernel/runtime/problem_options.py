"""Read optional pipeline settings from supported problem representations."""

from typing import Any


def threshold_bounds(problem: Any) -> tuple[float | None, float | None]:
    lower = getattr(problem, "lower_threshold", None)
    upper = getattr(problem, "upper_threshold", None)
    config = getattr(problem, "config", None)

    if hasattr(config, "overrides") and isinstance(config.overrides, dict):
        config = config.overrides
    if not isinstance(config, dict):
        return lower, upper

    return (
        config.get("lower_threshold", lower) if lower is None else lower,
        config.get("upper_threshold", upper) if upper is None else upper,
    )