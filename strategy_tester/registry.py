"""Method registry — @register_stage decorator + discovery."""
from __future__ import annotations

from typing import Callable

_VALID_STAGES = frozenset(
    {"s1", "s2_signal", "s2_optimize", "s3", "s4", "s5"}
)

_REGISTRY: dict[str, dict[str, Callable]] = {s: {} for s in _VALID_STAGES}


def register_stage(stage: str) -> Callable:
    """Decorator to register a function for a pipeline stage."""
    if stage not in _VALID_STAGES:
        raise ValueError(
            f"Unknown stage {stage!r}. "
            f"Valid: {sorted(_VALID_STAGES)}"
        )

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[stage][fn.__name__] = fn
        return fn

    return decorator


def get_method(stage: str, name: str) -> Callable:
    """Get a registered method by stage and name."""
    if stage not in _VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}")
    try:
        return _REGISTRY[stage][name]
    except KeyError:
        available = sorted(_REGISTRY[stage])
        raise KeyError(
            f"{name!r} not found in stage {stage!r}. "
            f"Available: {available}"
        ) from None


def list_methods(stage: str | None = None) -> dict[str, list[str]]:
    """List registered methods. Filter by stage if given."""
    if stage is not None:
        if stage not in _VALID_STAGES:
            raise ValueError(f"Unknown stage {stage!r}")
        return {stage: sorted(_REGISTRY[stage])}
    return {s: sorted(fns) for s, fns in _REGISTRY.items()}
