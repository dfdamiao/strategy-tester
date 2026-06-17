"""Tests for presets."""
from __future__ import annotations

import strategy_tester  # noqa: F401
from strategy_tester.pipeline import Pipeline
from strategy_tester.presets import PRESETS

_PIPELINE_KEYS = {"s1", "s2_signal", "s2_optim", "s3", "s4", "s5", "name"}


def test_all_presets_instantiate() -> None:
    for name, kwargs in PRESETS.items():
        pipe_kwargs = {k: v for k, v in kwargs.items() if k in _PIPELINE_KEYS}
        pipe = Pipeline(**pipe_kwargs)
        assert pipe.name, f"Preset {name} has no name"


def test_preset_count() -> None:
    assert len(PRESETS) >= 13
