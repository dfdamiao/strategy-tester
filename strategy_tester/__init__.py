"""Pipeline Library — composable backtesting framework.

Usage:
    from strategy_tester import Pipeline, PRESETS, list_methods
    pipe = Pipeline(**PRESETS["X-1"])
    result = pipe.run(prices, pairs, config)
"""
from __future__ import annotations

# Import stage packages to trigger @register_stage decorators
import strategy_tester.s1_screening  # noqa: F401
import strategy_tester.s2_optimize  # noqa: F401
import strategy_tester.s2_signal  # noqa: F401
import strategy_tester.s3_validation  # noqa: F401
import strategy_tester.s4_significance  # noqa: F401
import strategy_tester.s5_portfolio  # noqa: F401

# Public API
from strategy_tester.bridge import bridge_to_portfolio
from strategy_tester.compare import compare_pipelines
from strategy_tester.pipeline import Pipeline
from strategy_tester.presets import PRESETS
from strategy_tester.registry import list_methods

__all__ = [
    "Pipeline",
    "bridge_to_portfolio",
    "compare_pipelines",
    "PRESETS",
    "list_methods",
]
