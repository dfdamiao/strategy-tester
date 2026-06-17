"""Tests for S2 optimize methods."""
from __future__ import annotations
import numpy as np
import pandas as pd

import strategy_tester.s1_screening  # noqa: F401
import strategy_tester.s2_signal  # noqa: F401
import strategy_tester.s2_optimize  # noqa: F401
from strategy_tester.registry import get_method
from strategy_tester.interfaces import validate_interface


def _make_mr_pair(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    ratio_vals = np.ones(n)
    for i in range(1, n):
        ratio_vals[i] = (
            ratio_vals[i - 1]
            + 0.04 * (1.0 - ratio_vals[i - 1])
            + 0.01 * rng.normal()
        )
    a = b * ratio_vals
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    return prices


def test_grid_search_returns_s2_format():
    fn = get_method("s2_optimize", "grid_search")
    prices = _make_mr_pair()
    s1_result = pd.DataFrame([{
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "passed": True, "halflife": 15.0, "window": 10,
        "method": "chan_halflife",
    }])
    config = {
        "entry_grid": [-2.5, -2.0],
        "exit_grid": [1.0, 1.5],
        "stop_grid": [0.0],
        "slope_grid": [0.0],
        "is_ratio": 0.80,
        "cost_per_side": 0.001,
    }
    result = fn(prices, s1_result, **config)
    if not result.empty:
        validate_interface(result, "s2")
        assert result.iloc[0]["optim_method"] == "grid_search"
