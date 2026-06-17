"""End-to-end integration test: P-1 preset on synthetic pairs."""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategy_tester import Pipeline, PRESETS, list_methods


def _make_3_pairs(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)

    def _ou(theta, mu=1.0):
        x = np.ones(n)
        for i in range(1, n):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) + 0.01 * rng.normal()
        return x

    base = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    prices = pd.DataFrame({
        "A": base * _ou(0.04),
        "B": base.copy(),
        "C": base * _ou(0.06),
        "D": base * 1.05,
        "E": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))),
        "F": base * 0.95,
    }, index=idx)

    pairs = [
        {"pair": "A/B", "numerator": "A", "denominator": "B"},
        {"pair": "C/D", "numerator": "C", "denominator": "D"},
        {"pair": "E/F", "numerator": "E", "denominator": "F"},
    ]
    return prices, pairs


def test_list_methods_populated():
    methods = list_methods()
    assert len(methods["s1"]) >= 4
    assert len(methods["s2_signal"]) >= 4
    assert len(methods["s2_optimize"]) >= 2
    assert len(methods["s3"]) >= 7
    assert len(methods["s4"]) >= 8
    assert len(methods["s5"]) >= 9


def test_p1_stop_after_s1():
    """P-1 runs S1 without errors on synthetic data."""
    pipe = Pipeline(**PRESETS["P-1"])
    prices, pairs = _make_3_pairs()
    config = {
        "entry_grid": [-2.5, -2.0],
        "exit_grid": [1.0, 1.5],
        "stop_grid": [0.0],
        "slope_grid": [0.0],
        "adf_pvalue_threshold": 0.20,
    }
    result = pipe.run(prices, pairs, config, stop_after="s1", report=False)
    assert result.name == "Chan Pure"
    assert "s1" in result.stages
    s1_df = result.stages["s1"]["result"]
    assert len(s1_df) > 0
