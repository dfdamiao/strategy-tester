"""Tests for compare_pipelines."""
from __future__ import annotations
import numpy as np
import pandas as pd

from strategy_tester import PRESETS, compare_pipelines


def _make_prices_and_pairs():
    rng = np.random.default_rng(42)
    n = 1500
    idx = pd.bdate_range("2018-01-01", periods=n)
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    r = np.ones(n)
    for i in range(1, n):
        r[i] = r[i - 1] + 0.04 * (1.0 - r[i - 1]) + 0.01 * rng.normal()
    prices = pd.DataFrame({"A": b * r, "B": b}, index=idx)
    pairs = [{"pair": "A/B", "numerator": "A", "denominator": "B"}]
    return prices, pairs


def test_compare_returns_dict():
    """compare_pipelines returns dict of PipelineResults."""
    prices, pairs = _make_prices_and_pairs()
    config = {
        "entry_grid": [-2.0],
        "exit_grid": [1.0],
        "stop_grid": [0.0],
        "slope_grid": [0.0],
        "adf_pvalue_threshold": 0.20,
    }
    results = compare_pipelines(
        {"P-1": PRESETS["P-1"]},
        prices, pairs, config,
    )
    assert "P-1" in results
    assert results["P-1"].name == "Chan Pure"
