"""Tests for S1 screening methods."""
from __future__ import annotations
import numpy as np
import pandas as pd

import strategy_tester.s1_screening  # noqa: F401
from strategy_tester.registry import get_method
from strategy_tester.interfaces import validate_interface


def _make_mr_pair(
    n: int = 1500, theta: float = 0.03, seed: int = 42,
) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    ratio = np.ones(n)
    for i in range(1, n):
        ratio[i] = ratio[i - 1] + theta * (1.0 - ratio[i - 1]) + 0.01 * rng.normal()
    a = b * ratio
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    pairs = [{"pair": "A/B", "numerator": "A", "denominator": "B"}]
    return prices, pairs


def test_chan_halflife_mr_pair():
    fn = get_method("s1", "chan_halflife")
    prices, pairs = _make_mr_pair()
    config = {
        "min_halflife": 2, "max_halflife": 756,
        "adf_pvalue_threshold": 0.10, "min_common_rows": 252,
    }
    result = fn(prices, pairs, **config)
    validate_interface(result, "s1")
    assert result.iloc[0]["method"] == "chan_halflife"


def test_chan_hurst_mr_pair():
    fn = get_method("s1", "chan_hurst")
    prices, pairs = _make_mr_pair()
    config = {"hurst_threshold": 0.55, "min_common_rows": 252}
    result = fn(prices, pairs, **config)
    validate_interface(result, "s1")
    assert result.iloc[0]["method"] == "chan_hurst"


def test_kaufman_er_mr_pair():
    fn = get_method("s1", "kaufman_er")
    prices, pairs = _make_mr_pair()
    config = {"er_threshold": 0.50, "min_common_rows": 252}
    result = fn(prices, pairs, **config)
    validate_interface(result, "s1")
    assert result.iloc[0]["method"] == "kaufman_er"


def test_chan_combined_runs_all_three():
    fn = get_method("s1", "chan_combined")
    prices, pairs = _make_mr_pair()
    config = {
        "min_halflife": 2, "max_halflife": 756,
        "adf_pvalue_threshold": 0.10, "min_common_rows": 252,
        "hurst_threshold": 0.55, "er_threshold": 0.50,
    }
    result = fn(prices, pairs, **config)
    validate_interface(result, "s1")
    assert result.iloc[0]["method"] == "chan_combined"
    assert "hurst_exponent" in result.columns
    assert "efficiency_ratio" in result.columns
