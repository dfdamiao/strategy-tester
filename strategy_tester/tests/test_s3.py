"""Tests for S3 validation methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest  # noqa: F401

import strategy_tester.s3_validation  # noqa: F401
from strategy_tester.registry import get_method
from strategy_tester.interfaces import validate_interface


def _make_s2_result() -> pd.DataFrame:
    return pd.DataFrame([{
        "pair": "A/B",
        "numerator": "A",
        "denominator": "B",
        "halflife": 15.0,
        "window": 10,
        "entry_thresh": -2.0,
        "exit_thresh": 1.0,
        "stop_pct": 0.0,
        "slope_min": 0.0,
        "is_sharpe": 1.0,
        "is_penalized_sharpe": 0.9,
        "is_trades": 20,
        "oos_sharpe": 0.5,
        "oos_trades": 5,
        "passed": True,
        "signal_method": "zscore_robust_mad",
        "optim_method": "grid_search",
    }])


def _make_prices(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    r = np.ones(n)
    for i in range(1, n):
        r[i] = r[i - 1] + 0.04 * (1.0 - r[i - 1]) + 0.01 * rng.normal()
    a = b * r
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_chan_is_oos_s3_format() -> None:
    fn = get_method("s3", "chan_is_oos")
    result = fn(_make_prices(), _make_s2_result())
    if not result.empty:
        validate_interface(result, "s3")
        assert result.iloc[0]["val_method"] == "chan_is_oos"


def test_wfa_expanding_s3_format() -> None:
    fn = get_method("s3", "wfa_expanding")
    result = fn(_make_prices(), _make_s2_result())
    if not result.empty:
        validate_interface(result, "s3")
        assert result.iloc[0]["val_method"] == "wfa_expanding"
        assert result.iloc[0]["n_test_periods"] >= 2


def test_wfa_rolling_s3_format() -> None:
    fn = get_method("s3", "wfa_rolling")
    result = fn(
        _make_prices(), _make_s2_result(),
        wfa_rolling_is=400, wfa_rolling_oos=100,
    )
    if not result.empty:
        validate_interface(result, "s3")
        assert result.iloc[0]["val_method"] == "wfa_rolling"


def test_cpcv_s3_format() -> None:
    fn = get_method("s3", "cpcv")
    result = fn(_make_prices(), _make_s2_result())
    if not result.empty:
        validate_interface(result, "s3")
        assert result.iloc[0]["val_method"] == "cpcv"
        assert result.iloc[0]["n_test_periods"] >= 5


def test_sensitivity_s3_format() -> None:
    fn = get_method("s3", "sensitivity")
    result = fn(_make_prices(), _make_s2_result())
    if not result.empty:
        validate_interface(result, "s3")
        assert result.iloc[0]["val_method"] == "sensitivity"


def test_monte_carlo_s3_format() -> None:
    fn = get_method("s3", "monte_carlo")
    result = fn(_make_prices(), _make_s2_result(), mc_iterations=100)
    if not result.empty:
        validate_interface(result, "s3")
        assert result.iloc[0]["val_method"] == "monte_carlo"


def test_bootstrap_ci_s3_format() -> None:
    fn = get_method("s3", "bootstrap_ci")
    result = fn(
        _make_prices(), _make_s2_result(), bootstrap_iterations=100,
    )
    if not result.empty:
        validate_interface(result, "s3")
        assert result.iloc[0]["val_method"] == "bootstrap_ci"
