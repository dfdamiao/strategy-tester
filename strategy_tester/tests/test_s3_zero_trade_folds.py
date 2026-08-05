"""Zero-trade folds must not poison S3 gates with +inf Sharpe.

Bug (found 2026-08-04, first-rebuild verification pass): vectorbt
``pf.sharpe_ratio()`` returns +inf on a fold where the signal never
fires (0 trades, all-zero returns). ``backtest_vbt_fold`` /
``backtest_vbt_precomputed`` only guarded NaN, so inf leaked into
``wfa_expanding`` / ``wfa_rolling`` fold lists where it (a) counted as
a "positive" fold for the Pardo pct_positive gate — letting
never-trading nulls PASS — and (b) tripped the inf-mean guard that
rewrites mean_test_sharpe to 0.0, destroying real edges' scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest  # noqa: F401

import strategy_tester.s3_validation  # noqa: F401
from strategy_tester.backtest.vbt_runner import backtest_vbt_fold
from strategy_tester.registry import get_method


def _make_prices(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    r = np.ones(n)
    for i in range(1, n):
        r[i] = r[i - 1] + 0.04 * (1.0 - r[i - 1]) + 0.01 * rng.normal()
    return pd.DataFrame({"A": b * r, "B": b}, index=idx)


def _make_s2_never_fires() -> pd.DataFrame:
    """Entry threshold z <= -50 never triggers: a never-trading unit."""
    return pd.DataFrame([{
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "halflife": 15.0, "window": 10,
        "entry_thresh": -50.0, "exit_thresh": 1.0,
        "stop_pct": 0.0, "slope_min": 0.0,
        "is_sharpe": 1.0, "is_penalized_sharpe": 0.9, "is_trades": 20,
        "oos_sharpe": 0.5, "oos_trades": 5, "passed": True,
        "signal_method": "zscore_robust_mad", "optim_method": "grid_search",
    }])


def test_zero_trade_fold_sharpe_is_not_inf() -> None:
    prices = _make_prices(300)
    ratio = prices["A"] / prices["B"]
    bt = backtest_vbt_fold(
        prices["A"], ratio, 10, -50.0, 1.0,
        stop_pct=0.0, slope_min=0.0, fees=0.001,
    )
    assert bt["n_trades"] == 0
    assert not np.isinf(bt["sharpe"]), (
        "zero-trade fold must return NaN sharpe, not +inf"
    )


@pytest.mark.parametrize("method", ["wfa_expanding", "wfa_rolling"])
def test_never_trading_unit_fails_wfa(method: str) -> None:
    fn = get_method("s3", method)
    result = fn(
        _make_prices(), _make_s2_never_fires(), use_s2_window=True,
    )
    assert not result.empty
    row = result.iloc[0]
    assert not row["passed"], (
        f"{method}: unit that never trades must not pass the WFA gate"
    )
    assert row["mean_test_sharpe"] == 0.0
    assert np.isfinite(row["std_test_sharpe"])
