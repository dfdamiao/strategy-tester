"""Tests for backtest engine and signal helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from strategy_tester.backtest.vbt_runner import (
    compute_halflife,
    robust_zscore,
    zscore_slope,
    calculate_penalized_sharpe,
    build_is_oos_split,
)


def _make_ou_process(
    n: int = 1000, theta: float = 0.05, mu: float = 1.0,
    sigma: float = 0.01, seed: int = 42,
) -> pd.Series:
    """Synthetic OU process with known mean-reversion speed."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.normal()
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(x, index=idx, name="ratio")


def test_compute_halflife_ou():
    ratio = _make_ou_process(n=2000, theta=0.05)
    hl = compute_halflife(ratio)
    assert 5 < hl < 30, f"halflife={hl}, expected ~13.5"


def test_compute_halflife_random_walk():
    # Pure positive-slope drift: slope > 0 → NaN returned by compute_halflife
    idx = pd.bdate_range("2020-01-01", periods=500)
    rw = pd.Series(np.arange(500, dtype=float), index=idx)
    hl = compute_halflife(rw)
    assert np.isnan(hl), "Trending series should yield NaN halflife"


def test_robust_zscore_shape():
    ratio = _make_ou_process()
    z = robust_zscore(ratio, window=20)
    assert len(z) == len(ratio)
    assert z.iloc[:19].isna().all()
    assert z.iloc[50:].notna().any()


def test_zscore_slope():
    z = pd.Series([0.0, -1.0, -2.0, -1.5, -1.0, 0.0])
    slope = zscore_slope(z, n=2)
    assert slope.iloc[2] == pytest.approx(-1.0)
    assert slope.iloc[4] == pytest.approx(0.5)


def test_penalized_sharpe():
    assert calculate_penalized_sharpe(1.0, 4, 100) == pytest.approx(
        1.0 * np.sqrt(1 - 4 / 100), rel=1e-6
    )
    assert calculate_penalized_sharpe(1.0, 4, 3) == 0.0
    assert calculate_penalized_sharpe(1.0, 4, 0) == 0.0


def test_is_oos_split():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    is_idx, oos_idx = build_is_oos_split(idx, ratio=0.80)
    assert len(is_idx) == 800
    assert len(oos_idx) == 200
    assert is_idx[-1] < oos_idx[0]
