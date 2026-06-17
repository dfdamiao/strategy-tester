"""Tests for performance metrics."""
from __future__ import annotations
import numpy as np
from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    geometric_cagr,
    max_drawdown,
    calmar_ratio,
    psr_stat,
)


def test_sharpe_known():
    rets = np.full(252, 0.001)
    sr = annualized_sharpe(rets)
    assert sr > 10


def test_sharpe_zero_returns():
    rets = np.zeros(252)
    sr = annualized_sharpe(rets)
    assert sr == 0.0


def test_cagr():
    rets = np.full(252, 0.0004)
    cagr = geometric_cagr(rets)
    assert 0.09 < cagr < 0.12


def test_max_drawdown():
    rets = np.array([0.1, 0.1, -0.15, -0.10, 0.05])
    dd = max_drawdown(rets)
    assert dd < 0


def test_calmar():
    # Positive trend with a mid-series dip: CAGR > 0, MaxDD < 0 → Calmar > 0
    rets = np.concatenate([
        np.full(100, 0.002), np.full(20, -0.01), np.full(132, 0.002),
    ])
    c = calmar_ratio(rets)
    assert c > 0


def test_psr_high_sharpe():
    p = psr_stat(sharpe=2.0, n_obs=252, skew=0.0, kurtosis=3.0)
    assert p > 0.95


def test_psr_zero_obs():
    p = psr_stat(sharpe=2.0, n_obs=0, skew=0.0, kurtosis=3.0)
    assert p == 0.0
