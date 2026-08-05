"""Canonical Sharpe kernel — ported from the monorepo 2026-08-03 consolidation.

Locks the fix: ONE Sharpe implementation
(``strategy_tester/backtest/metrics.py::annualized_sharpe``) with explicit
``ddof`` and ``periods_per_year``; bars-per-year is detectable from the
panel calendar (multi-exchange fix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_tester.backtest.metrics import (
    annualized_sharpe,
    bars_per_year_from_index,
)

pytestmark = pytest.mark.unit


def test_kernel_ddof_parameter() -> None:
    rng = np.random.default_rng(0)
    r = rng.standard_normal(60) * 0.01 + 0.001
    s1 = annualized_sharpe(r, ddof=1)
    s0 = annualized_sharpe(r, ddof=0)
    # ddof=0 inflates by sqrt(n/(n-1))
    assert s0 == pytest.approx(s1 * np.sqrt(60 / 59), rel=1e-12)


def test_kernel_periods_per_year() -> None:
    rng = np.random.default_rng(1)
    r = rng.standard_normal(500) * 0.01 + 0.001
    s252 = annualized_sharpe(r, periods_per_year=252)
    s530 = annualized_sharpe(r, periods_per_year=530)
    assert s530 == pytest.approx(s252 * np.sqrt(530 / 252), rel=1e-12)


def test_kernel_degenerate_inputs() -> None:
    assert annualized_sharpe(np.array([])) == 0.0
    assert annualized_sharpe(np.zeros(10)) == 0.0


def test_bars_per_year_from_index() -> None:
    # ~252-density: business days
    idx = pd.bdate_range("2018-01-01", periods=1260)
    assert bars_per_year_from_index(idx) == pytest.approx(252, rel=0.05)
    # ~365-density: calendar days (proxy for multi-exchange unions >252)
    idx = pd.date_range("2018-01-01", periods=1460, freq="D")
    assert bars_per_year_from_index(idx) == pytest.approx(365, rel=0.05)
    # non-datetime index -> fallback
    assert bars_per_year_from_index(pd.RangeIndex(100)) == 252.0
    # too-short span -> fallback
    assert bars_per_year_from_index(pd.bdate_range("2024-01-01", periods=5)) == 252.0
