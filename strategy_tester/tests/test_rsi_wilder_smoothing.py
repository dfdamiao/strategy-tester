"""§ methods audit 2026-07-21: rsi_wilder must use RECURSIVE Wilder RMA
(ewm adjust=False), matching its docstring formula, not pandas' default
adjust=True bias-corrected EWMA. Matters most at short RSI windows (Connors 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy_tester.s2_signal.rsi_wilder import _compute_rsi_wilder

pytestmark = pytest.mark.unit


def _wilder_rsi_reference(prices: np.ndarray, n: int) -> np.ndarray:
    """Independent recursive Wilder RMA RSI: avg[t] = (avg[t-1]*(n-1)+x[t])/n,
    seeded at x[0] (matches ewm adjust=False), first n-1 masked NaN."""
    delta = np.diff(prices, prepend=prices[0])
    delta[0] = np.nan
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    a = 1.0 / n

    def rma(x):
        y = np.full(len(x), np.nan)
        y[1] = x[1]  # seed at first real value (index 0 is NaN from diff)
        for t in range(2, len(x)):
            y[t] = (1 - a) * y[t - 1] + a * x[t]
        y[:n] = np.nan  # min_periods=n
        return y

    g, ls = rma(gain), rma(loss)
    rs = g / np.where(ls == 0, np.nan, ls)
    return 100.0 - 100.0 / (1.0 + rs)


def test_wilder_rsi_matches_recursive_reference_not_adjust_true() -> None:
    rng = np.random.default_rng(1)
    prices = 100 + np.cumsum(rng.normal(0, 1, 300))
    s = pd.Series(prices)
    got = _compute_rsi_wilder(s, window=2).to_numpy()

    # adjust=True reference (the OLD buggy behavior) — must NOT match
    delta = s.diff()
    g_t = delta.clip(lower=0).ewm(alpha=0.5, min_periods=2).mean()
    l_t = (-delta).clip(lower=0).ewm(alpha=0.5, min_periods=2).mean()
    rs_t = g_t / l_t.replace(0, np.nan)
    adjust_true = (100 - 100 / (1 + rs_t)).to_numpy()

    valid = ~np.isnan(got) & ~np.isnan(adjust_true)
    # the fix changed behavior away from adjust=True
    assert not np.allclose(got[valid], adjust_true[valid])

    # and it now matches the recursive Wilder reference
    ref = _wilder_rsi_reference(prices, 2)
    v2 = ~np.isnan(got) & ~np.isnan(ref)
    assert np.allclose(got[v2], ref[v2], rtol=1e-9, atol=1e-9)
