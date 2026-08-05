"""Annualization must follow the bar frequency (fix 2026-08-05).

Methodology row 7 (METHODOLOGY_DECISIONS.md) locked "annualization
strategy-defined per universe, no hardcoded 252" — but the Pipeline s3
validators and probe/xs.py still hardcoded 252, inflating weekly-bar
Sharpe by sqrt(252/52) ~ 2.2x and monthly by ~4.6x.

Test design: run the SAME price values through each validator twice —
once on a daily index, once on a weekly (W-FRI) index. Per-bar returns
are identical, so the reported annualized Sharpes must differ by
sqrt(252/~52.2) ~ 2.2. Before the fix both used 252 (ratio 1.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import strategy_tester.s3_validation  # noqa: F401
from strategy_tester.registry import get_method

_EXPECT = np.sqrt(252.0 / 52.18)  # ~2.198


def _values(n: int = 600, seed: int = 42):
    rng = np.random.default_rng(seed)
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    r = np.ones(n)
    for i in range(1, n):
        r[i] = r[i - 1] + 0.04 * (1.0 - r[i - 1]) + 0.01 * rng.normal()
    return b * r, b


def _prices(freq: str, n: int = 600) -> pd.DataFrame:
    a, b = _values(n)
    if freq == "D":
        idx = pd.bdate_range("2016-01-04", periods=n)
    else:
        idx = pd.date_range("2012-01-06", periods=n, freq="W-FRI")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _s2(entry: float = -1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "A/B",
                "numerator": "A",
                "denominator": "B",
                "halflife": 17.0,
                "window": 20,
                "entry_thresh": entry,
                "exit_thresh": 0.0,
                "stop_pct": 0.0,
                "slope_min": 0.0,
                "is_sharpe": 1.0,
                "is_penalized_sharpe": 0.9,
                "is_trades": 30,
                "oos_sharpe": 0.5,
                "oos_trades": 8,
                "passed": True,
                "signal_method": "zscore_robust_mad",
                "optim_method": "grid_search",
            }
        ]
    )


@pytest.mark.parametrize(
    "method,extra",
    [
        ("chan_is_oos", {}),
        ("wfa_expanding", {}),
        ("wfa_rolling", {"wfa_rolling_is": 300, "wfa_rolling_oos": 100}),
        ("cpcv", {}),
        ("bootstrap_ci", {"bootstrap_iterations": 2000}),
        ("monte_carlo", {"mc_iterations": 500}),
    ],
)
def test_weekly_bars_annualize_at_52_not_252(method: str, extra: dict) -> None:
    fn = get_method("s3", method)
    rows = {}
    for freq in ("D", "W"):
        out = fn(_prices(freq), _s2(), use_s2_window=True, **extra)
        assert not out.empty, f"{method} returned empty for freq={freq}"
        rows[freq] = out.iloc[0]
    sr_d = float(rows["D"]["mean_test_sharpe"])
    sr_w = float(rows["W"]["mean_test_sharpe"])
    # Sign is irrelevant to the annualization ratio — both frequencies see
    # identical per-bar returns, so the SRs share sign.
    assert (
        abs(sr_d) > 0.15
    ), f"{method}: fixture SR too small (daily SR {sr_d}) for a ratio test"
    ratio = sr_d / sr_w
    assert 1.85 <= ratio <= 2.55, (
        f"{method}: daily/weekly SR ratio {ratio:.3f}, expected ~{_EXPECT:.2f}"
        " — weekly bars are not being annualized at ~52 bars/yr"
    )
