"""WFA active-fold convention (Q6, 2026-08-04) — align Pipeline s3 WFA
to the probe/P8 convention.

Zero-trade OOS windows are absence of evidence, not evidence of failure:
they are skipped (count neither for nor against), and the
conditioning-on-activity bias is guarded by hard floors instead —
``pct_positive`` becomes the ACTIVE win rate (Pardo, house convention:
>= 0.625) and the gate requires ``n_active >= 4``. Mirrors
``lib/probe/wfa.py`` (skip inactive folds + P8 floors) and the
fred_regime Pardo lock (active_win_rate = n_pos/n_active >= 0.625 AND
n_active >= 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest  # noqa: F401

import strategy_tester.s3_validation  # noqa: F401
from strategy_tester.registry import get_method


def _make_prices(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    r = np.ones(n)
    for i in range(1, n):
        r[i] = r[i - 1] + 0.04 * (1.0 - r[i - 1]) + 0.01 * rng.normal()
    return pd.DataFrame({"A": b * r, "B": b}, index=idx)


def _make_s2(entry: float = -2.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "pair": "A/B", "numerator": "A", "denominator": "B",
        "halflife": 17.0, "window": 20,
        "entry_thresh": entry, "exit_thresh": 0.0,
        "stop_pct": 0.0, "slope_min": 0.0,
        "is_sharpe": 1.0, "is_penalized_sharpe": 0.9, "is_trades": 30,
        "oos_sharpe": 0.5, "oos_trades": 8, "passed": True,
        "signal_method": "zscore_robust_mad", "optim_method": "grid_search",
    }])


def test_rolling_sparse_true_edge_passes() -> None:
    """A real edge that sits out some 180d windows must not be vetoed
    by inactive windows: active win rate + n_active floor decide.

    entry z <= -1.5 on the OU fixture is active in 7 of 11 rolling
    windows (seed 42) — sparse, but above the n_active >= 4 floor."""
    fn = get_method("s3", "wfa_rolling")
    result = fn(_make_prices(), _make_s2(entry=-1.5), use_s2_window=True)
    assert not result.empty
    row = result.iloc[0]
    assert row["n_active"] < row["n_folds_total"], "fixture must be sparse"
    assert row["n_active"] >= 4
    assert row["passed"], (
        "sparse-but-consistent edge must pass under active-fold Pardo"
    )


def test_rolling_too_sparse_edge_fails_floor() -> None:
    """Same edge at entry z <= -2.0 is active in only 3 of 11 windows
    (seed 42): below the n_active >= 4 floor -> must fail regardless
    of how good the 3 active windows look."""
    fn = get_method("s3", "wfa_rolling")
    result = fn(_make_prices(), _make_s2(entry=-2.0), use_s2_window=True)
    row = result.iloc[0]
    assert row["n_active"] < 4
    assert not row["passed"]


@pytest.mark.parametrize("method", ["wfa_expanding", "wfa_rolling"])
def test_active_fold_columns_reported(method: str) -> None:
    fn = get_method("s3", method)
    result = fn(_make_prices(), _make_s2(), use_s2_window=True)
    row = result.iloc[0]
    for col in ("n_active", "n_folds_total", "median_test_sharpe"):
        assert col in row, f"{method} missing {col}"
    assert row["n_active"] <= row["n_folds_total"]
    # n_test_periods now reports graded (active) folds only
    assert row["n_test_periods"] == row["n_active"]
    # pct_positive is the ACTIVE win rate: n_pos / n_active
    assert 0.0 <= row["pct_positive"] <= 1.0


@pytest.mark.parametrize("method", ["wfa_expanding", "wfa_rolling"])
def test_n_active_floor_blocks_barely_active_unit(method: str) -> None:
    """A unit active in fewer than 4 folds must fail regardless of how
    good those few folds look (anti 'only-acted-twice-both-wins')."""
    fn = get_method("s3", method)
    result = fn(
        _make_prices(), _make_s2(), use_s2_window=True,
        pardo_min_active_folds=10**6,
    )
    row = result.iloc[0]
    assert not row["passed"]
