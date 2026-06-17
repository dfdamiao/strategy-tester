"""Connors cumulative RSI signal for ETF mean-reversion.

Sums consecutive RSI(period) values over a short window. Persistent oversold
reads are a stronger entry signal than a single-day extreme. Exit on RSI
overbought OR price > 5-day MA, whichever first.

Conforms to the lib s2_signal interface required by grid_search:
    precompute(ratio: pd.Series, window: int, slope_window: int = 2) -> dict
    apply_thresholds(pre, entry_thresh, exit_thresh, slope_min=0.0)
        -> (entries, exits)

Grid mapping (per-asset via grid_search):
    window       = rsi_period       (Connors uses 2; grid [2, 3])
    entry_thresh = cum_rsi entry    (grid e.g. [5, 10, 20, 30, 40, 50])
    exit_thresh  = RSI exit level   (grid e.g. [60, 70, 80])

Pooled defaults (NOT gridded — fixed in precompute):
    sum_window     = 2  (Connors canonical: cum = RSI[t-1] + RSI[t])
    ma_exit_window = 5  (5-day MA exit per High Probability ETF Trading)
    slope_window   = 2  (falling-knife guard window)

Stop loss is applied at S5 (ATR-based), not at the signal level. Excluded
from the batched Numba fast path in grid_search because the entry uses
cum_rsi (compound) while the exit uses rsi OR ratio>ma_exit (compound).

References
----------
Connors & Alvarez (2009), High Probability ETF Trading.
Quantitativo (2024), "Squeezing more profits with cumulative RSI".
Wilder (1978), New Concepts in Technical Trading Systems (RSI formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage

SUM_WINDOW_DEFAULT = 2
MA_EXIT_WINDOW_DEFAULT = 5


def _rsi_simple_ma(price: pd.Series, period: int) -> pd.Series:
    """RSI with simple rolling-mean smoothing (matches Connors / Quantitativo)."""
    delta = price.diff()
    gain = delta.clip(lower=0.0).rolling(period, min_periods=period).mean()
    loss = (-delta).clip(lower=0.0).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def precompute(
    ratio: pd.Series,
    window: int,
    slope_window: int = 2,
) -> dict:
    """Compute RSI, cum_rsi, slope of cum_rsi, and the 5-day exit MA.

    Parameters
    ----------
    ratio : pd.Series
        Close prices or ratio series.
    window : int
        RSI period. grid_search sweeps this via window_grid (e.g. [2, 3]).
    slope_window : int
        Slope filter window for falling-knife guard.
    """
    rsi = _rsi_simple_ma(ratio, window)
    cum_rsi_series = rsi.rolling(
        SUM_WINDOW_DEFAULT, min_periods=SUM_WINDOW_DEFAULT,
    ).sum()
    return {
        "cum_rsi": cum_rsi_series,
        "rsi": rsi,
        "ma_exit": ratio.rolling(
            MA_EXIT_WINDOW_DEFAULT, min_periods=MA_EXIT_WINDOW_DEFAULT,
        ).mean(),
        "ratio": ratio,
        "slope": zscore_slope(cum_rsi_series, slope_window),
    }


def apply_thresholds(
    pre: dict,
    entry_thresh: float,
    exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Entry: cum_rsi <= entry_thresh AND slope >= slope_min.
    Exit:  RSI >= exit_thresh OR ratio > 5-day MA.

    Both shifted by 1 bar (no look-ahead).
    """
    cum = pre["cum_rsi"]
    rsi = pre["rsi"]
    ratio = pre["ratio"]
    ma_exit = pre["ma_exit"]
    slope = pre["slope"]

    entries = (
        (cum <= entry_thresh) & (slope >= slope_min)
    ).shift(1, fill_value=False)
    exits = (
        (rsi >= exit_thresh) | (ratio > ma_exit)
    ).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def cum_rsi(
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float,
    slope_min: float = 0.0,
    slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Connors cumulative RSI mean-reversion signal — registered S2 entry.

    entry_thresh = cum_rsi entry level (e.g. 30; lower = stricter oversold).
    exit_thresh  = RSI exit level (e.g. 70; upper bound on bounce).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
