"""ATR-normalized pullback signal.

Entry when ratio pulls back from recent high by more than
entry_thresh × ATR (mean-reversion on oversold pullback).
Exit when pullback narrows to exit_thresh × ATR.

References:
    Clenow, Following the Trend (2013) Ch.5 — ATR-normalized entries
        on equities/futures, "directly applicable to ETFs" (p.237).
    Murphy, Technical Analysis (1999) Ch.9 — pullback entry patterns.
    Wilder (1978) — ATR calculation with exponential smoothing.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.indicators import compute_atr
from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Compute ATR, rolling max, and pullback depth once per pair.

    pullback_depth = (rolling_max - ratio) / ATR
    High pullback_depth = deep oversold relative to volatility.
    """
    atr = compute_atr(ratio, period=14)
    rolling_max = ratio.rolling(window, min_periods=window).max()

    # Pullback depth normalized by ATR
    # Clenow Ch.5: ATR normalizes across all volatility regimes
    pullback_depth = (rolling_max - ratio) / atr.replace(
        0.0, float("nan"),
    )

    return {
        "pullback": pullback_depth,
        "slope": zscore_slope(pullback_depth, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Entry on deep pullback, exit when pullback narrows.

    entry_thresh = ATR multiples for entry (e.g. 2.0 = 2×ATR pullback).
    exit_thresh = ATR multiples for exit (e.g. 0.5 = pullback narrowed).

    Clenow Ch.5: ATR-based entries adapt automatically to vol regime.
    Murphy Ch.9: pullback-to-support is classic mean-reversion entry.
    """
    pullback, slope = pre["pullback"], pre["slope"]

    entries = ((pullback >= entry_thresh) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (pullback <= exit_thresh).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def atr_pullback(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """ATR-normalized pullback signal.

    Clenow Ch.5 + Murphy Ch.9.
    entry_thresh = min ATR multiples for entry (e.g. 2.0).
    exit_thresh = max ATR multiples for exit (e.g. 0.5).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
