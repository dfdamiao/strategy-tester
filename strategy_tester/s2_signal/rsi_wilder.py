"""RSI signal using Wilder's exponential smoothing (true RSI).

Entry when RSI drops below oversold threshold (mean-reversion).
Exit when RSI rises above overbought threshold.

Wilder's smoothing: ewm(alpha=1/period) — equivalent to the original
recursive formula: avg_gain[t] = (avg_gain[t-1] * (n-1) + gain[t]) / n.

Reference: Wilder, New Concepts in Technical Trading Systems (1978)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage


def _compute_rsi_wilder(
    ratio: pd.Series, window: int,
) -> pd.Series:
    """RSI with Wilder's exponential smoothing.

    Wilder's smoothing uses alpha = 1/period, which is equivalent to:
        avg_gain[t] = (avg_gain[t-1] * (period-1) + gain[t]) / period

    This produces a smoother, less noisy RSI than simple rolling mean.
    """
    delta = ratio.diff()
    gain = delta.clip(lower=0.0).ewm(
        alpha=1.0 / window, min_periods=window,
    ).mean()
    loss = (-delta).clip(lower=0.0).ewm(
        alpha=1.0 / window, min_periods=window,
    ).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: compute RSI + slope once per pair."""
    rsi = _compute_rsi_wilder(ratio, window)
    return {"rsi": rsi, "slope": zscore_slope(rsi, slope_window)}


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: threshold + shift. Called per grid combo.

    entry_thresh = oversold level (e.g. 30). Entry when RSI <= entry_thresh.
    exit_thresh = overbought level (e.g. 70). Exit when RSI >= exit_thresh.
    """
    rsi, slope = pre["rsi"], pre["slope"]
    entries = ((rsi <= entry_thresh) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (rsi >= exit_thresh).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def rsi_wilder(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """RSI mean-reversion signal (Wilder's exponential smoothing).

    entry_thresh = oversold level (e.g. 30).
    exit_thresh = overbought level (e.g. 70).
    Wilder (1978): true RSI with alpha=1/period EMA.
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
