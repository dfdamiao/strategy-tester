"""Bollinger Bands signal. Entry when ratio < lower band."""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.backtest.vbt_runner import zscore_slope


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: rolling mean/std + slope. Once per pair."""
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))
    return {
        "ratio": ratio, "mean": mean, "std": std,
        "z": z, "slope": zscore_slope(z, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: band thresholds + shift. Called per grid combo."""
    ratio, mean, std = pre["ratio"], pre["mean"], pre["std"]
    slope = pre["slope"]
    lower = mean + entry_thresh * std
    upper = mean + exit_thresh * std
    entries = ((ratio <= lower) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (ratio >= upper).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def bollinger(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Bollinger Bands signal.
    entry_thresh = number of sigma for lower band (e.g. -2.0).
    exit_thresh = 0.0 means mid band (mean)."""
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
