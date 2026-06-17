"""Standard z-score signal (mean + std)."""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.backtest.vbt_runner import zscore_slope


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: compute z + slope once per pair."""
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))
    return {"z": z, "slope": zscore_slope(z, slope_window)}


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: threshold + shift. Called per grid combo."""
    z, slope = pre["z"], pre["slope"]
    entries = ((z <= entry_thresh) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (z >= exit_thresh).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def zscore_standard(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Generate entry/exit signals from standard z-score (mean/std)."""
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
