"""Volatility-adaptive z-score signal.

Standard z-score with entry/exit thresholds scaled by the current
volatility percentile. Low vol → tighter thresholds (faster mean
reversion expected). High vol → wider thresholds (more noise).

References:
    Sinclair, Volatility Trading 2e (2013) Ch.7 — vol mean-reverts;
        low-vol periods have fastest mean-reversion.
    Carver, Systematic Trading (2015) Ch.10 — vol-adjusted triggers
        are key to multi-asset robustness.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Compute z-score + volatility percentile once per pair.

    vol_pctile = rolling 252-day percentile rank of realized vol.
    Range [0, 1]: 0 = lowest vol in past year, 1 = highest.
    """
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))

    # Realized volatility and its percentile rank over 1 year
    # Sinclair Ch.7: vol is more predictable than price direction
    realized_vol = ratio.pct_change().rolling(
        window, min_periods=window,
    ).std()
    vol_pctile = realized_vol.rolling(
        252, min_periods=window,
    ).rank(pct=True)

    return {
        "z": z,
        "vol_pctile": vol_pctile,
        "slope": zscore_slope(z, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Scale thresholds by vol percentile, then apply.

    Scaling: adj_thresh = base_thresh * (1 + 0.5 * (vol_pctile - 0.5))
    At median vol (pctile=0.5): adj = base (no change).
    At low vol (pctile=0.1): adj = base * 0.8 (tighter → enter sooner).
    At high vol (pctile=0.9): adj = base * 1.2 (wider → more room).

    Carver Ch.10: vol-adjusted triggers prevent over-trading in calm
    markets and under-trading in volatile ones.
    """
    z, slope = pre["z"], pre["slope"]
    vol_pctile = pre["vol_pctile"].fillna(0.5)

    scale = 1.0 + 0.5 * (vol_pctile - 0.5)
    adj_entry = entry_thresh * scale
    adj_exit = exit_thresh * scale

    entries = ((z <= adj_entry) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (z >= adj_exit).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def vol_adaptive_zscore(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Vol-adaptive z-score signal.

    Sinclair Ch.7 + Carver Ch.10: thresholds scale with vol percentile.
    entry_thresh = base z-score for entry (e.g. -2.0).
    exit_thresh = base z-score for exit (e.g. 0.5).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
