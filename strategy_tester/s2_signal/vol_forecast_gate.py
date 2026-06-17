"""Volatility-forecast gated z-score signal.

Standard z-score signal, but entries are only allowed when the
vol forecast percentile is in the bottom quartile (low vol =
faster mean reversion). Exits also trigger on vol spikes.

References:
    Sinclair, Volatility Trading 2e (2013) Ch.7 — vol mean-reverts
        strongly (finding 6.1); low-vol periods have highest
        mean-reversion profitability.
    Jansen, ML for Algo Trading 2e (2020) Ch.9 — EWMA vol forecast
        is predictive of next-period realized vol.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage

# Vol gate constants
# Sinclair Ch.7: trade when vol is cheap (bottom quartile)
_VOL_ENTRY_PCTILE = 0.30
# Exit on vol spike (top quartile) — position becomes risky
_VOL_EXIT_PCTILE = 0.75


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Compute z-score + EWMA vol forecast + vol percentile.

    Vol forecast blends short-term EWMA with long-term mean:
        vol_forecast = 0.7 × EWMA(window) + 0.3 × rolling_mean(252)
    Jansen Ch.9: EWMA blend captures both recent dynamics and
    long-term average for vol prediction.
    """
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))

    # Realized vol
    realized_vol = ratio.pct_change().rolling(
        window, min_periods=window,
    ).std()

    # EWMA vol forecast (Jansen Ch.9: blend short and long term)
    vol_ewma = realized_vol.ewm(span=window).mean()
    vol_mean_long = realized_vol.rolling(
        252, min_periods=window,
    ).mean()
    vol_forecast = 0.7 * vol_ewma + 0.3 * vol_mean_long.fillna(vol_ewma)

    # Vol percentile rank over past year
    vol_pctile = vol_forecast.rolling(
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
    """Z-score thresholds gated by vol percentile.

    Entries: z <= entry_thresh AND vol_pctile <= 0.30
        (Sinclair Ch.7: only trade when vol is low — faster reversion)
    Exits: z >= exit_thresh OR vol_pctile >= 0.75
        (exit on z-score reversion OR vol spike)
    """
    z, slope = pre["z"], pre["slope"]
    vol_pctile = pre["vol_pctile"].fillna(0.5)

    entries = (
        (z <= entry_thresh)
        & (slope >= slope_min)
        & (vol_pctile <= _VOL_ENTRY_PCTILE)
    ).shift(1, fill_value=False)

    exits = (
        (z >= exit_thresh) | (vol_pctile >= _VOL_EXIT_PCTILE)
    ).shift(1, fill_value=False)

    return entries, exits


@register_stage("s2_signal")
def vol_forecast_gate(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Vol-forecast gated z-score signal.

    Sinclair Ch.7 + Jansen Ch.9: only enter when vol forecast is in
    bottom quartile. Exit on z-score reversion or vol spike.
    entry_thresh = z-score entry level (e.g. -2.0).
    exit_thresh = z-score exit level (e.g. 0.5).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
