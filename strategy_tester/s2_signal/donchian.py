"""Donchian Channel Breakout signal (Turtle Trading System).

Entry: price breaks above N-bar highest high.
Exit: price breaks below N-bar lowest low.

entry_thresh = entry window (cast to int, e.g. 20 = Turtle System 1).
exit_thresh = exit window (cast to int, e.g. 10 = Turtle standard).

References:
    Curtis Faith, Way of the Turtle (2007) — original Turtle rules.
    Murphy, Technical Analysis (1999) Ch.9 — Donchian channels.
    Clenow, Following the Trend (2013) Ch.5 — modern implementation.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Store ratio and slope. Rolling max/min recomputed per threshold.

    Donchian's entry/exit windows vary across the grid, so the rolling
    max/min must be recomputed in apply_thresholds for each combo.
    Precompute stores ratio and diagnostic slope only.

    window is used for the slope computation (not for channels).
    """
    # Slope on ratio for falling-knife guard
    z_proxy = ratio / ratio.rolling(
        window, min_periods=window,
    ).mean() - 1.0  # Normalized deviation from MA
    return {
        "ratio": ratio,
        "slope": zscore_slope(z_proxy, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Donchian channel breakout with variable windows.

    entry_thresh = entry window (int): breakout above N-bar high.
    exit_thresh = exit window (int): breakdown below N-bar low.

    Turtle System 1: entry=20, exit=10.
    Turtle System 2: entry=55, exit=20.

    Faith (2007): entry on break of previous high, exit on break of
    previous low (shorter window for faster exit).
    """
    ratio, slope = pre["ratio"], pre["slope"]
    entry_window = max(int(entry_thresh), 5)
    exit_window = max(int(exit_thresh), 3)

    # Channel boundaries (use previous bar's high/low to avoid lookahead)
    # Murphy Ch.9: channel = highest high / lowest low over N bars
    high_n = ratio.rolling(
        entry_window, min_periods=entry_window,
    ).max().shift(1)
    low_n = ratio.rolling(
        exit_window, min_periods=exit_window,
    ).min().shift(1)

    # Entry: ratio breaks above previous N-bar high
    # Exit: ratio breaks below previous N-bar low
    # Already shifted by .shift(1) on the channel, so signal is
    # "today's close vs yesterday's channel" — no lookahead.
    # Apply one more shift for next-bar execution.
    entries = ((ratio >= high_n) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (ratio <= low_n).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def donchian(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Donchian Channel Breakout signal.

    Faith (2007) Turtle rules + Murphy Ch.9.
    entry_thresh = entry channel window (e.g. 20.0 → 20-bar high).
    exit_thresh = exit channel window (e.g. 10.0 → 10-bar low).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
