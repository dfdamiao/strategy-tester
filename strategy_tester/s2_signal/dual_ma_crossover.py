"""Dual MA crossover signal. Entry when short MA > long MA. Murphy (1999) Ch.9, Clenow (2013).

Signal logic:
    short_ma = SMA(ratio, short_window)  where short_window = max(int(window * 0.33), 5)
    long_ma  = SMA(ratio, window)
    short_ma(t) > long_ma(t) -> LONG at t+1 open

The dual crossover filters out whipsaws vs single-MA systems by requiring
agreement between fast and slow averages.

Bias guards:
    - min_periods=window prevents partial-window lookahead
    - Signals shifted +1 bar: entry/exit at next-bar open
    - entry_thresh / exit_thresh ignored (always 0 for pure MA crossover)

References:
    Murphy, Technical Analysis of the Financial Markets (1999) Ch.9
    Clenow, Following the Trend (2013)
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: dual rolling MAs + binary signal. Once per pair.

    Parameters
    ----------
    ratio : pd.Series
        Price series (single) or price_A / price_B (pairs).
    window : int
        LONG window. Short window = max(int(window * 0.33), 5).
    slope_window : int
        Bars for slope diagnostic (not used for gating).

    Reference: Murphy (1999) Ch.9, Clenow (2013)
    """
    short_window = max(int(window * 0.33), 5)
    long_ma = ratio.rolling(window, min_periods=window).mean()
    short_ma = ratio.rolling(
        short_window, min_periods=short_window,
    ).mean()
    signal = (short_ma > long_ma).astype(int)
    # Slope of ratio changes (diagnostic — not used for gating)
    slope = ratio.diff(slope_window) / ratio.shift(slope_window)
    return {
        "ratio": ratio,
        "short_ma": short_ma,
        "long_ma": long_ma,
        "short_window": short_window,
        "signal": signal,
        "slope": slope,
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: extract entry/exit from pre-computed signal.

    For dual MA crossover, thresholds are ignored (always 0).
    The signal is binary: short MA > long MA = long, else = flat.
    Next-bar execution via shift(1).
    """
    signal = pre["signal"]
    # Entry: signal flips to 1 (short MA crosses above long MA)
    entries = (signal == 1).shift(1, fill_value=False)
    # Exit: signal flips to 0 (short MA crosses below long MA)
    exits = (signal == 0).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def dual_ma_crossover(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Dual MA crossover signal (Murphy 1999, Clenow 2013).

    entry_thresh/exit_thresh/slope_min are accepted for interface
    compatibility but ignored — dual MA crossover is a binary signal
    determined solely by short_ma vs long_ma relationship.

    window = long MA period; short = max(int(window * 0.33), 5).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
