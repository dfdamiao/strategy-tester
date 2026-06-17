"""MA crossover signal. Entry when price/ratio > MA(n).

Signal logic (Murphy, Technical Analysis 1999 Ch.9):
    Single: Close(t) > MA(Close, n, t) → LONG at t+1 open
    Pair:   ratio(t) > MA(ratio, n, t) → LONG numerator at t+1 open

Bias guards:
    - min_periods=window prevents partial-window lookahead
    - Signals shifted +1 bar: entry/exit at next-bar open
    - entry_thresh / exit_thresh ignored (always 0 for pure MA crossover)
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: rolling MA + binary signal. Once per pair.

    For MA crossover, 'ratio' is the price series for singles
    or price_A / price_B for pairs.

    Reference: Murphy, Technical Analysis (1999) Ch.9
    """
    ma = ratio.rolling(window, min_periods=window).mean()
    signal = (ratio > ma).astype(int)
    # Slope of signal changes (diagnostic — not used for gating)
    slope = ratio.diff(slope_window) / ratio.shift(slope_window)
    return {
        "ratio": ratio, "ma": ma, "signal": signal, "slope": slope,
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: extract entry/exit from pre-computed signal.

    For MA crossover, thresholds are ignored (always 0).
    The signal is binary: above MA = long, below MA = flat.
    Next-bar execution via shift(1).
    """
    signal = pre["signal"]
    # Entry: signal flips to 1 (price crosses above MA)
    entries = (signal == 1).shift(1, fill_value=False)
    # Exit: signal flips to 0 (price crosses below MA)
    exits = (signal == 0).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def ma_crossover(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """MA crossover signal (Murphy 1999 Ch.9).

    entry_thresh/exit_thresh/slope_min are accepted for interface
    compatibility but ignored — MA crossover is a binary signal
    determined solely by price vs MA(window).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
