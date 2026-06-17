"""EMA crossover signal. Entry when price/ratio > EMA(n). Murphy (1999) Ch.9.

Signal logic (Murphy, Technical Analysis 1999 Ch.9):
    Single: Close(t) > EMA(Close, n, t) -> LONG at t+1 open
    Pair:   ratio(t) > EMA(ratio, n, t) -> LONG numerator at t+1 open

EMA weights recent prices exponentially, reacting faster than SMA.

Bias guards:
    - min_periods=window prevents partial-window lookahead
    - Signals shifted +1 bar: entry/exit at next-bar open
    - entry_thresh / exit_thresh ignored (always 0 for pure EMA crossover)
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: EMA + binary signal. Once per pair.

    For EMA crossover, 'ratio' is the price series for singles
    or price_A / price_B for pairs.

    Reference: Murphy, Technical Analysis (1999) Ch.9
    """
    ema = ratio.ewm(span=window, min_periods=window).mean()
    signal = (ratio > ema).astype(int)
    # Slope of ratio changes (diagnostic — not used for gating)
    slope = ratio.diff(slope_window) / ratio.shift(slope_window)
    return {
        "ratio": ratio, "ema": ema, "signal": signal, "slope": slope,
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: extract entry/exit from pre-computed signal.

    For EMA crossover, thresholds are ignored (always 0).
    The signal is binary: above EMA = long, below EMA = flat.
    Next-bar execution via shift(1).
    """
    signal = pre["signal"]
    # Entry: signal flips to 1 (price crosses above EMA)
    entries = (signal == 1).shift(1, fill_value=False)
    # Exit: signal flips to 0 (price crosses below EMA)
    exits = (signal == 0).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def ema_crossover(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """EMA crossover signal (Murphy 1999 Ch.9).

    entry_thresh/exit_thresh/slope_min are accepted for interface
    compatibility but ignored — EMA crossover is a binary signal
    determined solely by price vs EMA(window).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
