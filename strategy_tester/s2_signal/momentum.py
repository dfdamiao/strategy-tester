"""Momentum signal. Entry when N-bar return > 0.

Signal logic (time-series momentum):
    returns(t) = (price(t) - price(t-n)) / price(t-n)
    returns(t) > 0 -> LONG at t+1 open

Pure time-series momentum: assets that have risen over the lookback
window tend to continue rising.  Applied per-asset (not cross-sectional).

Bias guards:
    - pct_change(window) naturally requires `window` bars of history
    - Signals shifted +1 bar: entry/exit at next-bar open
    - entry_thresh / exit_thresh ignored (always 0 for pure momentum)

References:
    Moskowitz, Ooi & Pedersen, "Time Series Momentum" (JFE, 2012)
    Antonacci, Dual Momentum Investing (2014)
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: N-bar returns + binary signal. Once per pair.

    Parameters
    ----------
    ratio : pd.Series
        Price series (single) or price_A / price_B (pairs).
    window : int
        Lookback period for momentum return.
    slope_window : int
        Bars for slope diagnostic (not used for gating).

    Reference: Moskowitz, Ooi & Pedersen (JFE 2012)
    """
    returns = ratio.pct_change(window)
    signal = (returns > 0).astype(int)
    # NaN warmup bars -> 0 (no signal)
    signal = signal.fillna(0).astype(int)
    # Slope of ratio changes (diagnostic — not used for gating)
    slope = ratio.diff(slope_window) / ratio.shift(slope_window)
    return {
        "ratio": ratio, "returns": returns, "signal": signal,
        "slope": slope,
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: extract entry/exit from pre-computed signal.

    For momentum, thresholds are ignored (always 0).
    The signal is binary: positive N-bar return = long, else = flat.
    Next-bar execution via shift(1).
    """
    signal = pre["signal"]
    # Entry: signal flips to 1 (N-bar return turns positive)
    entries = (signal == 1).shift(1, fill_value=False)
    # Exit: signal flips to 0 (N-bar return turns non-positive)
    exits = (signal == 0).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def momentum(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Momentum signal (Moskowitz, Ooi & Pedersen 2012).

    entry_thresh/exit_thresh/slope_min are accepted for interface
    compatibility but ignored — momentum is a binary signal
    determined solely by the sign of the N-bar return.
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
