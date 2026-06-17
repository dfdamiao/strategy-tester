"""KAMA crossover signal. Entry when price/ratio > KAMA(n). Kaufman TSM 6e Ch.17.

Signal logic (Kaufman, Trading Systems and Methods 6e Ch.17):
    KAMA adapts its smoothing speed via the Efficiency Ratio (ER):
        ER = |direction| / volatility
        SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
        KAMA[t] = KAMA[t-1] + SC[t] * (price[t] - KAMA[t-1])

    When price is trending (high ER), KAMA tracks closely (fast).
    When price is noisy (low ER), KAMA smooths heavily (slow).

    Single: Close(t) > KAMA(Close, n, t) -> LONG at t+1 open
    Pair:   ratio(t) > KAMA(ratio, n, t) -> LONG numerator at t+1 open

Bias guards:
    - First `window` bars are NaN (warmup)
    - Signals shifted +1 bar: entry/exit at next-bar open
    - entry_thresh / exit_thresh ignored (always 0 for pure KAMA crossover)
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage


@numba.njit(cache=True)
def _kama_core(
    prices: np.ndarray,
    period: int,
    fast_sc: float,
    slow_sc: float,
) -> np.ndarray:
    """Compute KAMA values. Kaufman TSM 6e Ch.17.

    Returns
    -------
    kama : np.ndarray
        KAMA values (NaN for first `period` bars).
    """
    n = len(prices)
    kama = np.full(n, np.nan)

    if n <= period:
        return kama

    # Initialize KAMA at the first valid bar
    kama[period] = prices[period]

    for i in range(period, n):
        # Direction = |price[i] - price[i-period]|
        direction = abs(prices[i] - prices[i - period])

        # Volatility = sum of |price[j] - price[j-1]| over period
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(prices[j] - prices[j - 1])

        # Efficiency Ratio
        if volatility > 0:
            er = direction / volatility
        else:
            er = 0.0

        # Smoothing Constant: SC = [ER * (fast - slow) + slow]^2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # KAMA update
        if i == period:
            kama[i] = prices[i]
        else:
            kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])

    return kama


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: KAMA + binary signal. Once per pair.

    Parameters
    ----------
    ratio : pd.Series
        Price series (single) or price_A / price_B (pairs).
    window : int
        KAMA lookback period for Efficiency Ratio.
    slope_window : int
        Bars for slope diagnostic (not used for gating).

    Reference: Kaufman, Trading Systems and Methods 6e Ch.17
    """
    vals = ratio.values.astype(np.float64)
    fast_sc = 2.0 / (2 + 1)    # fast EMA constant (period=2)
    slow_sc = 2.0 / (30 + 1)   # slow EMA constant (period=30)

    kama_arr = _kama_core(vals, window, fast_sc, slow_sc)
    kama = pd.Series(kama_arr, index=ratio.index, name="kama")

    signal = (ratio > kama).astype(int)
    # NaN warmup bars -> 0 (no signal)
    signal = signal.fillna(0).astype(int)

    # Slope of ratio changes (diagnostic — not used for gating)
    slope = ratio.diff(slope_window) / ratio.shift(slope_window)

    return {
        "ratio": ratio, "kama": kama, "signal": signal, "slope": slope,
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: extract entry/exit from pre-computed signal.

    For KAMA crossover, thresholds are ignored (always 0).
    The signal is binary: above KAMA = long, below KAMA = flat.
    Next-bar execution via shift(1).
    """
    signal = pre["signal"]
    # Entry: signal flips to 1 (price crosses above KAMA)
    entries = (signal == 1).shift(1, fill_value=False)
    # Exit: signal flips to 0 (price crosses below KAMA)
    exits = (signal == 0).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def kama_crossover(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """KAMA crossover signal (Kaufman TSM 6e Ch.17).

    entry_thresh/exit_thresh/slope_min are accepted for interface
    compatibility but ignored — KAMA crossover is a binary signal
    determined solely by price vs KAMA(window).

    KAMA adapts speed via Efficiency Ratio: trending markets get fast
    response, choppy markets get heavy smoothing.
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
