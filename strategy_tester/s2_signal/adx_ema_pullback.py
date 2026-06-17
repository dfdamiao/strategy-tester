"""Raschke & Connors Holy Grail signal: ADX-confirmed trend, EMA pullback entry.

Long-only single-asset trend-continuation signal. ADX gates "is there a strong
trend"; EMA pullback gates the entry timing; Raschke 1-2-3 swing high gates
the exit.

Conforms to the lib s2_signal interface required by grid_search:
    precompute(ratio: pd.Series, window: int, slope_window: int = 2) -> dict
    apply_thresholds(pre, entry_thresh, exit_thresh, slope_min=0.0)
        -> (entries, exits)

Grid mapping (per-asset via grid_search):
    window       = ema_period (Raschke uses 20; grid [13, 20, 34])
    entry_thresh = adx_thresh (Raschke uses 30; grid [25, 30, 35, 40])
    exit_thresh  = unused (always 0; exits are structural via swing detection)

Pooled defaults (NOT gridded — fixed in precompute):
    adx_period         = 14  (Wilder canonical)
    pullback_lookback  = 5   (bars within which the EMA touch must occur)
    swing_lookback     = 3   (bars used to confirm the 1-2-3 high)
    slope_window       = 2   (compatibility; not used by this signal)

Close-only design notes:
    - True ADX needs OHLC, but the lib's grid_search interface passes only a
      single Series (`ratio`). We use the close-only `compute_adx(ratio)` from
      lib/indicators.py (matches `adx_regime_gate.py` precedent — same
      simplification).
    - 1-2-3 swing detected on close, not high. Minor accuracy loss vs Raschke's
      original (intraday wicks not captured), acceptable for daily-bar ETFs.
    - Stop-loss is applied at S5 (ATR(14) * k_stop below the pullback low),
      not at the signal level.

Excluded from the batched Numba fast path in grid_search because entry uses
ADX gate AND pullback re-cross (compound), and exit uses 1-2-3 swing pattern
(not a simple threshold).

Long-only direction. Raschke's original is L/S; long-only constraint applied
upfront, no shorts proposed.

References
----------
Raschke & Connors (1995), Street Smarts: High Probability Short-Term Trading
    Strategies, ch. "Holy Grail".
Quantitativo (2024), "The Holy Grail still works".
Wilder (1978), New Concepts in Technical Trading Systems (ADX, EMA formulas).
Murphy (1999), Technical Analysis of the Financial Markets ch.15 (ADX).
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.indicators import compute_adx
from strategy_tester.registry import register_stage

ADX_PERIOD_DEFAULT = 14
PULLBACK_LOOKBACK_DEFAULT = 5
SWING_LOOKBACK_DEFAULT = 3


def _ema(price: pd.Series, period: int) -> pd.Series:
    """Exponential moving average matching Raschke / Wilder convention."""
    return price.ewm(span=period, adjust=False, min_periods=period).mean()


def _pullback_recross(
    ratio: pd.Series, ema: pd.Series, lookback: int,
) -> pd.Series:
    """True on bars where ratio just crossed back above the EMA after touching
    or being below it within the last `lookback` bars.

    Conditions for trigger at bar t:
        1. ratio[t] > ema[t]                 (above now)
        2. ratio[t-1] <= ema[t-1]            (was at-or-below previous bar)
        3. min(ratio[t-lookback..t]) <= ema  (touched or dipped below recently)
    """
    above_now = ratio > ema
    above_prev = ratio.shift(1) <= ema.shift(1)
    touched_recently = (ratio <= ema).rolling(
        lookback, min_periods=1,
    ).max().astype(bool)
    return above_now & above_prev & touched_recently


def _swing_high_exit(close: pd.Series, swing_lookback: int) -> pd.Series:
    """Raschke 1-2-3 swing-high exit (close-based simplification).

    Trigger at bar t: close[t] < close[t-1] AND there were
    `swing_lookback - 1` higher-closes in a row immediately before bar t-1.

    Example with swing_lookback=3: HH, HH, then a lower close on bar t.
    """
    rises = (close > close.shift(1))
    consecutive_rises = rises.rolling(
        swing_lookback - 1, min_periods=swing_lookback - 1,
    ).sum()
    is_lower_close = close < close.shift(1)
    return is_lower_close & (consecutive_rises.shift(1) >= swing_lookback - 1)


def precompute(
    ratio: pd.Series,
    window: int,
    slope_window: int = 2,
) -> dict:
    """Compute ADX, EMA, and signal helpers.

    Parameters
    ----------
    ratio : pd.Series
        Close prices or ratio series.
    window : int
        EMA period. grid_search sweeps via window_grid (e.g. [13, 20, 34]).
    slope_window : int
        Unused by this signal but kept for interface compatibility.
    """
    adx = compute_adx(ratio, period=ADX_PERIOD_DEFAULT)
    ema = _ema(ratio, window)
    return {
        "adx": adx,
        "ema": ema,
        "ratio": ratio,
        "slope": zscore_slope(adx, slope_window),
    }


def apply_thresholds(
    pre: dict,
    entry_thresh: float,
    exit_thresh: float = 0.0,  # unused; kept for interface compat
    slope_min: float = 0.0,    # unused; kept for interface compat
) -> tuple[pd.Series, pd.Series]:
    """Entry: ADX > entry_thresh AND EMA pullback re-cross within last K bars.
    Exit:  Raschke 1-2-3 swing-high (close-based).

    Both shifted by 1 bar (no look-ahead). exit_thresh and slope_min are
    accepted to satisfy the grid_search call signature but ignored — the exit
    is structural, not threshold-based.
    """
    del exit_thresh, slope_min  # interface-compat only
    adx = pre["adx"]
    ratio = pre["ratio"]
    ema = pre["ema"]

    pullback = _pullback_recross(ratio, ema, PULLBACK_LOOKBACK_DEFAULT)
    entries = ((adx > entry_thresh) & pullback).shift(1, fill_value=False)
    exits = _swing_high_exit(ratio, SWING_LOOKBACK_DEFAULT).shift(
        1, fill_value=False,
    )
    return entries, exits


@register_stage("s2_signal")
def adx_ema_pullback(
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float = 0.0,
    slope_min: float = 0.0,
    slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Raschke Holy Grail trend-continuation signal — registered S2 entry.

    entry_thresh = ADX threshold (e.g. 30; Raschke canonical).
    exit_thresh  = unused (Raschke 1-2-3 swing exit is structural).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
