"""Connors RSI(2) single-asset mean-reversion signal — A1 canonical.

Verbatim implementation of TODO_STRATEGIES.md §A1 / Connors & Alvarez 2009 Ch.7:

    Entry: RSI(2) <= entry_thresh AND Close > SMA(200) AND no_position
    Exit:  Close > SMA(5) OR RSI(2) >= exit_thresh OR days_held >= max_hold

Wilder smoothing on RSI per spec (alpha=1/period; reuses
`_compute_rsi_wilder` from `rsi_wilder.py` — see Wilder 1978).

Conforms to the lib s2_signal interface required by grid_search:
    precompute(ratio: pd.Series, window: int, ...) -> dict
    apply_thresholds(pre, entry_thresh, exit_thresh, slope_min) -> (entries, exits)

`ratio` is the single-asset Close series for unit_type=single.

Grid mapping (per-ticker via grid_search):
    window       = RSI period         (Connors fixed at 2; grid [2] only)
    entry_thresh = RSI oversold level (grid [5, 10, 15])
    exit_thresh  = RSI overbought     (grid [65, 70, 75])

Pooled defaults (NOT gridded — Connors canonical):
    sma_short = 5,  sma_long = 200,  max_hold = 10

References
----------
Connors & Alvarez (2009), Short Term Trading Strategies That Work, Ch.7.
Kakushadze & Serur (2018), section 4.4 (RSI(2) survey).
Wilder (1978), New Concepts in Technical Trading Systems (RSI formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.s2_signal.rsi_wilder import _compute_rsi_wilder

SMA_SHORT_DEFAULT = 5
SMA_LONG_DEFAULT = 200
MAX_HOLD_DEFAULT = 10


def precompute(
    ratio: pd.Series,
    window: int,
    slope_window: int = 2,  # 3rd positional per lib contract; UNUSED here (A1 has no slope filter)
    sma_short: int = SMA_SHORT_DEFAULT,
    sma_long: int = SMA_LONG_DEFAULT,
) -> dict:
    """Compute Wilder RSI + short and long SMAs of the close series.

    NOTE: 3rd positional arg is `slope_window` (lib contract — grid_search
    passes it positionally). Connors A1 has no slope filter, so it is
    accepted-but-ignored. Earlier sma_short before slope_window in this
    signature caused grid_search to silently override SMA(5) -> SMA(2),
    inflating exit frequency and destroying Sharpe (bug discovered 2026-05-09).
    """
    del slope_window  # unused; required only for lib precompute contract
    rsi = _compute_rsi_wilder(ratio, window)
    sma_s = ratio.rolling(sma_short, min_periods=sma_short).mean()
    sma_l = ratio.rolling(sma_long, min_periods=sma_long).mean()
    return {
        "rsi": rsi,
        "sma_short": sma_s,
        "sma_long": sma_l,
        "close": ratio,
    }


def _inject_max_hold(
    entries: np.ndarray,
    natural_exits: np.ndarray,
    max_hold: int,
) -> np.ndarray:
    """Force-exit max_hold bars after each entry.

    Walks the bool arrays bar-by-bar tracking position state. Once an
    entry fires while flat, the day counter starts; if no natural exit
    fires within `max_hold` bars, force exits[entry+max_hold] = True.

    Conservative: never blocks a natural exit; only adds time-stop exits.
    """
    out = natural_exits.copy()
    n = len(entries)
    in_pos = False
    days_held = 0
    for i in range(n):
        if not in_pos:
            if entries[i]:
                in_pos = True
                days_held = 0
        else:
            days_held += 1
            if natural_exits[i]:
                in_pos = False
                days_held = 0
            elif days_held >= max_hold:
                out[i] = True
                in_pos = False
                days_held = 0
    return out


def apply_thresholds(
    pre: dict,
    entry_thresh: float,
    exit_thresh: float,
    slope_min: float = 0.0,  # required positional by grid_search contract; not used (A1 has no slope filter)
    max_hold: int = MAX_HOLD_DEFAULT,
    use_trend_filter: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """A1 entry/exit signals as bool series, shifted +1 to prevent look-ahead.

    Entry: (RSI <= entry_thresh) [AND (Close > SMA(sma_long)) if use_trend_filter]
    Exit:  (Close > SMA(sma_short)) OR (RSI >= exit_thresh) OR (days_held >= max_hold)

    use_trend_filter=False is for the A1 sensitivity sweep
    (TODO_STRATEGIES.md A1: "thresholds {5,10,15} x SMA filter {none, 200}").
    """
    del slope_min  # unused; required only for lib contract

    rsi = pre["rsi"]
    close = pre["close"]
    sma_s = pre["sma_short"]
    sma_l = pre["sma_long"]

    if use_trend_filter:
        raw_entries = (rsi <= entry_thresh) & (close > sma_l)
    else:
        raw_entries = rsi <= entry_thresh
    raw_exits = (close > sma_s) | (rsi >= exit_thresh)

    entries = raw_entries.shift(1, fill_value=False)
    natural_exits = raw_exits.shift(1, fill_value=False)

    final_exits_arr = _inject_max_hold(
        entries.values.astype(bool),
        natural_exits.values.astype(bool),
        max_hold,
    )
    final_exits = pd.Series(final_exits_arr, index=entries.index)
    return entries, final_exits


@register_stage("s2_signal")
def rsi_connors(
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float,
    slope_min: float = 0.0,
    slope_window: int = 2,  # accepted (lib contract); unused (A1 has no slope filter)
    sma_short: int = SMA_SHORT_DEFAULT,
    sma_long: int = SMA_LONG_DEFAULT,
    max_hold: int = MAX_HOLD_DEFAULT,
    use_trend_filter: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Connors RSI(2) single-asset MR — registered S2 entry.

    entry_thresh     = RSI oversold (Connors A1: 10).
    exit_thresh      = RSI overbought (Connors A1: 70).
    max_hold         = trading-day force-exit guard (Connors A1: 10).
    use_trend_filter = if False, drops the SMA(200) entry gate
                       (used by the sensitivity sweep only; A1 canonical = True).
    slope_window     = accepted but unused (A1 has no slope filter; required
                       by lib's backtest_vbt_fold call contract).
    """
    del slope_window  # unused; required only for lib contract
    pre = precompute(ratio, window, sma_short=sma_short, sma_long=sma_long)
    return apply_thresholds(
        pre, entry_thresh, exit_thresh, slope_min, max_hold, use_trend_filter,
    )
