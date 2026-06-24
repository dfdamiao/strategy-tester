"""N-day-low bounce ("Double 7's") mean-reversion signal — single-asset.

Covers the Connors/Alvarez "Double 7's" and "5-day-low" family in the
quantocracy KILLED_ARCHETYPE pile: while above a long trend filter, buy when the
close prints a new N-day low; exit when it prints a new N-day high.

Operationalisation (daily bars, long-only):
    Entry: close == rolling-min over `window` bars AND close > SMA(`entry_thresh`).
    Exit:  close == rolling-max over `window` bars, or `exit_thresh` days elapsed.

Conforms to the lib s2_signal contract used by the probe single path:
    fn(ratio, window, entry_thresh, exit_thresh, ...) -> (entries, exits)
Entries/exits shifted +1 (decision at close of t -> position at t+1; no look-ahead).

References
----------
Connors & Alvarez (2009), Short Term Trading Strategies That Work — "Double 7's".
Alvarez Quant Trading, "Double 7's Strategy"; "A SPY Setup Suggesting Upside Edge".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage


def _max_hold_exits(
    entries: np.ndarray, natural: np.ndarray, max_hold: int
) -> np.ndarray:
    """Force-exit max_hold bars after entry; never blocks a natural exit."""
    out = natural.copy()
    in_pos = False
    held = 0
    for i in range(len(entries)):
        if not in_pos:
            if entries[i]:
                in_pos = True
                held = 0
        else:
            held += 1
            if natural[i]:
                in_pos = False
                held = 0
            elif held >= max_hold:
                out[i] = True
                in_pos = False
                held = 0
    return out


@register_stage("s2_signal")
def nday_low(
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float,
    slope_min: float = 0.0,  # lib contract; unused
    slope_window: int = 2,  # lib contract; unused
) -> tuple[pd.Series, pd.Series]:
    """Double-7's N-day-low bounce on a single Close series.

    window       = N-day low/high lookback (Connors Double-7's: 7).
    entry_thresh = SMA trend-filter length in bars (200 = above 200-day MA).
    exit_thresh  = max hold in trading days (cast to int).
    """
    del slope_min, slope_window  # required by lib contract; unused
    w = int(window)
    nlow = ratio.rolling(w, min_periods=w).min()
    nhigh = ratio.rolling(w, min_periods=w).max()
    sma = ratio.rolling(int(entry_thresh), min_periods=int(entry_thresh)).mean()

    raw_entry = (ratio <= nlow) & (ratio > sma)
    raw_exit = ratio >= nhigh

    entries = raw_entry.shift(1, fill_value=False)
    natural = raw_exit.shift(1, fill_value=False)
    final_exits = _max_hold_exits(
        entries.to_numpy(dtype=bool), natural.to_numpy(dtype=bool), int(exit_thresh)
    )
    return entries, pd.Series(final_exits, index=ratio.index)
