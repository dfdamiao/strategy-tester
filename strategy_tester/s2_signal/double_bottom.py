"""Double-Bottom reversal signal — single-asset price pattern (KILLED_ARCHETYPE probe).

Quantpedia "Technical Analysis Report Methodology + Double Bottom Country Trading
Strategy": buy a (country) ETF when price forms a double bottom — two local lows
of near-equal level separated by an interim peak — then hold a short swing.

Operationalisation (faithful but simplified for daily ETF bars):
    * A *confirmed trough* at bar i-1: close[i-1] is the minimum of the prior
      `window` bars AND close turns up (close[i] > close[i-1]).
    * A *double bottom* fires when a new confirmed trough is within `entry_thresh`
      (fractional) of the PREVIOUS confirmed trough's level, the two troughs are
      separated by >= `window` bars, and price rose at least `entry_thresh` above
      the first trough between them (a genuine interim peak / neckline).
    * Entry on the turn-up bar; exit after `exit_thresh` trading days (swing hold)
      or earlier if price prints a fresh `window`-bar high (target reached).

Conforms to the lib s2_signal contract used by the probe single path:
    fn(ratio, window, entry_thresh, exit_thresh, ...) -> (entries, exits)
`ratio` is the single-asset Close series for unit_type=single. Entries/exits are
shifted +1 (decision at close of bar t -> position at t+1; no look-ahead).

References
----------
Quantpedia, "Double Bottom Country Trading Strategy" (technical-analysis report).
Bulkowski (2005), Encyclopedia of Chart Patterns — double-bottom definition.
Edwards & Magee, Technical Analysis of Stock Trends — reversal patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage


def _double_bottom_signals(
    close: np.ndarray,
    window: int,
    max_dist: float,
    max_hold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateful detector. Returns (entries, exits) aligned to `close` (pre-shift)."""
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    last_level = np.nan
    last_idx = -(10**9)
    peak_since = -np.inf
    in_pos = False
    held = 0
    for i in range(window + 1, n):
        prev = close[i - 1]
        window_min = close[i - 1 - window : i - 1].min()
        is_trough = (prev <= window_min) and (close[i] > prev)
        if not np.isnan(last_level):
            peak_since = max(peak_since, close[i])
        if is_trough:
            if (
                not np.isnan(last_level)
                and abs(prev - last_level) / last_level <= max_dist
                and (i - 1 - last_idx) >= window
                and peak_since >= last_level * (1.0 + max_dist)
                and not in_pos
            ):
                entries[i] = True
                in_pos = True
                held = 0
            last_level = prev
            last_idx = i - 1
            peak_since = close[i]
        if in_pos:
            held += 1
            is_new_high = close[i] == close[i - window + 1 : i + 1].max()
            if held >= max_hold or is_new_high:
                exits[i] = True
                in_pos = False
                held = 0
    return entries, exits


@register_stage("s2_signal")
def double_bottom(
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float,
    slope_min: float = 0.0,  # lib contract; unused
    slope_window: int = 2,  # lib contract; unused
) -> tuple[pd.Series, pd.Series]:
    """Double-bottom reversal entry on a single Close series.

    window       = local-trough lookback (bars) AND min trough separation.
    entry_thresh = max fractional distance between the two bottoms (0.03 = 3%).
    exit_thresh  = swing hold in trading days (cast to int).
    """
    del slope_min, slope_window  # required by lib contract; unused here
    close = ratio.to_numpy(dtype=np.float64)
    e, x = _double_bottom_signals(
        close, int(window), float(entry_thresh), int(exit_thresh)
    )
    entries = pd.Series(e, index=ratio.index).shift(1, fill_value=False)
    exits = pd.Series(x, index=ratio.index).shift(1, fill_value=False)
    return entries, exits
