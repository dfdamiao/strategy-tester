"""Consecutive-down-days mean-reversion signal — single-asset (KILLED_ARCHETYPE probe).

Covers the short-horizon "buy after N down closes" family in the quantocracy kill
pile: "A simple statistical edge in SPY" (3+ down days, exit next close),
"Three-day Pullback into Turnaround Tuesday", and "Monday's Strong Selling"
bounce setups. All reduce to: after `window` consecutive down closes while the
asset is still above its long trend, buy the bounce; exit on the first up close
or after a short hold.

Operationalisation (daily bars, long-only):
    Entry: `window` consecutive closes with daily return < `entry_thresh`
           AND close > SMA(200) (long-term-uptrend filter).
    Exit:  first up close after entry, or `exit_thresh` trading days elapsed.

Conforms to the lib s2_signal contract used by the probe single path:
    fn(ratio, window, entry_thresh, exit_thresh, ...) -> (entries, exits)
Entries/exits shifted +1 (decision at close of t -> position at t+1; no look-ahead).

References
----------
"A Simple Statistical Edge in SPY" (Trading with Python) — 3-down-days bounce.
Connors & Alvarez (2009), Short Term Trading Strategies That Work — pullback MR.
Quantifiable Edges — Turnaround-Tuesday / consecutive-lower-close setups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage

SMA_LONG_DEFAULT = 200


def _consec_count(down: np.ndarray) -> np.ndarray:
    """Running count of consecutive True values (resets to 0 on False)."""
    out = np.zeros(len(down), dtype=np.int64)
    run = 0
    for i in range(len(down)):
        run = run + 1 if down[i] else 0
        out[i] = run
    return out


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
def consec_down(
    ratio: pd.Series,
    window: int,
    entry_thresh: float,
    exit_thresh: float,
    slope_min: float = 0.0,  # lib contract; unused
    slope_window: int = 2,  # lib contract; unused
    sma_long: int = SMA_LONG_DEFAULT,
) -> tuple[pd.Series, pd.Series]:
    """Consecutive-down-days bounce on a single Close series.

    window       = required consecutive down closes (e.g. 3).
    entry_thresh = daily-return ceiling that counts as a "down" close
                   (0.0 = any down close; -0.005 = down by >0.5%).
    exit_thresh  = max hold in trading days (cast to int).
    """
    del slope_min, slope_window  # required by lib contract; unused
    ret = ratio.pct_change().to_numpy(dtype=np.float64)
    down = np.zeros(len(ret), dtype=bool)
    np.less(ret, float(entry_thresh), out=down, where=~np.isnan(ret))
    streak = _consec_count(down)
    sma_l = (
        ratio.rolling(sma_long, min_periods=sma_long).mean().to_numpy(dtype=np.float64)
    )
    close = ratio.to_numpy(dtype=np.float64)

    raw_entry = (streak >= int(window)) & (close > sma_l)
    raw_exit = ret > 0.0  # first up close mean-reverts the pullback

    entries = pd.Series(raw_entry, index=ratio.index).shift(1, fill_value=False)
    natural = pd.Series(raw_exit, index=ratio.index).shift(1, fill_value=False)
    final_exits = _max_hold_exits(
        entries.to_numpy(dtype=bool), natural.to_numpy(dtype=bool), int(exit_thresh)
    )
    return entries, pd.Series(final_exits, index=ratio.index)
