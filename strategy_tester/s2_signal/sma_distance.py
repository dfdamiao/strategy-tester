"""Distance from SMA signal. Entry when ratio more than `entry_thresh` below SMA.

Sister to `bollinger.py` per Kakushadze & Serur §4.4 formulas 154-158: Bollinger
family with σ-bands replaced by % distance bands. Simpler, parameter-stable
cousin — no rolling std, just rolling mean.

Author grounding:
  - Kakushadze & Serur §4.4 (formula 154-158): % distance bands
  - Hilpisch *Python for Algorithmic Trading* (2020) Ch.10: SMA-distance worked example
  - Faber *Quantitative Approach to TAA* SSRN 1585517 (2007): 10-month MA filter

Canonical for the `mr_distance` signal_class (introduced 2026-05-12 with the
sma200_distance_sweep A5 retest).
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


def precompute(ratio: pd.Series, window: int) -> dict:
    """Expensive part: rolling SMA + signed % distance. Once per pair."""
    sma = ratio.rolling(window, min_periods=window).mean()
    distance = ratio / sma - 1.0
    return {"ratio": ratio, "sma": sma, "distance": distance}


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: distance thresholds + lag-1 shift.

    entry_thresh: signed fraction (e.g. -0.05 = "long when 5% below SMA")
    exit_thresh : signed fraction (e.g. 0.00 = "exit when back at SMA")

    NOTE: this module provides the SIGNAL portion only. The compound
    `signal_cross_or_time_stop` exit (signal OR days_held >= max_hold) is
    enforced by the state-machine backtest in
    `docs/sma200_distance_sweep/validation/scripts/common.py` —
    the `max_hold` parameter is the caller's responsibility there.
    """
    distance = pre["distance"]
    entries = (distance < entry_thresh).shift(1, fill_value=False)
    exits = (distance > exit_thresh).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def sma_distance(
    ratio: pd.Series, window: int, entry_thresh: float, exit_thresh: float,
) -> tuple[pd.Series, pd.Series]:
    """SMA distance signal (registered for lib Pipeline use).

    entry_thresh: % distance for entry (e.g. -0.05 = -5%).
    exit_thresh : % distance for exit  (e.g.  0.00 = at SMA).
    """
    pre = precompute(ratio, window)
    return apply_thresholds(pre, entry_thresh, exit_thresh)
