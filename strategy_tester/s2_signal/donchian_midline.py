"""Donchian Channel Midline Reversion signal (A7 — inverted Turtle).

Entry: price at the bottom of its N-bar range (within `entry_thresh` fraction).
Exit: price recovers to `exit_thresh` fraction from the bottom (0.50 = midline).

Unlike the classic Donchian breakout (donchian.py), this fades the channel
touch rather than trading the channel break.

entry_thresh = fraction from bottom at entry (e.g. 0.10 → within 10% of LL).
exit_thresh  = fraction from bottom at exit (e.g. 0.50 = midline, 0.75 = ¾ up).

References:
    Curtis Faith, Way of the Turtle (2007) — Donchian channel structure.
    Clenow, Following the Trend (2013) §4 — channel as trend/MR tool.
    Chan, Algorithmic Trading (2013) Ch.2 — band-MR template.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.registry import register_stage


def precompute(ratio: pd.Series, lookback: int) -> dict:
    """Rolling max (HH), min (LL), midline, width. Once per pair per outer combo."""
    hh = ratio.rolling(lookback, min_periods=lookback).max()
    ll = ratio.rolling(lookback, min_periods=lookback).min()
    width = hh - ll
    mid = (hh + ll) / 2.0
    return {"ratio": ratio, "hh": hh, "ll": ll, "mid": mid, "width": width}


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: channel-position entry/exit. Called per grid combo.

    entry_thresh = fraction from bottom (e.g. 0.10 → within 10% of LL).
    exit_thresh  = fraction from bottom at exit (0.50 = midline, 0.75 = ¾ up).
    """
    ratio, ll, width = pre["ratio"], pre["ll"], pre["width"]
    entry_level = ll + entry_thresh * width
    exit_level = ll + exit_thresh * width
    entries = (ratio < entry_level).shift(1, fill_value=False)
    exits = (ratio >= exit_level).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def donchian_midline(
    ratio: pd.Series, lookback: int,
    entry_thresh: float, exit_thresh: float,
) -> tuple[pd.Series, pd.Series]:
    """Donchian Channel Midline Reversion signal.

    entry_thresh = fraction from bottom (0.10 = within 10% of N-bar low).
    exit_thresh  = fraction from bottom at exit (0.50 = midline, 0.75 = ¾ up).
    """
    pre = precompute(ratio, lookback)
    return apply_thresholds(pre, entry_thresh, exit_thresh)
