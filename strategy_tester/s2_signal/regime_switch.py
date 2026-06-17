"""ADX-based regime switching signal.

Switches between mean-reversion (z-score) and trend-following (MA)
based on ADX level:
- ADX > 25: trending regime → use MA crossover signal
- ADX < 20: ranging regime → use z-score mean-reversion
- Between 20-25: no signal (ambiguous regime)

FIX from legacy: vol detection now compares vol-to-vol (rolling
percentile rank), not vol-to-mean-returns (legacy bug in
enhanced_regime_switching.py:61).

References:
    Narang, Inside the Black Box (2011) Ch.10 — regime detection,
        adaptive strategy selection based on market state.
    Murphy, Technical Analysis (1999) Ch.15 — ADX interpretation.
    Jansen, ML for Algo Trading 2e (2020) Ch.9 — regime-based
        strategy switching.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.indicators import compute_adx
from strategy_tester.registry import register_stage

# ADX regime thresholds (Murphy Ch.15)
_ADX_TREND = 25  # ADX above this = trending
_ADX_RANGE = 20  # ADX below this = ranging


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Compute z-score + MA signal + ADX once per pair.

    Both MR and trend signals are precomputed. apply_thresholds
    selects which one to use based on ADX at each bar.
    """
    # Z-score for mean-reversion mode
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))

    # MA signal for trend mode (binary: price > SMA)
    ma_signal = (ratio > mean).astype(int)

    # ADX for regime detection — Murphy Ch.15
    adx = compute_adx(ratio, period=14)

    return {
        "z": z,
        "ma_signal": ma_signal,
        "adx": adx,
        "slope": zscore_slope(z, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Switch between MR and trend signals based on ADX.

    Narang Ch.10: matching trade logic to regime reduces losses
    during regime flips.

    MR mode (ADX < 20): z <= entry_thresh → entry, z >= exit_thresh → exit
    Trend mode (ADX > 25): ma_signal flips 0→1 → entry, 1→0 → exit
    Ambiguous (20-25): no signal
    """
    z, slope, adx = pre["z"], pre["slope"], pre["adx"]
    ma_signal = pre["ma_signal"]
    adx_filled = adx.fillna(22.5)  # Ambiguous during warmup

    # Regime masks
    is_trending = adx_filled > _ADX_TREND
    is_ranging = adx_filled < _ADX_RANGE

    # MR entries/exits (z-score, only in ranging regime)
    mr_entries = (z <= entry_thresh) & (slope >= slope_min) & is_ranging
    mr_exits = (z >= exit_thresh) & is_ranging

    # Trend entries/exits (MA crossover, only in trending regime)
    ma_entry = (ma_signal == 1) & (ma_signal.shift(1) == 0) & is_trending
    ma_exit = (ma_signal == 0) & (ma_signal.shift(1) == 1) & is_trending

    # Combine: regime determines which signal fires
    entries = (mr_entries | ma_entry).shift(1, fill_value=False)
    exits = (mr_exits | ma_exit).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def regime_switch(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """ADX-based regime switching signal.

    Narang Ch.10 + Murphy Ch.15 + Jansen Ch.9.
    MR mode (ADX<20): entry_thresh/exit_thresh are z-score levels.
    Trend mode (ADX>25): MA crossover (thresholds ignored).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
