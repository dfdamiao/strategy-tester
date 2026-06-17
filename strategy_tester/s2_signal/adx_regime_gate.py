"""ADX-gated z-score signal.

Standard z-score entry with exit threshold adjusted by ADX:
- ADX > 40 (strong trend): widen exit → hold position longer
  (mean-reversion should ride the reversal further).
- ADX < 20 (range-bound): tighten exit → quick take-profit
  (choppy markets = faster exits).
- Between 20-40: standard exit threshold.

References:
    Narang, Inside the Black Box (2011) Ch.10 — regime change detection.
    Murphy, Technical Analysis (1999) Ch.15 — ADX interpretation:
        ADX > 40 = strong trend, ADX < 20 = no trend.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.indicators import compute_adx
from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Compute z-score + ADX(14) once per pair."""
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))

    # ADX(14) — Murphy Ch.15 standard period
    adx = compute_adx(ratio, period=14)

    return {
        "z": z,
        "adx": adx,
        "slope": zscore_slope(z, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Z-score entry; exit threshold scales with ADX.

    ADX > 40: exit_adj = exit_thresh * 1.5 (wider, hold longer)
    ADX < 20: exit_adj = exit_thresh * 0.5 (tighter, quick profit)
    Otherwise: exit_adj = exit_thresh (standard)

    Murphy Ch.15: ADX measures trend strength, not direction.
    Narang Ch.10: adapting to regime improves risk-adjusted returns.
    """
    z, slope, adx = pre["z"], pre["slope"], pre["adx"]
    adx_filled = adx.fillna(30.0)  # Default to mid-range during warmup

    # Scale exit threshold by ADX regime
    exit_scale = pd.Series(1.0, index=z.index)
    exit_scale = exit_scale.where(adx_filled <= 40, 1.5)
    exit_scale = exit_scale.where(adx_filled >= 20, 0.5)
    adj_exit = exit_thresh * exit_scale

    entries = ((z <= entry_thresh) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (z >= adj_exit).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def adx_regime_gate(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """ADX-gated z-score signal.

    Narang Ch.10 + Murphy Ch.15: exit threshold adapts to trend strength.
    entry_thresh = z-score entry level (e.g. -2.0).
    exit_thresh = base z-score exit level (e.g. 0.5), scaled by ADX.
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
