"""Keltner Channel signal. Entry when ratio < lower band (ATR-normalized).

The Keltner channel uses an EMA midline ± k × ATR for bands, where ATR is
the Wilder average true range. Structurally identical to Bollinger (see
`bollinger.py`) but the channel width is set by realized intraday range
(ATR) instead of close-to-close standard deviation.

References:
    Chan, Algorithmic Trading (2013) Ch.2 §"Linear Mean-Reversion Strategy
        (Bollinger Bands)" p.303-323 — band-MR template. Keltner = same
        template with `std → ATR`.
    Clenow, Following the Trend (2013) Ch.5 — ATR captures realized intraday
        range, doesn't compress in low-vol regimes like close-to-close stdev.
    Sinclair, Volatility Trading §7 — hard stops hurt MR strategies on
        whipsaw days; time_stop preferred over hard ATR-stop.
    Murphy, Technical Analysis (1999) Appendix A — Keltner Channels as
        STARC bands (one-sentence reference).
    Wilder (1978) — ATR calculation with exponential smoothing.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.indicators import compute_atr
from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, ema_lookback: int, atr_lookback: int = 14,
    slope_window: int = 2,
) -> dict:
    """Expensive part: EMA midline + ATR + ATR-normalized distance. Once per pair.

    z_atr = (ratio - EMA(ratio, ema_lookback)) / ATR(ratio, atr_lookback)

    Returns dict with the precomputed arrays. The cheap part
    (`apply_thresholds`) sweeps entry/exit multiples without recomputing.
    """
    ema = ratio.ewm(span=ema_lookback, min_periods=ema_lookback, adjust=False).mean()
    atr = compute_atr(ratio, period=atr_lookback)
    z_atr = (ratio - ema) / atr.replace(0.0, float("nan"))
    return {
        "ratio": ratio, "ema": ema, "atr": atr,
        "z_atr": z_atr, "slope": zscore_slope(z_atr, slope_window),
    }


def apply_thresholds(
    pre: dict, atr_mult_entry: float, exit_mult: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: ATR-multiple thresholds + shift. Called per grid combo.

    atr_mult_entry : float
        Long entry when z_atr <= -atr_mult_entry (positive value; signal flips
        sign in the comparison). Typical range 1.5 - 3.0.
    exit_mult : float
        Exit when z_atr >= exit_mult. 0.0 = exit at EMA midline (Keltner
        canonical). 0.5 = exit when (close - EMA) / ATR >= 0.5 (slight
        overshoot above midline).
    slope_min : float
        Falling-knife guard on z_atr slope. Default 0.0 (no filter).

    NO hard ATR-stop on loss side (Sinclair §7 + TODO §A6 spec). The caller
    (state machine) implements time_stop at max_hold bars instead.
    """
    ratio = pre["ratio"]
    z_atr = pre["z_atr"]
    slope = pre["slope"]
    entries = ((z_atr <= -atr_mult_entry) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (z_atr >= exit_mult).shift(1, fill_value=False)
    # Drop the first valid value of `ratio` to keep type consistent with
    # bollinger.py contract (returns aligned Series); ratio is unused
    # downstream but retained in `pre` for transparency.
    _ = ratio  # tag as referenced for linters
    return entries, exits


@register_stage("s2_signal")
def keltner(
    ratio: pd.Series, ema_lookback: int, atr_mult_entry: float,
    exit_mult: float, atr_lookback: int = 14,
    slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Keltner Channel signal (ATR-normalized z-score from EMA midline).

    Long entry when `(close - EMA(ema_lookback)) / ATR(atr_lookback) <= -atr_mult_entry`.
    Exit when `(close - EMA) / ATR >= exit_mult` (0.0 = at midline).

    NO hard ATR-stop. Caller handles time_stop separately.
    """
    pre = precompute(ratio, ema_lookback, atr_lookback, slope_window)
    return apply_thresholds(pre, atr_mult_entry, exit_mult, slope_min)
