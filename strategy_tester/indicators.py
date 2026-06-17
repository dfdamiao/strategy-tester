"""Shared technical indicators for signal modules.

ADX: Murphy, Technical Analysis (1999) Ch.15
ATR: Wilder, New Concepts in Technical Trading Systems (1978)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(
    ratio: pd.Series, period: int = 14,
) -> pd.Series:
    """ATR from close-to-close prices (Wilder 1978).

    Uses |close[t] - close[t-1]| as True Range proxy when only
    close prices are available (no OHLC). Smoothed with Wilder's
    exponential moving average: ewm(alpha=1/period).

    Parameters
    ----------
    ratio : pd.Series
        Close prices or ratio series.
    period : int
        ATR lookback (default 14, Wilder standard).

    Returns
    -------
    pd.Series
        ATR series, NaN for first `period` bars.
    """
    tr = ratio.diff().abs()
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr


def compute_atr_ohlcv(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR with proper True Range (Wilder 1978).

    TR = max(H-L, |H-Pc|, |L-Pc|) where Pc = previous close.
    Smoothed with Wilder's EMA (alpha=1/period).

    Parameters
    ----------
    high, low, close : pd.Series
        OHLCV price series for a single ticker.
    period : int
        ATR lookback (default 14, Wilder standard).

    Returns
    -------
    pd.Series
        ATR series, NaN for first ``period`` bars.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def compute_adx_ohlcv(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX with proper directional movement (Wilder 1978, Murphy Ch.15).

    DM+ = max(H[t] - H[t-1], 0)  when  H[t]-H[t-1] > L[t-1]-L[t]
    DM- = max(L[t-1] - L[t], 0)  when  L[t-1]-L[t] > H[t]-H[t-1]

    Parameters
    ----------
    high, low, close : pd.Series
        OHLCV price series for a single ticker.
    period : int
        ADX lookback (default 14, Wilder standard).

    Returns
    -------
    pd.Series
        ADX series (0-100 scale), NaN for warmup bars.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Directional movement (Wilder/Murphy: keep only the larger)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    dm_plus = pd.Series(0.0, index=high.index)
    dm_minus = pd.Series(0.0, index=high.index)
    dm_plus[(up_move > down_move) & (up_move > 0)] = up_move
    dm_minus[(down_move > up_move) & (down_move > 0)] = down_move

    alpha = 1.0 / period
    smooth_tr = tr.ewm(alpha=alpha, min_periods=period).mean()
    smooth_dm_plus = dm_plus.ewm(alpha=alpha, min_periods=period).mean()
    smooth_dm_minus = dm_minus.ewm(alpha=alpha, min_periods=period).mean()

    di_plus = 100.0 * smooth_dm_plus / smooth_tr.replace(0.0, np.nan)
    di_minus = 100.0 * smooth_dm_minus / smooth_tr.replace(0.0, np.nan)

    di_sum = di_plus + di_minus
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum.replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, min_periods=period).mean()


def compute_adx(
    ratio: pd.Series, period: int = 14,
) -> pd.Series:
    """ADX via Wilder smoothing. Murphy (1999) Ch.15.

    Adapted for close-only data: uses close[t]-close[t-1] direction
    as proxy for directional movement (no high/low available).

    DM+ = max(close[t] - close[t-1], 0)
    DM- = max(close[t-1] - close[t], 0)
    When both are positive, keep only the larger (Murphy Ch.15 rule).

    Parameters
    ----------
    ratio : pd.Series
        Close prices or ratio series.
    period : int
        ADX lookback (default 14, Wilder standard).

    Returns
    -------
    pd.Series
        ADX series (0-100 scale), NaN for warmup bars.
    """
    diff = ratio.diff()

    # Directional movement (Murphy Ch.15: keep larger of DM+/DM-)
    dm_plus_raw = diff.clip(lower=0.0)
    dm_minus_raw = (-diff).clip(lower=0.0)

    # When both positive, keep only the larger
    both_positive = (dm_plus_raw > 0) & (dm_minus_raw > 0)
    larger_is_plus = dm_plus_raw >= dm_minus_raw
    dm_plus = dm_plus_raw.copy()
    dm_minus = dm_minus_raw.copy()
    dm_plus[both_positive & ~larger_is_plus] = 0.0
    dm_minus[both_positive & larger_is_plus] = 0.0

    # True Range proxy (close-to-close)
    tr = diff.abs()

    # Wilder smoothing (EMA with alpha=1/period)
    alpha = 1.0 / period
    smooth_tr = tr.ewm(alpha=alpha, min_periods=period).mean()
    smooth_dm_plus = dm_plus.ewm(alpha=alpha, min_periods=period).mean()
    smooth_dm_minus = dm_minus.ewm(alpha=alpha, min_periods=period).mean()

    # Directional indicators
    di_plus = 100.0 * smooth_dm_plus / smooth_tr.replace(0.0, np.nan)
    di_minus = 100.0 * smooth_dm_minus / smooth_tr.replace(0.0, np.nan)

    # DX = |DI+ - DI-| / (DI+ + DI-)
    di_sum = di_plus + di_minus
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum.replace(0.0, np.nan)

    # ADX = Wilder smooth of DX
    adx = dx.ewm(alpha=alpha, min_periods=period).mean()
    return adx
