"""Stochastic %K extreme reversion signal (A9 — stochastic_k_sweep).

    %K = 100 × (Close − LL[k_period]) / (HH[k_period] − LL[k_period])   # range [0, 100]
    smoothed_K = SMA(%K, k_smooth)   # k_smooth=1 → raw %K (no smoothing)

Entry when smoothed_K < entry_threshold (oversold).
Exit when smoothed_K > exit_threshold (recovery to mid-range).

Sister-sweep convention: signal_class is labeled `mr_zscore` even though %K is
not literally a z-score — the precompute + apply_thresholds two-phase interface
is the same contract used by all s2_signal modules.

Note: raw %K (k_smooth=1) is a sign-and-shift of Williams %R: %K = 100 + %R.
k_smooth > 1 ("Slow Stochastic") produces a genuinely different signal by
dampening whipsaw; this is the primary differentiation vs A8.

References:
    Lane, George (1984) — original %K/%D publication.
    Murphy, J.J. (1999) Technical Analysis of the Financial Markets §3.4.

Close-only approximation: uses rolling max/min of Close as HH/LL proxy.
Same convention as A4/A6/A7/A8 sisters for parity on the shared cohort.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage


@nb.njit(cache=True)
def _compute_stochastic_k_raw(price: np.ndarray, k_period: int) -> np.ndarray:
    """Raw %K over a k_period-bar range, Close-only approximation.

    Returns ndarray of shape (n,), NaN for warmup bars i < k_period - 1.
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        start = i - k_period + 1
        hi = price[start]
        lo = price[start]
        for j in range(start + 1, i + 1):
            if price[j] > hi:
                hi = price[j]
            if price[j] < lo:
                lo = price[j]
        rng = hi - lo
        if rng > 0:
            out[i] = 100.0 * (price[i] - lo) / rng
    return out


@nb.njit(cache=True)
def _apply_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with NaN passthrough for warmup bars."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        start = max(0, i - window + 1)
        total = 0.0
        count = 0
        for j in range(start, i + 1):
            if not np.isnan(arr[j]):
                total += arr[j]
                count += 1
        if count == window:
            out[i] = total / window
    return out


def precompute(ratio: pd.Series, k_period: int, k_smooth: int = 1) -> dict:
    """Compute %K (and optional SMA smoothing) once per unit per outer combo.

    k_smooth=1 produces raw %K; k_smooth=3 produces classic Slow Stochastic.
    Returns dict with `ratio` (passthrough) and `sk` (the smoothed %K Series).
    """
    raw_k = _compute_stochastic_k_raw(ratio.to_numpy(), k_period)
    if k_smooth > 1:
        smoothed = _apply_sma(raw_k, k_smooth)
    else:
        smoothed = raw_k
    return {"ratio": ratio, "sk": pd.Series(smoothed, index=ratio.index)}


def apply_thresholds(
    pre: dict,
    entry_threshold: float,
    exit_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: threshold-crossing entry/exit. Called per inner grid combo.

    entry_threshold: %K level for long entry (e.g. 20 → enter when smoothed_K < 20).
    exit_threshold:  %K level for exit    (e.g. 70 → exit  when smoothed_K > 70).

    Bar-shifted (shift +1) so no look-ahead.
    """
    sk = pre["sk"]
    entries = (sk < entry_threshold).shift(1, fill_value=False)
    exits = (sk > exit_threshold).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def stochastic_k(
    ratio: pd.Series,
    k_period: int,
    k_smooth: int,
    entry_threshold: float,
    exit_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K Extreme Reversion signal (A9).

    entry_threshold: smoothed %K level for entry (default 20).
    exit_threshold:  smoothed %K level for exit (default 70).
    k_smooth=1 → raw %K; k_smooth=3 → Slow Stochastic.
    """
    pre = precompute(ratio, k_period, k_smooth)
    return apply_thresholds(pre, entry_threshold, exit_threshold)
