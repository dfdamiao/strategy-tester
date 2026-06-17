"""High-minus-spread band mean-reversion signal (hl_spread_band_sweep, A10).

Asymmetric band trigger isolated from the Quantitativo "2.11 Sharpe MR" rule
set. The lower band is anchored on a recent rolling high (NOT the mean) and
offset by a Parkinson high-minus-low spread (NOT close-only sigma, NOT ATR):

    HH(t)     = rolling_max(close, N)[t-1]
    SPREAD(t) = rolling_mean(high - low, M)[t-1]
    band(t)   = HH(t) - k * SPREAD(t)

    entry(t)  : close[t] < band(t)
    exit(t)   : close[t] > high[t-1]
                OR  (R > 0 AND close[t] < rolling_mean(close, R)[t])

R = 0 disables the regime stop. R in {200, 300} engages a long-trend SMA
filter that closes positions when the asset itself drops below its own slow
SMA (Murphy TA Ch.10 regime overlay).

References:
    Sinclair, E. (2013) Volatility Trading, §1.2 — Parkinson H-L estimator
        (~5x more efficient than close-only sigma).
    Quantitativo (2024) "A Mean Reversion Strategy with 2.11 Sharpe" — band
        rule isolated from the IBS leg.
    Murphy, J.J. (1999) Technical Analysis, Ch.10 — regime SMA overlay.

Sister-sweep convention: signal_class is labeled `mr_zscore` in
strategy_shape.yaml as the category label only. The downstream port from
williams_r_sweep calls this kernel directly with OHLC arrays; there is no
precompute / apply_thresholds split because all four sweep params (k, N, M, R)
participate in the band, leaving no cheap inner threshold sweep to factor.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
def compute_hl_spread_band(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    k: float,
    N: int,
    M: int,
    R: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (entry_mask, exit_signal_mask) bool arrays of shape (n,).

    All rolling references on the entry side use prior-bar data (t-1) so the
    signal at bar t does not leak today's close into today's band. The exit
    side mixes a t-1 reference (high[t-1]) with a t reference (regime SMA at
    t); the regime SMA includes today's close, matching the close-of-bar exit
    convention of the sister sweeps (williams_r, sma200_distance, stochastic_k).

    Args:
        close: 1-D close-price array.
        high:  1-D high-price array (same length as close).
        low:   1-D low-price array  (same length as close).
        k:     band multiplier on the H-L spread.
        N:     rolling-max window for the close reference.
        M:     rolling-mean window for the H-L spread.
        R:     regime-SMA window. R = 0 disables the regime exit.

    Returns:
        (entry, exit_signal) — bool ndarrays, length n. Warmup bars where the
        required rolling window is not yet full are left False.
    """
    n = len(close)
    entry = np.zeros(n, dtype=np.bool_)
    exit_signal = np.zeros(n, dtype=np.bool_)

    spread = np.empty(n, dtype=np.float64)
    for i in range(n):
        spread[i] = high[i] - low[i]

    entry_warmup = max(N, M)
    for i in range(1, n):
        if i >= entry_warmup:
            hh = close[i - N]
            for j in range(i - N + 1, i):
                if close[j] > hh:
                    hh = close[j]
            sp_sum = 0.0
            for j in range(i - M, i):
                sp_sum += spread[j]
            band = hh - k * (sp_sum / M)
            if close[i] < band:
                entry[i] = True

        cross_exit = close[i] > high[i - 1]
        regime_exit = False
        if R > 0 and i >= R - 1:
            c_sum = 0.0
            for j in range(i - R + 1, i + 1):
                c_sum += close[j]
            if close[i] < c_sum / R:
                regime_exit = True
        if cross_exit or regime_exit:
            exit_signal[i] = True

    return entry, exit_signal
