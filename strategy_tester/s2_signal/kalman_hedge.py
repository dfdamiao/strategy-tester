"""Kalman-filtered hedge for pair trading. Two variants:

1) Legacy 1D — `precompute()` / `apply_thresholds()` / `kalman_hedge()`:
   Kalman smoothing on the RATIO (A/B). Single-dimensional state. Used by
   the original kalman_hedge signal class (preserved for backward compat).

2) **New 2D — `kalman_beta_hedge()`**: canonical Halls-Moore AAT Ch.28
   implementation for catalog B2 (`engle_granger_kalman_pairs`). State =
   [β_t, α_t] evolves as random walk; observation A_t = β_t·B_t + α_t + ε_t.
   Returns time-varying β_t, α_t, and residual ε_t. Numba-compiled hot loop.

Use (2) for cointegration-with-dynamic-hedge-ratio strategies. Use (1)
only for the older single-variate ratio-smoothing approach.
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Expensive part: Kalman filter + rolling z-score. Once per pair."""
    from pykalman import KalmanFilter

    vals = ratio.values.reshape(-1, 1)
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=float(vals[0]),
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01,
    )
    state_means, _ = kf.filter(vals)
    residual = ratio.values - state_means.flatten()
    residual_series = pd.Series(residual, index=ratio.index)

    med = residual_series.rolling(window, min_periods=window).median()
    mad = (
        (residual_series - med).abs()
        .rolling(window, min_periods=window).median()
    )
    z = (residual_series - med) / (
        1.4826 * mad.replace(0.0, float("nan"))
    )
    return {"z": z, "slope": zscore_slope(z, slope_window)}


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: threshold + shift. Called per grid combo."""
    z, slope = pre["z"], pre["slope"]
    entries = ((z <= entry_thresh) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (z >= exit_thresh).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def kalman_hedge(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Kalman-filtered z-score on ratio residual.
    Uses pykalman 1D state-space model."""
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)


# ============================================================================
# 2D Kalman β hedge — canonical Halls-Moore AAT Ch.28 / catalog B2 spec
# ============================================================================
# State-space model:
#   State:        x_t = [β_t, α_t]^T,  x_t = x_{t-1} + w_t,  w_t ~ N(0, δ·I)
#   Observation:  A_t = [B_t, 1] · x_t + v_t,                v_t ~ N(0, V_eps)
#
# Initialization:  x_0 = [1, 0]  (β starts at 1, α at 0 — catalog spec)
#                  P_0 = I       (initial covariance — diffuse prior)
#
# Catalog defaults: δ = 1e-4  (transition variance — β/α drift rate)
#                   V_eps = 1e-3  (observation noise)


@numba.njit(cache=True)
def _kalman_beta_filter(
    A: np.ndarray, B: np.ndarray, delta: float, V_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba 2D Kalman filter inner loop.

    Hand-rolled to avoid pykalman's per-step Python overhead. All matrix ops
    are unrolled for the 2x2 state — no np.linalg calls in the hot loop.

    Returns (beta, alpha, residual) as length-N float64 arrays.
    """
    n = A.shape[0]
    beta = np.empty(n)
    alpha = np.empty(n)
    resid = np.empty(n)

    # Initial state x = [beta, alpha]
    x0 = 1.0
    x1 = 0.0
    # Initial covariance P (2x2 symmetric, diffuse prior)
    P00 = 1.0
    P01 = 0.0
    P10 = 0.0
    P11 = 1.0

    for t in range(n):
        # Predict step: x_pred = x (random walk identity), P_pred = P + δ·I
        Pp00 = P00 + delta
        Pp01 = P01
        Pp10 = P10
        Pp11 = P11 + delta

        # Observation matrix H = [B_t, 1]
        h0 = B[t]
        h1 = 1.0

        # Predicted observation: H · x_pred
        A_pred = h0 * x0 + h1 * x1

        # Innovation (residual) y = A_t - A_pred
        y = A[t] - A_pred

        # Innovation variance S = H · P_pred · H^T + V_eps
        # = h0·(h0·Pp00 + h1·Pp01) + h1·(h0·Pp10 + h1·Pp11) + V_eps
        S = h0 * (h0 * Pp00 + h1 * Pp01) + h1 * (h0 * Pp10 + h1 * Pp11) + V_eps

        if S <= 0.0:
            # Numerical breakdown — return what we have, mark rest as NaN
            for k in range(t, n):
                beta[k] = np.nan
                alpha[k] = np.nan
                resid[k] = np.nan
            return beta, alpha, resid

        # Kalman gain K = P_pred · H^T / S  (2x1 vector)
        K0 = (Pp00 * h0 + Pp01 * h1) / S
        K1 = (Pp10 * h0 + Pp11 * h1) / S

        # Update state: x = x_pred + K · y
        x0 = x0 + K0 * y
        x1 = x1 + K1 * y

        # Update covariance: P = (I - K·H) · P_pred
        # I - K·H = [[1 - K0·h0, -K0·h1], [-K1·h0, 1 - K1·h1]]
        a00 = 1.0 - K0 * h0
        a01 = -K0 * h1
        a10 = -K1 * h0
        a11 = 1.0 - K1 * h1

        P00 = a00 * Pp00 + a01 * Pp10
        P01 = a00 * Pp01 + a01 * Pp11
        P10 = a10 * Pp00 + a11 * Pp10
        P11 = a10 * Pp01 + a11 * Pp11

        beta[t] = x0
        alpha[t] = x1
        resid[t] = y

    return beta, alpha, resid


def kalman_beta_hedge(
    A: pd.Series,
    B: pd.Series,
    delta: float = 1.0e-4,
    V_eps: float = 1.0e-3,
) -> dict[str, pd.Series]:
    """2D Kalman filter for time-varying hedge ratio β_t and intercept α_t.

    Catalog B2 spec (Halls-Moore AAT Ch.28):
        Observation:  A_t = β_t · B_t + α_t + ε_t
        State:        x_t = [β_t, α_t]; random-walk transition with cov δ·I
        Observation noise: V_eps

    Parameters
    ----------
    A, B : pd.Series
        Numerator and denominator price series. Aligned on intersection of
        their indexes; NaN rows dropped.
    delta : float, default 1e-4
        Transition covariance (random-walk variance of β and α). Catalog default.
        Larger → β/α adapt faster; smaller → β/α more stable.
    V_eps : float, default 1e-3
        Observation noise variance. Catalog default.

    Returns
    -------
    dict with keys {"beta", "alpha", "residual"} — each a pd.Series aligned
    to the common (A, B) date index.

    Notes
    -----
    - Initial state [β_0, α_0] = [1, 0] per catalog. First ~30-50 bars are
      burn-in; downstream filters should drop them or use min_periods.
    - δ and V_eps are HELD FIXED at catalog defaults in engle_granger_kalman_pairs
      per intake.yaml::concerns.kalman_params_tunable (n_trials hygiene).
    """
    common = A.index.intersection(B.index)
    df = pd.concat([A.loc[common], B.loc[common]], axis=1).dropna()
    A_vals = np.ascontiguousarray(df.iloc[:, 0].to_numpy(), dtype=np.float64)
    B_vals = np.ascontiguousarray(df.iloc[:, 1].to_numpy(), dtype=np.float64)

    if len(A_vals) < 2:
        empty = pd.Series([], dtype=float, index=df.index)
        return {"beta": empty, "alpha": empty, "residual": empty}

    beta, alpha, resid = _kalman_beta_filter(A_vals, B_vals, delta, V_eps)

    return {
        "beta": pd.Series(beta, index=df.index, name="beta"),
        "alpha": pd.Series(alpha, index=df.index, name="alpha"),
        "residual": pd.Series(resid, index=df.index, name="residual"),
    }
