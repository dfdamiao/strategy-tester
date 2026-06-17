"""Cointegration-gated z-score signal.

Standard z-score on the ratio, but entries are masked to False
if the ADF test fails (p > 0.05), meaning the ratio is NOT
stationary and mean-reversion is unreliable.

This preserves the precompute/apply_thresholds interface by running
the ADF test in precompute and storing the p-value. apply_thresholds
then gates entries on the test result.

References:
    Chan, Algorithmic Trading (2013) Ch.7 — cointegration as
        prerequisite for pairs trading. ADF test on the spread.
    Engle & Granger (1987) — two-step cointegration test.
    Chan, Quantitative Trading (2009) Ch.7 — half-life validation.
"""
from __future__ import annotations

import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage


def _adf_pvalue(series: pd.Series) -> float:
    """ADF test p-value. Chan AT Ch.7: p < 0.05 = stationary.

    Returns 1.0 if test fails or insufficient data.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        clean = series.dropna()
        if len(clean) < 30:
            return 1.0
        result = adfuller(clean.values, autolag="AIC")
        return float(result[1])
    except Exception:
        return 1.0


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Compute z-score + ADF test once per pair.

    Chan AT Ch.7: run ADF on the ratio (or spread). If p > 0.05,
    the series is NOT stationary → mean-reversion signals unreliable.
    """
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))

    # ADF stationarity test on the ratio
    # Engle & Granger (1987): stationarity is the prerequisite
    adf_p = _adf_pvalue(ratio)

    return {
        "z": z,
        "adf_pvalue": adf_p,
        "slope": zscore_slope(z, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Z-score thresholds, gated by ADF stationarity test.

    If ADF p-value > 0.05: all entries are False (not cointegrated).
    Otherwise: standard z-score entry/exit.

    Chan AT Ch.7: only trade pairs where the spread is stationary
    (ADF p < 0.05). Non-stationary spreads will trend away.
    """
    z, slope = pre["z"], pre["slope"]
    adf_p = pre["adf_pvalue"]

    if adf_p > 0.05:
        # Not stationary — no entries allowed
        entries = pd.Series(False, index=z.index)
        exits = pd.Series(False, index=z.index)
        return entries, exits

    entries = ((z <= entry_thresh) & (slope >= slope_min)).shift(
        1, fill_value=False,
    )
    exits = (z >= exit_thresh).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def cointegration_spread(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Cointegration-gated z-score signal.

    Chan AT Ch.7 + Engle & Granger (1987): entries only allowed
    when ADF test confirms stationarity (p < 0.05).
    entry_thresh = z-score entry (e.g. -2.0).
    exit_thresh = z-score exit (e.g. 0.5).
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
