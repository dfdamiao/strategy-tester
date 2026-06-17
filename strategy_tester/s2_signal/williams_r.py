"""Williams %R extreme reversion signal (A8 — TODO_STRATEGIES.md §A8).

Williams %R is a re-scaled stochastic %K oscillator over an N-bar range:

    %R = -100 × (HH − Close) / (HH − LL)        # range [-100, 0]

Entry when %R < entry_threshold (deeply oversold; default -90).
Exit when %R > exit_threshold (recovery to mid-range; default -50).

References:
    Williams, Larry (1973) — original publication.
    Murphy, J.J. (1999) Technical Analysis of the Financial Markets, Ch.5.

Murphy himself flags %R as "Redundant if using Stochastic" — %R = -%K - 100.
A8 (this module) and A9 (Stochastic %K reversion) are mathematically equivalent;
the threshold crossings differ only by a constant shift. If A8 ships, A9 should
be retired from the TODO queue.

Sister-sweep convention: signal_class is labeled `mr_zscore` even though %R is
not literally a z-score — the contract with lib/s2_signal + s3_validation is
the same (precompute + apply_thresholds two-phase interface).
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage


@nb.njit(cache=True)
def compute_williams_r(price: np.ndarray, lookback: int) -> np.ndarray:
    """Williams %R over an N-bar range, computed from Close-only.

    Standard %R uses High/Low/Close, but on daily-close inputs the rolling max
    and min of Close approximate HH/LL within a 1-bar lag. Sister sweeps A4/A6/A7
    use Close-only on the same cohort for parity; A8 follows suit.

    Returns ndarray of shape (n,), NaN for warmup bars i < lookback - 1.
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback - 1, n):
        start = i - lookback + 1
        hi = price[start]
        lo = price[start]
        for j in range(start + 1, i + 1):
            if price[j] > hi:
                hi = price[j]
            if price[j] < lo:
                lo = price[j]
        rng = hi - lo
        if rng > 0:
            out[i] = -100.0 * (hi - price[i]) / rng
    return out


def precompute(ratio: pd.Series, lookback: int) -> dict:
    """Compute %R once per unit per outer combo (cheap on njit).

    Returns dict with `ratio` (passthrough, for downstream return computation)
    and `wr` (the %R array). Sister-sweep precompute convention.
    """
    wr = compute_williams_r(ratio.to_numpy(), lookback)
    return {"ratio": ratio, "wr": pd.Series(wr, index=ratio.index)}


def apply_thresholds(
    pre: dict, entry_threshold: float, exit_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Cheap part: threshold-crossing entry/exit. Called per inner grid combo.

    entry_threshold: %R level for entry (e.g. -90 → enter when %R < -90).
    exit_threshold:  %R level for exit (e.g. -50 → exit when %R > -50).

    Bar-shifted (shift +1) so no look-ahead.
    """
    wr = pre["wr"]
    entries = (wr < entry_threshold).shift(1, fill_value=False)
    exits = (wr > exit_threshold).shift(1, fill_value=False)
    return entries, exits


@register_stage("s2_signal")
def williams_r(
    ratio: pd.Series, lookback: int,
    entry_threshold: float, exit_threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Williams %R Extreme Reversion signal (A8).

    entry_threshold: %R level for entry (default -90).
    exit_threshold:  %R level for exit (default -50).
    """
    pre = precompute(ratio, lookback)
    return apply_thresholds(pre, entry_threshold, exit_threshold)
