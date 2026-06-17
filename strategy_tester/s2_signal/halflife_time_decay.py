"""Z-score signal with time-decay exit based on half-life.

Standard z-score entry. Exit threshold tightens as bars_since_entry
approaches 2× halflife. Forced exit at 2× halflife.

The AR(1) half-life tells us how fast the spread mean-reverts.
If it hasn't reverted by 2× halflife, the regime likely changed.

References:
    Chan, Quantitative Trading (2009) Ch.7 p.170-172 —
        "The half-life of mean reversion is the time it takes for
        a spread to revert halfway." GLD-GDX example: ~7.8 days.
    López de Prado, MLAM (2020) p.150-152 — OTR framework:
        mean-reverting strategies need tighter profit-taking,
        wider stop-loss. Time-decay exit implements tighter PT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import zscore_slope
from strategy_tester.registry import register_stage

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _compute_halflife(ratio: pd.Series) -> float:
    """AR(1) half-life estimation. Chan QT Ch.7.

    spread[t] = c + phi * spread[t-1] + epsilon
    halflife = -ln(2) / ln(phi)

    Returns NaN if phi >= 1 (not mean-reverting) or < 2 days.
    """
    y = ratio.values
    y_lag = np.roll(y, 1)
    y_lag[0] = np.nan
    mask = ~(np.isnan(y) | np.isnan(y_lag))
    if mask.sum() < 30:
        return float("nan")

    y_clean = y[mask]
    y_lag_clean = y_lag[mask]
    delta = y_clean - y_lag_clean

    # OLS: delta = alpha + beta * y_lag
    x = np.column_stack([np.ones(len(y_lag_clean)), y_lag_clean])
    try:
        beta = np.linalg.lstsq(x, delta, rcond=None)[0][1]
    except np.linalg.LinAlgError:
        return float("nan")

    if beta >= 0:
        return float("nan")  # Not mean-reverting

    halflife = -np.log(2) / beta
    return halflife if halflife >= 2.0 else float("nan")


if HAS_NUMBA:
    @numba.njit(cache=True)
    def _apply_time_decay_numba(
        z_arr: np.ndarray,
        raw_entries: np.ndarray,
        raw_exits: np.ndarray,
        exit_thresh: float,
        halflife: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba loop: track bars_since_entry, tighten exit, force exit.

        Chan Ch.7: if not reverted by 2× halflife, regime changed.
        LdP MLAM: tighter PT for mean-reverting strategies.
        """
        n = len(z_arr)
        entries = np.zeros(n, dtype=numba.boolean)
        exits = np.zeros(n, dtype=numba.boolean)
        in_position = False
        entry_bar = 0
        max_hold = int(2.0 * halflife)

        for t in range(n):
            if raw_entries[t] and not in_position:
                entries[t] = True
                in_position = True
                entry_bar = t

            if in_position and t > entry_bar:
                bars = t - entry_bar
                # Linear decay: exit_adj = exit_thresh at entry,
                # tightens to exit_thresh * 0.1 at 2× halflife
                decay = max(1.0 - bars / max_hold, 0.1)
                adj_exit = exit_thresh * decay

                if z_arr[t] >= adj_exit or bars >= max_hold:
                    exits[t] = True
                    in_position = False

                if raw_exits[t]:
                    exits[t] = True
                    in_position = False

        return entries, exits
else:
    def _apply_time_decay_numba(
        z_arr: np.ndarray,
        raw_entries: np.ndarray,
        raw_exits: np.ndarray,
        exit_thresh: float,
        halflife: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pure Python fallback (slow)."""
        n = len(z_arr)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        in_position = False
        entry_bar = 0
        max_hold = int(2.0 * halflife)

        for t in range(n):
            if raw_entries[t] and not in_position:
                entries[t] = True
                in_position = True
                entry_bar = t

            if in_position and t > entry_bar:
                bars = t - entry_bar
                decay = max(1.0 - bars / max_hold, 0.1)
                adj_exit = exit_thresh * decay

                if z_arr[t] >= adj_exit or bars >= max_hold:
                    exits[t] = True
                    in_position = False

                if raw_exits[t]:
                    exits[t] = True
                    in_position = False

        return entries, exits


def precompute(
    ratio: pd.Series, window: int, slope_window: int = 2,
) -> dict:
    """Compute z-score + AR(1) halflife once per pair."""
    mean = ratio.rolling(window, min_periods=window).mean()
    std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - mean) / std.replace(0.0, float("nan"))
    halflife = _compute_halflife(ratio)

    return {
        "z": z,
        "halflife": halflife,
        "slope": zscore_slope(z, slope_window),
    }


def apply_thresholds(
    pre: dict, entry_thresh: float, exit_thresh: float,
    slope_min: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Z-score entry with time-decay exit.

    Entry: standard z-score (z <= entry_thresh) with slope guard.
    Exit: tightening threshold + forced exit at 2× halflife.

    If halflife is NaN (not mean-reverting), falls back to standard
    z-score exit without time decay.
    """
    z, slope = pre["z"], pre["slope"]
    halflife = pre["halflife"]

    # Standard z-score entry
    raw_entries = (
        (z <= entry_thresh) & (slope >= slope_min)
    ).fillna(False)

    # Standard z-score exit (fallback)
    raw_exits = (z >= exit_thresh).fillna(False)

    if np.isnan(halflife) or halflife < 2.0:
        # No valid halflife — use standard z-score exit
        entries = raw_entries.shift(1, fill_value=False)
        exits = raw_exits.shift(1, fill_value=False)
        return entries, exits

    # Time-decay exit via Numba loop
    z_arr = z.fillna(0.0).values
    entry_arr = raw_entries.values.astype(bool)
    exit_arr = raw_exits.values.astype(bool)

    entries_np, exits_np = _apply_time_decay_numba(
        z_arr, entry_arr, exit_arr, exit_thresh, halflife,
    )

    entries = pd.Series(entries_np, index=z.index).shift(
        1, fill_value=False,
    )
    exits = pd.Series(exits_np, index=z.index).shift(
        1, fill_value=False,
    )
    return entries, exits


@register_stage("s2_signal")
def halflife_time_decay(
    ratio: pd.Series, window: int, entry_thresh: float,
    exit_thresh: float, slope_min: float = 0.0, slope_window: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Z-score with time-decay exit based on AR(1) half-life.

    Chan QT Ch.7 + LdP MLAM: exit tightens as trade ages,
    forced exit at 2× halflife.
    entry_thresh = z-score entry (e.g. -2.0).
    exit_thresh = base z-score exit (e.g. 0.5), decays over time.
    """
    pre = precompute(ratio, window, slope_window)
    return apply_thresholds(pre, entry_thresh, exit_thresh, slope_min)
