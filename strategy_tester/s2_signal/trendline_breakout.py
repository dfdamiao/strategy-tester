"""Trendline breakout signal (Murphy 1999 Ch. trendlines).

Entry (long numerator) when the ratio closes above a detected resistance
trendline for ``entry_conf`` consecutive bars. Exit when the ratio closes
below a detected support trendline for ``exit_conf`` consecutive bars, OR
when a numerator-based ATR trailing stop fires (Wilder 1978, k × ATR(14)).

Key rules:
    - Only ONE open trade per pair at a time. A new entry signal that fires
      while a position is already open is ignored.
    - Streak counters are per-line: each detected trendline tracks its own
      consecutive-breach count independently. First line to reach
      ``entry_conf`` bars wins.
    - Next-bar execution: entries/exits are shifted +1 bar so fills occur at
      next-bar open, not the confirmation bar's close.
    - Detection runs on whatever window ``precompute`` is called with — no
      lookahead. For walk-forward / CPCV, call ``precompute`` per fold.

Trendline detector is fixed-parameter (see spec
``docs/superpowers/specs/2026-04-17-ratio-breakout-refactor-design.md``):
    min_points=4, max_error_pct=2.0, min_period_days=21.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_tester.detectors import detect_resistance, detect_support
from strategy_tester.indicators import compute_atr_ohlcv
from strategy_tester.registry import register_stage

DEFAULT_MIN_POINTS = 4
DEFAULT_MAX_ERROR_PCT = 2.0
DEFAULT_MIN_PERIOD_DAYS = 21
DEFAULT_ATR_PERIOD = 14


def _ratio_ohlc(
    num: pd.DataFrame, den: pd.DataFrame,
) -> pd.DataFrame:
    """Elementwise ratio OHLC on the shared date index.

    ratio_O = num_O / den_O, etc. Assumes ``num`` and ``den`` share an index;
    caller aligns them.
    """
    idx = num.index.intersection(den.index)
    n = num.loc[idx]
    d = den.loc[idx]
    return pd.DataFrame(
        {
            "Open": n["Open"].to_numpy() / d["Open"].to_numpy(),
            "High": n["High"].to_numpy() / d["Low"].to_numpy(),   # max ratio
            "Low": n["Low"].to_numpy() / d["High"].to_numpy(),    # min ratio
            "Close": n["Close"].to_numpy() / d["Close"].to_numpy(),
        },
        index=idx,
    )


def _line_projection_matrix(
    lines: list, n_bars: int,
) -> np.ndarray:
    """Build a (n_bars, n_lines) float32 matrix of projected line prices.

    For each line l and each bar k > line.end_idx, value = slope*k + intercept.
    Before end_idx (line not yet "alive"), value = NaN. This lets the streak
    walker do a vectorised compare on each bar.
    """
    n_lines = len(lines)
    if n_lines == 0:
        return np.empty((n_bars, 0), dtype=np.float32)
    proj = np.full((n_bars, n_lines), np.nan, dtype=np.float32)
    for j, line in enumerate(lines):
        start = line.end_idx + 1
        if start >= n_bars:
            continue
        k = np.arange(start, n_bars, dtype=np.float32)
        proj[start:, j] = line.slope * k + line.intercept
    return proj


def precompute(
    num_ohlc: pd.DataFrame,
    den_ohlc: pd.DataFrame,
    *,
    min_points: int = DEFAULT_MIN_POINTS,
    max_error_pct: float = DEFAULT_MAX_ERROR_PCT,
    min_period_days: int = DEFAULT_MIN_PERIOD_DAYS,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> dict:
    """Expensive work: trendline detection + ATR, once per pair per fold.

    Parameters
    ----------
    num_ohlc, den_ohlc : pd.DataFrame
        Numerator and denominator OHLC. Must have columns
        ``[Open, High, Low, Close]`` and a shared DatetimeIndex.
    min_points, max_error_pct, min_period_days : trendline detector params
        Fixed for production (see spec). Exposed for debugging only.
    atr_period : int
        ATR lookback for the trailing stop. Default 14 (Wilder 1978).

    Returns
    -------
    dict with keys:
        ``index``: aligned DatetimeIndex
        ``ratio_close``: float64 array — ratio closes on aligned index
        ``num_close``: float64 array — numerator closes (for trailing stop)
        ``num_atr``: float64 array — numerator ATR(atr_period)
        ``resistance_proj``: (n_bars, n_r_lines) float32 — projected prices
        ``support_proj``: (n_bars, n_s_lines) float32 — projected prices
    """
    ratio = _ratio_ohlc(num_ohlc, den_ohlc)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < 100:
        return _empty_pre(ratio.index)

    num_aligned = num_ohlc.loc[ratio.index]

    r_lines = detect_resistance(
        ratio,
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        n_cores=1,
    )
    s_lines = detect_support(
        ratio,
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        n_cores=1,
    )

    n_bars = len(ratio)
    return {
        "index": ratio.index,
        "ratio_close": ratio["Close"].to_numpy(dtype=np.float64),
        "num_close": num_aligned["Close"].to_numpy(dtype=np.float64),
        "num_atr": compute_atr_ohlcv(
            num_aligned["High"],
            num_aligned["Low"],
            num_aligned["Close"],
            period=atr_period,
        ).to_numpy(dtype=np.float64),
        "resistance_proj": _line_projection_matrix(r_lines, n_bars),
        "support_proj": _line_projection_matrix(s_lines, n_bars),
        "n_resistance": len(r_lines),
        "n_support": len(s_lines),
    }


def _empty_pre(index: pd.Index) -> dict:
    n = len(index)
    return {
        "index": index,
        "ratio_close": np.zeros(n, dtype=np.float64),
        "num_close": np.zeros(n, dtype=np.float64),
        "num_atr": np.zeros(n, dtype=np.float64),
        "resistance_proj": np.empty((n, 0), dtype=np.float32),
        "support_proj": np.empty((n, 0), dtype=np.float32),
        "n_resistance": 0,
        "n_support": 0,
    }


def _walk_signals(
    ratio_close: np.ndarray,
    num_close: np.ndarray,
    num_atr: np.ndarray,
    resistance_proj: np.ndarray,  # (n, n_r) float32
    support_proj: np.ndarray,     # (n, n_s) float32
    entry_conf: int,
    exit_conf: int,
    atr_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python reference walker. See grid_breakout for Numba version.

    Implements:
        - per-line streak counters (reset on bars that don't confirm)
        - one-trade-at-a-time: entry signals ignored while in position
        - ATR trailing stop: stop ratchets up, never down
        - exit on support-streak OR ATR-stop-hit, whichever fires first
        - next-bar execution via shift(1): entry/exit bar t marks confirmation,
          the actual fill happens on bar t+1 open (handled by the caller or
          VectorBT by convention)

    Returns ``(entries, exits)`` as bool arrays indexed on the confirmation bar.
    """
    n = len(ratio_close)
    entries = np.zeros(n, dtype=np.bool_)
    exits = np.zeros(n, dtype=np.bool_)

    n_r = resistance_proj.shape[1]
    n_s = support_proj.shape[1]
    r_streak = np.zeros(n_r, dtype=np.int32)
    s_streak = np.zeros(n_s, dtype=np.int32)

    in_position = False
    trail_stop = -np.inf  # numerator price

    for t in range(n):
        # --- update resistance streaks (entry candidate) ---
        entry_fires = False
        for j in range(n_r):
            proj = resistance_proj[t, j]
            if np.isnan(proj):
                r_streak[j] = 0
                continue
            if ratio_close[t] > proj:
                r_streak[j] += 1
                if r_streak[j] >= entry_conf:
                    entry_fires = True
            else:
                r_streak[j] = 0

        # --- update support streaks (exit candidate) ---
        support_exit_fires = False
        for j in range(n_s):
            proj = support_proj[t, j]
            if np.isnan(proj):
                s_streak[j] = 0
                continue
            if ratio_close[t] < proj:
                s_streak[j] += 1
                if s_streak[j] >= exit_conf:
                    support_exit_fires = True
            else:
                s_streak[j] = 0

        # --- trailing stop update (only meaningful when in position) ---
        if in_position:
            atr_t = num_atr[t]
            if not np.isnan(atr_t):
                new_trail = num_close[t] - atr_k * atr_t
                if new_trail > trail_stop:
                    trail_stop = new_trail

        atr_exit_fires = in_position and num_close[t] <= trail_stop

        # --- resolve (one-trade-at-a-time) ---
        if in_position:
            if support_exit_fires or atr_exit_fires:
                exits[t] = True
                in_position = False
                trail_stop = -np.inf
        else:
            if entry_fires:
                entries[t] = True
                in_position = True
                atr_t = num_atr[t]
                if not np.isnan(atr_t):
                    trail_stop = num_close[t] - atr_k * atr_t
                else:
                    trail_stop = -np.inf

    return entries, exits


def apply_thresholds(
    pre: dict,
    entry_conf: int,
    exit_conf: int,
    atr_k: float,
) -> tuple[pd.Series, pd.Series]:
    """Cheap per-combo sweep: walk precomputed lines + ATR, return entries/exits.

    Parameters
    ----------
    pre : dict from :func:`precompute`
    entry_conf : int — consecutive bars above resistance to confirm entry
    exit_conf : int — consecutive bars below support to confirm exit
    atr_k : float — ATR multiplier for trailing stop (e.g. 2.0)

    Returns
    -------
    (entries, exits) : tuple[pd.Series, pd.Series]
        Boolean series on the precomputed index. Next-bar execution via
        ``shift(1)``; fills occur at next-bar open by convention of the
        downstream VectorBT backtest.
    """
    idx = pre["index"]
    entries_arr, exits_arr = _walk_signals(
        pre["ratio_close"],
        pre["num_close"],
        pre["num_atr"],
        pre["resistance_proj"],
        pre["support_proj"],
        int(entry_conf),
        int(exit_conf),
        float(atr_k),
    )
    entries = pd.Series(entries_arr, index=idx).shift(1, fill_value=False).astype(bool)
    exits = pd.Series(exits_arr, index=idx).shift(1, fill_value=False).astype(bool)
    return entries, exits


@register_stage("s2_signal")
def trendline_breakout(
    num_ohlc: pd.DataFrame,
    den_ohlc: pd.DataFrame,
    entry_conf: int,
    exit_conf: int,
    atr_k: float,
    *,
    min_points: int = DEFAULT_MIN_POINTS,
    max_error_pct: float = DEFAULT_MAX_ERROR_PCT,
    min_period_days: int = DEFAULT_MIN_PERIOD_DAYS,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> tuple[pd.Series, pd.Series]:
    """One-shot helper: precompute + apply_thresholds.

    Use this for ad-hoc calls. For grid search, call :func:`precompute` once
    per pair and :func:`apply_thresholds` per combo.
    """
    pre = precompute(
        num_ohlc, den_ohlc,
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        atr_period=atr_period,
    )
    return apply_thresholds(pre, entry_conf, exit_conf, atr_k)
