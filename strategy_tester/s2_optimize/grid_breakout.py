"""Exhaustive grid search for ratio-breakout S2 parameters.

Grid: entry_conf ∈ {3..7} × exit_conf ∈ {3..7} × atr_k ∈ {1.5, 2.0, 2.5, 3.0, 3.5}
(125 combos per pair).

Pipeline per pair:
    1. Build ratio OHLC from numerator + denominator OHLC.
    2. 80/20 IS/OOS split on the shared date index.
    3. ``precompute`` on the IS window only — detect trendlines, ATR(14) on
       numerator. No lookahead.
    4. Numba-JIT'd walker sweeps 125 combos, returns Sharpe / CAGR / trade
       count / max DD per combo.
    5. Pick best combo by penalized IS Sharpe (penalty = sqrt(1 - p/n_trades)
       with p=3 params); reject if IS Sharpe < min_is_sharpe or IS trades <
       min_is_trades.
    6. Evaluate the chosen combo on the OOS window — re-detect trendlines on
       OOS-only bars (zero lookahead). Return S2_REQUIRED row.

Parallel across pairs with ``mp.get_context("fork") + Pool(initializer=...)``
— each worker loads its slice of the OHLC dict once, then runs 125 combos on
many pairs.

References:
    Pardo (2008) EOTS 2e — exhaustive grid when feasible
    Chan (2008) QT — per-asset optimization
    Murphy (1999) Ch. trendlines — breakout signal basis
    Wilder (1978) — ATR trailing stop
    Bailey & LdP (2012) — PSR / IS Sharpe floor
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time as _time
from datetime import datetime

import numba
import numpy as np
import pandas as pd

from strategy_tester.registry import register_stage
from strategy_tester.s2_signal.trendline_breakout import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_MAX_ERROR_PCT,
    DEFAULT_MIN_PERIOD_DAYS,
    DEFAULT_MIN_POINTS,
    _line_projection_matrix,
)
from strategy_tester.detectors import detect_resistance, detect_support

ANNUALIZE = 252

# Default grid — matches production: fixed stop-loss on numerator.
# stop_pct=0.0 means "no stop" (Murphy pure signal); 0.10, 0.15, 0.20 are
# the disaster-only stops from ratio_breakout_strategy/backtest_config.py
# (OLD used 0.08 as a default — we replace with 0.10).
DEFAULT_ENTRY_CONFS = (3, 4, 5, 6, 7)
DEFAULT_EXIT_CONFS = (3, 4, 5, 6, 7)
DEFAULT_STOP_PCTS = (0.0, 0.10, 0.15, 0.20)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


# ---------------------------------------------------------------------------
# Numba walker — runs one (entry_conf, exit_conf, atr_k) combo on one window
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _walk_and_score(
    ratio_close: np.ndarray,
    num_close: np.ndarray,
    resistance_proj: np.ndarray,  # (n, n_r) float32; NaN before line is live
    support_proj: np.ndarray,     # (n, n_s) float32
    entry_conf: int,
    exit_conf: int,
    stop_pct: float,              # fixed stop-loss fraction (0.0 = no stop)
    stop_slippage_bps: float,     # slippage applied on stop fills (e.g. 5.0 = 5 bps)
    tx_cost_per_side: float,      # round-trip cost deducted on exit (e.g. 0.001 = 10 bps/side)
):
    """Run streak walker + compute Sharpe / CAGR / max DD / trade count.

    Matches OLD production semantics (ratio_breakout_strategy/):
        - Signal fires on confirmation bar (streak == conf). Entry at close.
        - Signal exit: fill at num_close[t] (same-bar close) — matches
          backtest_ratio_strategy.py:684.
        - Stop exit: theoretical stop price × (1 - slippage_bps/1e4) — matches
          backtest_ratio_strategy.py:574-582 (OLD books stop at theoretical
          price, not realized close).
        - Tx costs: 2 × tx_cost_per_side deducted from exit-day return
          (entry+exit combined — equivalent to splitting per Chan AT Ch.3).

    Returns: (sharpe, cagr, max_dd, n_trades, hit_rate, avg_trade_dur).
    """
    n = len(ratio_close)
    n_r = resistance_proj.shape[1]
    n_s = support_proj.shape[1]

    r_streak = np.zeros(n_r, dtype=np.int32)
    s_streak = np.zeros(n_s, dtype=np.int32)

    in_position = False
    entry_price = 0.0  # numerator close at entry — fixed for trade lifetime

    daily_rets = np.zeros(n - 1, dtype=np.float64)
    n_trades = 0
    n_winning = 0
    total_duration = 0
    trade_duration = 0
    trade_ret_sum = 0.0
    round_trip_cost = 2.0 * tx_cost_per_side
    slippage_factor = 1.0 - stop_slippage_bps / 10000.0

    for t in range(n - 1):
        # --- update resistance streaks ---
        entry_fires = False
        for j in range(n_r):
            proj = resistance_proj[t, j]
            if np.isnan(proj):
                r_streak[j] = 0
                continue
            if ratio_close[t] > proj:
                r_streak[j] += 1
                # Fire ONLY on the confirmation bar (streak == entry_conf),
                # not on every bar afterwards. Matches production semantics
                # (ratio_breakout_strategy/backtest_utils.py:76).
                if r_streak[j] == entry_conf:
                    entry_fires = True
            else:
                r_streak[j] = 0

        # --- update support streaks ---
        support_exit_fires = False
        for j in range(n_s):
            proj = support_proj[t, j]
            if np.isnan(proj):
                s_streak[j] = 0
                continue
            if ratio_close[t] < proj:
                s_streak[j] += 1
                if s_streak[j] == exit_conf:
                    support_exit_fires = True
            else:
                s_streak[j] = 0

        # --- fixed stop-loss check (numerator close vs entry_price) ---
        stop_fires = False
        if in_position and stop_pct > 0.0 and entry_price > 0.0:
            if num_close[t] <= entry_price * (1.0 - stop_pct):
                stop_fires = True

        # --- resolve (one-trade-at-a-time). Same-bar close fill semantics. ---
        will_enter = False
        will_exit_signal = False
        will_exit_stop = False
        if in_position:
            # Stop takes precedence (OLD production checks stop before signal,
            # backtest_ratio_strategy.py:565-638).
            if stop_fires:
                will_exit_stop = True
            elif support_exit_fires:
                will_exit_signal = True
        else:
            if entry_fires:
                will_enter = True

        will_exit = will_exit_signal or will_exit_stop

        # --- compute return for this bar ---
        # Entering position at bar t close: no MTM return on bar t itself;
        # first MTM hits on bar t+1. Holding through bar t: mark-to-market from
        # today's close to tomorrow's close, unless exiting.
        if in_position and not will_exit:
            num_c_t = num_close[t]
            num_c_tp1 = num_close[t + 1]
            if num_c_t > 0.0:
                r = num_c_tp1 / num_c_t - 1.0
                daily_rets[t] = r
                trade_ret_sum += r
                trade_duration += 1
        elif in_position and will_exit:
            # Book the exit return for bar t. Use entry_price reference-based
            # accounting: total trade return is (exit_price / entry_price - 1).
            # daily_rets has already captured the compounded return up to bar
            # t-1, so the "exit bar" return is (exit_price / num_close[t-1] - 1)
            # where exit_price is either theoretical-stop+slippage or close.
            # Subtract round-trip cost here.
            prev_c = num_close[t - 1] if t > 0 else num_close[t]
            if will_exit_stop:
                exit_price = entry_price * (1.0 - stop_pct) * slippage_factor
            else:
                exit_price = num_close[t]
            if prev_c > 0.0:
                r_exit = exit_price / prev_c - 1.0 - round_trip_cost
                daily_rets[t] = r_exit
                trade_ret_sum += r_exit
                trade_duration += 1

        # --- state transitions at end of day ---
        if will_exit:
            in_position = False
            total_duration += trade_duration
            if trade_ret_sum > 0.0:
                n_winning += 1
            trade_ret_sum = 0.0
            trade_duration = 0
            entry_price = 0.0
        elif will_enter:
            in_position = True
            n_trades += 1
            trade_ret_sum = 0.0
            trade_duration = 0
            entry_price = num_close[t]

    # Close any open trade at end
    if in_position:
        total_duration += trade_duration
        if trade_ret_sum > 0.0:
            n_winning += 1

    # --- metrics on daily_rets ---
    if n - 1 < 2 or n_trades == 0:
        return 0.0, 0.0, 0.0, n_trades, 0.0, 0.0

    mean_r = 0.0
    for r in daily_rets:
        mean_r += r
    mean_r /= float(n - 1)

    var_r = 0.0
    for r in daily_rets:
        var_r += (r - mean_r) ** 2
    var_r /= float(n - 2)
    std_r = var_r ** 0.5

    if std_r < 1e-10:
        return 0.0, 0.0, 0.0, n_trades, 0.0, 0.0

    sharpe = (mean_r / std_r) * (ANNUALIZE ** 0.5)
    cagr = (1.0 + mean_r) ** ANNUALIZE - 1.0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in daily_rets:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd

    hit_rate = float(n_winning) / float(n_trades) if n_trades > 0 else 0.0
    avg_dur = float(total_duration) / float(n_trades) if n_trades > 0 else 0.0
    return sharpe, cagr, max_dd, n_trades, hit_rate, avg_dur


# ---------------------------------------------------------------------------
# Returns-series helper (for S3B bootstrap + S4 significance)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _walk_returns(
    ratio_close: np.ndarray,
    num_close: np.ndarray,
    resistance_proj: np.ndarray,
    support_proj: np.ndarray,
    entry_conf: int,
    exit_conf: int,
    stop_pct: float,
    stop_slippage_bps: float,
    tx_cost_per_side: float,
) -> np.ndarray:
    """Return the per-bar daily-return array from the SAME walker as
    ``_walk_and_score``, for downstream stats (bootstrap, PSR, t-test, DSR).

    Identical semantics to ``_walk_and_score`` — same stop-slippage + tx-cost
    accounting — just returns the raw daily_rets vector instead of summary
    statistics. Shape: (n - 1,).
    """
    n = len(ratio_close)
    n_r = resistance_proj.shape[1]
    n_s = support_proj.shape[1]
    r_streak = np.zeros(n_r, dtype=np.int32)
    s_streak = np.zeros(n_s, dtype=np.int32)
    in_position = False
    entry_price = 0.0
    daily_rets = np.zeros(n - 1, dtype=np.float64)
    round_trip_cost = 2.0 * tx_cost_per_side
    slippage_factor = 1.0 - stop_slippage_bps / 10000.0

    for t in range(n - 1):
        entry_fires = False
        for j in range(n_r):
            proj = resistance_proj[t, j]
            if np.isnan(proj):
                r_streak[j] = 0
                continue
            if ratio_close[t] > proj:
                r_streak[j] += 1
                if r_streak[j] == entry_conf:
                    entry_fires = True
            else:
                r_streak[j] = 0
        support_exit_fires = False
        for j in range(n_s):
            proj = support_proj[t, j]
            if np.isnan(proj):
                s_streak[j] = 0
                continue
            if ratio_close[t] < proj:
                s_streak[j] += 1
                if s_streak[j] == exit_conf:
                    support_exit_fires = True
            else:
                s_streak[j] = 0
        stop_fires = False
        if in_position and stop_pct > 0.0 and entry_price > 0.0:
            if num_close[t] <= entry_price * (1.0 - stop_pct):
                stop_fires = True

        will_enter = False
        will_exit_signal = False
        will_exit_stop = False
        if in_position:
            if stop_fires:
                will_exit_stop = True
            elif support_exit_fires:
                will_exit_signal = True
        else:
            if entry_fires:
                will_enter = True
        will_exit = will_exit_signal or will_exit_stop

        if in_position and not will_exit:
            num_c_t = num_close[t]
            num_c_tp1 = num_close[t + 1]
            if num_c_t > 0.0:
                daily_rets[t] = num_c_tp1 / num_c_t - 1.0
        elif in_position and will_exit:
            prev_c = num_close[t - 1] if t > 0 else num_close[t]
            if will_exit_stop:
                exit_price = entry_price * (1.0 - stop_pct) * slippage_factor
            else:
                exit_price = num_close[t]
            if prev_c > 0.0:
                daily_rets[t] = exit_price / prev_c - 1.0 - round_trip_cost

        if will_exit:
            in_position = False
            entry_price = 0.0
        elif will_enter:
            in_position = True
            entry_price = num_close[t]

    return daily_rets


# ---------------------------------------------------------------------------
# Trace helper (visualization only — NOT JIT'd, not used in hot path)
# ---------------------------------------------------------------------------


def _walk_and_trace(
    ratio_close: np.ndarray,
    num_close: np.ndarray,
    resistance_proj: np.ndarray,
    support_proj: np.ndarray,
    entry_conf: int,
    exit_conf: int,
    stop_pct: float,
    stop_slippage_bps: float = 5.0,
    tx_cost_per_side: float = 0.001,
) -> list[dict]:
    """Pure-Python walker that returns a per-trade list for visualization.

    Mirrors ``_walk_and_score`` semantics exactly (same-bar close fills,
    theoretical stop + slippage, round-trip tx cost). For each trade, records:
        entry_idx, exit_idx, entry_price, exit_price, pnl_pct, exit_reason
    where ``exit_reason`` is 'signal' (support breakdown) or 'stop' (fixed SL)
    or 'eod' (end-of-data close-out).

    Pure Python — slow but only runs once per pair for viz, not in inner loop.
    """
    n = len(ratio_close)
    n_r = resistance_proj.shape[1]
    n_s = support_proj.shape[1]
    r_streak = np.zeros(n_r, dtype=np.int32)
    s_streak = np.zeros(n_s, dtype=np.int32)
    in_position = False
    entry_price = 0.0
    entry_idx = -1
    trades: list[dict] = []
    slippage_factor = 1.0 - stop_slippage_bps / 10000.0
    round_trip_cost = 2.0 * tx_cost_per_side

    for t in range(n - 1):
        entry_fires = False
        for j in range(n_r):
            proj = resistance_proj[t, j]
            if np.isnan(proj):
                r_streak[j] = 0
                continue
            if ratio_close[t] > proj:
                r_streak[j] += 1
                if r_streak[j] == entry_conf:
                    entry_fires = True
            else:
                r_streak[j] = 0

        support_exit_fires = False
        for j in range(n_s):
            proj = support_proj[t, j]
            if np.isnan(proj):
                s_streak[j] = 0
                continue
            if ratio_close[t] < proj:
                s_streak[j] += 1
                if s_streak[j] == exit_conf:
                    support_exit_fires = True
            else:
                s_streak[j] = 0

        stop_fires = False
        if in_position and stop_pct > 0.0 and entry_price > 0.0:
            if num_close[t] <= entry_price * (1.0 - stop_pct):
                stop_fires = True

        will_enter = False
        will_exit = False
        exit_reason = ""
        if in_position:
            # Stop precedence: OLD checks stop before signal
            # (backtest_ratio_strategy.py:565-638).
            if stop_fires:
                will_exit = True
                exit_reason = "stop"
            elif support_exit_fires:
                will_exit = True
                exit_reason = "signal"
        elif entry_fires:
            will_enter = True

        if will_exit:
            # Signal exit: fill at confirmation-bar close (same-bar close).
            # Stop exit: fill at theoretical stop × (1 - slippage_bps/1e4).
            if exit_reason == "stop":
                exit_price = float(entry_price * (1.0 - stop_pct) * slippage_factor)
            else:
                exit_price = float(num_close[t])
            gross_pnl = (exit_price / entry_price - 1.0) if entry_price > 0 else 0.0
            pnl_pct = gross_pnl - round_trip_cost
            trades.append({
                "entry_idx": int(entry_idx),
                "exit_idx": int(t),
                "entry_price": float(entry_price),
                "exit_price": exit_price,
                "pnl_pct": float(pnl_pct),
                "exit_reason": exit_reason,
            })
            in_position = False
            entry_price = 0.0
            entry_idx = -1
        elif will_enter:
            in_position = True
            entry_price = float(num_close[t])
            entry_idx = t

    # Close any open trade at end-of-data
    if in_position and entry_idx >= 0:
        last = n - 1
        exit_price = float(num_close[last])
        gross_pnl = (exit_price / entry_price - 1.0) if entry_price > 0 else 0.0
        pnl_pct = gross_pnl - round_trip_cost
        trades.append({
            "entry_idx": int(entry_idx),
            "exit_idx": int(last),
            "entry_price": float(entry_price),
            "exit_price": exit_price,
            "pnl_pct": float(pnl_pct),
            "exit_reason": "eod",
        })

    return trades


# ---------------------------------------------------------------------------
# Per-pair flow (detection + grid sweep + OOS eval)
# ---------------------------------------------------------------------------


def _detect_and_project(
    ratio_ohlc: pd.DataFrame,
    *,
    min_points: int,
    max_error_pct: float,
    min_period_days: int,
    _diag_pair: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Run detector + project; returns (resistance_proj, support_proj).

    Diagnostic print: if detection takes > 5s OR produces > 50k lines, emit a
    line with pair name, n_bars, n_r, n_s, wall time — to stderr so it shows
    up in the run log without interleaving with the progress bar.
    """
    n = len(ratio_ohlc)
    t0 = _time.time()
    r_lines = detect_resistance(
        ratio_ohlc,
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        n_cores=1,
    )
    t1 = _time.time()
    s_lines = detect_support(
        ratio_ohlc,
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        n_cores=1,
    )
    t2 = _time.time()
    n_r, n_s = len(r_lines), len(s_lines)
    total_s = t2 - t0
    if total_s > 5.0 or (n_r + n_s) > 50_000:
        import sys as _sys
        print(
            f"  [DIAG] pair={_diag_pair!s} n_bars={n} "
            f"n_r={n_r} n_s={n_s} "
            f"r_time={t1 - t0:.1f}s s_time={t2 - t1:.1f}s "
            f"total={total_s:.1f}s",
            file=_sys.stderr, flush=True,
        )
    return (
        _line_projection_matrix(r_lines, n),
        _line_projection_matrix(s_lines, n),
    )


def _build_ratio_ohlc(
    num_oh: dict,
    den_oh: dict,
) -> pd.DataFrame | None:
    """Align num + den OHLC on shared index; compute ratio OHLC.

    ``num_oh`` / ``den_oh`` are dicts ``{"Open": Series, "High": Series, ...}``.
    """
    try:
        idx = num_oh["Close"].index.intersection(den_oh["Close"].index)
        if len(idx) < 504:
            return None
        n_open = num_oh["Open"].loc[idx].to_numpy()
        n_high = num_oh["High"].loc[idx].to_numpy()
        n_low = num_oh["Low"].loc[idx].to_numpy()
        n_close = num_oh["Close"].loc[idx].to_numpy()
        d_open = den_oh["Open"].loc[idx].to_numpy()
        d_high = den_oh["High"].loc[idx].to_numpy()
        d_low = den_oh["Low"].loc[idx].to_numpy()
        d_close = den_oh["Close"].loc[idx].to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pd.DataFrame(
                {
                    "Open": n_open / d_open,
                    "High": n_high / d_low,   # max ratio
                    "Low": n_low / d_high,    # min ratio
                    "Close": n_close / d_close,
                },
                index=idx,
            )
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) < 504:
            return None
        return ratio
    except Exception:
        return None


def _penalized_sharpe(
    sharpe: float, n_params: int, n_trades: int,
) -> float:
    """SR × sqrt(1 - p/n). Analogous to adjusted-R²."""
    if n_trades < n_params or n_trades == 0:
        return -999.0
    penalty = np.sqrt(1.0 - n_params / n_trades)
    return sharpe * penalty


def _process_one(
    prices_ohlc: dict[str, dict],
    pair_row: dict,
    is_ratio: float,
    entry_confs: tuple,
    exit_confs: tuple,
    stop_pcts: tuple,
    min_is_trades: int,
    min_oos_trades: int,
    min_is_sharpe: float,
    min_oos_sharpe: float,
    min_points: int,
    max_error_pct: float,
    min_period_days: int,
    atr_period: int,
    stop_slippage_bps: float = 5.0,
    tx_cost_per_side: float = 0.001,
) -> dict | None:
    """Full per-pair flow: detect once on full history → IS grid → OOS eval.

    OLD-production semantics: trendline detection runs once on the entire
    ratio series (no per-window re-detect). IS/OOS walkers slice the
    projection matrices by time; each line is still gated to fire only
    after its terminal pivot (end_idx + 1) via _line_projection_matrix.

    Mild informational lookahead: the detector sees OOS bars when fitting
    lines. This matches OLD production (ratio_breakout_strategy/
    backtest_ratio_strategy.py:396-404 — cached once per pair) and is the
    accepted trade-off for trendline strategies where lines span years.
    """
    num = pair_row["numerator"]
    den = pair_row["denominator"]
    pair_name = pair_row["pair"]

    def _early_reject(reason: str) -> dict:
        return {
            "pair": pair_name, "numerator": num, "denominator": den,
            "halflife": float("nan"), "window": 0,
            "entry_thresh": 0.0, "exit_thresh": 0.0,
            "slope_min": 0.0,
            "is_sharpe": float("nan"), "is_penalized_sharpe": float("nan"),
            "is_trades": 0, "is_cagr": float("nan"),
            "is_max_dd": float("nan"), "is_hit_rate": float("nan"),
            "is_avg_trade_dur": 0.0,
            "oos_sharpe": float("nan"), "oos_trades": 0,
            "oos_cagr": float("nan"), "oos_max_dd": float("nan"),
            "oos_hit_rate": float("nan"), "oos_avg_trade_dur": 0.0,
            "entry_conf": 0, "exit_conf": 0, "stop_pct": 0.0,
            "atr_period": int(atr_period),
            "passed": False, "fail_reason": reason,
            "signal_method": "trendline_breakout",
            "optim_method": "grid_breakout",
            "full_grid": [],
        }

    if num not in prices_ohlc or den not in prices_ohlc:
        return _early_reject("missing_prices")

    ratio = _build_ratio_ohlc(prices_ohlc[num], prices_ohlc[den])
    if ratio is None:
        return _early_reject("short_shared_history")

    n_total = len(ratio)
    split_idx = int(n_total * is_ratio)
    if split_idx < 252 or (n_total - split_idx) < 60:
        return _early_reject(f"short_is_oos(n={n_total})")

    # Align numerator Close on ratio index. ATR no longer consumed (OLD uses
    # fixed % stop, not ATR trailing) but `atr_period` retained in signature
    # for interface compat.
    num_close_all = prices_ohlc[num]["Close"].loc[ratio.index].to_numpy(
        dtype=np.float64
    )
    _ = atr_period  # intentionally unused — signature compat

    # --- FULL-HISTORY detection (matches OLD production). Detect once, slice
    # projection matrices by window. Each line still gated to fire only after
    # its terminal pivot via _line_projection_matrix(end_idx + 1). ---
    full_r_proj, full_s_proj = _detect_and_project(
        ratio,
        min_points=min_points,
        max_error_pct=max_error_pct,
        min_period_days=min_period_days,
        _diag_pair=pair_name,
    )

    is_ratio_close = ratio["Close"].to_numpy(dtype=np.float64)[:split_idx]
    is_num_close = num_close_all[:split_idx]
    is_r_proj = full_r_proj[:split_idx, :]
    is_s_proj = full_s_proj[:split_idx, :]

    # Slice OOS views once before the grid loop — every combo evaluates OOS too.
    oos_ratio_close = ratio["Close"].to_numpy(dtype=np.float64)[split_idx:]
    oos_num_close = num_close_all[split_idx:]
    oos_r_proj_full = full_r_proj[split_idx:, :]
    oos_s_proj_full = full_s_proj[split_idx:, :]

    best = None
    full_grid: list[dict] = []
    for ec in entry_confs:
        for xc in exit_confs:
            for sp in stop_pcts:
                is_sh, is_cg, is_dd, is_nt, is_hr, is_ad = _walk_and_score(
                    is_ratio_close, is_num_close,
                    is_r_proj, is_s_proj,
                    int(ec), int(xc), float(sp),
                    stop_slippage_bps, tx_cost_per_side,
                )
                # Evaluate OOS for this combo too (Pardo §6.3 neighborhood needs
                # OOS Sharpe across all combos, not just the winner).
                oos_sh, oos_cg, oos_dd, oos_nt, oos_hr, oos_ad = _walk_and_score(
                    oos_ratio_close, oos_num_close,
                    oos_r_proj_full, oos_s_proj_full,
                    int(ec), int(xc), float(sp),
                    stop_slippage_bps, tx_cost_per_side,
                )
                full_grid.append({
                    "entry_conf": int(ec),
                    "exit_conf": int(xc),
                    "stop_pct": float(sp),
                    "is_sharpe": float(is_sh) if not np.isnan(is_sh) else float("nan"),
                    "is_trades": int(is_nt),
                    "is_cagr": float(is_cg) if not np.isnan(is_cg) else float("nan"),
                    "is_max_dd": float(is_dd) if not np.isnan(is_dd) else float("nan"),
                    "oos_sharpe": float(oos_sh) if not np.isnan(oos_sh) else float("nan"),
                    "oos_trades": int(oos_nt),
                    "oos_cagr": float(oos_cg) if not np.isnan(oos_cg) else float("nan"),
                    "oos_max_dd": float(oos_dd) if not np.isnan(oos_dd) else float("nan"),
                })
                if np.isnan(is_sh) or is_nt < min_is_trades:
                    continue
                pen = _penalized_sharpe(float(is_sh), 3, int(is_nt))
                if best is None or pen > best["pen"]:
                    best = {
                        "pen": pen,
                        "entry_conf": int(ec),
                        "exit_conf": int(xc),
                        "stop_pct": float(sp),
                        "is_sharpe": float(is_sh),
                        "is_cagr": float(is_cg),
                        "is_max_dd": float(is_dd),
                        "is_trades": int(is_nt),
                        "is_hit_rate": float(is_hr),
                        "is_avg_dur": float(is_ad),
                    }

    def _rejected_row(reason: str) -> dict:
        """Rejected row with NO combo selected (used when best is None)."""
        return {
            "pair": pair_name, "numerator": num, "denominator": den,
            "halflife": float("nan"), "window": 0,
            "entry_thresh": 0.0, "exit_thresh": 0.0,
            "slope_min": 0.0,
            "is_sharpe": float("nan"), "is_penalized_sharpe": float("nan"),
            "is_trades": 0, "is_cagr": float("nan"),
            "is_max_dd": float("nan"), "is_hit_rate": float("nan"),
            "is_avg_trade_dur": 0.0,
            "oos_sharpe": float("nan"), "oos_trades": 0,
            "oos_cagr": float("nan"), "oos_max_dd": float("nan"),
            "oos_hit_rate": float("nan"), "oos_avg_trade_dur": 0.0,
            "entry_conf": 0, "exit_conf": 0, "stop_pct": 0.0,
            "atr_period": int(atr_period),
            "passed": False, "fail_reason": reason,
            "signal_method": "trendline_breakout",
            "optim_method": "grid_breakout",
            "full_grid": [],
        }

    if best is None:
        return _rejected_row("no_valid_is_combo")

    if best["is_sharpe"] < min_is_sharpe:
        # Preserve the best-combo IS metrics even though the gate fails
        # (so downstream comparison / diagnostics keep signal).
        return {
            "pair": pair_name, "numerator": num, "denominator": den,
            "halflife": float("nan"), "window": 0,
            "entry_thresh": 0.0, "exit_thresh": 0.0,
            "slope_min": 0.0,
            "is_sharpe": float(best["is_sharpe"]),
            "is_penalized_sharpe": float(best["pen"]),
            "is_trades": int(best["is_trades"]),
            "is_cagr": float(best["is_cagr"]),
            "is_max_dd": float(best["is_max_dd"]),
            "is_hit_rate": float(best["is_hit_rate"]),
            "is_avg_trade_dur": float(best["is_avg_dur"]),
            "oos_sharpe": float("nan"), "oos_trades": 0,
            "oos_cagr": float("nan"), "oos_max_dd": float("nan"),
            "oos_hit_rate": float("nan"), "oos_avg_trade_dur": 0.0,
            "entry_conf": best["entry_conf"],
            "exit_conf": best["exit_conf"],
            "stop_pct": best["stop_pct"],
            "atr_period": int(atr_period),
            "passed": False,
            "fail_reason": f"is_sharpe<{min_is_sharpe}",
            "signal_method": "trendline_breakout",
            "optim_method": "grid_breakout",
        }

    # --- OOS metrics for the selected combo come from full_grid (already eval'd).
    winner_row = next(
        (
            g for g in full_grid
            if g["entry_conf"] == best["entry_conf"]
            and g["exit_conf"] == best["exit_conf"]
            and g["stop_pct"] == best["stop_pct"]
        ),
        None,
    )
    if winner_row is None:
        return _rejected_row("winner_missing_from_grid")
    oos_sh = winner_row["oos_sharpe"]
    oos_cg = winner_row["oos_cagr"]
    oos_dd = winner_row["oos_max_dd"]
    oos_nt = winner_row["oos_trades"]
    # hit_rate + avg_dur are not stored per-combo to keep grid lean; re-run
    # walker once on winner for those (interface compat with S2_REQUIRED).
    _sh2, _cg2, _dd2, _nt2, oos_hr, oos_ad = _walk_and_score(
        ratio["Close"].to_numpy(dtype=np.float64)[split_idx:],
        num_close_all[split_idx:],
        full_r_proj[split_idx:, :], full_s_proj[split_idx:, :],
        best["entry_conf"], best["exit_conf"], best["stop_pct"],
        stop_slippage_bps, tx_cost_per_side,
    )

    # --- gates ---
    fail_reasons: list[str] = []
    if np.isnan(oos_sh):
        fail_reasons.append("oos_sr_nan")
    elif oos_sh <= 0:
        fail_reasons.append("oos_sr<=0")
    if oos_nt < min_oos_trades:
        fail_reasons.append(f"oos_trades<{min_oos_trades}")
    passed = len(fail_reasons) == 0

    return {
        "pair": pair_name,
        "numerator": num,
        "denominator": den,
        "halflife": float("nan"),  # not computed for breakout; interface compat
        "window": 0,                # not used; interface compat
        "entry_thresh": float(best["entry_conf"]),
        "exit_thresh": float(best["exit_conf"]),
        "stop_pct": float(best["stop_pct"]),
        "slope_min": 0.0,
        "is_sharpe": round(best["is_sharpe"], 4),
        "is_penalized_sharpe": round(best["pen"], 4),
        "is_trades": int(best["is_trades"]),
        "is_cagr": round(best["is_cagr"] * 100, 4),
        "is_max_dd": round(best["is_max_dd"] * 100, 4),
        "is_hit_rate": round(best["is_hit_rate"] * 100, 2),
        "is_avg_trade_dur": round(best["is_avg_dur"], 1),
        "oos_sharpe": round(float(oos_sh), 4),
        "oos_trades": int(oos_nt),
        "oos_cagr": round(float(oos_cg) * 100, 4),
        "oos_max_dd": round(float(oos_dd) * 100, 4),
        "oos_hit_rate": round(float(oos_hr) * 100, 2),
        "oos_avg_trade_dur": round(float(oos_ad), 1),
        "entry_conf": int(best["entry_conf"]),
        "exit_conf": int(best["exit_conf"]),
        "atr_period": int(atr_period),
        "passed": passed,
        "fail_reason": "|".join(fail_reasons) if fail_reasons else "",
        "signal_method": "trendline_breakout",
        "optim_method": "grid_breakout",
        "full_grid": full_grid,   # all 75 combos — consumed by phase3_neighborhood.py
    }


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

_worker_prices_ohlc: dict[str, dict] = {}


def _worker_init(prices_ohlc: dict[str, dict]) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    global _worker_prices_ohlc
    _worker_prices_ohlc = prices_ohlc


def _pair_to_cache_key(pair: str) -> str:
    """Filename-safe key for a pair (e.g. 'SPY/TLT' -> 'SPY__TLT')."""
    return pair.replace("/", "__").replace(" ", "_")


def _worker_process(args: tuple) -> dict | None:
    (
        pair_row, is_ratio, entry_confs, exit_confs, stop_pcts,
        min_is_trades, min_oos_trades, min_is_sr, min_oos_sr,
        min_points, max_error_pct, min_period_days, atr_period,
        cache_dir, stop_slippage_bps, tx_cost_per_side,
    ) = args
    import os as _os
    import pickle as _pickle
    import sys as _sys
    import time as _t
    _start = _t.time()
    _pid = _os.getpid()
    _pair = pair_row.get("pair", "?")
    print(f"  [START pid={_pid}] {_pair}", file=_sys.stderr, flush=True)
    result = _process_one(
        _worker_prices_ohlc,
        pair_row,
        is_ratio,
        entry_confs,
        exit_confs,
        stop_pcts,
        min_is_trades,
        min_oos_trades,
        min_is_sr,
        min_oos_sr,
        min_points,
        max_error_pct,
        min_period_days,
        atr_period,
        stop_slippage_bps,
        tx_cost_per_side,
    )
    _elapsed = _t.time() - _start
    _passed = bool(result.get("passed", False)) if result else False
    # Save per-pair checkpoint so re-runs skip completed work.
    if cache_dir is not None and result is not None:
        try:
            key = _pair_to_cache_key(_pair)
            ckpt = f"{cache_dir}/{key}.pkl"
            with open(ckpt, "wb") as _f:
                _pickle.dump(result, _f)
        except Exception as e:  # noqa: BLE001
            print(
                f"  [CACHE-WARN pid={_pid}] {_pair}: {e}",
                file=_sys.stderr, flush=True,
            )
    print(
        f"  [DONE  pid={_pid}] {_pair}  {_elapsed:.1f}s  "
        f"pass={_passed}",
        file=_sys.stderr, flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Registered entry point
# ---------------------------------------------------------------------------


@register_stage("s2_optimize")
def grid_breakout(
    prices_ohlc: dict[str, dict],
    s1_result: pd.DataFrame,
    signal_fn=None,  # noqa: ARG001 (interface compat)
    **config,
) -> pd.DataFrame:
    """Exhaustive 3-D grid for ratio breakout: entry_conf × exit_conf × atr_k.

    ``prices_ohlc`` is a dict ``{ticker: {"Open": Series, "High": Series,
    "Low": Series, "Close": Series}}``. Differs from ``grid_ma`` (which takes
    close-only DataFrame) because breakout needs full OHLC.

    Config keys:
        entry_confs (tuple): default (3,4,5,6,7)
        exit_confs (tuple): default (3,4,5,6,7)
        stop_pcts (tuple): default (1.5, 2.0, 2.5, 3.0, 3.5)
        is_ratio (float): default 0.80
        min_is_trades (int): default 5
        min_oos_trades (int): default 5
        min_is_sharpe (float): default 0.5  (Bailey & LdP 2012)
        min_oos_sharpe (float): default 0.0 (used only to flag; > 0 required)
        min_points (int): default 4
        max_error_pct (float): default 2.0
        min_period_days (int): default 21
        atr_period (int): default 14 (Wilder)
        n_workers (int): default cpu_count - 2
    """
    entry_confs = tuple(config.get("entry_confs", DEFAULT_ENTRY_CONFS))
    exit_confs = tuple(config.get("exit_confs", DEFAULT_EXIT_CONFS))
    stop_pcts = tuple(config.get("stop_pcts", DEFAULT_STOP_PCTS))
    is_ratio = config.get("is_ratio", 0.80)
    min_is_trades = config.get("min_is_trades", 5)
    min_oos_trades = config.get("min_oos_trades", 5)
    min_is_sharpe = config.get("min_is_sharpe", 0.5)
    min_oos_sharpe = config.get("min_oos_sharpe", 0.0)
    min_points = config.get("min_points", DEFAULT_MIN_POINTS)
    max_error_pct = config.get("max_error_pct", DEFAULT_MAX_ERROR_PCT)
    min_period_days = config.get("min_period_days", DEFAULT_MIN_PERIOD_DAYS)
    atr_period = config.get("atr_period", DEFAULT_ATR_PERIOD)
    stop_slippage_bps = config.get("stop_slippage_bps", 5.0)
    tx_cost_per_side = config.get("tx_cost_per_side", 0.001)

    passed_pairs = s1_result[s1_result["passed"]].to_dict("records")
    n_combos = len(entry_confs) * len(exit_confs) * len(stop_pcts)

    # Per-pair checkpointing: re-runs load completed pairs from disk.
    # `cache_dir` is an absolute path (str); caller owns creation + clearing.
    cache_dir = config.get("cache_dir", None)
    cached_rows: list[dict] = []
    if cache_dir is not None:
        import pickle as _pickle
        from pathlib import Path as _Path
        _cdir = _Path(cache_dir)
        _cdir.mkdir(parents=True, exist_ok=True)
        remaining: list[dict] = []
        for pr in passed_pairs:
            key = _pair_to_cache_key(pr["pair"])
            ckpt = _cdir / f"{key}.pkl"
            if ckpt.exists():
                try:
                    with open(ckpt, "rb") as _f:
                        cached_rows.append(_pickle.load(_f))
                    continue
                except Exception:  # noqa: BLE001
                    pass  # corrupted — re-run
            remaining.append(pr)
        if cached_rows:
            _log(
                f"grid_breakout: {len(cached_rows)} pairs loaded from cache, "
                f"{len(remaining)} remaining (cache_dir={_cdir})"
            )
        passed_pairs = remaining

    n_pairs = len(passed_pairs)
    _log(
        f"grid_breakout: {n_pairs} pairs x {n_combos} combos "
        f"[entry={entry_confs}, exit={exit_confs}, atr_k={stop_pcts}]"
    )

    # If everything was cached, short-circuit
    if n_pairs == 0 and cached_rows:
        _log("grid_breakout: all pairs satisfied from cache")
        return pd.DataFrame(cached_rows)

    # Warm up Numba JIT in main process before forking
    _walk_and_score(
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.full((3, 1), np.nan, dtype=np.float32),
        np.full((3, 1), np.nan, dtype=np.float32),
        3, 3, 0.10, 5.0, 0.001,
    )

    work_args = [
        (
            pr, is_ratio, entry_confs, exit_confs, stop_pcts,
            min_is_trades, min_oos_trades, min_is_sharpe, min_oos_sharpe,
            min_points, max_error_pct, min_period_days, atr_period,
            cache_dir, stop_slippage_bps, tx_cost_per_side,
        )
        for pr in passed_pairs
    ]

    rows: list[dict] = []
    n_pass = 0
    t_start = _time.time()

    ctx = mp.get_context("fork")
    n_workers = config.get("n_workers", None)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
    n_workers = min(n_workers, max(1, n_pairs))

    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(prices_ohlc,),
    ) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_worker_process, work_args, chunksize=4)
        ):
            if result is not None:
                rows.append(result)
                if result["passed"]:
                    n_pass += 1
            if (i + 1) % 25 == 0 or i == n_pairs - 1:
                elapsed = _time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_pairs - i - 1) / rate if rate > 0 else 0
                pass_pct = 100.0 * n_pass / (i + 1) if (i + 1) > 0 else 0.0
                med_is = float("nan")
                med_oos_sr = float("nan")
                med_oos_tr = float("nan")
                if rows:
                    is_arr = np.array(
                        [r["is_sharpe"] for r in rows if not np.isnan(r["is_sharpe"])]
                    )
                    if is_arr.size:
                        med_is = float(np.median(is_arr))
                    passing = [r for r in rows if r["passed"]]
                    if passing:
                        oos_sr_arr = np.array([r["oos_sharpe"] for r in passing])
                        oos_tr_arr = np.array([r["oos_trades"] for r in passing])
                        med_oos_sr = float(np.median(oos_sr_arr))
                        med_oos_tr = float(np.median(oos_tr_arr))
                _log(
                    f"grid_breakout: {i + 1}/{n_pairs} "
                    f"({n_pass} pass, {pass_pct:.1f}%) "
                    f"med_is_SR={med_is:.2f} "
                    f"med_oos_SR={med_oos_sr:.2f} "
                    f"med_oos_tr={med_oos_tr:.0f} "
                    f"[{rate:.1f}/s, ETA {eta:.0f}s]"
                )

    # Merge cached rows (from prior runs) with freshly-computed rows.
    all_rows = cached_rows + rows
    n_total_pass = sum(1 for r in all_rows if r.get("passed", False))
    _log(
        f"grid_breakout complete: {n_total_pass} passed / "
        f"{len(all_rows)} total ({len(cached_rows)} from cache + "
        f"{len(rows)} fresh)"
    )

    if all_rows:
        return pd.DataFrame(all_rows)

    return pd.DataFrame(
        columns=[
            "pair", "numerator", "denominator", "halflife", "window",
            "entry_thresh", "exit_thresh", "stop_pct", "slope_min",
            "is_sharpe", "is_penalized_sharpe", "is_trades",
            "is_cagr", "is_max_dd", "is_hit_rate", "is_avg_trade_dur",
            "oos_sharpe", "oos_trades", "oos_cagr", "oos_max_dd",
            "oos_hit_rate", "oos_avg_trade_dur",
            "entry_conf", "exit_conf", "stop_pct", "atr_period",
            "passed", "fail_reason", "signal_method", "optim_method",
        ]
    )
