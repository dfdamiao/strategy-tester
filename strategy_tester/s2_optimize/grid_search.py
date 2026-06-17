"""Exhaustive grid search optimizer. Chan (2013); LdP AFML Ch.14.

ThreadPoolExecutor for parallel pair processing.
Pre-computes signal (z-score/Kalman/etc.) ONCE per pair, then only
varies thresholds in the inner loop — massive speedup for expensive
signal methods like kalman_hedge.
"""
from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from strategy_tester.backtest.vbt_runner import (
    backtest_numba_entries_exits,
    backtest_vbt_fold,
    build_is_oos_split,
    calculate_penalized_sharpe,
    compute_ratio,
    grid_sweep_regime,
    grid_sweep_threshold,
)
from strategy_tester.registry import register_stage

# Check if batched Numba grid sweep is available
try:
    from strategy_tester.backtest.vbt_runner import (
        HAS_NUMBA_GRID_SWEEP,
    )
except ImportError:
    HAS_NUMBA_GRID_SWEEP = False


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _get_signal_module(signal_fn: Callable | None):
    """Get the module of a signal function for precompute/apply."""
    if signal_fn is None:
        return None
    mod_name = signal_fn.__module__
    return importlib.import_module(mod_name)


def _process_one_pair(
    prices: pd.DataFrame,
    pair_row: dict,
    entry_grid: list,
    exit_grid: list,
    stop_grid: list,
    slope_grid: list,
    is_ratio: float,
    fees: float,
    slope_window: int,
    min_is_trades: int,
    min_oos_trades: int,
    n_params: int,
    signal_fn: Callable | None,
    signal_module,
    min_is_sharpe: float = 0.5,
    min_oos_sharpe: float = 0.5,
    window_grid: list | None = None,
) -> dict | None:
    """Grid search one pair: IS optimize + OOS evaluate.

    If signal_module has precompute/apply_thresholds, uses two-phase
    approach (precompute per window + N threshold applications).
    If window_grid is provided, sweeps windows too (e.g. RSI period).
    """
    pair_name = pair_row["pair"]
    num = pair_row["numerator"]
    den = pair_row["denominator"]
    hl = pair_row["halflife"]
    base_window = pair_row["window"]

    if num not in prices.columns:
        return None
    # Singles (num == den) handled by compute_ratio: ratio = price itself.
    # Without this, ratio = num/den = 1.0 constant → NaN signals → 0 trades.
    if num != den and den not in prices.columns:
        return None

    try:
        ratio, p_num, common = compute_ratio(prices, pair_row)
    except KeyError:
        return None

    is_idx, oos_idx = build_is_oos_split(common, is_ratio)
    ratio_is = ratio.loc[is_idx]
    num_is = p_num.loc[is_idx]
    ratio_oos = ratio.loc[oos_idx]
    num_oos = p_num.loc[oos_idx]

    has_precompute = (
        signal_module is not None
        and hasattr(signal_module, "precompute")
        and hasattr(signal_module, "apply_thresholds")
    )

    # Windows to sweep: if window_grid provided, use it; else fixed
    windows = window_grid if window_grid else [base_window]

    # Determine if batched Numba grid sweep is possible
    # Requires: precompute module, Numba available, slope_grid=[0.0],
    # AND signal uses simple threshold logic (signal ≤/≥ thresh).
    # Donchian/ATR use entry_thresh as a window size, not a level — excluded.
    # cum_rsi: compound entry (cum) + compound exit (rsi OR ratio>ma_exit).
    # adx_ema_pullback: ADX gate + structural pullback re-cross + 1-2-3 swing.
    _BATCH_EXCLUDED = {
        "donchian", "atr_pullback", "cum_rsi", "adx_ema_pullback",
        # rsi_connors: compound entry (rsi <= thresh AND close > sma_long)
        # and compound exit (close > sma_short OR rsi >= thresh).
        # Fast path picks first non-slope precompute key (rsi) and would
        # silently bypass the SMA filter — incorrect.
        "rsi_connors",
    }
    mod_name = (
        signal_module.__name__.rsplit(".", 1)[-1]
        if signal_module is not None else ""
    )
    use_batch = (
        has_precompute
        and HAS_NUMBA_GRID_SWEEP
        and slope_grid == [0.0]
        and mod_name not in _BATCH_EXCLUDED
    )
    use_regime_batch = (
        mod_name == "regime_switch"
        and HAS_NUMBA_GRID_SWEEP
        and slope_grid == [0.0]
    )

    # Detect signal type for Numba: RSI uses oversold (entry ≤ thresh)
    # z-score uses entry ≤ -thresh
    is_oversold = "rsi" in mod_name or "bollinger" in mod_name

    # Grid search on IS
    best_pen = -np.inf
    best_params = None
    best_is_result = None
    best_window = base_window

    for window in windows:
        if window < 2:
            continue

        if use_regime_batch:
            # ── REGIME FAST PATH: batched Numba for regime_switch ──
            pre_is = signal_module.precompute(
                ratio_is, window, slope_window,
            )
            z_arr = pre_is["z"].fillna(0.0).values
            ma_arr = pre_is["ma_signal"].fillna(0.0).values
            adx_arr = pre_is["adx"].fillna(22.5).values
            close_is = num_is.values

            result = grid_sweep_regime(
                z_arr, ma_arr, adx_arr, close_is,
                entry_grid, exit_grid, stop_grid,
                fees=fees,
                min_is_trades=min_is_trades,
                n_params=n_params,
            )

            if result is not None and result["pen_sharpe"] > best_pen:
                best_pen = result["pen_sharpe"]
                best_params = {
                    "entry_thresh": result["entry_thresh"],
                    "exit_thresh": result["exit_thresh"],
                    "stop_pct": result["stop_pct"],
                    "slope_min": 0.0,
                }
                best_is_result = {
                    "sharpe": result["sharpe"],
                    "n_trades": result["n_trades"],
                    "hit_rate": 0.0,
                }
                best_window = window
        elif use_batch:
            # ── FAST PATH: batched Numba grid sweep ──
            pre_is = signal_module.precompute(
                ratio_is, window, slope_window,
            )
            # Extract signal array (RSI, z-score, etc.)
            signal_key = next(
                k for k in pre_is
                if k not in ("slope", "slope_arr")
            )
            signal_arr = pre_is[signal_key].values
            close_is = num_is.values

            result = grid_sweep_threshold(
                signal_arr, close_is,
                entry_grid, exit_grid, stop_grid,
                fees=fees,
                min_is_trades=min_is_trades,
                n_params=n_params,
                is_oversold=is_oversold,
            )

            if result is not None and result["pen_sharpe"] > best_pen:
                best_pen = result["pen_sharpe"]
                best_params = {
                    "entry_thresh": result["entry_thresh"],
                    "exit_thresh": result["exit_thresh"],
                    "stop_pct": result["stop_pct"],
                    "slope_min": 0.0,
                }
                best_is_result = {
                    "sharpe": result["sharpe"],
                    "n_trades": result["n_trades"],
                    "hit_rate": 0.0,
                }
                best_window = window
        else:
            # ── SLOW PATH: per-combo Python loop ──
            # Pre-convert num_is to numpy once (avoid repeated
            # .values.astype() inside backtest_numba_entries_exits)
            close_np = num_is.values.astype(np.float64)

            pre_is = None
            if has_precompute:
                pre_is = signal_module.precompute(
                    ratio_is, window, slope_window,
                )

            for entry_thresh in entry_grid:
                for exit_thresh in exit_grid:
                    for stop_pct in stop_grid:
                        for slope_min in slope_grid:
                            if has_precompute:
                                entries, exits = (
                                    signal_module.apply_thresholds(
                                        pre_is, entry_thresh,
                                        exit_thresh, slope_min,
                                    )
                                )
                                # Call raw Numba directly
                                # (skip pandas conversion)
                                if HAS_NUMBA_GRID_SWEEP:
                                    from strategy_tester.backtest.vbt_runner import (  # noqa: E501
                                        _backtest_entries_exits_numba,
                                    )
                                    ent = entries.values.astype(
                                        bool,
                                    )
                                    ext = exits.values.astype(
                                        bool,
                                    )
                                    s, nt, hr = (
                                        _backtest_entries_exits_numba(
                                            ent, ext, close_np,
                                            stop_pct, fees,
                                        )
                                    )
                                    bt = {
                                        "sharpe": round(s, 4),
                                        "n_trades": nt,
                                        "hit_rate": round(
                                            hr * 100, 2,
                                        ),
                                    }
                                else:
                                    bt = backtest_numba_entries_exits(
                                        num_is, entries, exits,
                                        stop_pct=stop_pct,
                                        fees=fees,
                                    )
                            else:
                                bt = backtest_vbt_fold(
                                    num_is, ratio_is, window,
                                    entry_thresh, exit_thresh,
                                    stop_pct=stop_pct,
                                    slope_min=slope_min,
                                    slope_window=slope_window,
                                    fees=fees,
                                    signal_fn=signal_fn,
                                )
                            sharpe = bt["sharpe"]
                            n_trades = bt["n_trades"]

                            if (
                                pd.isna(sharpe)
                                or n_trades < min_is_trades
                            ):
                                continue

                            pen = calculate_penalized_sharpe(
                                sharpe, n_params, n_trades,
                            )
                            if pen > best_pen:
                                best_pen = pen
                                best_params = {
                                    "entry_thresh": entry_thresh,
                                    "exit_thresh": exit_thresh,
                                    "stop_pct": stop_pct,
                                    "slope_min": slope_min,
                                }
                                best_is_result = bt
                                best_window = window

    if best_params is None or best_is_result is None:
        return None

    # IS Sharpe gate — Bailey & LdP (2012): SR < 0.5 likely spurious
    if best_is_result["sharpe"] < min_is_sharpe:
        return None

    # Evaluate best on OOS (re-precompute with best window)
    if has_precompute:
        pre_oos_best = signal_module.precompute(
            ratio_oos, best_window, slope_window,
        )
        oos_entries, oos_exits = signal_module.apply_thresholds(
            pre_oos_best,
            best_params["entry_thresh"],
            best_params["exit_thresh"],
            best_params["slope_min"],
        )
        oos_bt = backtest_numba_entries_exits(
            num_oos, oos_entries, oos_exits,
            stop_pct=best_params["stop_pct"], fees=fees,
        )
    else:
        oos_bt = backtest_vbt_fold(
            num_oos, ratio_oos, best_window,
            best_params["entry_thresh"],
            best_params["exit_thresh"],
            stop_pct=best_params["stop_pct"],
            slope_min=best_params["slope_min"],
            slope_window=slope_window,
            fees=fees,
            signal_fn=signal_fn,
        )

    passed = (
        not pd.isna(oos_bt["sharpe"])
        and oos_bt["sharpe"] >= min_oos_sharpe  # Chan QT Ch.5
        and oos_bt["n_trades"] >= min_oos_trades
    )

    return {
        "pair": pair_name,
        "numerator": num,
        "denominator": den,
        "halflife": hl,
        "window": best_window,
        "entry_thresh": best_params["entry_thresh"],
        "exit_thresh": best_params["exit_thresh"],
        "stop_pct": best_params["stop_pct"],
        "slope_min": best_params["slope_min"],
        "is_sharpe": best_is_result["sharpe"],
        "is_penalized_sharpe": round(best_pen, 4),
        "is_trades": best_is_result["n_trades"],
        "oos_sharpe": oos_bt["sharpe"],
        "oos_trades": oos_bt["n_trades"],
        "passed": passed,
        "signal_method": (
            signal_fn.__name__ if signal_fn else "zscore_robust_mad"
        ),
        "optim_method": "grid_search",
    }


@register_stage("s2_optimize")
def grid_search(
    prices: pd.DataFrame,
    s1_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """Exhaustive grid search over entry/exit/stop/slope on IS data."""
    entry_grid = config.get("entry_grid", [-2.5, -2.0, -1.5])
    exit_grid = config.get("exit_grid", [0.5, 1.0, 1.5])
    stop_grid = config.get("stop_grid", [0.0])
    slope_grid = config.get("slope_grid", [0.0])
    is_ratio = config.get("is_ratio", 0.80)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)
    min_is_trades = config.get("min_is_trades", 3)
    min_oos_trades = config.get("min_oos_trades", 3)
    min_is_sharpe = config.get("min_is_sharpe", 0.5)
    min_oos_sharpe = config.get("min_oos_sharpe", 0.5)
    window_grid = config.get("window_grid", None)
    parallel = config.get("parallel", True)
    n_params = 4 + (1 if window_grid else 0)

    n_window = len(window_grid) if window_grid else 1
    n_combos = (
        n_window * len(entry_grid) * len(exit_grid)
        * len(stop_grid) * len(slope_grid)
    )
    passed_pairs = s1_result[s1_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)

    # Pre-warm Numba JIT (first call compiles, ~300ms penalty)
    if HAS_NUMBA_GRID_SWEEP and n_pairs > 0:
        _dummy = np.zeros(10, dtype=np.float64)
        _db = np.zeros(10, dtype=bool)
        try:
            from strategy_tester.backtest.vbt_runner import (
                _backtest_entries_exits_numba,
                _grid_sweep_threshold_numba,
            )
            _backtest_entries_exits_numba(_db, _db, _dummy, 0.0, 0.001)
            _grid_sweep_threshold_numba(
                _dummy, _dummy,
                np.array([1.0]), np.array([1.0]), np.array([0.0]),
                0.001, 1, 4, True,
            )
        except Exception:
            pass
        try:
            from strategy_tester.backtest.vbt_runner import (
                _grid_sweep_regime_numba,
            )
            _grid_sweep_regime_numba(
                _dummy, _dummy, _dummy, _dummy,
                np.array([1.0]), np.array([1.0]), np.array([0.0]),
                0.001, 1, 4, 25.0, 20.0,
            )
        except Exception:
            pass

    signal_module = _get_signal_module(signal_fn)
    mode = "precompute" if (
        signal_module and hasattr(signal_module, "precompute")
    ) else "per-combo"
    _log(
        f"grid_search: {n_pairs} pairs x {n_combos} combos "
        f"= {n_pairs * n_combos:,} backtests (signal: {mode})"
    )

    rows: list[dict] = []

    if parallel and n_pairs > 10:
        n_workers = min(os.cpu_count() or 4, 6)
        _log(f"grid_search: {n_workers} workers started")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_pair, prices, pr,
                    entry_grid, exit_grid, stop_grid, slope_grid,
                    is_ratio, fees, slope_window,
                    min_is_trades, min_oos_trades, n_params,
                    signal_fn, signal_module,
                    min_is_sharpe, min_oos_sharpe,
                    window_grid,
                ): pr
                for pr in passed_pairs
            }
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                if (i + 1) % 10 == 0 or i == n_pairs - 1:
                    n_pass = sum(1 for r in rows if r["passed"])
                    _log(
                        f"grid_search: {i + 1}/{n_pairs} "
                        f"({n_pass} pass so far)"
                    )
    else:
        for i, pr in enumerate(passed_pairs):
            result = _process_one_pair(
                prices, pr,
                entry_grid, exit_grid, stop_grid, slope_grid,
                is_ratio, fees, slope_window,
                min_is_trades, min_oos_trades, n_params,
                signal_fn, signal_module,
                min_is_sharpe, min_oos_sharpe,
                window_grid,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 50 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                _log(
                    f"grid_search: {i + 1}/{n_pairs} "
                    f"({n_pass} pass so far)"
                )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            "pair", "numerator", "denominator", "halflife",
            "window", "entry_thresh", "exit_thresh", "stop_pct",
            "slope_min", "is_sharpe", "is_penalized_sharpe",
            "is_trades", "oos_sharpe", "oos_trades", "passed",
            "signal_method", "optim_method",
        ]
    )
