"""Optuna TPE optimizer. Bayesian search, 50 trials.

ThreadPoolExecutor for parallel pair processing.
Pre-computes signal ONCE per pair, then only varies thresholds
in Optuna's objective function.
"""
from __future__ import annotations

import importlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import pandas as pd

from strategy_tester.backtest.vbt_runner import (
    backtest_vbt_fold,
    backtest_vbt_precomputed,
    build_is_oos_split,
    calculate_penalized_sharpe,
)
from strategy_tester.registry import register_stage


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts}  {msg}", flush=True)


def _get_signal_module(signal_fn: Callable | None):
    """Get the module of a signal function for precompute/apply."""
    if signal_fn is None:
        return None
    return importlib.import_module(signal_fn.__module__)


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
    n_trials: int,
    random_state: int,
    min_is_trades: int,
    min_oos_trades: int,
    n_params: int,
    signal_fn: Callable | None,
    signal_module,
    min_is_sharpe: float = 0.5,
    min_oos_sharpe: float = 0.5,
) -> dict | None:
    """Optuna TPE optimization for one pair."""
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pair_name = pair_row["pair"]
    num = pair_row["numerator"]
    den = pair_row["denominator"]
    hl = pair_row["halflife"]
    window = pair_row["window"]

    if num not in prices.columns or den not in prices.columns:
        return None

    p_num = prices[num].dropna()
    p_den = prices[den].dropna()
    common = p_num.index.intersection(p_den.index)
    ratio = p_num.loc[common] / p_den.loc[common]

    is_idx, oos_idx = build_is_oos_split(common, is_ratio)
    ratio_is = ratio.loc[is_idx]
    num_is = p_num.loc[is_idx]
    ratio_oos = ratio.loc[oos_idx]
    num_oos = p_num.loc[oos_idx]

    # Pre-compute signal if module supports it
    has_precompute = (
        signal_module is not None
        and hasattr(signal_module, "precompute")
        and hasattr(signal_module, "apply_thresholds")
    )

    pre_is = None
    pre_oos = None
    if has_precompute:
        pre_is = signal_module.precompute(
            ratio_is, window, slope_window,
        )
        pre_oos = signal_module.precompute(
            ratio_oos, window, slope_window,
        )

    def objective(trial: "optuna.Trial") -> float:
        entry = trial.suggest_categorical("entry", entry_grid)
        exit_ = trial.suggest_categorical("exit", exit_grid)
        stop = trial.suggest_categorical("stop", stop_grid)
        slope = trial.suggest_categorical("slope", slope_grid)

        if has_precompute:
            entries, exits = signal_module.apply_thresholds(
                pre_is, entry, exit_, slope,
            )
            bt = backtest_vbt_precomputed(
                num_is, entries, exits,
                stop_pct=stop, fees=fees,
            )
        else:
            bt = backtest_vbt_fold(
                num_is, ratio_is, window,
                entry, exit_, stop_pct=stop, slope_min=slope,
                slope_window=slope_window, fees=fees,
                signal_fn=signal_fn,
            )
        if pd.isna(bt["sharpe"]) or bt["n_trades"] < min_is_trades:
            return -999.0
        return calculate_penalized_sharpe(
            bt["sharpe"], n_params, bt["n_trades"],
        )

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
    )
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False,
    )

    if study.best_value <= -999:
        return None

    bp = study.best_params

    # Best IS result
    if has_precompute:
        is_e, is_x = signal_module.apply_thresholds(
            pre_is, bp["entry"], bp["exit"], bp["slope"],
        )
        best_is = backtest_vbt_precomputed(
            num_is, is_e, is_x,
            stop_pct=bp["stop"], fees=fees,
        )
        oos_e, oos_x = signal_module.apply_thresholds(
            pre_oos, bp["entry"], bp["exit"], bp["slope"],
        )
        oos_bt = backtest_vbt_precomputed(
            num_oos, oos_e, oos_x,
            stop_pct=bp["stop"], fees=fees,
        )
    else:
        best_is = backtest_vbt_fold(
            num_is, ratio_is, window,
            bp["entry"], bp["exit"],
            stop_pct=bp["stop"], slope_min=bp["slope"],
            slope_window=slope_window, fees=fees,
            signal_fn=signal_fn,
        )
        oos_bt = backtest_vbt_fold(
            num_oos, ratio_oos, window,
            bp["entry"], bp["exit"],
            stop_pct=bp["stop"], slope_min=bp["slope"],
            slope_window=slope_window, fees=fees,
            signal_fn=signal_fn,
        )

    # IS Sharpe gate — Bailey & LdP (2012): SR < 0.5 likely spurious
    if best_is["sharpe"] < min_is_sharpe:
        return None

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
        "window": window,
        "entry_thresh": bp["entry"],
        "exit_thresh": bp["exit"],
        "stop_pct": bp["stop"],
        "slope_min": bp["slope"],
        "is_sharpe": best_is["sharpe"],
        "is_penalized_sharpe": round(study.best_value, 4),
        "is_trades": best_is["n_trades"],
        "oos_sharpe": oos_bt["sharpe"],
        "oos_trades": oos_bt["n_trades"],
        "passed": passed,
        "signal_method": (
            signal_fn.__name__
            if signal_fn else "zscore_robust_mad"
        ),
        "optim_method": "optuna_tpe",
    }


@register_stage("s2_optimize")
def optuna_tpe(
    prices: pd.DataFrame,
    s1_result: pd.DataFrame,
    signal_fn: Callable | None = None,
    **config,
) -> pd.DataFrame:
    """Bayesian optimization via Optuna TPE sampler. 50 trials/pair."""
    entry_grid = config.get("entry_grid", [-2.5, -2.0, -1.5])
    exit_grid = config.get("exit_grid", [0.5, 1.0, 1.5])
    stop_grid = config.get("stop_grid", [0.0])
    slope_grid = config.get("slope_grid", [0.0])
    is_ratio = config.get("is_ratio", 0.80)
    fees = config.get("cost_per_side", 0.001)
    slope_window = config.get("slope_window", 2)
    n_trials = config.get("optuna_n_trials", 50)
    random_state = config.get("random_state", 42)
    min_is_trades = config.get("min_is_trades", 3)
    min_oos_trades = config.get("min_oos_trades", 3)
    min_is_sharpe = config.get("min_is_sharpe", 0.5)
    min_oos_sharpe = config.get("min_oos_sharpe", 0.5)
    parallel = config.get("parallel", True)
    n_params = 4

    passed_pairs = s1_result[s1_result["passed"]].to_dict("records")
    n_pairs = len(passed_pairs)

    signal_module = _get_signal_module(signal_fn)
    mode = "precompute" if (
        signal_module and hasattr(signal_module, "precompute")
    ) else "per-combo"
    _log(f"optuna_tpe: {n_pairs} pairs x {n_trials} trials "
         f"(signal: {mode})")

    rows: list[dict] = []

    if parallel and n_pairs > 10:
        n_workers = min(os.cpu_count() or 4, 6)
        _log(f"optuna_tpe: {n_workers} workers started")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_pair, prices, pr,
                    entry_grid, exit_grid, stop_grid, slope_grid,
                    is_ratio, fees, slope_window,
                    n_trials, random_state,
                    min_is_trades, min_oos_trades, n_params,
                    signal_fn, signal_module,
                    min_is_sharpe, min_oos_sharpe,
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
                        f"optuna_tpe: {i + 1}/{n_pairs} "
                        f"({n_pass} pass so far)"
                    )
    else:
        for i, pr in enumerate(passed_pairs):
            result = _process_one_pair(
                prices, pr,
                entry_grid, exit_grid, stop_grid, slope_grid,
                is_ratio, fees, slope_window,
                n_trials, random_state,
                min_is_trades, min_oos_trades, n_params,
                signal_fn, signal_module,
                min_is_sharpe, min_oos_sharpe,
            )
            if result is not None:
                rows.append(result)
            if (i + 1) % 50 == 0 or i == n_pairs - 1:
                n_pass = sum(1 for r in rows if r["passed"])
                _log(
                    f"optuna_tpe: {i + 1}/{n_pairs} "
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
